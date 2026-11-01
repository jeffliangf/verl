"""Check this Slurm canary once a minute, independently of the chat session."""

import argparse
import datetime
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from canary_ops import atomic_json, evaluation_status, verify_checkpoint
from run_budget import final_target, required_evaluations

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
LIMITS = {"startup": 2700, "training": 5400, "checkpoint": 3600, "evaluation": 14400, "finishing": 900}
MIN_FREE_BYTES = 450 * 1024**3


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(command):
    return subprocess.run(command, check=True, capture_output=True, text=True, timeout=20).stdout.strip()


def scheduler(job):
    try:
        queued = run(["squeue", "--noheader", f"--jobs={job}", "--format=%T|%M|%l|%R|%S"])
    except subprocess.CalledProcessError as exc:
        # Slurm removes terminal jobs from squeue before accounting expires.
        # Other query failures must remain monitoring errors, not job failures.
        if "Invalid job id specified" not in (exc.stderr or ""):
            raise
        queued = ""
    if queued:
        state, elapsed, limit, reason, start = queued.splitlines()[0].split("|", 4)
        result = {"state": state, "elapsed": elapsed, "limit": limit, "nodes_or_reason": reason}
        if state == "RUNNING" and start != "Unknown":
            result["started_at"] = (
                datetime.datetime.fromisoformat(start).replace(tzinfo=datetime.timezone.utc).timestamp()
            )
        return result
    rows = run(
        ["sacct", "-X", "--noheader", "--parsable2", f"--jobs={job}", "--format=State,ExitCode,Elapsed"]
    ).splitlines()
    if not rows:
        return {"state": "UNKNOWN"}
    state, code, elapsed = rows[0].split("|")[:3]
    return {"state": state.split()[0], "exit_code": code, "elapsed": elapsed}


def json_lines(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_bytes().splitlines():
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            pass
    return rows


def progress(root, job, running_since):
    """Use actual rollout/metric/checkpoint progress, never periodic GPU log writes."""
    target = final_target(root, 20)
    candidates = [(running_since, "startup", 0)]
    latest_step, bad_metric, last_actor_step, last_actor_metrics = 0, None, 0, {}
    # Older attempts can include updates lost since the last checkpoint.
    # Only this job's metrics describe its completed updates and health.
    for path in root.glob(f"metrics-{job}.jsonl"):
        for row in json_lines(path):
            step = int(row["step"])
            latest_step = max(latest_step, step)
            if "actor/loss" in row.get("data", {}) and step >= last_actor_step:
                last_actor_step, last_actor_metrics = step, row["data"]
            for key in ("actor/loss", "actor/grad_norm"):
                value = row.get("data", {}).get(key)
                if key in row.get("data", {}) and (value is None or not math.isfinite(value)):
                    bad_metric = {"step": step, "metric": key, "value": str(value)}
        if path.stat().st_size:
            candidates.append((path.stat().st_mtime, "finishing" if latest_step >= target else "training", latest_step))
    dispatches = json_lines(root / "timings.jsonl")
    for row in dispatches:
        if row.get("stage") == "dispatch" and row.get("event") == "start":
            stamp = datetime.datetime.fromisoformat(row["timestamp"]).timestamp()
            candidates.append((stamp, "evaluation" if row.get("validate") else "training", int(row["policy_step"])))
    log = root / f"slurm-{job}.log"
    if log.exists():
        # Checkpoint start carries its own timestamp; regular Ray statistics do not reset this clock.
        text = log.read_text(errors="replace")
        pattern = r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3}):Saving checkpoint to .*global_step_(\d+)"
        for stamp, step in re.findall(pattern, text):
            when = (
                datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S,%f")
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
            )
            candidates.append((when, "checkpoint", int(step)))
    # NFS can make a step-N metrics write appear newer than dispatch N+1.
    # Compare policy steps first, timestamps within a step. Exclude events
    # before this allocation so a resumed job does not inherit stale phases.
    current = [candidate for candidate in candidates if candidate[0] >= running_since]
    started, phase, step = max(current, key=lambda candidate: (candidate[2], candidate[0]))
    latest_activity = started
    if phase == "training":
        # Replacement groups belong to the same optimizer step. Their dispatch
        # proves liveness but must not grant another full training deadline.
        started = min(stamp for stamp, kind, number in current if kind == phase and number == step)
    evaluations = [
        evaluation_status(root, step_number)
        for step_number in required_evaluations(root, target)
        if step_number <= max(latest_step, step)
    ]
    completed_evals = [result["step"] for result in evaluations if result["complete"]]
    marker = root / "checkpoints/latest_checkpointed_iteration.txt"
    try:
        checkpoint_step = int(marker.read_text())
    except (OSError, ValueError):
        checkpoint_step = None
    # Stage logs provide liveness while an evaluation is processing 200 prompts.
    useful_times = [latest_activity]
    partition = "val" if phase == "evaluation" else "train"
    active = root / partition / f"step_{step}"
    if active.exists():
        useful_times.extend(p.stat().st_mtime for p in active.rglob("timings*.jsonl"))
    fatal_failure = None
    if active.exists():
        for failure_path in active.glob("*/failure.json"):
            if failure_path.stat().st_mtime >= running_since:
                failure = json.loads(failure_path.read_text())
                if failure.get("dropped") is False and failure.get("stage") == "internal":
                    fatal_failure = {"path": str(failure_path), "error": failure.get("error")}
                    break
    budget_path = root / "run-budget.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else {}
    return {
        "target_step": target,
        "budget_deadline_epoch": budget.get("deadline_epoch"),
        "phase": phase,
        "phase_step": step,
        "phase_started": started,
        "last_progress_at": max(useful_times),
        "last_logged_step": latest_step,
        "checkpoint_step": checkpoint_step,
        "completed_evaluations": completed_evals,
        "evaluations": evaluations,
        "degraded_evaluations": [result["step"] for result in evaluations if result["complete"] and result["degraded"]],
        "nonfinite_metric": bad_metric,
        "fatal_failure": fatal_failure,
        "last_actor_step": last_actor_step,
        "last_actor_metrics": {
            key: last_actor_metrics[key]
            for key in (
                "actor/loss",
                "actor/grad_norm",
                "actor/perf/max_memory_allocated_gb",
                "actor/perf/cpu_memory_used_gb",
                "training/reward_extra/valid_call/mean",
                "timing_s/step",
            )
            if key in last_actor_metrics
        },
    }


def stop_reason(state, observed, now, free_bytes):
    if state in {"RUNNING", "PENDING"} and observed.get("budget_deadline_epoch") is not None and now >= observed["budget_deadline_epoch"]:
        return "72-hour continuation allocation budget exhausted"
    if state != "RUNNING":
        return None
    if observed.get("fatal_failure"):
        return "Trainer recorded a fatal internal group error; this step cannot complete"
    if observed.get("nonfinite_metric"):
        return "Nonfinite actor metric"
    if free_bytes < MIN_FREE_BYTES:
        return "Less than 450 GiB free; preserving space for checkpoint recovery"
    phase = observed["phase"]
    if now - observed["phase_started"] > LIMITS[phase]:
        return f"{phase} exceeded {LIMITS[phase] // 60} minute deadline"
    if phase != "startup" and now - observed["last_progress_at"] > 2400:
        return "No meaningful stage progress for 40 minutes"
    return None


def cancel_owned_job(root, manifest, reason):
    job = str(manifest["job_id"])
    # Refuse to cancel any job whose owner, name, or batch script is not this canary.
    details = run(["scontrol", "show", "job", job, "--oneliner"])
    expected = [
        f"JobId={job} ",
        f"JobName={manifest['job_name']} ",
        f"UserId=jeff({os.getuid()}) ",
        f"Command={root / 'train.sbatch'} ",
    ]
    if not all(value in details + " " for value in expected):
        raise RuntimeError("Job identity mismatch; refusing cancellation")
    atomic_json(
        root / "watchdog-alert.json",
        {"time": utc(), "job_id": job, "reason": reason, "action": "cancelling this canary only"},
    )
    run(["scancel", job])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "launch.json").read_text())
    if manifest.get("controller_host") not in (None, os.uname().nodename):
        print(json.dumps({"action": "skipped", "reason": "Watchdog is assigned to another host"}), flush=True)
        return
    job = str(manifest["job_id"])
    with (root / "watchdog.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_file = root / "watchdog-state.json"
        previous = json.loads(state_file.read_text()) if state_file.exists() else {}
        # Shared filesystem metadata can briefly expose a prior attempt's state
        # after it has been archived. Never inherit its lifecycle timestamps.
        if str(previous.get("job_id")) != job:
            previous = {}
        running_since = previous.get("running_since") or (datetime.datetime.fromisoformat(manifest["submitted_at"]).timestamp() if manifest.get("submitted_at") else None)
        terminal_since = previous.get("terminal_since") if previous.get("state") in TERMINAL else None
        consecutive_errors = 0
        while True:
            now = time.time()
            status = {
                "checked_at": utc(),
                "job_id": job,
                "monitor_pid": os.getpid(),
                "check_interval_seconds": 60,
                "monitor_host": os.uname().nodename,
                "running_since": running_since,
                "terminal_since": terminal_since,
            }
            try:
                status.update(scheduler(job))
                if status["state"] not in TERMINAL:
                    terminal_since = None
                    status["terminal_since"] = None
                if status["state"] == "RUNNING":
                    running_since = status.get("started_at", running_since or now)
                status["running_since"] = running_since
                observed = progress(root, job, running_since or now)
                status.update(observed)
                status["phase_age_seconds"] = round(now - observed["phase_started"])
                status["progress_age_seconds"] = round(now - observed["last_progress_at"])
                free = shutil.disk_usage(root).free
                for name, specification in manifest.get("bulk_storage", {}).items():
                    reserve = specification["reserve_bytes"]
                    path = root / name
                    if not os.path.ismount(path):
                        observed["fatal_failure"] = f"Bulk-storage mount missing: {path}"
                    elif shutil.disk_usage(path).free < reserve:
                        observed["fatal_failure"] = f"Bulk-storage reserve exhausted: {path}"
                status["fatal_failure"] = observed.get("fatal_failure")
                status["disk_free_gib"] = round(free / 1024**3, 2)
                reason = stop_reason(status["state"], observed, now, free)
                status["stop_reason"] = reason
                if reason and not args.dry_run:
                    cancel_owned_job(root, manifest, reason)
                terminal = status["state"] in TERMINAL
                if terminal:
                    terminal_since = terminal_since or now
                    status["terminal_since"] = terminal_since
                    # Allow shared filesystem logs to become visible after Slurm exits.
                    if now - terminal_since >= 180:
                        passed = (
                            status["state"] == "COMPLETED"
                            and status.get("exit_code") == "0:0"
                            and observed["last_logged_step"] == observed["target_step"]
                            and observed["completed_evaluations"] == required_evaluations(root, observed["target_step"])
                        )
                        if passed:
                            try:
                                status["final_checkpoint"] = verify_checkpoint(
                                    root / "checkpoints",
                                    observed["target_step"],
                                    manifest["nodes"] * manifest["gpus_per_node"],
                                )
                            except Exception as exc:
                                passed = False
                                status["checkpoint_verification_error"] = f"{type(exc).__name__}: {exc}"
                        status["target_completed"] = passed
                        passed = passed and not observed["degraded_evaluations"]
                        status["canary_passed"] = passed
                        atomic_json(root / "final-status.json", status)
                        if not passed and not (root / "watchdog-alert.json").exists():
                            atomic_json(
                                root / "watchdog-alert.json",
                                {
                                    "time": utc(),
                                    "job_id": job,
                                    "reason": "Job ended before all canary success checks passed",
                                    "action": "inspect logs",
                                },
                            )
                        atomic_json(state_file, status)
                        print(json.dumps(status), flush=True)
                        return
                consecutive_errors = 0
            except Exception as exc:
                status["monitor_error"] = f"{type(exc).__name__}: {exc}"
                # A failed scheduler query is not evidence that training is stuck.
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    atomic_json(
                        root / "monitor-warning.json",
                        {
                            "time": utc(),
                            "job_id": job,
                            "reason": status["monitor_error"],
                            "consecutive_errors": consecutive_errors,
                        },
                    )
            atomic_json(state_file, status)
            print(json.dumps(status), flush=True)
            if args.once:
                return
            time.sleep(60)


if __name__ == "__main__":
    main()
