"""Persist experiment health every 30 minutes and recover classified transient failures."""
import argparse
import csv
import datetime
import json
import math
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from canary_ops import atomic_json
from run_budget import final_target
from watchdog import TERMINAL, json_lines, scheduler
from recover import recover


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def collect(root):
    manifest = json.loads((root / "launch.json").read_text())
    job = str(manifest["job_id"])
    state = scheduler(job)
    initial, target = manifest.get("recovery_checkpoint_step", manifest["initial_checkpoint_step"]), final_target(root)
    records = {
        int(row["step"]): row["data"] for row in json_lines(root / f"metrics-{job}.jsonl")
        if initial < int(row["step"]) <= target and "timing_s/step" in row.get("data", {})
    }
    dispatches, coverage = defaultdict(list), {}
    for row in json_lines(root / "timings.jsonl"):
        step = row.get("policy_step")
        if row.get("stage") == "dispatch" and row.get("event") == "end" and not row.get("validate"):
            dispatches[step].append(row)
        if row.get("stage") == "judge_coverage" and row.get("event") == "summarized":
            coverage[step] = row
    covered = [row for step, row in coverage.items() if step in records]
    comparisons = sum(row["comparisons"] for row in covered)
    fallbacks = sum(row["fallbacks"] for row in covered)
    watchdog_path = root / "watchdog-state.json"
    watchdog = json.loads(watchdog_path.read_text()) if watchdog_path.exists() else {}
    durations = [row["timing_s/step"] for row in records.values()]
    latest = records[max(records)] if records else {}
    peaks, gpu_last = {}, {}
    for path in root.glob(f"gpus-{job}-*.csv"):
        with path.open() as handle:
            for row in csv.DictReader(handle, skipinitialspace=True):
                try:
                    gpu = f"{path.stem.removeprefix(f'gpus-{job}-')}:{row['index']}"
                    used = float(row["memory.used [MiB]"].split()[0]) / 1024
                    peaks[gpu] = max(peaks.get(gpu, 0), used)
                    gpu_last[gpu] = row
                except (KeyError, TypeError, ValueError):
                    continue  # A sampling line may still be in flight.
    nonfinite = [
        {"step": step, "metric": key, "value": str(row[key])}
        for step, row in records.items() for key in ("actor/loss", "actor/grad_norm")
        if key in row and (row[key] is None or not math.isfinite(row[key]))
    ]
    final_path = root / "final-status.json"
    final = json.loads(final_path.read_text()) if final_path.exists() else None
    budget_path = root / "run-budget.json"
    budget = json.loads(budget_path.read_text()) if budget_path.exists() else None
    return {
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_id": job, "scheduler": state, "initial_step": initial,
        "target_step": target if (root / "finish-request.json").exists() else None,
        "step_ceiling": manifest["total_training_steps"],
        "requested_duration_hours": manifest["requested_duration_hours"], "allocation_budget": budget,
        "group_size": manifest["group_size"], "judge_concurrency": manifest["judge_concurrency"],
        "judge_streaming": manifest.get("judge_streaming", False),
        "completed_updates": len(records), "completed_steps": sorted(records),
        "latest_step": max(records, default=initial),
        "steps_needing_replacement": [step for step in sorted(records) if len(dispatches[step]) > 1],
        "judge_comparisons": comparisons, "judge_fallbacks": fallbacks,
        "judge_fallback_fraction": fallbacks / comparisons if comparisons else None,
        "step_seconds": {
            "count": len(durations), "mean": statistics.mean(durations) if durations else None,
            "p50": quantile(durations, 0.5), "p75": quantile(durations, 0.75),
            "p99": quantile(durations, 0.99), "maximum": max(durations, default=None),
            "note": "Includes checkpoint saves and excludes evaluation; tail percentiles need a large sample.",
        },
        "latest_actor": {key: value for key, value in latest.items() if key.startswith("actor/")},
        "nonfinite_metrics": nonfinite, "peak_gpu_memory_gib": peaks, "last_gpu_samples": gpu_last,
        "watchdog": watchdog,
        "watchdog_age_seconds": round(time.time() - watchdog_path.stat().st_mtime) if watchdog_path.exists() else None,
        "final_status": final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "launch.json").read_text())
    report = collect(root)
    reports = root / "checks-30min"
    reports.mkdir(exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    atomic_json(reports / f"{stamp}.json", report)
    atomic_json(root / "check-latest.json", report)
    print(json.dumps({key: report[key] for key in (
        "checked_at", "job_id", "scheduler", "latest_step", "completed_updates",
        "steps_needing_replacement", "judge_fallback_fraction", "step_seconds", "peak_gpu_memory_gib",
    )}), flush=True)
    try:
        recovery = recover(root, report)
    except Exception as exc:
        recovery = {"action": "blocked", "reason": f"{type(exc).__name__}: {exc}"}
        atomic_json(root / "recovery-state.json", recovery)
    print(json.dumps({"recovery": recovery}), flush=True)
    if recovery["action"] in {"complete", "budget_exhausted", "stopped"}:
        subprocess.run(["systemctl", "--user", "disable", "--now", manifest["check_timer"]], check=True)


if __name__ == "__main__":
    main()
