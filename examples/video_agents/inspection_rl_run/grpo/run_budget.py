"""One immutable 72-hour allocation budget, shared by every recovery attempt."""

import argparse
import datetime
import fcntl
import json
import math
import subprocess
import time
from pathlib import Path

from canary_ops import atomic_json


def initialize(root, job_id, started_at=None):
    root = Path(root)
    with (root / "run-budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "run-budget.json"
        if path.exists():
            budget = json.loads(path.read_text())
            if budget["deadline_epoch"] <= time.time():
                raise RuntimeError("The original 72-hour experiment budget has expired")
            return budget
        if started_at is None:
            value = subprocess.check_output(
                ["squeue", "-h", "-j", str(job_id), "-o", "%S"], text=True, timeout=20
            ).strip()
            started_at = datetime.datetime.fromisoformat(value).replace(tzinfo=datetime.timezone.utc).timestamp()
        budget = {
            "first_job_id": str(job_id),
            "started_at_epoch": started_at,
            "deadline_epoch": started_at + 72 * 3600,
            "budget_seconds": 72 * 3600,
            "finish_reserve_seconds": 7200,
            "scope": "Elapsed from first allocation, including recovery downtime; never reset on relaunch",
        }
        atomic_json(path, budget)
        return budget


def final_target(root, default=1000):
    root = Path(root)
    stop = root / "finish-request.json"
    if stop.exists():
        return int(json.loads(stop.read_text())["final_step"])
    launch = root / "launch.json"
    return int(json.loads(launch.read_text()).get("total_training_steps", default)) if launch.exists() else default


def required_evaluations(root, target=None):
    root = Path(root)
    manifest = json.loads((root / "launch.json").read_text()) if (root / "launch.json").exists() else {}
    target = final_target(root, 20) if target is None else target
    initial = int(manifest.get("initial_checkpoint_step", 0))
    frequency = int(manifest.get("test_freq", 5))
    result = [initial] if manifest.get("evaluate_initial", False) else []
    result.extend(range((initial // frequency + 1) * frequency, target + 1, frequency))
    if target > initial and target not in result:
        result.append(target)
    return result


def finish_decision(root, step, save_frequency, now=None):
    root = Path(root)
    now = time.time() if now is None else now
    target = final_target(root)
    budget = json.loads((root / "run-budget.json").read_text())
    # Forecast the next checkpoint interval from actual complete step timings,
    # including bounded refills. The Slurm limit remains the hard upper bound.
    durations = []
    for file in root.glob("metrics-*.jsonl"):
        for line in file.read_text().splitlines():
            try:
                row = json.loads(line)
                value = row.get("data", {}).get("timing_s/step")
                if max(0, step - 10) <= row["step"] <= step and value is not None and math.isfinite(value):
                    durations.append(float(value))
            except (ValueError, KeyError):
                continue
    estimated_interval = save_frequency * max([1200.0, *durations]) * 1.25
    remaining = budget["deadline_epoch"] - now
    requested = (root / "stop-request.json").exists()
    should_finish = step >= target or requested or remaining <= budget["finish_reserve_seconds"] + estimated_interval
    return {
        "finish": should_finish,
        "final_step": step,
        "reason": "requested stop at verified checkpoint"
        if requested
        else "step ceiling reached"
        if step >= target
        else "reserve time for final evaluation before 72-hour limit",
        "remaining_seconds": remaining,
        "estimated_next_checkpoint_interval_seconds": estimated_interval,
        "finish_reserve_seconds": budget["finish_reserve_seconds"],
        "deadline_epoch": budget["deadline_epoch"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if args.initialize:
        if not args.job_id:
            parser.error("--job-id is required for initialization")
        print(json.dumps(initialize(args.run_dir, args.job_id)), flush=True)


if __name__ == "__main__":
    main()
