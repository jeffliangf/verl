"""Validate the frozen launch and start a single 72-hour allocation budget."""
import datetime
import hashlib
import json
import os
import socket
import subprocess
import time
from pathlib import Path

from canary_ops import atomic_json
from run_budget import initialize

root = Path(os.environ["RL_RUN_DIR"])
manifest = json.loads((root / "launch.json").read_text())
code = Path(os.environ["RL_CHECKOUT"])
assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code).decode().strip() == manifest["code_commit"]
assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=code)
for name, digest in manifest["launcher_sha256"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, f"Launch file changed: {name}"
assert int(os.environ["SLURM_JOB_NUM_NODES"]) == 2
hosts = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).splitlines()
assert set(hosts) == set(manifest["requested_nodes"]), "Allocation differs from the requested H100 nodes"
budget = initialize(root, os.environ["SLURM_JOB_ID"])
assert budget["budget_seconds"] == 72 * 3600
# A recovery can spend hours queued. Bound Slurm itself by the original
# deadline, even if the controller-side watchdog later becomes unavailable.
allocation_start = subprocess.check_output(
    ["squeue", "-h", "-j", os.environ["SLURM_JOB_ID"], "-o", "%S"], text=True, timeout=20
).strip()
allocation_epoch = datetime.datetime.fromisoformat(allocation_start).replace(tzinfo=datetime.timezone.utc).timestamp()
remaining_minutes = int((budget["deadline_epoch"] - allocation_epoch) // 60)
assert 0 < remaining_minutes <= 72 * 60, "No valid allocation time remains"
subprocess.run(
    ["scontrol", "update", f"JobId={os.environ['SLURM_JOB_ID']}", f"TimeLimit={remaining_minutes}"],
    check=True, timeout=20,
)
atomic_json(root / f"allocation-start-{os.environ['SLURM_JOB_ID']}.json", {
    "job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
    "started_at_epoch": time.time(), "original_deadline_epoch": budget["deadline_epoch"],
    "code_commit": manifest["code_commit"], "initial_checkpoint_step": manifest["initial_checkpoint_step"],
    "slurm_time_limit_minutes": remaining_minutes,
})
print("CONTINUATION_START " + json.dumps({
    "job_id": os.environ["SLURM_JOB_ID"], "node": socket.gethostname(),
    "initial_step": manifest["initial_checkpoint_step"], "budget_hours": 72,
    "deadline_epoch": budget["deadline_epoch"],
}), flush=True)
