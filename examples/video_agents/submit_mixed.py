# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Submit a frozen mixed run only after complete native-data and gateway checks."""

import argparse
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT
from .data_preflight import require_preflight
from .h3 import server_status
from .mixed_group16 import require_h3_pool
from .storage import file_sha256, read_json, write_json
from .tournament import load_protocol


def submit(run, budget_from=None):
    manifest = read_json(run / "launch.json")
    if manifest.get("job_id"):
        raise ValueError("This attempt already has a scheduler job; use a fresh attempt directory")
    require_preflight(run)
    checkout = Path(manifest["code_checkout"])
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip() != manifest[
        "code_commit"
    ] or subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout):
        raise ValueError("The frozen runtime checkout changed")
    for name, digest in manifest["input_sha256"].items():
        if file_sha256(Path(name)) != digest:
            raise ValueError(f"Launch input changed: {name}")
    status = server_status(load_protocol(run / "protocol.json"))
    require_h3_pool(status, manifest["expected_h3_workers"])
    write_json(run.parent / "h3-prelaunch-status.json", status)
    if budget_from is not None:
        if (run / "run-budget.json").exists():
            raise ValueError("An attempt's allocation budget cannot be overwritten")
        write_json(run / "run-budget.json", read_json(budget_from))
    minutes = manifest["requested_duration_hours"] * 60
    if (run / "run-budget.json").exists():
        remaining = read_json(run / "run-budget.json")["deadline_epoch"] - time.time()
        if remaining <= manifest["finish_reserve_seconds"]:
            raise ValueError("The original allocation budget has reached its finish reserve")
        minutes = min(minutes, math.floor(remaining / 60))
    template = ROOT / "examples/video_agents/grpo_group16_run/systemd/video-agents-grpo-group16-16h100-20261104.service"
    previous = read_json(Path(manifest["source_configuration"]).parent / "launch.json")
    unit = (
        template.read_text()
        .replace(str(Path(previous["run_dir"]).parent), str(run.parent))
        .replace(previous["code_checkout"], str(checkout))
        .replace("Monitor fresh Qwen group-16 GRPO run", "Monitor mixed Qwen group-16 GRPO run")
    )
    unit_source = run.parent / "systemd" / manifest["watchdog_service"]
    unit_source.parent.mkdir(parents=True, exist_ok=True)
    unit_source.write_text(unit)
    unit_target = Path.home() / ".config/systemd/user" / manifest["watchdog_service"]
    if unit_target.exists() and unit_target.read_text() != unit:
        raise ValueError("Unexpected existing watchdog service")
    shutil.copyfile(unit_source, unit_target)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    command = [
        "sbatch",
        "--parsable",
        "--job-name=" + manifest["job_name"],
        "--nodelist=" + ",".join(manifest["requested_nodes"]),
        f"--time={minutes}",
        "--output=" + str(run / "slurm-%j.log"),
        manifest["batch_script"],
        str(run),
    ]
    job_id = int(subprocess.check_output(command, text=True).strip().split(";", 1)[0])
    submitted_at = datetime.now(timezone.utc).isoformat()
    manifest.update(job_id=job_id, submitted_at=submitted_at, slurm_time_limit_minutes=minutes)
    write_json(run / "launch.json", manifest)
    write_json(
        run / "submission.json",
        {
            "job_id": job_id,
            "command": command,
            "submitted_at": submitted_at,
            "data_preflight": str(run / "data-preflight.json"),
            "allocation_budget_inherited_from": str(budget_from) if budget_from else None,
        },
    )
    subprocess.run(["systemctl", "--user", "start", manifest["watchdog_service"]], check=True)
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-from", type=Path)
    args = parser.parse_args()
    print(f"Submitted Slurm job {submit(args.run_dir.resolve(), args.budget_from)}", flush=True)


if __name__ == "__main__":
    main()
