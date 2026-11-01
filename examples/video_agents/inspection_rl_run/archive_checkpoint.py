"""Keep one verified checkpoint copy on the controller, independently of the GPU node."""

import argparse
import fcntl
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from canary_ops import atomic_json, retain_latest, verify_checkpoint


def archive(root, backup):
    marker = root / "checkpoints/verified.json"
    if not marker.exists():
        return {"status": "waiting_for_checkpoint"}
    current = json.loads(marker.read_text())
    step = int(current["step"])
    complete = backup / "verified.json"
    if complete.exists() and int(json.loads(complete.read_text())["step"]) >= step:
        return {"status": "current", "step": step, "backup": str(backup)}
    manifest = json.loads((root / "launch.json").read_text())
    storage = manifest["bulk_storage"]["checkpoints"]
    destination = backup / f"global_step_{step}"
    # An interrupted incoming copy is not a verified backup. Reclaim only that
    # exact directory; the prior verified checkpoint remains until replacement.
    if destination.exists():
        if destination.is_symlink() or destination.resolve().parent != backup.resolve():
            raise ValueError("Unsafe incoming backup directory")
        shutil.rmtree(destination)
    if shutil.disk_usage(backup).free < current["bytes"] + 10 * 1024**3:
        raise RuntimeError("Controller backup lacks space for a new complete copy; keeping the prior backup")
    started = time.time()
    status = {"status": "copying", "step": step, "destination": str(destination), "started_epoch": started}
    atomic_json(root / "checkpoint-backup.json", status)
    ssh = shlex.join([
        "ssh", "-i", "/home/jeff/.ssh/gmi", "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
    ])
    subprocess.run([
        "rsync", "-a", "--whole-file", "--ignore-times", "--partial", "-e", ssh,
        f"jeff@{storage['address']}:{storage['path']}/global_step_{step}/", str(destination) + "/",
    ], check=True, timeout=3 * 3600)
    verified = verify_checkpoint(backup, step, manifest["nodes"] * manifest["gpus_per_node"])
    if verified["bytes"] != current["bytes"]:
        raise ValueError("The backup size differs from the verified source checkpoint")
    temporary = backup / ".latest.tmp"
    temporary.write_text(str(step) + "\n")
    temporary.replace(backup / "latest_checkpointed_iteration.txt")
    result = retain_latest(backup, step, manifest["nodes"] * manifest["gpus_per_node"], keep=1)
    status.update(status="verified", bytes=result["bytes"], completed_epoch=time.time(),
                  seconds=time.time() - started, backup=str(backup))
    atomic_json(root / "checkpoint-backup.json", status)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.backup_dir.mkdir(parents=True, exist_ok=True)
    with (args.backup_dir / ".archive.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                result = archive(args.run_dir, args.backup_dir)
            except Exception as error:
                result = {"status": "failed", "error": f"{type(error).__name__}: {error}", "time": time.time()}
                atomic_json(args.run_dir / "checkpoint-backup.json", result)
            print(json.dumps(result), flush=True)
            if args.once:
                return
            final = args.run_dir / "final-status.json"
            if final.exists() and json.loads(final.read_text()).get("target_completed"):
                if result["status"] in {"current", "verified"}:
                    return
            time.sleep(60)


if __name__ == "__main__":
    main()
