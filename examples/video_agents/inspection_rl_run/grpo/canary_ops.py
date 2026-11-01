"""Checkpoint verification and conservative retention for this canary only."""

import json
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
            name = stream.name
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def evaluation_status(root, step, expected=200, min_valid_fraction=0.9):
    """Require the whole cohort to settle and the trainer to publish its valid rows."""
    root = Path(root)
    status = {"step": step, "complete": False, "valid": None, "dropped": None, "degraded": False}
    try:
        rows = [
            json.loads(line)
            for line in (root / "validation" / f"{step}.jsonl").read_text().splitlines()
            if line.strip()
        ]
        if not all(isinstance(row, dict) for row in rows):
            return status
        summary_path = root / "val" / f"step_{step}" / "eval-summary.json"
        if not summary_path.exists():
            # Compatibility with earlier attempts that required every prompt to score.
            if len(rows) == expected:
                status.update(complete=True, valid=expected, dropped=0)
            return status
        summary = json.loads(summary_path.read_text())
        valid, dropped = summary["valid"], summary["dropped"]
        if any(type(value) is not int for value in (summary["policy_step"], summary["prompts"], valid, dropped)):
            return status
        examples = summary["dropped_examples"]
        if (
            summary["policy_step"] != step
            or summary["prompts"] != expected
            or not 0 < valid <= expected
            or dropped != expected - valid
            or len(examples) != dropped
            or len({row["sample_id"] for row in examples}) != dropped
        ):
            return status
        status.update(
            complete=len(rows) == valid,
            valid=valid,
            dropped=dropped,
            degraded=bool(summary.get("degraded")) or valid / expected < min_valid_fraction,
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return status


def verify_checkpoint(root, step, world_size):
    root = Path(root).resolve()
    checkpoint = root / f"global_step_{step}"
    if checkpoint.is_symlink() or checkpoint.resolve().parent != root:
        raise ValueError("Checkpoint must be a real child of this run's checkpoint directory")
    actor = checkpoint / "actor"
    if actor.is_symlink():
        raise ValueError("Refusing a linked actor directory")
    config = json.loads((actor / "fsdp_config.json").read_text())
    if config["world_size"] != world_size:
        raise ValueError("Checkpoint world size does not match the canary")
    paths = [checkpoint / "data.pt"]
    paths += [
        actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        for kind in ("model", "optim", "extra_state")
    ]
    total = 0
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Unsafe checkpoint file: {path}")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Checkpoint file is not fully visible: {path}")
        with zipfile.ZipFile(path) as archive:
            if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                raise ValueError(f"Invalid PyTorch archive: {path}")
        total += path.stat().st_size
    return {
        "step": step,
        "path": str(checkpoint),
        "archives_checked": len(paths),
        "bytes": total,
        "verified_at": time.time(),
        "scope": "Required shards and archive directories; not a tensor CRC or GPU restore test",
    }


def retain_latest(root, step, world_size, keep=1):
    """Prune only this run, after verifying the checkpoints that will survive."""
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("Checkpoint retention must be a positive integer")
    if Path(root).is_symlink():
        raise ValueError("Refusing a linked checkpoint root")
    root = Path(root).resolve()
    verified = verify_checkpoint(root, step, world_size)
    marker = root / "latest_checkpointed_iteration.txt"
    if int(marker.read_text().strip()) != step:
        raise ValueError("Latest checkpoint marker does not match the completed save")
    previous = []
    for path in root.glob("global_step_*"):
        match = re.fullmatch(r"global_step_(\d+)", path.name)
        if match and int(match[1]) < step:
            if path.is_symlink() or path.resolve().parent != root or not path.is_dir():
                raise ValueError(f"Refusing to delete unsafe checkpoint path: {path}")
            previous.append(path)
    previous.sort(key=lambda path: int(path.name.removeprefix("global_step_")), reverse=True)
    retained = previous[:keep - 1]
    for path in retained:
        verify_checkpoint(root, int(path.name.removeprefix("global_step_")), world_size)
    victims = previous[keep - 1:]
    verified["retained_checkpoints"] = [f"global_step_{step}", *[path.name for path in retained]]
    # Commit a durable verification record before deleting any previous state.
    atomic_json(root / "verified.json", verified)
    for path in victims:
        shutil.rmtree(path)
    verified["deleted_previous_checkpoints"] = [path.name for path in victims]
    atomic_json(root / "verified.json", verified)
    return verified
