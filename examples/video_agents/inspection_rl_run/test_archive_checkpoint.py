import json
import shutil
import zipfile
from pathlib import Path

import pytest

import archive_checkpoint as archive
from canary_ops import atomic_json, verify_checkpoint


def checkpoint(root, step):
    directory = root / f"global_step_{step}"
    (directory / "actor").mkdir(parents=True)
    atomic_json(directory / "actor/fsdp_config.json", {"world_size": 2})
    files = [directory / "data.pt"]
    files += [directory / "actor" / f"{kind}_world_size_2_rank_{rank}.pt"
              for kind in ("model", "optim", "extra_state") for rank in range(2)]
    for file in files:
        with zipfile.ZipFile(file, "w") as output:
            output.writestr("archive/data.pkl", f"step {step}")
    atomic_json(root / "verified.json", verify_checkpoint(root, step, 2))


def setup(tmp_path, monkeypatch):
    root, backup = tmp_path / "run", tmp_path / "backup"
    root.mkdir(); backup.mkdir()
    atomic_json(root / "launch.json", {
        "nodes": 2, "gpus_per_node": 1,
        "bulk_storage": {"checkpoints": {"address": "storage.invalid", "path": str(root / "checkpoints")}},
    })
    def copy(command, **kwargs):
        source = Path(command[-2].split(":", 1)[1])
        shutil.copytree(source, command[-1], dirs_exist_ok=True)
    monkeypatch.setattr(archive.subprocess, "run", copy)
    return root, backup


def test_backup_verifies_before_retaining_only_latest(tmp_path, monkeypatch):
    root, backup = setup(tmp_path, monkeypatch)
    checkpoint(root / "checkpoints", 5)
    assert archive.archive(root, backup)["status"] == "verified"
    assert archive.archive(root, backup)["status"] == "current"
    checkpoint(root / "checkpoints", 10)
    assert archive.archive(root, backup)["step"] == 10
    assert not (backup / "global_step_5").exists()
    assert (root / "checkpoints/global_step_5").exists()
    assert verify_checkpoint(backup, 10, 2)["bytes"] > 0


def test_failed_copy_keeps_prior_verified_checkpoint(tmp_path, monkeypatch):
    root, backup = setup(tmp_path, monkeypatch)
    checkpoint(root / "checkpoints", 5)
    archive.archive(root, backup)
    checkpoint(root / "checkpoints", 10)
    def failed(*args, **kwargs):
        raise RuntimeError("copy interrupted")
    monkeypatch.setattr(archive.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="copy interrupted"):
        archive.archive(root, backup)
    assert json.loads((backup / "verified.json").read_text())["step"] == 5
    assert verify_checkpoint(backup, 5, 2)["bytes"] > 0
