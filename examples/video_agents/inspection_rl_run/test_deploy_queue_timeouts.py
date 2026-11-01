"""Verify checkpoint gating and isolation of the planned timeout deployment."""

import pytest

import deploy_queue_timeouts as deploy
from canary_ops import atomic_json


@pytest.fixture
def run(tmp_path, monkeypatch):
    plan = {
        "status": "waiting",
        "source_job": 3573,
        "source_commit": "before",
        "target_commit": "after",
        "code_checkout": str(tmp_path / "code"),
        "staged_checkout": str(tmp_path / "staged"),
        "minimum_checkpoint_step": 5,
        "budgets": {"queue": 600, "generation": 150, "download": 120},
    }
    manifest = {
        "job_id": 3573,
        "job_name": "rl-inspection-v1-1-16h100",
        "code_commit": "before",
        "code_checkout": plan["code_checkout"],
        "watchdog_service": "test-watchdog",
    }
    config = {"inspection_rl": {"turn_max_tokens": 8192}, "video_rl": {"timeouts": {"h3": 150}}}
    for name, value in {
        "queue-timeout-deployment.json": plan,
        "launch.json": manifest,
        "training-config.json": config,
        "run-budget.json": {"deadline_epoch": 12345},
    }.items():
        atomic_json(tmp_path / name, value)
    monkeypatch.setattr(deploy.recover, "command", lambda *a, **k: pytest.fail("No commands before verification"))
    return tmp_path, plan, manifest, config


def checkpoint(root, verified=5):
    atomic_json(root / "checkpoints/verified.json", {"step": verified})
    (root / "checkpoints/latest_checkpointed_iteration.txt").write_text("5\n")
    atomic_json(root / "queue-timeout-smoke/result.json", {"status": "succeeded"})
    atomic_json(
        root / "queue-timeout-smoke/protocol.json",
        {"h3_phase_timeouts": {"queue": 600, "generation": 150, "download": 120}},
    )


def test_waits_without_checkpoint(run):
    root, _, _, _ = run
    assert deploy.deploy(root)["status"] == "waiting"
    assert not (root / "recovery-disabled.json").exists()


def test_partial_checkpoint_cannot_trigger_cancellation(run):
    root, _, _, _ = run
    checkpoint(root, verified=4)
    assert deploy.deploy(root)["status"] == "waiting"


def test_another_allocation_cannot_be_cancelled(run):
    root, plan, _, _ = run
    checkpoint(root)
    plan["source_job"] = 3517
    atomic_json(root / "queue-timeout-deployment.json", plan)
    with pytest.raises(ValueError, match="Allocation or code changed"):
        deploy.deploy(root)


def test_verification_failure_keeps_training_running(run, monkeypatch):
    root, _, _, _ = run
    checkpoint(root)

    def fail(*args):
        raise ValueError("incomplete checkpoint")

    monkeypatch.setattr(deploy.recover, "verify_launch", fail)
    with pytest.raises(ValueError, match="incomplete checkpoint"):
        deploy.deploy(root)
    assert not (root / "recovery-disabled.json").exists()


def test_deployment_preserves_config_budget_and_old_artifacts(run, monkeypatch):
    root, plan, manifest, original = run
    checkpoint(root)
    events = []

    def verify(*args):
        events.append("verified")
        return 5

    def command(args, **kwargs):
        events.append(args)
        if args[:3] == ["git", "rev-parse", "HEAD"]:
            return "after"
        if args[:3] == ["scontrol", "show", "job"]:
            return f"JobId=3573 JobName={manifest['job_name']} Command={root / 'train.sbatch'}"
        return ""

    states = iter(["RUNNING", "CANCELLED"])

    def submit(directory, reason, *, automatic):
        events.append("submit")
        assert not automatic and directory == root
        assert deploy.read(root / "launch.json")["code_commit"] == "after"
        assert deploy.read(root / "run-budget.json") == {"deadline_epoch": 12345}
        assert deploy.read(root / "training-config.json") == deploy.updated_config(original, plan["budgets"])
        return {"job_id": 9999}

    monkeypatch.setattr(deploy.recover, "verify_launch", verify)
    monkeypatch.setattr(deploy.recover, "command", command)
    monkeypatch.setattr(deploy.recover, "scheduler", lambda job: {"state": next(states)})
    monkeypatch.setattr(deploy.recover, "submit", submit)
    result = deploy.deploy(root)
    assert result["status"] == "submitted" and result["new_job"] == 9999
    assert events.index("verified") < events.index(["scancel", "3573"]) < events.index("submit")
    assert not (root / "recovery-disabled.json").exists()
    archive = root / "attempts/3573/before-queue-timeout-fix"
    assert deploy.read(archive / "training-config.json") == original
    assert deploy.read(archive / "launch.json") == manifest
    before = list(events)
    assert deploy.deploy(root) == result
    assert events == before


def test_config_update_is_limited_to_two_timeout_fields():
    original = {"inspection_rl": {"keep": 1}, "video_rl": {"timeouts": {"h3": 150}}, "unrelated": [1, 2]}
    result = deploy.updated_config(original, {"queue": 600, "generation": 150, "download": 120})
    assert original["inspection_rl"] == {"keep": 1}
    assert result["unrelated"] == original["unrelated"]
    assert result["inspection_rl"] == {"keep": 1, "h3_queue_timeout_seconds": 600, "h3_download_timeout_seconds": 120}
    with pytest.raises(ValueError, match="preserve"):
        deploy.updated_config(original, {"queue": 600, "generation": 200, "download": 120})
