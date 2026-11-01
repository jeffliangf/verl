import datetime
import json
import os
import subprocess
import time
import zipfile
from types import SimpleNamespace

import pytest
from canary_ops import retain_latest, verify_checkpoint
from watchdog import MIN_FREE_BYTES, cancel_owned_job, progress, stop_reason


def checkpoint(root, step, world=2):
    actor = root / f"global_step_{step}" / "actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text(json.dumps({"world_size": world}))
    files = [actor.parent / "data.pt"] + [
        actor / f"{kind}_world_size_{world}_rank_{rank}.pt"
        for rank in range(world)
        for kind in ("model", "optim", "extra_state")
    ]
    for path in files:
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("archive/data.pkl", b"payload")
    (root / "latest_checkpointed_iteration.txt").write_text(str(step))
    return actor.parent


def test_incomplete_checkpoint_never_deletes_previous(tmp_path):
    old = checkpoint(tmp_path, 5)
    new = checkpoint(tmp_path, 10)
    (new / "actor/optim_world_size_2_rank_1.pt").unlink()
    with pytest.raises(FileNotFoundError):
        retain_latest(tmp_path, 10, 2)
    assert old.exists()


def test_complete_save_prunes_previous_across_restarts(tmp_path):
    checkpoint(tmp_path, 5)
    checkpoint(tmp_path, 10)
    checkpoint(tmp_path, 15)
    result = retain_latest(tmp_path, 15, 2)
    assert sorted(result["deleted_previous_checkpoints"]) == ["global_step_10", "global_step_5"]
    assert [p.name for p in tmp_path.glob("global_step_*")] == ["global_step_15"]
    assert verify_checkpoint(tmp_path, 15, 2)["archives_checked"] == 7


def test_symlink_retention_never_touches_external_data(tmp_path):
    root = tmp_path / "checkpoints"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "important").write_text("keep")
    checkpoint(root, 10)
    (root / "global_step_5").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        retain_latest(root, 10, 2)
    assert (outside / "important").read_text() == "keep"


def test_stale_latest_marker_preserves_previous(tmp_path):
    old = checkpoint(tmp_path, 5)
    checkpoint(tmp_path, 10)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("5")
    with pytest.raises(ValueError):
        retain_latest(tmp_path, 10, 2)
    assert old.exists()


def test_one_node_checkpoint_requires_all_eight_ranks(tmp_path):
    checkpoint(tmp_path, 5, world=8)
    assert verify_checkpoint(tmp_path, 5, 8)["archives_checked"] == 25
    with pytest.raises(ValueError, match="world size"):
        verify_checkpoint(tmp_path, 5, 16)
    (tmp_path / "global_step_5/actor/optim_world_size_8_rank_7.pt").unlink()
    with pytest.raises(FileNotFoundError):
        verify_checkpoint(tmp_path, 5, 8)


def observed(phase="training", started=0, last_progress=100):
    return {"phase": phase, "phase_started": started, "last_progress_at": last_progress}


def test_slow_active_evaluation_is_allowed():
    assert stop_reason("RUNNING", observed("evaluation", 0, 7000), 7200, MIN_FREE_BYTES * 2) is None


def test_real_training_stall_stops_even_if_gpu_log_is_alive():
    assert "deadline" in stop_reason("RUNNING", observed(last_progress=5500), 5501, MIN_FREE_BYTES * 2)


def test_lack_of_progress_and_low_space_stop():
    assert "No meaningful" in stop_reason("RUNNING", observed(), 2501, MIN_FREE_BYTES * 2)
    assert "450 GiB" in stop_reason("RUNNING", observed(), 101, MIN_FREE_BYTES - 1)


def test_queued_job_is_not_cancelled_by_training_deadlines():
    assert stop_reason("PENDING", observed(), 100000, 0) is None


@pytest.mark.parametrize("state", ["PENDING", "RUNNING"])
def test_recovery_cannot_extend_original_budget(state):
    status = observed(started=999, last_progress=999)
    status["budget_deadline_epoch"] = 1000
    assert stop_reason(state, status, 999, MIN_FREE_BYTES * 2) is None
    assert "budget exhausted" in stop_reason(state, status, 1000, MIN_FREE_BYTES * 2)
    assert stop_reason("COMPLETED", status, 1001, MIN_FREE_BYTES * 2) is None


def test_parser_separates_validation_and_checkpoint(tmp_path):
    now = time.time()
    stamp = __import__("datetime").datetime.fromtimestamp(now, __import__("datetime").timezone.utc).isoformat()
    (tmp_path / "timings.jsonl").write_text(
        json.dumps({"stage": "dispatch", "event": "start", "timestamp": stamp, "validate": True, "policy_step": 5})
        + "\n"
    )
    (tmp_path / "gpus-1-node.csv").write_text("this must not reset progress")
    parsed = progress(tmp_path, "1", now - 5000)
    assert parsed["phase"] == "evaluation" and parsed["phase_step"] == 5
    assert parsed["last_progress_at"] == pytest.approx(now)


def dispatch_row(step, timestamp, validate=False):
    return (
        json.dumps(
            {
                "stage": "dispatch",
                "event": "start",
                "timestamp": datetime.datetime.fromtimestamp(timestamp, datetime.timezone.utc).isoformat(),
                "validate": validate,
                "policy_step": step,
            }
        )
        + "\n"
    )


def test_delayed_metrics_cannot_hide_next_step_progress(tmp_path):
    now = time.time()
    metrics = tmp_path / "metrics-1.jsonl"
    metrics.write_text(json.dumps({"step": 1, "data": {"actor/loss": 0.1}}) + "\n")
    os.utime(metrics, (now - 10, now - 10))
    (tmp_path / "timings.jsonl").write_text(dispatch_row(2, now - 20))
    active = tmp_path / "train/step_2/group/timings.jsonl"
    active.parent.mkdir(parents=True)
    active.write_text("{}\n")
    os.utime(active, (now, now))
    parsed = progress(tmp_path, "1", now - 4000)
    assert parsed["last_logged_step"] == 1 and parsed["phase_step"] == 2
    assert parsed["last_progress_at"] == pytest.approx(now)


def test_old_attempt_metrics_and_dispatch_do_not_override_resume(tmp_path):
    now = time.time()
    (tmp_path / "metrics-old.jsonl").write_text(json.dumps({"step": 19, "data": {"actor/loss": None}}) + "\n")
    (tmp_path / "timings.jsonl").write_text(dispatch_row(19, now - 1000) + dispatch_row(6, now - 10))
    parsed = progress(tmp_path, "new", now - 100)
    assert parsed["phase_step"] == 6
    assert parsed["last_logged_step"] == 0 and parsed["nonfinite_metric"] is None


def test_resume_startup_does_not_inherit_old_phase(tmp_path):
    now = time.time()
    (tmp_path / "timings.jsonl").write_text(dispatch_row(9, now - 1000))
    assert progress(tmp_path, "new", now - 10)["phase"] == "startup"


def test_replacement_dispatch_cannot_extend_optimizer_step_deadline(tmp_path):
    now = time.time()
    (tmp_path / "timings.jsonl").write_text(dispatch_row(1, now - 5501) + dispatch_row(1, now - 10))
    parsed = progress(tmp_path, "1", now - 6000)
    assert parsed["phase_started"] == pytest.approx(now - 5501)
    assert parsed["last_progress_at"] == pytest.approx(now - 10)
    assert "deadline" in stop_reason("RUNNING", parsed, now, MIN_FREE_BYTES * 2)


def test_next_optimizer_step_gets_its_own_training_deadline(tmp_path):
    now = time.time()
    (tmp_path / "timings.jsonl").write_text(
        dispatch_row(1, now - 5501) + dispatch_row(1, now - 1000) + dispatch_row(2, now - 10)
    )
    parsed = progress(tmp_path, "1", now - 6000)
    assert parsed["phase_step"] == 2
    assert parsed["phase_started"] == pytest.approx(now - 10)
    assert stop_reason("RUNNING", parsed, now, MIN_FREE_BYTES * 2) is None


def test_actual_slurm_start_preserves_events_before_first_poll(monkeypatch, tmp_path):
    import watchdog

    monkeypatch.setattr(watchdog, "run", lambda command: "RUNNING|0:55|36:00:00|node|2026-10-23T05:51:04")
    started = watchdog.scheduler("1")["started_at"]
    (tmp_path / "timings.jsonl").write_text(dispatch_row(1, started + 20))
    parsed = progress(tmp_path, "1", started)
    assert parsed["phase"] == "training" and parsed["phase_step"] == 1


def test_finished_job_missing_from_squeue_uses_accounting(monkeypatch):
    import watchdog

    calls = []

    def run(command):
        calls.append(command)
        if command[0] == "squeue":
            raise subprocess.CalledProcessError(1, command, stderr="slurm_load_jobs error: Invalid job id specified")
        return "CANCELLED by 10016|0:0|01:37:58"

    monkeypatch.setattr(watchdog, "run", run)
    assert watchdog.scheduler("2370") == {"state": "CANCELLED", "exit_code": "0:0", "elapsed": "01:37:58"}
    assert [command[0] for command in calls] == ["squeue", "sacct"]


def test_scheduler_connection_failure_is_not_a_terminal_job(monkeypatch):
    import watchdog

    calls = []

    def run(command):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command, stderr="Unable to contact slurm controller")

    monkeypatch.setattr(watchdog, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        watchdog.scheduler("2370")
    assert len(calls) == 1


def test_watchdog_wrong_host_never_queries_or_writes_state(monkeypatch, tmp_path):
    import watchdog

    (tmp_path / "launch.json").write_text(json.dumps({"job_id": "new", "controller_host": "controller"}))
    monkeypatch.setattr("sys.argv", ["watchdog", "--run-dir", str(tmp_path), "--once"])
    monkeypatch.setattr(watchdog.os, "uname", lambda: SimpleNamespace(nodename="compute-node"))
    monkeypatch.setattr(watchdog, "scheduler", lambda job: pytest.fail("Wrong host queried scheduler"))
    watchdog.main()
    assert not (tmp_path / "watchdog.lock").exists()
    assert not (tmp_path / "watchdog-state.json").exists()


@pytest.mark.parametrize(
    "prior_job,prior_state,current_state,expected_terminal_age",
    [
        ("old", "RUNNING", "PENDING", None),
        ("old", "CANCELLED", "FAILED", 0),
        ("new", "RUNNING", "RUNNING", None),
        ("new", "CANCELLED", "RUNNING", None),
        ("new", "RUNNING", "FAILED", 0),
        ("new", "CANCELLED", "CANCELLED", 30),
    ],
)
def test_watchdog_lifecycle_state_isolated_across_recovery(
    monkeypatch, tmp_path, prior_job, prior_state, current_state, expected_terminal_age
):
    import watchdog

    now = time.time()
    (tmp_path / "launch.json").write_text(json.dumps({"job_id": "new"}))
    (tmp_path / "watchdog-state.json").write_text(
        json.dumps(
            {
                "job_id": prior_job,
                "state": prior_state,
                "running_since": now - 100,
                "terminal_since": now - 30,
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["watchdog", "--run-dir", str(tmp_path), "--once", "--dry-run"])
    monkeypatch.setattr(watchdog.time, "time", lambda: now)
    monkeypatch.setattr(watchdog, "scheduler", lambda job: {"state": current_state, "started_at": now - 10})
    monkeypatch.setattr(watchdog.shutil, "disk_usage", lambda root: SimpleNamespace(free=MIN_FREE_BYTES * 2))
    watchdog.main()
    state = json.loads((tmp_path / "watchdog-state.json").read_text())
    assert state["job_id"] == "new" and "monitor_error" not in state
    expected = None if expected_terminal_age is None else now - expected_terminal_age
    assert state["terminal_since"] == expected
    if prior_job == "old" and current_state != "RUNNING":
        assert state["running_since"] is None
    assert not (tmp_path / "final-status.json").exists()


def test_checkpoint_then_evaluation_within_same_step(tmp_path):
    now = time.time()
    stamp = datetime.datetime.fromtimestamp(now - 20, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S,%f")[:23]
    (tmp_path / "slurm-1.log").write_text(f"{stamp}:Saving checkpoint to /tmp/checkpoints/global_step_5\n")
    (tmp_path / "timings.jsonl").write_text(dispatch_row(5, now - 10, validate=True))
    parsed = progress(tmp_path, "1", now - 100)
    assert parsed["phase"] == "evaluation" and parsed["phase_step"] == 5


def test_cancel_checks_ownership_and_script(monkeypatch, tmp_path):
    import watchdog

    calls = []
    monkeypatch.setattr(watchdog, "run", lambda command: calls.append(command) or "JobId=1 JobName=another-job")
    with pytest.raises(RuntimeError, match="identity mismatch"):
        cancel_owned_job(tmp_path, {"job_id": 1, "job_name": "canary"}, "stalled")
    assert len(calls) == 1


def test_cancel_targets_only_verified_job(monkeypatch, tmp_path):
    import watchdog

    calls = []
    detail = f"JobId=123 JobName=canary UserId=jeff({os.getuid()}) Command={tmp_path / 'train.sbatch'} "
    monkeypatch.setattr(watchdog, "run", lambda command: calls.append(command) or detail)
    cancel_owned_job(tmp_path, {"job_id": 123, "job_name": "canary"}, "stalled")
    assert calls[-1] == ["scancel", "123"]



@pytest.mark.parametrize("dropped,age,expected", [(False, 0, True), (True, 0, False), (False, 1000, False)])
def test_only_current_fatal_group_errors_stop_the_job(tmp_path, dropped, age, expected):
    now = time.time()
    (tmp_path / "timings.jsonl").write_text(dispatch_row(201, now - 20))
    failure = tmp_path / "train/step_201/group/failure.json"
    failure.parent.mkdir(parents=True)
    failure.write_text(json.dumps({"dropped": dropped, "stage": "internal", "error": "provider classification error"}))
    os.utime(failure, (now-age, now-age))
    parsed = progress(tmp_path, "1", now - 100)
    assert bool(parsed["fatal_failure"]) is expected
    assert bool(stop_reason("RUNNING", parsed, now, MIN_FREE_BYTES * 2)) is expected
