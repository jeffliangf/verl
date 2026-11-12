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
"""Balanced evaluation scores and durable best-plus-latest checkpoint retention."""

import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from examples.video_agents import checkpoints
from examples.video_agents.checkpoints import COMPLETE_MARKER, OVERALL_SCORE, SCORE_METRICS, VideoCheckpointCallback
from examples.video_agents.rl import CONFIG_DIR
from examples.video_agents.storage import read_json, write_json
from verl.trainer.ppo.checkpoint_callback import build_checkpoint_callback


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_AGENT_TENSORBOARD_RUNS_DIR", str(tmp_path / "registry"))
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="rl",
            overrides=[
                f"trainer.default_local_dir={tmp_path / 'checkpoints'}",
                "trainer.n_gpus_per_node=2",
                "video_rl.eval_media.enabled=false",
                f"video_rl.output_dir={tmp_path / 'run'}",
                f"video_rl.protocol={tmp_path / 'protocol.json'}",
            ],
        )


def checkpoint(callback, step, publish=True):
    directory = callback.root / f"global_step_{step}"
    actor = directory / "actor"
    actor.mkdir(parents=True)
    write_json(actor / "fsdp_config.json", {"world_size": 2})
    files = [directory / "data.pt"]
    files.extend(
        actor / f"{kind}_world_size_2_rank_{rank}.pt" for kind in ("model", "optim", "extra_state") for rank in range(2)
    )
    for file in files:
        with zipfile.ZipFile(file, "w") as archive:
            archive.writestr("archive/data.pkl", f"step {step}")
    if publish:
        (callback.root / "latest_checkpointed_iteration.txt").write_text(str(step))
        callback.on_save(None, step, str(directory))
    return directory


def evaluate(callback, step, aa, movie):
    metrics = dict(zip(SCORE_METRICS, (aa, movie), strict=True))
    metrics["val-core/aa_eval/reward/mean@1"] = 0.99
    callback.on_validation(None, step, metrics)
    return metrics


def retained(callback):
    return {int(path.name.removeprefix("global_step_")) for path in callback.root.glob("global_step_*")}


def test_default_callback_logs_exact_average_without_a_checkpoint(config):
    callback = build_checkpoint_callback(config)
    assert isinstance(callback, VideoCheckpointCallback)
    metrics = evaluate(callback, 0, 0.2, 0.8)
    assert metrics[OVERALL_SCORE] == pytest.approx(0.5)
    assert not callback.root.exists()  # Initial base-model evaluation is logging only.


@pytest.mark.parametrize("val_only", [False, True])
def test_evaluation_media_published_before_score_or_checkpoint_guards(config, tmp_path, monkeypatch, val_only):
    calls = []

    def publish(*args, **kwargs):
        calls.append((args, kwargs))
        return {"errors": []}

    monkeypatch.setitem(
        sys.modules, "examples.video_agents.eval_media", SimpleNamespace(publish_evaluation_media=publish)
    )
    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path / "custom-tensorboard"))
    config.video_rl.eval_media.enabled = True
    config.video_rl.output_dir = str(tmp_path / "run")
    config.video_rl.protocol = str(tmp_path / "protocol.json")
    config.trainer.val_only = val_only
    callback = build_checkpoint_callback(config)
    metrics = {}
    callback.on_validation(None, 0, metrics)
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (tmp_path / "run", 0, tmp_path / "protocol.json")
    assert kwargs["log_dir"] == tmp_path / "custom-tensorboard"
    assert kwargs["settings"] == {
        "enabled": True,
        "samples_per_task": 4,
        "fps": 6,
        "max_size": 384,
        "timeout": 60,
    }
    assert not metrics and not callback.root.exists()


def test_evaluation_media_remains_automatic_for_existing_configs(config, tmp_path, monkeypatch):
    calls = []

    def publish(*args, **kwargs):
        calls.append((args, kwargs))
        return {"errors": []}

    monkeypatch.setitem(
        sys.modules, "examples.video_agents.eval_media", SimpleNamespace(publish_evaluation_media=publish)
    )
    monkeypatch.delenv("TENSORBOARD_DIR", raising=False)
    with open_dict(config.video_rl):
        del config.video_rl["eval_media"]
    config.video_rl.output_dir = str(tmp_path / "run")
    config.video_rl.protocol = str(tmp_path / "protocol.json")
    callback = build_checkpoint_callback(config)
    evaluate(callback, 0, 0.2, 0.8)
    assert len(calls) == 1
    assert calls[0][1] == {"settings": {}, "log_dir": tmp_path / "run/tensorboard"}


@pytest.mark.parametrize("disabled", ["setting", "backend"])
def test_evaluation_media_can_be_disabled(config, monkeypatch, disabled):
    def publish(*args, **kwargs):
        pytest.fail("Media publisher must not run when disabled")

    monkeypatch.setitem(
        sys.modules, "examples.video_agents.eval_media", SimpleNamespace(publish_evaluation_media=publish)
    )
    if disabled == "backend":
        config.video_rl.eval_media.enabled = True
        config.trainer.logger = ["console"]
    callback = build_checkpoint_callback(config)
    assert evaluate(callback, 0, 0.2, 0.8)[OVERALL_SCORE] == pytest.approx(0.5)


@pytest.mark.parametrize("failure", ["exception", "report"])
def test_evaluation_media_failure_does_not_block_checkpoint_selection(config, tmp_path, monkeypatch, caplog, failure):
    def publish(*args, **kwargs):
        if failure == "exception":
            raise RuntimeError("preview encoder unavailable")
        return {"errors": ["preview encoder unavailable"]}

    monkeypatch.setitem(
        sys.modules, "examples.video_agents.eval_media", SimpleNamespace(publish_evaluation_media=publish)
    )
    config.video_rl.eval_media.enabled = True
    config.video_rl.output_dir = str(tmp_path / "run")
    config.video_rl.protocol = str(tmp_path / "protocol.json")
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 1)
    assert evaluate(callback, 1, 0.2, 0.8)[OVERALL_SCORE] == pytest.approx(0.5)
    assert read_json(callback.best_path)["step"] == 1
    assert "preview encoder unavailable" in caplog.text


def test_tensorboard_registry_keeps_existing_runs_and_stable_collision_names(config, tmp_path):
    first = checkpoints.register_tensorboard_run(tmp_path / "experiment/grpo", tmp_path / "logs-first")
    assert first == tmp_path / "registry/experiment"
    assert first.resolve() == tmp_path / "logs-first"
    assert checkpoints.register_tensorboard_run(tmp_path / "experiment/grpo", tmp_path / "logs-first") == first
    second = checkpoints.register_tensorboard_run(tmp_path / "other/experiment/drgrpo", tmp_path / "logs-second")
    assert second != first and second.name.startswith("experiment-")
    assert second.resolve() == tmp_path / "logs-second"
    assert first.resolve() == tmp_path / "logs-first"
    assert (
        checkpoints.register_tensorboard_run(tmp_path / "other/experiment/drgrpo", tmp_path / "logs-second") == second
    )


def test_tensorboard_registry_failure_does_not_block_media(config, tmp_path, monkeypatch, caplog):
    calls = []

    def publish(*args, **kwargs):
        calls.append(args)
        return {"errors": []}

    monkeypatch.setitem(
        sys.modules, "examples.video_agents.eval_media", SimpleNamespace(publish_evaluation_media=publish)
    )
    registry = tmp_path / "registry"
    registry.write_text("a file, not a directory")
    config.video_rl.eval_media.enabled = True
    callback = build_checkpoint_callback(config)
    assert evaluate(callback, 0, 0.2, 0.8)[OVERALL_SCORE] == pytest.approx(0.5)
    assert len(calls) == 1
    assert "Could not register TensorBoard run" in caplog.text


def test_native_metrics_write_overall_scalar_to_tensorboard(config, tmp_path, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from verl.trainer.ppo.v1.trainer_base import PPOTrainer
    from verl.utils.tracking import Tracking

    # Unequal dataset sizes distinguish the requested split average from pooling
    # all valid prompts. Use the real trainer metric names and TensorBoard writer.
    metrics = PPOTrainer._val_metrics_update(
        None,
        ["aa_eval", "aa_eval", "movie_gen_video_bench"],
        ["a", "b", "c"],
        {"score": [0.2, 0.4, 0.8], "reward": [0.2, 0.4, 0.8]},
        [],
    )
    callback = build_checkpoint_callback(config)
    callback.on_validation(None, 0, metrics)
    expected = ((0.2 + 0.4) / 2 + 0.8) / 2
    assert metrics[OVERALL_SCORE] == pytest.approx(expected)
    log_dir = tmp_path / "tensorboard"
    monkeypatch.setenv("TENSORBOARD_DIR", str(log_dir))
    tracking = Tracking("test", "overall-score", default_backend=["tensorboard"])
    tracking.log(metrics, step=0)
    tracking.logger["tensorboard"].finish()
    events = EventAccumulator(str(log_dir)).Reload().Scalars(OVERALL_SCORE)
    assert [(event.step, event.value) for event in events] == [(0, pytest.approx(expected))]


def mixed_evaluation(config, step):
    from examples.video_agents.agent import evaluation_summary
    from examples.video_agents.config import REFERENCE_FIELDS
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    sources = ["t2v/aa_eval", "t2v/aa_eval", "t2v/movie_gen_video_bench", "i2v/aa-i2v", "i2v/aa-i2v-batch2"]
    scores = [0.1, 0.3, 0.9, 0.6, 0.8]
    records = [
        {
            "sample_id": str(i),
            "task": source.split("/")[0],
            "dropped": False,
            "rewards": [{"score": score, "valid_call": 1, **{f"vs_{r}": score for r in REFERENCE_FIELDS}}],
        }
        for i, (source, score) in enumerate(zip(sources, scores, strict=True))
    ]
    config.video_rl.output_dir = str(Path(config.trainer.default_local_dir).parent / "run")
    config.video_rl.protocol = str(Path(config.video_rl.output_dir) / "protocol.json")
    write_json(Path(config.video_rl.protocol), {"samples": {"val": records}})
    path = Path(config.video_rl.output_dir) / "val" / f"step_{step}" / "eval-summary.json"
    write_json(path, evaluation_summary(step, records, 0.9))
    metrics = PPOTrainer._val_metrics_update(
        None, sources, [str(i) for i in range(len(scores))], {"score": scores, "reward": scores}, []
    )
    return metrics, path


def test_degraded_evaluation_still_publishes_media(config, monkeypatch):
    calls = []

    def publish(*args, **kwargs):
        calls.append(args[1])
        return {"errors": []}

    monkeypatch.setitem(
        sys.modules, "examples.video_agents.eval_media", SimpleNamespace(publish_evaluation_media=publish)
    )
    config.video_rl.eval_media.enabled = True
    metrics, path = mixed_evaluation(config, 5)
    summary = read_json(path)
    summary["degraded"] = True
    write_json(path, summary)
    callback = build_checkpoint_callback(config)
    callback.on_validation(None, 5, metrics)
    assert calls == [5]
    assert OVERALL_SCORE not in metrics
    assert not callback.best_path.exists()


def test_mixed_checkpoint_balances_tasks_and_survives_restart(config):
    metrics, path = mixed_evaluation(config, 0)
    callback = build_checkpoint_callback(config)
    callback.on_validation(None, 0, metrics)
    expected = ((0.1 + 0.3 + 0.9) / 3 + (0.6 + 0.8) / 2) / 2
    assert metrics[OVERALL_SCORE] == pytest.approx(expected)
    assert metrics[OVERALL_SCORE] == read_json(path)["task_macro_score"]
    assert not callback.root.exists()
    checkpoint(callback, 1)
    metrics, _ = mixed_evaluation(config, 1)
    callback.on_validation(None, 1, metrics)
    best = read_json(callback.best_path)
    assert best["step"] == 1 and best["score"] == pytest.approx(expected)
    assert set(best["metrics"]) == set(checkpoints.TASK_SCORE_METRICS.values())
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 2)
    metrics, _ = mixed_evaluation(config, 2)
    callback.on_validation(None, 2, metrics)
    assert read_json(callback.best_path) == best  # A tied task macro keeps the prior winner.


@pytest.mark.parametrize(
    "problem", ["missing", "stale", "partial", "missing_task", "degraded", "task_degraded", "no_score"]
)
def test_incomplete_mixed_evaluation_cannot_select_a_checkpoint(config, problem):
    metrics, path = mixed_evaluation(config, 1)
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 1)
    summary = read_json(path)
    if problem == "missing":
        path.unlink()
    else:
        if problem == "stale":
            summary["policy_step"] = 0
        elif problem == "partial":
            summary["by_task"]["t2v"]["prompts"] -= 1
        elif problem == "missing_task":
            del summary["by_task"]["i2v"]
        elif problem == "degraded":
            summary["degraded"] = True
        elif problem == "task_degraded":
            summary["by_task"]["i2v"]["degraded"] = True
        else:
            summary["by_task"]["i2v"]["score"] = None
        write_json(path, summary)
    metrics.update(dict.fromkeys(SCORE_METRICS, 1.0))  # Never fall back to T2V-only scores.
    callback.on_validation(None, 1, metrics)
    assert OVERALL_SCORE not in metrics
    assert not callback.best_path.exists()


def test_keep_latest_two_and_best_then_replace_winner(config):
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 5)
    checkpoint(callback, 10)
    assert evaluate(callback, 10, 0.5, 0.9)[OVERALL_SCORE] == pytest.approx(0.7)
    checkpoint(callback, 15)
    checkpoint(callback, 20)
    # AA alone improves, but the equally weighted overall score gets worse.
    evaluate(callback, 20, 0.8, 0.4)
    checkpoint(callback, 25)
    assert retained(callback) == {10, 20, 25}
    assert read_json(callback.best_path)["step"] == 10
    assert (callback.root / "latest_checkpointed_iteration.txt").read_text() == "25"

    checkpoint(callback, 30)
    evaluate(callback, 30, 0.8, 0.8)
    assert retained(callback) == {25, 30}
    best = read_json(callback.best_path)
    assert best["checkpoint"] == "global_step_30"
    assert best["score"] == pytest.approx(0.8)
    assert best["metrics"] == dict.fromkeys(SCORE_METRICS, 0.8)


def test_restart_preserves_best_and_ties_keep_existing_winner(config):
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 10)
    evaluate(callback, 10, 0.6, 0.6)
    checkpoint(callback, 15)
    checkpoint(callback, 20)
    callback = build_checkpoint_callback(config)
    evaluate(callback, 20, 0.6, 0.6)
    checkpoint(callback, 25)
    assert retained(callback) == {10, 20, 25}
    assert read_json(callback.best_path)["step"] == 10


@pytest.mark.parametrize("invalid", [None, float("nan"), float("inf"), -float("inf")])
def test_missing_or_nonfinite_split_cannot_replace_best(config, invalid):
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 10)
    evaluate(callback, 10, 0.5, 0.5)
    checkpoint(callback, 20)
    metrics = {SCORE_METRICS[0]: 1.0}
    if invalid is not None:
        metrics[SCORE_METRICS[1]] = invalid
    callback.on_validation(None, 20, metrics)
    assert OVERALL_SCORE not in metrics
    assert read_json(callback.best_path)["step"] == 10


def test_scores_are_not_applied_to_another_steps_checkpoint(config):
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 10)
    evaluate(callback, 10, 0.5, 0.5)
    metrics = evaluate(callback, 12, 1.0, 1.0)
    assert metrics[OVERALL_SCORE] == 1.0
    assert read_json(callback.best_path)["step"] == 10


def test_validation_only_logs_without_changing_selection(config):
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 10)
    evaluate(callback, 10, 0.5, 0.5)
    config.trainer.val_only = True
    callback = build_checkpoint_callback(config)
    assert evaluate(callback, 10, 1.0, 1.0)[OVERALL_SCORE] == 1.0
    assert read_json(callback.best_path)["score"] == 0.5


def test_incomplete_save_cannot_prune_previous_checkpoints(config):
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 5)
    checkpoint(callback, 10)
    directory = checkpoint(callback, 15, publish=False)
    (directory / "actor/optim_world_size_2_rank_1.pt").unlink()
    (callback.root / "latest_checkpointed_iteration.txt").write_text("15")
    with pytest.raises(FileNotFoundError):
        callback.on_save(None, 15, str(directory))
    assert retained(callback) == {5, 10, 15}
    assert not (directory / COMPLETE_MARKER).exists()


def test_failed_best_publication_keeps_previous_winner(config, monkeypatch):
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 5)
    evaluate(callback, 5, 0.5, 0.5)
    checkpoint(callback, 10)
    checkpoint(callback, 15)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(checkpoints, "write_json", fail)
    with pytest.raises(OSError, match="disk full"):
        evaluate(callback, 15, 0.9, 0.9)
    assert retained(callback) == {5, 10, 15}
    assert read_json(callback.best_path)["step"] == callback.best["step"] == 5


def test_unmanaged_and_interrupted_checkpoints_are_not_deleted(config):
    callback = build_checkpoint_callback(config)
    checkpoint(callback, 1, publish=False)
    checkpoint(callback, 5)
    checkpoint(callback, 10)
    checkpoint(callback, 15)
    assert retained(callback) == {1, 10, 15}


@pytest.mark.parametrize(
    "override",
    [
        "trainer.max_actor_ckpt_to_keep=1",
        "trainer.max_critic_ckpt_to_keep=1",
        "trainer.del_local_ckpt_after_load=true",
        "actor_rollout_ref.actor.checkpoint.async_save=true",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model]",
        "trainer.v1.trainer_mode=colocate_async",
        "video_rl.checkpoint_keep_latest=0",
    ],
)
def test_rejects_settings_that_could_lose_best_checkpoint(config, override):
    config = OmegaConf.merge(config, OmegaConf.from_dotlist([override]))
    with pytest.raises(ValueError):
        build_checkpoint_callback(config)


def test_symlink_checkpoint_is_never_pruned(config, tmp_path):
    callback = build_checkpoint_callback(config)
    directory = checkpoint(callback, 5)
    outside = tmp_path / "outside"
    directory.rename(outside)
    directory.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Unsafe"):
        checkpoint(callback, 10)
    assert (outside / "actor/model_world_size_2_rank_0.pt").is_file()


def test_damaged_best_is_detected_before_pruning_on_restart(config):
    callback = build_checkpoint_callback(config)
    directory = checkpoint(callback, 5)
    evaluate(callback, 5, 0.5, 0.5)
    checkpoint(callback, 10)
    (directory / "actor/model_world_size_2_rank_0.pt").unlink()
    with pytest.raises(FileNotFoundError):
        build_checkpoint_callback(config)
    assert retained(callback) == {5, 10}


@pytest.mark.parametrize("problem", [None, "manifest", "queue_step", "storage", "controller"])
def test_cohort_checkpoint_publishes_only_with_matching_durable_queue(config, problem):
    config.video_rl.async_lookahead = True
    config.trainer.v1.trainer_mode = "colocate_async"
    config.trainer.v1.trainer_class = "examples.video_agents.cohort_trainer.VideoCohortTrainer"
    callback = build_checkpoint_callback(config)
    directory = checkpoint(callback, 1, publish=False)
    write_json(directory / "video-cohorts.json", {"step": 2 if problem == "manifest" else 1})
    write_json(
        directory / "transfer_queue/metadata.json",
        {"storage_saved": problem != "storage", "user_metadata": {"global_steps": 2 if problem == "queue_step" else 1}},
    )
    if problem != "controller":
        (directory / "transfer_queue/controller_state.pkl").write_bytes(b"saved queue")
    (callback.root / "latest_checkpointed_iteration.txt").write_text("1")
    if problem:
        with pytest.raises(ValueError, match="must match"):
            callback.on_save(None, 1, str(directory))
        assert not (directory / COMPLETE_MARKER).exists()
    else:
        callback.on_save(None, 1, str(directory))
        assert read_json(directory / COMPLETE_MARKER) == {"step": 1}
