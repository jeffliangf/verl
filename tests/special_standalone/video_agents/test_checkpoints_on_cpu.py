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

import zipfile

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from examples.video_agents import checkpoints
from examples.video_agents.checkpoints import COMPLETE_MARKER, OVERALL_SCORE, SCORE_METRICS, VideoCheckpointCallback
from examples.video_agents.rl import CONFIG_DIR
from examples.video_agents.storage import read_json, write_json
from verl.trainer.ppo.checkpoint_callback import build_checkpoint_callback


@pytest.fixture
def config(tmp_path):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return compose(
            config_name="rl",
            overrides=[f"trainer.default_local_dir={tmp_path / 'checkpoints'}", "trainer.n_gpus_per_node=2"],
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
