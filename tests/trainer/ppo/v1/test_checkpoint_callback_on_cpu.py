# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback
from verl.trainer.ppo.v1 import trainer_base
from verl.trainer.ppo.v1.trainer_base import PPOTrainer


class _StubTrainer(PPOTrainer):
    def on_step_end(self):
        pass

    def on_sample_end(self):
        pass


class _RecordingCallback(CheckpointCallback):
    def __init__(self, config=None, events=None):
        super().__init__(config=config)
        self.events = events if events is not None else []

    def on_save(self, trainer, global_step, checkpoint_dir, async_save=False, **kwargs):
        self.events.append(("on_save", global_step, checkpoint_dir, async_save))

    def on_validation(self, trainer, global_step, metrics, **kwargs):
        self.events.append(("on_validation", global_step, metrics.copy()))
        metrics["combined_score"] = 0.75


def _make_trainer(tmp_path, events, async_save=False):
    trainer = _StubTrainer.__new__(_StubTrainer)
    trainer.trainer_mode = "sync"
    trainer.global_steps = 3
    trainer.use_critic = False
    trainer.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path),
                "default_hdfs_dir": None,
                "resume_mode": "auto",
                "resume_from_path": None,
                "del_local_ckpt_after_load": False,
                "checkpoint_callback_class": None,
            },
            "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": async_save}}},
        }
    )
    trainer.checkpoint_callback = _RecordingCallback(config=trainer.config, events=events)
    trainer.actor_rollout_wg = MagicMock()
    trainer.actor_rollout_wg.save_checkpoint.side_effect = lambda *a, **k: events.append(("wg_save",))
    dataloader = MagicMock()
    dataloader.state_dict.return_value = {}
    trainer.train_dataloader = dataloader
    return trainer


def test_save_fires_on_save_after_worker_save(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)

    trainer._save_checkpoint()

    expected_dir = os.path.join(str(tmp_path), "global_step_3")
    assert events == [
        ("wg_save",),
        ("on_save", 3, expected_dir, False),
    ]
    tracker = os.path.join(str(tmp_path), "latest_checkpointed_iteration.txt")
    with open(tracker) as f:
        assert f.read() == "3"


def test_async_save_fires_on_save_with_flag(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events, async_save=True)

    trainer._save_checkpoint()

    expected_dir = os.path.join(str(tmp_path), "global_step_3")
    assert events[-1] == ("on_save", 3, expected_dir, True)
    assert not os.path.exists(os.path.join(str(tmp_path), "latest_checkpointed_iteration.txt"))


def test_on_save_not_fired_on_worker_failure(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)
    trainer.actor_rollout_wg.save_checkpoint.side_effect = RuntimeError("save failed")

    with pytest.raises(RuntimeError, match="save failed"):
        trainer._save_checkpoint()

    assert events == []


def test_callback_exception_propagates(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)
    trainer.checkpoint_callback.on_save = MagicMock(side_effect=RuntimeError("callback failed"))

    with pytest.raises(RuntimeError, match="callback failed"):
        trainer._save_checkpoint()


def test_validation_callback_observes_same_step_metrics_before_logging(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)
    trainer.on_validate_begin = lambda: events.append(("begin",))
    trainer._validate = lambda: {"dataset_score": 0.5}
    trainer.on_validate_end = lambda: events.append(("end",))

    metrics = trainer._run_validation()

    assert events == [("begin",), ("end",), ("on_validation", 3, {"dataset_score": 0.5})]
    assert metrics == {"dataset_score": 0.5, "combined_score": 0.75}


def test_failed_validation_does_not_notify_callback(tmp_path):
    events = []
    trainer = _make_trainer(tmp_path, events)
    trainer._validate = MagicMock(side_effect=RuntimeError("validation failed"))
    with pytest.raises(RuntimeError, match="validation failed"):
        trainer._run_validation()
    assert events == []


@pytest.mark.parametrize(
    "save_on_validation,save_freq,expected_saves", [(True, 3, [2, 3, 4]), (False, 3, [3, 4]), (True, -1, [])]
)
def test_fit_saves_evaluated_weights_and_logs_callback_metrics(
    tmp_path, monkeypatch, save_on_validation, save_freq, expected_saves
):
    events = []
    trainer = _make_trainer(tmp_path, events)
    trainer.config = OmegaConf.merge(
        trainer.config,
        {
            "trainer": {
                "project_name": "test",
                "experiment_name": "test",
                "logger": [],
                "val_before_train": True,
                "total_epochs": 1,
                "save_freq": save_freq,
                "test_freq": 2,
            },
            "global_profiler": {"steps": None},
        },
    )
    trainer.global_steps = 0
    trainer.total_training_steps = trainer.steps_per_epoch = 4
    trainer.checkpoint_callback.save_on_validation = save_on_validation
    trainer._save_checkpoint = lambda: events.append(("save", trainer.global_steps))
    trainer._validate = lambda: {"dataset_score": trainer.global_steps / 10}
    trainer.step = lambda *a: SimpleNamespace(keys=[], partition_id="train")
    for method in ("_reissue_inflight_prompts", "_start_profiling", "_stop_profiling", "_shutdown_dump_executor"):
        monkeypatch.setattr(trainer, method, lambda: None)
    trainer._compute_metrics = lambda *a, **kw: None
    trainer._consume_sync_metrics = lambda: {}
    logs = []
    monkeypatch.setattr(
        trainer_base, "Tracking", lambda **kw: SimpleNamespace(log=lambda data, step: logs.append((step, data.copy())))
    )
    monkeypatch.setattr(trainer_base, "ValidationGenerationsLogger", MagicMock())
    monkeypatch.setattr(trainer_base, "DapoFilteredRewardTableLogger", MagicMock())
    monkeypatch.setattr(trainer_base, "SkipManager", MagicMock())
    monkeypatch.setattr(trainer_base.tq, "kv_clear", MagicMock())

    trainer.fit(MagicMock())

    assert [event[1] for event in events if event[0] == "save"] == expected_saves
    assert [event[1] for event in events if event[0] == "on_validation"] == [0, 2, 4]
    for step in (0, 2, 4):
        logged = dict(logs)[step]
        assert logged["dataset_score"] == step / 10
        assert logged["combined_score"] == 0.75
        if step in expected_saves:
            save_index = events.index(("save", step))
            validation_index = next(i for i, event in enumerate(events) if event[:2] == ("on_validation", step))
            assert save_index < validation_index
