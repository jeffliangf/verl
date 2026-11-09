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
"""Exercise two-cohort scheduling, actual version checks and durable response replay."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from examples.video_agents import cohort_trainer as module
from examples.video_agents.cohort_config import validate_config
from examples.video_agents.policy_cache import PolicyCache
from examples.video_agents.storage import item_key, read_json, write_json
from verl.utils import tensordict_utils as tu
from verl.workers.rollout.replica import TokenOutput


@pytest.fixture
def config(tmp_path):
    root = Path(__file__).resolve().parents[3]
    with initialize_config_dir(config_dir=str(root / "examples/video_agents/configs"), version_base=None):
        value = compose(
            config_name="rl",
            overrides=[
                f"video_rl.output_dir={tmp_path}",
                "video_rl.async_lookahead=true",
                "video_rl.judge_streaming=true",
                "video_rl.judge_background_cleanup=true",
                "trainer.v1.trainer_mode=colocate_async",
                "trainer.v1.trainer_class=examples.video_agents.cohort_trainer.VideoCohortTrainer",
                "algorithm.rollout_correction.rollout_is=token",
                "data.train_batch_size=2",
                "data.gen_batch_size=1",
                f"trainer.default_local_dir={tmp_path}/checkpoints",
            ],
        )
    return value


@pytest.mark.parametrize(
    "field,value",
    [
        ("actor_rollout_ref.rollout.calculate_log_probs", False),
        ("algorithm.rollout_correction.rollout_is", None),
        ("algorithm.rollout_correction.bypass_mode", True),
        ("video_rl.judge_background_cleanup", False),
        ("video_rl.judge_limits.cleanup_backlog", 4),
        ("trainer.v1.sampler.sync_refill_failed_groups", False),
    ],
)
def test_unsafe_lookahead_configs_are_rejected(config, field, value):
    from omegaconf import OmegaConf

    validate_config(config)
    OmegaConf.update(config, field, value)
    with pytest.raises(ValueError):
        validate_config(config)


@pytest.mark.asyncio
async def test_policy_response_is_replayed_without_regeneration(tmp_path):
    calls = []

    async def generate(**kwargs):
        calls.append(kwargs)
        return TokenOutput(
            token_ids=[1, 2],
            log_probs=[-0.5, -1.5],
            stop_reason="completed",
            extra_fields={"min_global_steps": 0, "max_global_steps": 0},
        )

    request = {"request_id": "first", "prompt_ids": [10], "sampling_params": {"temperature": 0.7}}
    cache = PolicyCache(tmp_path, 0, {"uid": "a"})
    original = await cache.generate(SimpleNamespace(generate=generate), request)
    resumed = PolicyCache(tmp_path, 0, {"uid": "a"}, required=True)
    replay = await resumed.generate(SimpleNamespace(generate=generate), {**request, "request_id": "resumed"})
    assert replay == original and len(calls) == 1
    with pytest.raises(ValueError, match="Frozen input"):
        await resumed.generate(SimpleNamespace(generate=generate), {**request, "prompt_ids": [11]})
    saved = read_json(tmp_path / "response.json")
    saved["output"]["token_ids"] = [7, 8]
    write_json(tmp_path / "response.json", saved)
    with pytest.raises(ValueError, match="checksum"):
        await resumed.generate(SimpleNamespace(generate=generate), request)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["missing", "partial", "version", "logprobs"])
async def test_invalid_or_missing_policy_responses_fail_closed(tmp_path, bad):
    async def generate(**kwargs):
        assert bad != "missing", "A resumed response must not be regenerated"
        return TokenOutput(
            token_ids=[1],
            log_probs=None if bad == "logprobs" else [-0.5],
            stop_reason="aborted" if bad == "partial" else "completed",
            extra_fields={"min_global_steps": 0, "max_global_steps": 1 if bad == "version" else 0},
        )

    cache = PolicyCache(tmp_path, 0, {}, required=bad == "missing")
    with pytest.raises(RuntimeError):
        await cache.generate(SimpleNamespace(generate=generate), {"prompt_ids": [0], "sampling_params": {}})
    assert not (tmp_path / "response.json").exists()


class Queue:
    def __init__(self):
        self.tags, self.fields = {}, {}

    def put(self, keys, partition_id, tags, fields=None):
        assert partition_id == "train"
        for i, (key, tag) in enumerate(zip(keys, tags, strict=True)):
            self.tags.setdefault(key, {}).update(tag)
            if fields is not None:
                self.fields[key] = fields[i].clone()

    def clear(self, keys, partition_id):
        for key in keys:
            self.tags.pop(key, None)
            self.fields.pop(key, None)

    def get(self, keys, partition_id):
        return torch.stack([self.fields[key] for key in keys])


class Manager:
    def __init__(self, queue, events, n):
        self.queue, self.events, self.n = queue, events, n
        self.jobs, self.failures = {}, set()

    def complete(self, uid):
        job = self.jobs[uid]
        if job["record"] is not None:
            return
        if uid in self.failures:
            job["record"] = {"uid": uid, "dropped": True, "stage": "h3"}
            self.queue.tags[uid]["status"] = "failure"
            return
        job["record"] = {"uid": uid, "dropped": False}
        self.queue.tags[uid]["status"] = "finished"
        for i in range(self.n):
            self.queue.tags[f"{uid}_{i}_0"] = {
                "status": "success",
                "global_steps": job["step"],
                "min_global_steps": job["version"],
                "max_global_steps": job["version"],
            }

    def generate_sequences(self, batch):
        step = tu.get_non_tensor_data(batch, "cohort_step", None)
        version = tu.get_non_tensor_data(batch, "behavior_version", None)
        self.events.append(("dispatch", step, version, len(batch)))
        for uid in batch["uid"]:
            self.jobs[uid] = {"step": step, "version": version, "policy_ready": True, "record": None}
        # External work A remains live until the next batch is admitted. This
        # makes waiting for A before dispatching B a reproducible deadlock.
        maximum = max(job["step"] for job in self.jobs.values())
        for uid, job in self.jobs.items():
            if job["step"] < maximum:
                self.complete(uid)

    def cohort_status(self, uids, release=False):
        if self.current_step() == 3:
            for uid in uids:
                self.complete(uid)
        result = {uid: deepcopy(self.jobs[uid]) for uid in uids}
        if release:
            for uid in uids:
                del self.jobs[uid]
        return result

    def _check_judge_coverage(self, step, records):
        assert all(self.jobs[row["uid"]]["step"] == step for row in records)
        self.events.append(("coverage", step, len(records)))


@pytest.fixture
def trainer(config, monkeypatch):
    queue, events = Queue(), []
    monkeypatch.setattr(module.tq, "kv_batch_put", queue.put)
    monkeypatch.setattr(module.tq, "kv_batch_get", queue.get)
    monkeypatch.setattr(module.tq, "kv_clear", queue.clear)
    monkeypatch.setattr(module.tq, "kv_list", lambda *args: {"train": deepcopy(queue.tags)})
    value = object.__new__(module.VideoCohortTrainer)
    value.config, value.trainer_mode = config, "colocate_async"
    value.cohorts, value.restored_failures, value.consumed_uids = {}, {}, []
    value.global_steps, value.total_training_steps = 1, 3
    value.steps_per_epoch = 3
    value.timing_raw, value.curr_step_profile = {}, False
    value.replay_buffer = value._build_replay_buffer()
    value.agent_loop_manager = Manager(queue, events, config.actor_rollout_ref.rollout.n)
    value.agent_loop_manager.current_step = lambda: value.global_steps
    value.checkpoint_manager = SimpleNamespace(
        abort_replicas=lambda: events.append(("abort", value.global_steps)),
        sleep_replicas=lambda: events.append(("sleep", value.global_steps)),
        update_weights=lambda step: events.append(("weights", step)),
        resume_generation_replicas=lambda: events.append(("resume", value.global_steps)),
    )
    counter = 0

    def fetch(count=None):
        nonlocal counter
        count = count or config.data.train_batch_size
        ids = [f"u{i}" for i in range(counter, counter + count)]
        counter += count
        return tu.get_tensordict({"uid": np.array(ids, dtype=object), "index": np.arange(count)})

    value._next_train_batch = fetch
    return value, queue, events


def test_two_cohorts_overlap_external_work_with_updates_and_stop_at_final_batch(trainer):
    trainer, queue, events = trainer
    consumed = []
    for step in (1, 2, 3):
        trainer.global_steps = step
        metrics = trainer.prepare_step()
        batch, lag = trainer.replay_buffer.sample(step, "train", 2)
        assert len(batch.keys) == 8
        assert lag["training/cohort/policy_lag_max"] == (0 if step == 1 else 1)
        consumed.extend(key.split("_")[0] for key in batch.keys[::4])
        trainer.on_sample_end()
        if step < 3:
            assert len(trainer.agent_loop_manager.jobs) == 2
            assert all(job["record"] is None for job in trainer.agent_loop_manager.jobs.values())
        events.append(("train", step))
        trainer.on_step_end()
        queue.clear(batch.keys, "train")
        assert metrics["training/cohort/prefetched_groups"] == (2 if step < 3 else 0)
    assert len(consumed) == len(set(consumed)) == 6
    assert [(e[1], e[2]) for e in events if e[0] == "dispatch"] == [(1, 0), (2, 0), (3, 1)]
    assert not trainer.cohorts and not queue.tags
    assert events.index(("dispatch", 2, 0, 2)) < events.index(("train", 1))


def test_epoch_limit_does_not_leave_an_unconsumed_cohort(trainer):
    trainer, _, events = trainer
    trainer.total_training_steps = 10
    trainer.config.trainer.total_epochs = 1
    trainer.global_steps = trainer.steps_per_epoch
    metrics = trainer.prepare_step()
    assert metrics["training/cohort/prefetched_groups"] == 0
    assert [event[1] for event in events if event[0] == "dispatch"] == [3]
    assert not trainer.cohorts


@pytest.mark.parametrize("bad", ["missing", "duplicate_session", "stale", "mixed", "unknown"])
def test_sampler_rejects_partial_or_wrong_version_groups(trainer, bad):
    trainer, queue, _ = trainer
    trainer.prepare_step()
    key = "u0_0_0"
    if bad == "missing":
        queue.tags.pop(key)
    elif bad == "duplicate_session":
        queue.tags["u0_0_1"] = queue.tags.pop(key)
    elif bad == "stale":
        trainer.global_steps = 3
    elif bad == "mixed":
        queue.tags[key]["max_global_steps"] = 1
    else:
        queue.tags[key]["min_global_steps"] = None
    with pytest.raises(RuntimeError):
        trainer.replay_buffer.sample(trainer.global_steps, "train", 2)
    assert "u0" in queue.tags  # Validation happens before consuming prompts.


def test_failed_group_is_replaced_inside_its_cohort(trainer):
    trainer, queue, events = trainer
    trainer.agent_loop_manager.failures.add("u0")
    metrics = trainer.prepare_step()
    batch, _ = trainer.replay_buffer.sample(1, "train", 2)
    assert {key.split("_")[0] for key in batch.keys} == {"u1", "u4"}
    assert "u0" not in queue.tags and "u0" not in trainer.agent_loop_manager.jobs
    assert metrics["training/cohort/replaced_groups"] == 1
    assert ("dispatch", 1, 0, 1) in events
    assert set(trainer.cohorts[2]["members"]) == {"u2", "u3"}


def test_checkpoint_preserves_only_unconsumed_cohort_and_replays_cached_policy(trainer, monkeypatch):
    trainer, queue, events = trainer
    trainer.prepare_step()
    trainer.replay_buffer.sample(1, "train", 2)
    trainer.on_sample_end()
    path = Path(trainer.config.trainer.default_local_dir) / "global_step_1/video-cohorts.json"

    def native_save(self):
        assert read_json(path)["cohorts"]["2"]["members"] == {"u2": 0, "u3": 0}

    monkeypatch.setattr(module.PPOTrainerSync, "_save_checkpoint", native_save)
    trainer._save_checkpoint()
    for uid in ("u2", "u3"):
        for i in range(4):
            write_json(
                Path(trainer.config.video_rl.output_dir)
                / "train/groups"
                / item_key(uid)
                / "policy"
                / str(i)
                / "response.json",
                {"fixture": True},
            )
    monkeypatch.setattr(module.PPOTrainerSync, "_load_checkpoint", lambda self: None)
    trainer.cohorts = {}
    trainer._load_checkpoint()
    assert set(trainer.cohorts) == {2}
    trainer.agent_loop_manager.jobs = {}
    trainer.global_steps = 2
    assert trainer._reissue_inflight_prompts() == 2
    assert set(queue.tags) == {"u2", "u3"}  # Consumed/partially published trajectories were discarded.
    assert [(e[1], e[2]) for e in events if e[0] == "dispatch"][-2:] == [(2, 0), (2, 0)]
