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
"""One fixed batch of lookahead, with complete responses and bounded policy lag."""

import time
from collections import defaultdict
from pathlib import Path

import torch
import transfer_queue as tq
from tensordict import NonTensorData, TensorDict

from verl.trainer.ppo.v1.replay_buffer import ReplayBuffer
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu
from verl.utils.debug import marked_timer

from .cohort_config import validate_config
from .observability import record_event
from .storage import item_key, read_json, write_json


def advantages_without_dropped(uids, scores, dropped, *, norm_adv_by_std_in_grpo=True, epsilon=1e-6):
    """GRPO outcome advantages computed over each group's kept responses only.

    Matches verl's compute_grpo_outcome_advantage on the kept responses. Dropped responses get 0, and so does a
    group with fewer than two kept responses: it has nothing to compare (verl would use the raw score).
    """
    advantages = torch.zeros(len(uids), dtype=torch.float32)
    groups = defaultdict(list)
    for i, (uid, drop) in enumerate(zip(uids, dropped, strict=True)):
        if not drop:
            groups[uid].append(i)
    for members in groups.values():
        if len(members) < 2:
            continue
        group = scores[members].float()
        centered = group - torch.mean(group)
        advantages[members] = centered / (torch.std(group) + epsilon) if norm_adv_by_std_in_grpo else centered
    return advantages


def _as_list(value):
    return value.tolist() if hasattr(value, "tolist") else list(value)


class CohortReplayBuffer(ReplayBuffer):
    """Consume exactly the nominated cohort; validation retains native behavior."""

    def select(self, members, group_size):
        self.members, self.group_size = dict(members), group_size

    def sample(self, global_steps, partition_id, batch_size):
        if partition_id == "val":
            return super().sample(global_steps, partition_id, batch_size)
        if len(self.members) != batch_size:
            raise RuntimeError("Cohort size differs from the learner batch")
        self._sync_metadata_from_transfer_queue()
        if not set(self.members) <= self.finished_keys[partition_id]:
            raise RuntimeError("Cannot train an incomplete cohort")
        counts, lags = dict.fromkeys(self.members, 0), []
        for key, tag in self.partitions[partition_id].items():
            uid = key.split("_", 1)[0]
            if uid not in self.members:
                continue
            expected = self.members[uid]
            low, high = tag.get("min_global_steps"), tag.get("max_global_steps")
            if type(low) is not int or type(high) is not int or low != high or low != expected:
                raise RuntimeError(f"Group {uid} has unknown, mixed, or mismatched behavior versions: {low}, {high}")
            lag = global_steps - 1 - low
            if lag not in {0, 1}:
                raise RuntimeError(f"Group {uid} has policy lag {lag}; only zero or one is allowed")
            counts[uid] += 1
            lags.append(lag)
        expected_keys = {f"{uid}_{i}_0" for uid in self.members for i in range(self.group_size)}
        actual_keys = {key for key in self.partitions[partition_id] if key.split("_", 1)[0] in self.members}
        if actual_keys != expected_keys or any(count != self.group_size for count in counts.values()):
            raise RuntimeError("A cohort must contain every sibling exactly once, without padding")
        batch = self._materialize_batch(partition_id, list(self.members), self.partitions[partition_id])
        return batch, {
            "training/cohort/policy_lag_mean": sum(lags) / len(lags),
            "training/cohort/policy_lag_max": max(lags),
        }


class VideoCohortTrainer(PPOTrainerSync):
    """Share the existing GPUs; external H3/judge work crosses optimizer updates."""

    def __init__(self, config):
        validate_config(config)
        self.cohorts, self.restored_failures = {}, {}
        self.consumed_uids = []
        super().__init__(config)

    def _build_replay_buffer(self):
        return CohortReplayBuffer(
            trainer_mode=self.trainer_mode,
            trainer_config={},
            max_off_policy_threshold=None,
            max_off_policy_strategy="wait",
            sampler_kwargs={},
        )

    def _checkpoint_path(self):
        if self.config.trainer.resume_mode == "resume_path":
            return Path(self.config.trainer.resume_from_path)
        return Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"

    def _load_checkpoint(self):
        super()._load_checkpoint()
        if not self.global_steps:
            return
        path = self._checkpoint_path() / "video-cohorts.json"
        if not path.exists():
            if self._restored_tq_prompt_count:
                raise RuntimeError("Async checkpoint is missing its video-cohort manifest")
            return  # A synchronous checkpoint has no prefetched work.
        state = read_json(path)
        if state["step"] != self.global_steps or state["group_size"] != self.config.actor_rollout_ref.rollout.n:
            raise ValueError("Video cohort checkpoint identity mismatch")
        self.cohorts = {int(step): value for step, value in state["cohorts"].items()}
        if set(self.cohorts) - {self.global_steps + 1}:
            raise ValueError("Checkpoint contains more than one batch of lookahead")
        for cohort in self.cohorts.values():
            if len(cohort["members"]) != self.config.data.train_batch_size:
                raise ValueError("Checkpoint cohort cardinality changed")
            if any(self.global_steps - version not in {0, 1} for version in cohort["members"].values()):
                raise ValueError("Checkpoint contains stale behavior policies")
        self.restored_failures = state["failures"]

    def _dispatch(self, batch, step, version, *, restored=False):
        tu.assign_non_tensor_data(batch, "global_steps", step)
        tu.assign_non_tensor_data(batch, "cohort_step", step)
        tu.assign_non_tensor_data(batch, "behavior_version", version)
        tu.assign_non_tensor_data(batch, "require_cached_policy", restored)
        uids = list(batch["uid"])
        tags = [
            {
                "is_prompt": True,
                "status": "pending",
                "global_steps": step,
                "cohort_step": step,
                "behavior_version": version,
            }
            for _ in uids
        ]
        tq.kv_batch_put(
            keys=uids,
            partition_id="train",
            tags=tags,
            fields=batch.select(*[key for key in batch.keys() if not isinstance(batch.get(key), NonTensorData)]),
        )
        self.agent_loop_manager.generate_sequences(batch)
        self.cohorts[step]["members"].update(dict.fromkeys(uids, version))

    def _reissue_inflight_prompts(self, partition_id="train"):
        items = (tq.kv_list(partition_id) or {}).get(partition_id, {})
        members = {uid for cohort in self.cohorts.values() for uid in cohort["members"]}
        prompts = {key for key, tag in items.items() if tag.get("is_prompt")}
        if members != prompts:
            raise RuntimeError("TransferQueue prompts do not match the checkpointed cohort membership")
        # Rebuild trajectories from complete response caches and durable H3/pair
        # artifacts. This tolerates a TQ snapshot taken during group publication.
        trajectories = [key for key, tag in items.items() if not tag.get("is_prompt")]
        if trajectories:
            tq.kv_clear(keys=trajectories, partition_id=partition_id)
        for step, cohort in self.cohorts.items():
            for uid, version in list(cohort["members"].items()):
                if uid in self.restored_failures:
                    continue
                root = Path(self.config.video_rl.output_dir) / "train/groups" / item_key(uid) / "policy"
                if not all(
                    (root / str(i) / "response.json").exists() for i in range(self.config.actor_rollout_ref.rollout.n)
                ):
                    raise RuntimeError(f"Checkpointed policy responses are missing for {uid}")
                batch = tq.kv_batch_get(keys=[uid], partition_id=partition_id)
                self._dispatch(batch, step, version, restored=True)
        return len(members)

    def _status(self, uids):
        live = [uid for uid in uids if uid not in self.restored_failures]
        status = self.agent_loop_manager.cohort_status(live) if live else {}
        status.update(
            {
                uid: {"record": self.restored_failures[uid], "policy_ready": True}
                for uid in uids
                if uid in self.restored_failures
            }
        )
        return status

    def _all_uids(self):
        return [uid for cohort in self.cohorts.values() for uid in cohort["members"]]

    def _drain(self, uids):
        with marked_timer("cohort_policy_drain", self.timing_raw):
            while not all(value["policy_ready"] for value in self._status(uids).values()):
                time.sleep(0.1)

    def _new_cohort(self, step):
        self.cohorts[step] = {"members": {}, "dropped": 0}
        self._dispatch(self._next_train_batch(), step, self.global_steps - 1)

    def prepare_step(self):
        if self.global_steps not in self.cohorts:
            self._new_cohort(self.global_steps)
        current = self.cohorts[self.global_steps]
        # Only one full cohort enters token generation at once. H3/judging for
        # the previous cohort already run independently of its token requests.
        self._drain(list(current["members"]))
        last_step = min(self.total_training_steps, self.steps_per_epoch * self.config.trainer.total_epochs)
        if self.global_steps < last_step:
            self._new_cohort(self.global_steps + 1)
        with marked_timer("cohort_reward_wait", self.timing_raw):
            while True:
                statuses = self._status(self._all_uids())  # Observe errors in BOTH cohorts.
                dropped = [
                    uid
                    for uid in current["members"]
                    if statuses[uid]["record"] is not None and statuses[uid]["record"]["dropped"]
                ]
                for uid in dropped:
                    current["dropped"] += 1
                    if current["dropped"] > self.config.video_rl.max_dropped_groups_per_step:
                        raise RuntimeError("Cohort provider failures exceeded the replacement limit")
                    self.replay_buffer._sync_metadata_from_transfer_queue()
                    self.replay_buffer._clear_groups("train", {uid})
                    if uid not in self.restored_failures:
                        self.agent_loop_manager.cohort_status([uid], release=True)
                    self.restored_failures.pop(uid, None)
                    del current["members"][uid]
                    self._dispatch(self._next_train_batch(1), self.global_steps, self.global_steps - 1)
                if dropped:
                    continue
                if all(statuses[uid]["record"] is not None for uid in current["members"]):
                    records = [statuses[uid]["record"] for uid in current["members"]]
                    break
                time.sleep(0.1)
        self.agent_loop_manager._check_judge_coverage(self.global_steps, records)
        self.replay_buffer.select(current["members"], self.config.actor_rollout_ref.rollout.n)
        self.consumed_uids = list(current["members"])
        del self.cohorts[self.global_steps]
        return {
            "training/cohort/replaced_groups": current["dropped"],
            "training/cohort/prefetched_groups": len(self._all_uids()),
        }

    def _compute_advantage(self, batch, metrics):
        batch = super()._compute_advantage(batch, metrics)
        if self.config.video_rl.get("no_call", "forfeit") == "drop":
            batch = self._drop_no_call_turns(batch, metrics)
        return batch

    def _drop_no_call_turns(self, batch, metrics):
        """Give turns without a tool call no gradient and keep them out of their group's GRPO statistics.

        The tournament marks them "no_call_dropped" and gives their siblings no wins over them. Groups without a
        dropped turn keep verl's advantages unchanged; in the others the kept responses are re-normalized among
        themselves, and the dropped turns get zero advantage and an empty loss mask (no policy, KL or entropy
        term, and no share of the token-mean denominator).
        """
        if any(tag.get("is_padding", False) for tag in batch.tags):
            raise RuntimeError("Dropping turns without a tool call expects complete cohorts without padding")
        names = ("response_mask", "loss_mask", "advantages", "returns")
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["uid", "rm_scores", "extra_fields", *names]
        )
        extra = [value if isinstance(value, dict) else {} for value in _as_list(data["extra_fields"])]
        dropped = [bool((value.get("reward_extra_info") or {}).get("no_call_dropped")) for value in extra]
        uids = [str(uid) for uid in _as_list(data["uid"])]
        affected = {uid for uid, drop in zip(uids, dropped, strict=True) if drop}
        kept = defaultdict(int)
        for uid, drop in zip(uids, dropped, strict=True):
            kept[uid] += not drop
        metrics["training/no_call/dropped_fraction"] = sum(dropped) / len(dropped)
        metrics["training/no_call/groups_with_drops"] = len(affected)
        metrics["training/no_call/groups_without_signal"] = sum(kept[uid] < 2 for uid in affected)
        if not affected:
            return batch
        scores = torch.stack([row.sum() for row in data["rm_scores"].unbind()])
        advantages = advantages_without_dropped(
            uids,
            scores,
            dropped,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
        )
        rows = {name: list(data[name].unbind()) for name in names}
        for i, uid in enumerate(uids):
            if uid not in affected:
                continue
            if dropped[i]:
                rows["response_mask"][i] = torch.zeros_like(rows["response_mask"][i])
                rows["loss_mask"][i] = torch.zeros_like(rows["loss_mask"][i])
            mask = rows["response_mask"][i].to(rows["advantages"][i].dtype)
            rows["advantages"][i] = advantages[i].to(mask.dtype) * mask
            rows["returns"][i] = rows["advantages"][i].clone()
        output = TensorDict(
            {name: torch.nested.as_nested_tensor(values, layout=torch.jagged) for name, values in rows.items()},
            batch_size=len(batch.keys),
        )
        return tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

    def on_sample_end(self):
        self._drain(self._all_uids())
        # All legitimate requests have complete, durable responses. Abort only
        # now, to also retire server-side remnants of a timed-out failed request.
        self.checkpoint_manager.abort_replicas()
        super().on_sample_end()
        self.agent_loop_manager.cohort_status(self.consumed_uids, release=True)
        record_event(
            Path(self.config.video_rl.output_dir) / "timings.jsonl",
            "cohort_update",
            "begin",
            learner_step=self.global_steps,
            prefetched_groups=len(self._all_uids()),
        )

    def on_step_end(self):
        super().on_step_end()
        self.checkpoint_manager.resume_generation_replicas()

    def _save_checkpoint(self):
        statuses = self._status(self._all_uids())
        if not all(value["policy_ready"] for value in statuses.values()):
            raise RuntimeError("Cannot checkpoint a partially generated policy response")
        state = {
            "step": self.global_steps,
            "group_size": self.config.actor_rollout_ref.rollout.n,
            "cohorts": self.cohorts,
            "failures": {
                uid: value["record"]
                for uid, value in statuses.items()
                if value["record"] is not None and value["record"]["dropped"]
            },
        }
        path = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}" / "video-cohorts.json"
        write_json(path, state)
        # Native latest-checkpoint publication follows model, data cursor and TQ
        # persistence; the cohort manifest is already durable before that point.
        super()._save_checkpoint()
