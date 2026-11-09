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
"""Offline validation of the complete-response lookahead contract."""

TRAINER_CLASS = "examples.video_agents.cohort_trainer.VideoCohortTrainer"


def validate_config(config):
    """Refuse configurations that would silently change the two-cohort contract."""
    rollout, algorithm = config.actor_rollout_ref.rollout, config.algorithm
    correction = algorithm.rollout_correction
    if (
        not config.video_rl.async_lookahead
        or config.trainer.v1.trainer_mode != "colocate_async"
        or config.trainer.v1.trainer_class != TRAINER_CLASS
        or not config.video_rl.judge_streaming
        or not config.video_rl.judge_background_cleanup
        or not config.video_rl.judge_limits
        or not rollout.calculate_log_probs
        or correction.bypass_mode
        or correction.rollout_is != "token"
        or correction.rollout_rs is not None
    ):
        raise ValueError("Use --async-lookahead with streaming cleanup and decoupled token importance correction")
    if (
        config.algorithm.get("filter_groups", {}).get("enable", False)
        or config.trainer.v1.colocate_async.get("parameter_sync_step", 1) != 1
        or config.actor_rollout_ref.actor.checkpoint.get("async_save", False)
        or config.skip.rollout_tq.enable
        or config.trainer.v1.sampler.custom_sampler.path
    ):
        raise ValueError(
            "Video cohorts require one update per batch, synchronous checkpoint saving, and their own sampler"
        )
    if config.video_rl.judge_limits.cleanup_backlog <= 2 * config.data.train_batch_size:
        raise ValueError("Cleanup backlog limit must exceed the two-cohort window so live readers can drain")
    if not config.trainer.v1.sampler.sync_refill_failed_groups:
        raise ValueError("Video cohorts require bounded replacement of failed groups")
