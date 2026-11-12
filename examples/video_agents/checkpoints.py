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
"""Keep recent video-agent checkpoints and the best balanced held-out score."""

import logging
import math
import os
import re
import shutil
import zipfile
from collections import Counter
from hashlib import sha256
from pathlib import Path

from verl.trainer.ppo.checkpoint_callback import CheckpointCallback

from .cohort_config import TRAINER_CLASS
from .storage import read_json, write_json

logger = logging.getLogger(__name__)

SCORE_METRICS = (
    "val-aux/aa_eval/score/mean@1",
    "val-aux/movie_gen_video_bench/score/mean@1",
)
TASK_SCORE_METRICS = {task: f"val-core/{task}/score/mean@1" for task in ("t2v", "i2v")}
OVERALL_SCORE = "val-core/overall/score/mean@1"
COMPLETE_MARKER = "checkpoint_complete.json"


def register_tensorboard_run(run_dir, log_dir):
    """Expose future runs under a directory watched by the TensorBoard server."""
    run_dir, log_dir = Path(run_dir).resolve(), Path(log_dir).resolve()
    registry_root = (
        os.environ.get("VIDEO_AGENT_TENSORBOARD_RUNS_DIR") or Path.home() / ".cache/verl/tensorboard/video-agents"
    )
    registry = Path(registry_root).expanduser().resolve()
    if registry.is_relative_to(log_dir):
        raise ValueError("TensorBoard registry must be outside the run's log directory")
    registry.mkdir(parents=True, exist_ok=True)
    label = run_dir.parent.name if run_dir.name in {"grpo", "drgrpo"} else run_dir.name
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip(".") or "video-agent-run"
    suffix = sha256(str(log_dir).encode()).hexdigest()[:10]
    for name in (label, f"{label}-{suffix}"):
        link = registry / name
        try:
            link.symlink_to(log_dir, target_is_directory=True)
        except FileExistsError:
            if not link.is_symlink() or link.resolve() != log_dir:
                continue
        return link
    raise FileExistsError(f"TensorBoard registry entries already exist for {label}")


def overall_score(metrics):
    """Balance the evaluated tasks (one or both) for unified runs, or the two legacy T2V datasets."""
    keys = [key for key in TASK_SCORE_METRICS.values() if key in metrics] or SCORE_METRICS
    scores = [metrics.get(key) for key in keys]
    if any(value is None or not math.isfinite(value) for value in scores):
        return None
    return sum(float(value) for value in scores) / len(scores)


class VideoCheckpointCallback(CheckpointCallback):
    """Retain the latest N complete FSDP saves plus the highest evaluation score.

    Worker-side rotation must be disabled so it cannot delete the best save.
    Completion markers identify the directories owned by this callback; existing
    checkpoints without these markers are left alone. Selection is persisted in
    ``best_checkpoint.json`` and restored when the trainer restarts.
    """

    save_on_validation = True

    def __init__(self, config=None):
        super().__init__(config=config)
        self.root = Path(config.trainer.default_local_dir).resolve()
        self.keep_latest = config.video_rl.checkpoint_keep_latest
        if type(self.keep_latest) is not int or self.keep_latest < 1:
            raise ValueError("video_rl.checkpoint_keep_latest must be a positive integer")
        actor = config.actor_rollout_ref.actor
        self.video_cohorts = (
            config.video_rl.async_lookahead
            and config.trainer.v1.trainer_mode == "colocate_async"
            and config.trainer.v1.trainer_class == TRAINER_CLASS
        )
        if not config.trainer.use_v1 or (config.trainer.v1.trainer_mode != "sync" and not self.video_cohorts):
            raise ValueError("Video checkpoint selection requires synchronous V1 or the video cohort trainer")
        if actor.strategy not in ("fsdp", "fsdp2") or actor.checkpoint.get("async_save", False):
            raise ValueError("Video checkpoint retention requires synchronous FSDP saves")
        if not {"model", "optimizer", "extra"}.issubset(actor.checkpoint.save_contents):
            raise ValueError("Video checkpoint retention requires model, optimizer, and extra state")
        if (
            config.trainer.get("max_actor_ckpt_to_keep") is not None
            or config.trainer.get("max_critic_ckpt_to_keep") is not None
            or config.trainer.get("remove_previous_ckpt_in_save", False)
            or config.trainer.del_local_ckpt_after_load
        ):
            raise ValueError("Disable worker checkpoint deletion; the video callback owns retention")
        self.world_size = config.trainer.nnodes * config.trainer.n_gpus_per_node
        self.best_path = self.root / "best_checkpoint.json"
        self.best = read_json(self.best_path) if self.best_path.exists() else None
        if self.best is not None:
            score = overall_score(self.best["metrics"])
            if score is None or score != self.best["score"]:
                raise ValueError("Best checkpoint record has inconsistent scores")
            if self.best["checkpoint"] != f"global_step_{self.best['step']}":
                raise ValueError("Best checkpoint record has an inconsistent path")
            self._verify(self.best["step"])

    def _directory(self, step):
        if type(step) is not int or step < 0:
            raise ValueError("Checkpoint step must be a nonnegative integer")
        directory = self.root / f"global_step_{step}"
        if directory.is_symlink() or directory.resolve().parent != self.root:
            raise ValueError(f"Unsafe checkpoint directory: {directory}")
        return directory

    def _verify(self, step):
        """Verify all required shards before publishing or deleting checkpoints."""
        directory = self._directory(step)
        actor = directory / "actor"
        if actor.is_symlink() or read_json(actor / "fsdp_config.json")["world_size"] != self.world_size:
            raise ValueError("Checkpoint actor directory or world size is inconsistent")
        paths = [directory / "data.pt"]
        paths.extend(
            actor / f"{kind}_world_size_{self.world_size}_rank_{rank}.pt"
            for rank in range(self.world_size)
            for kind in ("model", "optim", "extra_state")
        )
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Unsafe checkpoint file: {path}")
            with zipfile.ZipFile(path) as archive:
                if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                    raise ValueError(f"Invalid PyTorch checkpoint archive: {path}")

    def _completed_steps(self):
        """Read only checkpoints that this callback has successfully verified."""
        steps = []
        for marker in self.root.glob(f"global_step_*/{COMPLETE_MARKER}"):
            step = read_json(marker)["step"]
            if marker.is_symlink() or marker.parent != self._directory(step):
                raise ValueError(f"Unsafe checkpoint completion marker: {marker}")
            steps.append(step)
        return sorted(steps, reverse=True)

    def _prune(self):
        steps = self._completed_steps()
        retained = set(steps[: self.keep_latest])
        if self.best is not None:
            retained.add(self.best["step"])
        for step in retained:
            self._verify(step)
        for step in steps:
            if step not in retained:
                shutil.rmtree(self._directory(step))

    def on_save(self, trainer, global_step, checkpoint_dir, async_save=False, **kwargs):
        directory = self._directory(global_step)
        if async_save or Path(checkpoint_dir).resolve() != directory:
            raise ValueError("Expected a synchronous checkpoint in this run's directory")
        if int((self.root / "latest_checkpointed_iteration.txt").read_text()) != global_step:
            raise ValueError("Latest checkpoint marker does not match the completed save")
        if any(step > global_step for step in self._completed_steps()):
            raise ValueError("Use a new run directory when restarting from an older checkpoint")
        if self.video_cohorts:
            state = read_json(directory / "video-cohorts.json")
            queue = read_json(directory / "transfer_queue/metadata.json")
            if (
                state["step"] != global_step
                or queue["user_metadata"].get("global_steps") != global_step
                or not queue["storage_saved"]
                or not (directory / "transfer_queue/controller_state.pkl").is_file()
            ):
                raise ValueError("Video cohort and TransferQueue checkpoints must match the saved policy")
        self._verify(global_step)
        write_json(directory / COMPLETE_MARKER, {"step": global_step})
        self._prune()

    def _selection_metrics(self, global_step, metrics):
        if not any(key.startswith(("val-aux/t2v/", "val-aux/i2v/")) for key in metrics):
            return {key: metrics.get(key) for key in SCORE_METRICS}
        # Native metrics are grouped by source, whose batch sizes can differ.
        # Use the task means already computed from individual valid prompts.
        path = Path(self.config.video_rl.output_dir) / "val" / f"step_{global_step}" / "eval-summary.json"
        if not path.exists():
            return {}
        summary = read_json(path)
        samples = read_json(Path(self.config.video_rl.protocol))["samples"]["val"]
        expected = Counter(sample["task"] for sample in samples)
        tasks = summary.get("by_task", {})
        # A unified run may evaluate T2V or I2V alone; select on the tasks it evaluates.
        if (
            not expected
            or not set(expected) <= set(TASK_SCORE_METRICS)
            or set(tasks) != set(expected)
            or summary.get("policy_step") != global_step
            or summary.get("prompts") != len(samples)
            or summary.get("degraded", True)
            or any(
                tasks[task].get("prompts") != count or tasks[task].get("degraded", True)
                for task, count in expected.items()
            )
        ):
            return {}
        return {metric: tasks[task].get("score") for task, metric in TASK_SCORE_METRICS.items() if task in expected}

    def _publish_evaluation_media(self, global_step):
        """Publish qualitative samples once on the driver, without affecting selection."""
        if "tensorboard" not in self.config.trainer.logger:
            return
        try:
            output_dir = Path(self.config.video_rl.output_dir)
            log_dir = Path(os.environ.get("TENSORBOARD_DIR", str(output_dir / "tensorboard")))
            try:
                register_tensorboard_run(output_dir, log_dir)
            except Exception:
                logger.warning("Could not register TensorBoard run %s", output_dir, exc_info=True)
            settings = dict(self.config.video_rl.get("eval_media", {}) or {})
            if not settings.get("enabled", True):
                return
            from .eval_media import publish_evaluation_media

            report = publish_evaluation_media(
                output_dir,
                global_step,
                Path(self.config.video_rl.protocol),
                settings=settings,
                log_dir=log_dir,
            )
            if report.get("errors"):
                logger.warning("Evaluation media at step %s: %s", global_step, "; ".join(report["errors"]))
        except Exception:
            # Checkpoint integrity failures still propagate from the rest of the
            # callback. Missing preview dependencies or media must not abort it.
            logger.warning("Could not publish evaluation media at step %s", global_step, exc_info=True)

    def on_validation(self, trainer, global_step, metrics, **kwargs):
        self._publish_evaluation_media(global_step)
        selection = self._selection_metrics(global_step, metrics)
        score = overall_score(selection)
        if score is None:
            logger.warning(
                "Skipping overall score at step %s: complete, finite evaluation scores are required", global_step
            )
            return
        # The trainer logs this same dict to TensorBoard, including initial and
        # validation-only passes that do not have a training checkpoint.
        metrics.update(selection)
        metrics[OVERALL_SCORE] = score
        if self.config.trainer.get("val_only", False):
            return
        directory = self._directory(global_step)
        if not (directory / COMPLETE_MARKER).exists():
            return  # Initial base-model evaluation has no saved training state.
        if self.best is not None and score <= self.best["score"]:
            return  # Ties keep the existing winner.
        self._verify(global_step)
        best = {
            "step": global_step,
            "checkpoint": directory.name,
            "score": score,
            "metrics": {key: float(value) for key, value in selection.items()},
        }
        # Publish the new winner before releasing the previous best checkpoint.
        write_json(self.best_path, best)
        self.best = best
        self._prune()
        logger.info("Best checkpoint is step %s with overall score %s", global_step, score)
