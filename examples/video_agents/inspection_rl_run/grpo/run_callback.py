"""Start from base weights, retain two verified saves, and finish within 72 hours."""
import json
import time
from pathlib import Path

from canary_ops import atomic_json, evaluation_status, retain_latest, verify_checkpoint
from run_budget import finish_decision
from verl.trainer.ppo.checkpoint_callback import CheckpointCallback


class ExperimentCheckpoint(CheckpointCallback):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.root = Path(config.trainer.default_local_dir).resolve()
        self.manifest = json.loads((self.root.parent / "launch.json").read_text())
        self.world = config.trainer.nnodes * config.trainer.n_gpus_per_node
        if self.manifest["initial_checkpoint_step"] != 0 or self.manifest["source_checkpoint"] is not None:
            raise ValueError("This experiment must start from the baseline model")
        if config.trainer.resume_mode != "disable" or config.trainer.resume_from_path is not None:
            raise ValueError("The launch must not restore another experiment")
        if config.trainer.del_local_ckpt_after_load:
            raise ValueError("Checkpoints must be preserved during restoration")
        if config.trainer.total_training_steps != self.manifest["total_training_steps"]:
            raise ValueError("Unexpected step ceiling")
        if config.algorithm.norm_adv_by_std_in_grpo != self.manifest["norm_adv_by_std_in_grpo"]:
            raise ValueError("Unexpected GRPO variant")
        marker = self.root / "latest_checkpointed_iteration.txt"
        selected, step = None, 0
        if marker.exists():
            step = int(marker.read_text())
            if not 0 < step <= self.manifest["total_training_steps"]:
                raise ValueError("Invalid recovery checkpoint step")
            selected = self.root / f"global_step_{step}"
            verify_checkpoint(self.root, step, self.world)
            config.trainer.resume_mode = "resume_path"
            config.trainer.resume_from_path = str(selected)
        elif (self.root / "verified.json").exists() or any(self.root.glob("global_step_*")):
            raise ValueError("Checkpoint history without a latest marker; refusing to discard it")
        scheduled = step == 0 or step % config.trainer.test_freq == 0
        config.trainer.val_before_train = scheduled and not evaluation_status(self.root.parent, step)["complete"]
        finish_path = self.root.parent / "finish-request.json"
        if finish_path.exists():
            final_step = int(json.loads(finish_path.read_text())["final_step"])
            if selected is None or final_step != step:
                raise ValueError("Final budget-stop step differs from the verified checkpoint")
            if evaluation_status(self.root.parent, step)["complete"]:
                raise RuntimeError("Final evaluation already complete; refusing redundant training")
            config.trainer.total_training_steps = step
            config.trainer.val_only = True
            config.trainer.val_before_train = True
        atomic_json(self.root.parent / "restore-request.json", {
            "checkpoint": str(selected) if selected else None,
            "requested_step": step,
            "initialization": "own checkpoint recovery" if selected else "baseline Qwen weights; fresh optimizer",
            "evaluate_before_next_update": config.trainer.val_before_train,
            "checked_at_epoch": time.time(),
        })

    def on_save(self, trainer, global_step, checkpoint_dir, async_save=False, **kwargs):
        if async_save or Path(checkpoint_dir).resolve() != self.root / f"global_step_{global_step}":
            raise ValueError("Expected a synchronous checkpoint in this experiment's own directory")
        if not 0 < global_step <= self.manifest["total_training_steps"]:
            raise ValueError("Save is outside this experiment")
        deadline = time.monotonic() + 180
        while True:
            try:
                result = retain_latest(self.root, global_step, self.world, keep=2)
                break
            except FileNotFoundError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(5)
        print(f"EXPERIMENT_CHECKPOINT_VERIFIED step={global_step} bytes={result['bytes']} "
              f"retained={result['retained_checkpoints']}", flush=True)
        decision = finish_decision(self.root.parent, global_step, self.config.trainer.save_freq)
        if decision["finish"]:
            atomic_json(self.root.parent / "finish-request.json", {**decision, "requested_at_epoch": time.time()})
            trainer.total_training_steps = global_step
            self.config.trainer.total_training_steps = global_step
            self.config.trainer.test_freq = 1
            print(f"EXPERIMENT_BUDGET_FINISH step={global_step} "
                  f"remaining_seconds={decision['remaining_seconds']:.0f}", flush=True)
