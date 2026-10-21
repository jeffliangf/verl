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
"""Native SFT with full video evaluation using the resident model after each epoch."""

from .resident import ResidentEvaluator, capture_generation_forwards


def attach_evaluation(trainer, evaluator, *, eval_before_training=False):
    """Evaluate the initial model or finish a saved epoch's evaluation before training."""
    step, size = trainer.resume_global_step, trainer.steps_per_epoch
    if size < 1 or step % size:
        raise ValueError("Expected a checkpoint at a complete SFT epoch boundary")
    save = trainer.ckpt_handler.save_checkpoint

    def save_and_evaluate(step):
        save(step=step)
        evaluator.evaluate(step)

    trainer.ckpt_handler.save_checkpoint = save_and_evaluate
    if step:
        # Saved before the exhausted iterator's final StopIteration. fit() starts
        # at the next epoch; rebuilding avoids skipping that epoch after a restart.
        trainer._build_dataloader()
        trainer.ckpt_handler.train_dataloader = trainer.train_dataloader
        evaluator.evaluate(step)
    elif eval_before_training:
        evaluator.evaluate(0)


def main():
    """Keep the standard optimizer, LR schedule, loss validation, and checkpoints."""
    from verl.trainer import sft_trainer

    class VideoSFTTrainer(sft_trainer.SFTTrainer):
        def _build_engine(self):
            self.generation_forwards = capture_generation_forwards(self.model_config.hf_config)
            super()._build_engine()

        def __init__(self, config):
            super().__init__(config)
            evaluator = ResidentEvaluator(self, self.generation_forwards)
            attach_evaluation(self, evaluator, eval_before_training=config.trainer.eval_before_training)

    sft_trainer.SFTTrainer = VideoSFTTrainer
    sft_trainer.main()


if __name__ == "__main__":
    main()
