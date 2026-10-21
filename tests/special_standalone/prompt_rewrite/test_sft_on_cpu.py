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
"""Check SFT drop-last accounting, complete video cohorts, and baseline/epoch callbacks."""

from types import SimpleNamespace

import pytest

from examples.prompt_rewrite import sft
from examples.prompt_rewrite.prepare_sft import prepare_sft
from examples.prompt_rewrite.storage import read_json, write_jsonl
from examples.prompt_rewrite.train import attach_evaluation


@pytest.fixture
def inputs(tmp_path):
    train, validation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    write_jsonl(
        train,
        [{"id": f"train-{i}", "prompt": f"Training prompt {i}", "h3_context_ir": f"Teacher {i}"} for i in range(4)],
    )
    write_jsonl(
        validation,
        [
            {
                "id": f"eval-{i}",
                "prompt": f"Held-out prompt {i}",
                "h3_context_ir": f"Eval teacher {i}",
                "source": source,
            }
            for i, source in enumerate(("aa_eval", "movie_gen_video_bench"))
        ],
    )
    prepared = tmp_path / "data"
    prepare_sft(train, validation, prepared)
    return [
        "--run-dir",
        str(tmp_path / "run"),
        "--dataset",
        str(validation),
        "--nproc-per-node",
        "2",
        "--",
        f"data.train_files={prepared / 'train.parquet'}",
        f"data.val_files={prepared / 'val.parquet'}",
        "data.train_batch_size=2",
        "trainer.total_epochs=3",
    ]


def test_preset_and_complete_validation_coverage(inputs):
    args = sft.parse_args(inputs)
    config, _ = sft.training_config(args)
    summary = sft.validate_data(config, args.dataset)
    assert summary["steps_per_epoch"] == 2
    assert summary["eval_sources"] == {"aa_eval": 1, "movie_gen_video_bench": 1}
    assert summary["new_videos_per_epoch"] == 2 and summary["judgments_per_epoch"] == 8
    assert config.optim.lr == 2e-5
    assert config.data.max_length == 4096 and config.data.use_dynamic_bsz is False
    assert "hf_model" in config.checkpoint.save_contents
    assert config.trainer.test_freq == config.trainer.save_freq == "after_each_epoch"
    config.data.train_batch_size = 4
    with pytest.raises(ValueError, match="val:.*one complete global batch"):
        sft.validate_data(config, args.dataset)


def test_dropped_sft_rows_do_not_shrink_video_cohort(tmp_path):
    train, validation = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    write_jsonl(
        train,
        [{"id": f"train-{i}", "prompt": f"Train {i}", "h3_context_ir": f"Teacher {i}"} for i in range(130)],
    )
    write_jsonl(
        validation,
        [
            {
                "id": f"eval-{i}",
                "prompt": f"Eval {i}",
                "h3_context_ir": f"Eval teacher {i}",
                "source": "aa_eval" if i % 2 else "movie_gen_video_bench",
            }
            for i in range(66)
        ],
    )
    prepared = tmp_path / "data"
    prepare_sft(train, validation, prepared)
    args = sft.parse_args(
        [
            "--run-dir",
            str(tmp_path / "run"),
            "--dataset",
            str(validation),
            "--eval-before-training",
            "--",
            f"data.train_files={prepared / 'train.parquet'}",
            f"data.val_files={prepared / 'val.parquet'}",
        ]
    )
    config, _ = sft.training_config(args)
    summary = sft.validate_data(config, args.dataset)
    assert config.data.train_batch_size == 64 and config.trainer.eval_before_training is True
    assert summary["steps_per_epoch"] == 2
    assert summary["files"]["train"]["used_rows_per_epoch"] == 128
    assert summary["files"]["val"]["used_rows_per_epoch"] == 64
    assert summary["files"]["train"]["dropped_rows_per_epoch"] == 2
    assert summary["files"]["val"]["dropped_rows_per_epoch"] == 2
    assert summary["new_videos_per_epoch"] == 66 and summary["judgments_per_epoch"] == 264


def test_baseline_flag_can_be_disabled(inputs):
    config, _ = sft.training_config(sft.parse_args(["--no-eval-before-training", *inputs]))
    assert config.trainer.eval_before_training is False


def test_rejects_different_eval_targets(inputs):
    args = sft.parse_args(inputs)
    config, _ = sft.training_config(args)
    from examples.prompt_rewrite.storage import read_jsonl

    rows = read_jsonl(args.dataset)
    rows[0]["h3_context_ir"] = "Changed teacher target"
    write_jsonl(args.dataset, rows)
    with pytest.raises(ValueError, match="validation text/source differs"):
        sft.validate_data(config, args.dataset)


def test_preview_never_launches_training_or_remote_work(inputs, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sft.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected launch"))
    assert sft.main(inputs) == 0
    assert "examples.prompt_rewrite.train" in capsys.readouterr().out
    assert not (tmp_path / "run").exists()


def test_single_resident_trainer_launch_and_frozen_resume(inputs, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sft, "check_tokens", lambda config: calls.append("tokens"))
    monkeypatch.setattr(sft.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    assert sft.main(["--execute", *inputs]) == 0
    assert calls[0] == "tokens" and len(calls) == 2
    command, kwargs = calls[1]
    assert "examples.prompt_rewrite.train" in command
    assert kwargs["check"] is True and "TENSORBOARD_DIR" in kwargs["env"]
    saved = read_json(tmp_path / "run/sft-inputs.json")
    assert saved["config"]["trainer"]["total_epochs"] == 3
    assert saved["data"]["steps_per_epoch"] == 2
    # A new schedule must not silently resume this experiment's optimizer state.
    assert sft.main(["--execute", *inputs[:-1], "trainer.total_epochs=4"]) == 1
    assert len(calls) == 2


def test_every_saved_epoch_is_evaluated_and_resume_keeps_scheduler():
    events = []
    trainer = SimpleNamespace(
        resume_global_step=2,
        steps_per_epoch=2,
        total_training_steps=6,
        optimizer_config=SimpleNamespace(total_training_steps=6),
        train_dataloader="exhausted",
        ckpt_handler=SimpleNamespace(save_checkpoint=lambda step: events.append(("save", step))),
    )

    def rebuild():
        trainer.train_dataloader = "fresh next epoch"

    trainer._build_dataloader = rebuild
    evaluator = SimpleNamespace(evaluate=lambda step: events.append(("evaluate", step)))
    attach_evaluation(trainer, evaluator, eval_before_training=True)
    assert events == [("evaluate", 2)]
    assert trainer.ckpt_handler.train_dataloader == "fresh next epoch"
    trainer.ckpt_handler.save_checkpoint(step=4)
    trainer.ckpt_handler.save_checkpoint(step=6)
    assert events == [("evaluate", 2), ("save", 4), ("evaluate", 4), ("save", 6), ("evaluate", 6)]
    assert trainer.total_training_steps == trainer.optimizer_config.total_training_steps == 6


def test_baseline_runs_before_training_without_saving_or_advancing_a_step():
    events = []
    trainer = SimpleNamespace(
        resume_global_step=0,
        steps_per_epoch=2,
        ckpt_handler=SimpleNamespace(save_checkpoint=lambda step: events.append(("save", step))),
    )
    evaluator = SimpleNamespace(evaluate=lambda step: events.append(("evaluate", step)))
    attach_evaluation(trainer, evaluator, eval_before_training=True)
    assert events == [("evaluate", 0)]
    assert trainer.resume_global_step == 0
    trainer.ckpt_handler.save_checkpoint(step=2)
    assert events == [("evaluate", 0), ("save", 2), ("evaluate", 2)]


def test_incomplete_baseline_stops_before_training():
    trainer = SimpleNamespace(
        resume_global_step=0,
        steps_per_epoch=2,
        ckpt_handler=SimpleNamespace(save_checkpoint=lambda step: pytest.fail("Unexpected save")),
    )

    def fail(step):
        assert step == 0
        raise RuntimeError("Baseline incomplete")

    with pytest.raises(RuntimeError, match="Baseline incomplete"):
        attach_evaluation(trainer, SimpleNamespace(evaluate=fail), eval_before_training=True)


def test_evaluation_failure_stops_at_saved_epoch():
    saved = []
    trainer = SimpleNamespace(
        resume_global_step=0,
        steps_per_epoch=2,
        ckpt_handler=SimpleNamespace(save_checkpoint=lambda step: saved.append(step)),
    )

    def failed(step):
        raise RuntimeError("Incomplete Gemini judgments")

    attach_evaluation(trainer, SimpleNamespace(evaluate=failed))
    with pytest.raises(RuntimeError, match="Incomplete Gemini"):
        trainer.ckpt_handler.save_checkpoint(step=2)
    assert saved == [2]
