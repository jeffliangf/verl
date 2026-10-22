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
"""Check SFT drop-last accounting, held-out isolation, and native trainer checkpoint exports."""

import pytest

from examples.video_agents import sft
from examples.video_agents.prepare_sft import prepare_sft
from examples.video_agents.storage import read_json, write_jsonl


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
    assert summary["new_videos_per_epoch"] == 0 and summary["judgments_per_epoch"] == 0
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
            "--",
            f"data.train_files={prepared / 'train.parquet'}",
            f"data.val_files={prepared / 'val.parquet'}",
        ]
    )
    config, _ = sft.training_config(args)
    summary = sft.validate_data(config, args.dataset)
    assert config.data.train_batch_size == 64
    assert summary["steps_per_epoch"] == 2
    assert summary["files"]["train"]["used_rows_per_epoch"] == 128
    assert summary["files"]["val"]["used_rows_per_epoch"] == 64
    assert summary["files"]["train"]["dropped_rows_per_epoch"] == 2
    assert summary["files"]["val"]["dropped_rows_per_epoch"] == 2
    assert summary["new_videos_per_epoch"] == 0 and summary["judgments_per_epoch"] == 0


def test_retired_video_baseline_flag_is_not_silently_ignored(inputs):
    with pytest.raises(SystemExit) as error:
        sft.parse_args(["--eval-before-training", *inputs])
    assert error.value.code == 2


def test_rejects_different_eval_targets(inputs):
    args = sft.parse_args(inputs)
    config, _ = sft.training_config(args)
    from examples.video_agents.storage import read_jsonl

    rows = read_jsonl(args.dataset)
    rows[0]["h3_context_ir"] = "Changed teacher target"
    write_jsonl(args.dataset, rows)
    with pytest.raises(ValueError, match="validation text/source differs"):
        sft.validate_data(config, args.dataset)


def test_preview_never_launches_training_or_remote_work(inputs, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(sft.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected launch"))
    assert sft.main(inputs) == 0
    assert "verl.trainer.sft_trainer" in capsys.readouterr().out
    assert not (tmp_path / "run").exists()


def test_native_sft_launch_and_frozen_resume(inputs, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(sft, "check_tokens", lambda config: calls.append("tokens"))
    monkeypatch.setattr(sft.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    assert sft.main(["--execute", *inputs]) == 0
    assert calls[0] == "tokens" and len(calls) == 2
    command, kwargs = calls[1]
    assert "verl.trainer.sft_trainer" in command
    assert kwargs["check"] is True and "TENSORBOARD_DIR" in kwargs["env"]
    saved = read_json(tmp_path / "run/sft-inputs.json")
    assert saved["config"]["trainer"]["total_epochs"] == 3
    assert saved["data"]["steps_per_epoch"] == 2
    # A new schedule must not silently resume this experiment's optimizer state.
    assert sft.main(["--execute", *inputs[:-1], "trainer.total_epochs=4"]) == 1
    assert len(calls) == 2


def test_sft_reader_masks_the_exact_generation_prefix(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from examples.video_agents.config import build_sft_messages
    from examples.video_agents.sft_dataset import PromptRewriteSFTDataset

    class CompleteConversationTokenizer:
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            assert any(m["role"] == "user" for m in messages), "No user query found in messages"
            text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
            if add_generation_prompt:
                text += "<assistant>"
            return list(text.encode())

    messages = build_sft_messages("An original prompt") + [{"role": "assistant", "content": "SECRET teacher target"}]
    path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([{"id": "train-1", "messages": messages, "enable_thinking": False}]), path)
    config = {"pad_mode": "no_padding", "truncation": "error", "max_length": 8192}
    data = PromptRewriteSFTDataset(path, CompleteConversationTokenizer(), config)
    item = data[0]
    prompt = bytes(item["input_ids"][item["loss_mask"] == 0].tolist()).decode()
    target = bytes(item["input_ids"][item["loss_mask"] == 1].tolist()).decode()
    assert prompt.endswith("<assistant>") and "SECRET" not in prompt
    assert target == "SECRET teacher target</assistant>"
    assert len(item["input_ids"]) < config["max_length"]
    data.max_length = 2
    with pytest.raises(ValueError, match="exceeds max_length"):
        data[0]
