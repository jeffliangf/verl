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
"""Check SFT export, teacher-input isolation, and validation-cohort separation."""

import pandas as pd
import pyarrow.parquet as pq
import pytest

from examples.video_agents.config import build_sft_messages
from examples.video_agents.prepare_sft import prepare_sft
from examples.video_agents.storage import read_json, write_jsonl


def test_parquet_preserves_text_and_sources_and_repairs_target_key(tmp_path):
    train_file, val_file = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    target = "integrated_multimodal_description: café\noverall_soundscape: quiet\nnon_diegetic_music: none"
    write_jsonl(train_file, [{"id": "train-1", "prompt": "A café", "source": "train", "h3_context_ir, ": target}])
    write_jsonl(
        val_file,
        [
            {
                "id": "eval-1",
                "prompt": "Rain",
                "source": "aa_eval",
                "h3_context_ir": "SECRET eval teacher",
                "prompt_h3_max_video": "/reference.mp4",
            }
        ],
    )
    original_train, original_val = train_file.read_bytes(), val_file.read_bytes()
    output = tmp_path / "sft"
    manifest = prepare_sft(train_file, val_file, output)
    train = pq.read_table(output / "train.parquet").to_pylist()
    val = pq.read_table(output / "val.parquet").to_pylist()
    assert train[0]["messages"] == build_sft_messages("A café") + [{"role": "assistant", "content": target}]
    assert val[0]["messages"] == build_sft_messages("Rain") + [{"role": "assistant", "content": "SECRET eval teacher"}]
    assert (train[0]["id"], train[0]["source"], train[0]["split"]) == ("train-1", "train", "train")
    assert val[0]["split"] == "val"
    assert train[0]["enable_thinking"] is False and val[0]["enable_thinking"] is False
    # Exercise the reader and Arrow dtype backend used by MultiTurnSFTDataset.
    dataframe = pd.read_parquet(output / "train.parquet", dtype_backend="pyarrow")
    assert dataframe.iloc[0]["messages"] == train[0]["messages"]
    assert manifest["splits"]["train"]["repaired_target_keys"] == 1
    assert manifest["splits"]["val"]["rows"] == 1
    assert read_json(output / "manifest.json") == manifest
    assert train_file.read_bytes() == original_train
    assert val_file.read_bytes() == original_val


@pytest.mark.parametrize("overlap", [{"id": "train-1"}, {"prompt": "  Training prompt\n"}])
def test_rejects_validation_leakage_before_writing(tmp_path, overlap):
    train_file, val_file = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    write_jsonl(train_file, [{"id": "train-1", "prompt": "Training prompt", "h3_context_ir": "Teacher"}])
    write_jsonl(val_file, [{"id": "eval-1", "prompt": "Validation prompt", "h3_context_ir": "Teacher", **overlap}])
    with pytest.raises(ValueError, match="Train/validation overlap"):
        prepare_sft(train_file, val_file, tmp_path / "sft")
    assert not (tmp_path / "sft").exists()


@pytest.mark.parametrize(
    "target_fields, error",
    [
        ({}, "Missing nonempty h3_context_ir"),
        ({"h3_context_ir": "  "}, "Missing nonempty h3_context_ir"),
        ({"h3_context_ir": "one", "h3_context_ir, ": "two"}, "Conflicting target fields"),
    ],
)
def test_rejects_missing_or_ambiguous_supervision(tmp_path, target_fields, error):
    train_file, val_file = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    write_jsonl(train_file, [{"id": "train-1", "prompt": "Training prompt", **target_fields}])
    write_jsonl(val_file, [{"id": "eval-1", "prompt": "Validation prompt", "h3_context_ir": "Teacher"}])
    with pytest.raises(ValueError, match=error):
        prepare_sft(train_file, val_file, tmp_path / "sft")
    assert not (tmp_path / "sft").exists()
