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
"""Regression checks for the approved mixed experiment's inherited launch settings."""

from copy import deepcopy

import pytest

from examples.video_agents import mixed_group16
from examples.video_agents.config import ROOT
from examples.video_agents.storage import file_sha256, read_json


def previous_overrides():
    return read_json(ROOT / "examples/video_agents/grpo_group16_run/common-overrides.json")


def test_reuses_group16_training_settings_and_changes_only_approved_limits():
    previous = previous_overrides()
    original = previous.copy()
    overrides = mixed_group16.mixed_overrides(previous)
    before = dict(item.split("=", 1) for item in previous)
    after = dict(item.split("=", 1) for item in overrides)
    assert previous == original
    assert {key: value for key, value in after.items() if before.get(key) != value} == {
        "video_rl.max_judge_fallback_fraction": "0.25",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image": "1",
    }
    assert all(after[key] == value for key, value in before.items())


@pytest.mark.parametrize("setting", ["video_rl.judge_concurrency=100", "actor_rollout_ref.rollout.n=8"])
def test_rejects_a_different_previous_launch(setting):
    key = setting.split("=", 1)[0]
    previous = [item for item in previous_overrides() if not item.startswith(key + "=")]
    with pytest.raises(ValueError, match="approved"):
        mixed_group16.mixed_overrides(previous + [setting])


@pytest.mark.parametrize("limit", [32, 64, 128])
def test_shared_h3_limit_keeps_rollout_groups_and_judge_capacity(limit):
    settings = dict(item.split("=", 1) for item in mixed_group16.mixed_overrides(previous_overrides(), limit))
    assert settings["video_rl.h3_max_inflight"] == str(limit)
    assert settings["data.train_batch_size"] == "32"
    assert settings["actor_rollout_ref.rollout.n"] == "16"
    assert settings["video_rl.judge_concurrency"] == "500"
    assert settings["video_rl.max_judge_fallback_fraction"] == "0.25"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_shared_h3_limit_rejects_invalid_capacity(limit):
    with pytest.raises(ValueError, match="positive integer"):
        mixed_group16.mixed_overrides(previous_overrides(), limit)


def test_run_local_files_preserve_frozen_bytes_and_original_previews(tmp_path):
    original_data = tmp_path / "approved"
    original_data.mkdir()
    original_preview = original_data / "train.preview.json"
    original_preview.write_text('[{"expanded": "user preview"}]\n')
    source = {"samples": {"train": [{"task": "t2v"}], "val": [{"task": "i2v"}]}, "datasets": {}}
    for split in ("train", "val"):
        path = original_data / f"{split}.jsonl"
        path.write_bytes(b'{"prompt": "Keep exact whitespace and UTF-8: \xc3\xa9"}\n')
        source["datasets"][split] = {"path": str(path), "sha256": file_sha256(path)}
    before = deepcopy(source)
    protocol = mixed_group16.mixed_protocol(source, tmp_path / "credential", tmp_path / "run/data")
    assert source == before
    assert protocol["samples"] == source["samples"]
    for split in ("train", "val"):
        local = tmp_path / "run/data" / f"{split}.jsonl"
        assert protocol["datasets"][split] == {"path": str(local), "sha256": source["datasets"][split]["sha256"]}
        assert local.read_bytes() == (original_data / local.name).read_bytes()
    (tmp_path / "run/data/train.preview.json").write_text("token check preview")
    assert original_preview.read_text() == '[{"expanded": "user preview"}]\n'


def test_rejects_dataset_changed_after_annotation(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("original")
    source = {"datasets": {"train": {"path": str(path), "sha256": file_sha256(path)}}}
    path.write_text("changed")
    with pytest.raises(ValueError, match="changed during copying"):
        mixed_group16.mixed_protocol(source, tmp_path / "credential", tmp_path / "run/data")


@pytest.mark.parametrize(
    "ready,workers,settings,accepted",
    [
        (True, 32, "verified", True),
        (True, 32, "unverified", True),
        (True, 31, "verified", False),
        (False, 32, "verified", False),
        (True, 32, "mismatch", False),
    ],
)
def test_requires_complete_compatible_pool(ready, workers, settings, accepted):
    status = {
        "all_ready": ready,
        "endpoints": [{"health": {"workers_ready": workers}, "settings_check": {"status": settings}}],
    }
    if accepted:
        mixed_group16.require_h3_pool(status, 32)
    else:
        with pytest.raises(ValueError):
            mixed_group16.require_h3_pool(status, 32)
