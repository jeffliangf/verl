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
"""Reproduce JSON chunk inference failures and validate complete mixed observations."""

from copy import deepcopy

import datasets
import pytest
from omegaconf import OmegaConf

from examples.video_agents import data_preflight, rl, submit_mixed
from examples.video_agents.native_dataset import VideoRLDataset
from examples.video_agents.storage import file_sha256, fingerprint, read_json, write_json, write_jsonl
from verl.utils.dataset.rl_dataset import RLHFDataset


@pytest.fixture
def chunked_json(monkeypatch):
    load = datasets.load_dataset

    def small_chunks(*args, **kwargs):
        if args[0] == "json":
            kwargs["chunksize"] = 1024
        return load(*args, **kwargs)

    monkeypatch.setattr(datasets, "load_dataset", small_chunks)


def mixed_rows():
    schema = read_json(rl.CONFIG_DIR / "sol_h3_unified_tool.json")
    samples = [
        {
            "id": str(i),
            "task": "t2v",
            "prompt": "Scene " + "x" * 1024,
            "request": "Make this scene.",
            "source": "test",
            "image": None,
            "target": {},
        }
        for i in range(8)
    ]
    samples.append(
        {
            "id": "image",
            "task": "i2v",
            "prompt": "Animate.",
            "request": "Animate this photo.",
            "source": "test",
            "image": {"path": "/tmp/source.png"},
            "target": {},
        }
    )
    return samples, rl.dataset_records(samples, "train", [schema])


@pytest.mark.parametrize("order", ["t2v_first", "i2v_first", "t2v_only"])
def test_typed_images_survive_json_chunks_without_changing_rows(tmp_path, chunked_json, order):
    _, rows = mixed_rows()
    if order == "i2v_first":
        rows.reverse()
    elif order == "t2v_only":
        rows = rows[:-1]
    path = tmp_path / "train.jsonl"
    write_jsonl(path, rows)
    config = OmegaConf.create({"filter_overlong_prompts": False})
    if order == "t2v_first":
        with pytest.raises(datasets.exceptions.DatasetGenerationError):
            RLHFDataset(str(path), tokenizer=None, config=config)
    dataset = VideoRLDataset(str(path), tokenizer=None, processor=object(), config=config)
    assert dataset.dataframe.to_list() == rows
    assert dataset.dataframe.features["images"].feature["image"].dtype == "string"
    for index, row in enumerate(rows):
        item = dataset[index]
        images = [
            part
            for message in item["raw_prompt"]
            if isinstance(message["content"], list)
            for part in message["content"]
            if part["type"] == "image"
        ]
        assert len(images) == len(row["images"])


def test_native_preflight_checks_all_rows_and_refuses_a_stale_report(tmp_path, monkeypatch, chunked_json):
    samples, rows = mixed_rows()
    protocol = {"samples": {"train": samples, "val": samples}, "datasets": {}}
    config = {
        "data": {
            "filter_overlong_prompts": False,
            "max_prompt_length": 4096,
            "custom_cls": {"path": "pkg://examples.video_agents.native_dataset", "name": "VideoRLDataset"},
        }
    }
    for split in ("train", "val"):
        path = tmp_path / f"{split}.jsonl"
        write_jsonl(path, rows)
        protocol["datasets"][split] = {"path": str(path), "sha256": file_sha256(path)}
        config["data"][f"{split}_files"] = str(path)
    checked = data_preflight.validate_native_data(OmegaConf.create(config), protocol, processor=object())
    assert checked["train"]["native_items_checked"] == len(rows)
    assert checked["val"]["by_task"] == {"t2v": 8, "i2v": 1}
    write_json(tmp_path / "protocol.json", protocol)
    write_json(tmp_path / "preview-config.json", config)
    write_json(tmp_path / "launch.json", {"code_commit": "test", "python_environment": "/production"})
    monkeypatch.setattr(data_preflight, "load_protocol", read_json)
    report = {
        "passed": True,
        "runtime_commit": "test",
        "python_environment": "/production",
        "configuration_sha256": fingerprint(config),
        "protocol_sha256": file_sha256(tmp_path / "protocol.json"),
        "datasets": protocol["datasets"],
        "native": checked,
        "max_prompt_tokens": {"train": 100, "val": 100},
    }
    write_json(tmp_path / "data-preflight.json", report)
    assert data_preflight.require_preflight(tmp_path) == report
    incomplete = deepcopy(report)
    incomplete["native"]["train"]["native_items_checked"] -= 1
    write_json(tmp_path / "data-preflight.json", incomplete)
    with pytest.raises(ValueError, match="Incomplete"):
        data_preflight.require_preflight(tmp_path)
    write_json(tmp_path / "data-preflight.json", report)
    config["data"]["max_prompt_length"] = 2048
    write_json(tmp_path / "preview-config.json", config)
    with pytest.raises(ValueError, match="stale"):
        data_preflight.require_preflight(tmp_path)


def test_submission_without_preflight_cannot_allocate_gpus(tmp_path, monkeypatch):
    write_json(tmp_path / "launch.json", {"job_id": None})

    def unexpected_command(*args, **kwargs):
        pytest.fail("Submission ran a command before validating the complete dataset")

    monkeypatch.setattr(submit_mixed.subprocess, "run", unexpected_command)
    monkeypatch.setattr(submit_mixed.subprocess, "check_output", unexpected_command)
    with pytest.raises(FileNotFoundError, match="data-preflight.json"):
        submit_mixed.submit(tmp_path)
