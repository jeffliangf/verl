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
"""Accuracy denominators and parity with the real parser and argument validator."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from examples.video_agents.control_constraints import FIRST_FRAME, allowed_arguments
from examples.video_agents.h3_request import tool_request
from examples.video_agents.image_assets import image_asset
from examples.video_agents.storage import read_json
from examples.video_agents.tool_call_audit import assess_calls, rates, stream_generations, summarize
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema

SCHEMA = read_json(Path(__file__).resolve().parents[3] / "examples/video_agents/configs/sol_h3_unified_tool.json")


def sample(tmp_path, task="t2v", constrained=True):
    image = None
    if task == "i2v":
        path = tmp_path / "frame.png"
        Image.new("RGB", (80, 120), "red").save(path)
        image = image_asset(path)
    row = {"id": "test", "task": task, "request": "A person waving.", "target": {}, "image": image}
    row["allowed_video_gen_args"] = allowed_arguments(
        row,
        frames=[243] if constrained else None,
        ratios=["3:4", "9:16"] if constrained else None,
        require_duration=constrained,
        require_aspect=constrained,
    )
    return row


def call(parameters):
    return [{"name": "sol_h3", "arguments": json.dumps(parameters)}]


@pytest.mark.parametrize("task", ["t2v", "i2v"])
@pytest.mark.parametrize("duration,correct", [(10, True), (10.125, True), (8, False), (15, False), (None, False)])
def test_accuracy_matches_native_validator(tmp_path, task, duration, correct):
    row = sample(tmp_path, task)
    parameters = {
        "task": "fl2va" if task == "i2v" else "t2va",
        "prompt": "Wave.",
        "conditions": [FIRST_FRAME] if task == "i2v" else [],
        "target": {"aspect_ratio": "3:4", **({"duration_seconds": duration} if duration is not None else {})},
    }
    result = assess_calls(call(parameters), SCHEMA, row)
    assert result["parsed_sol_h3_call"]
    assert result["requested_duration_correct"] is correct
    assert result["requested_aspect_correct"]
    assert result["all_args_valid"] is correct
    if correct:
        tool_request(parameters, SCHEMA, sample=row)
    else:
        with pytest.raises(ValueError):
            tool_request(parameters, SCHEMA, sample=row)


def test_free_controls_do_not_inflate_accuracy_denominator(tmp_path):
    free = assess_calls(call({"task": "t2va", "prompt": "Wave."}), SCHEMA, sample(tmp_path, constrained=False))
    assert free["all_args_valid"]
    assert free["requested_duration_correct"] is free["requested_aspect_correct"] is None
    missing = assess_calls([], SCHEMA, sample(tmp_path))
    for result in (free, missing):
        result.update(split="train", task="t2v", source="test", finish_reason="stop")
    report = summarize([free, missing], 2)
    assert report["complete"]
    assert report["overall"]["parsed_sol_h3_call"] == {"passed": 1, "total": 2, "percent": 50}
    assert report["overall"]["requested_duration_correct"] == {"passed": 0, "total": 1, "percent": 0}
    assert report["among_parsed_calls"]["train"]["requested_duration_correct"]["percent"] is None


def test_aspect_labels_remain_broad_and_default_is_not_explicit(tmp_path):
    row = sample(tmp_path)
    for ratio, expected in [("3:4", True), ("9:16", True), ("16:9", False), (None, False)]:
        parameters = {"task": "t2va", "prompt": "Wave.", "target": {"duration_seconds": 10}}
        if ratio is not None:
            parameters["target"]["aspect_ratio"] = ratio
        result = assess_calls(call(parameters), SCHEMA, row)
        assert result["requested_aspect_correct"] is expected
        assert result["all_args_valid"] is expected
        assert result["worker_compatible"]


def test_eval_matches_actual_canvas_rule(tmp_path):
    row = sample(tmp_path, constrained=False)
    del row["allowed_video_gen_args"]
    row["target"] = {"duration_seconds": 10, "aspect_ratio": "16:9"}
    for ratio, expected in [("16:9", True), ("auto", True), ("9:16", False), (None, False)]:
        parameters = {"task": "t2va", "prompt": "Wave.", "target": {"duration_seconds": 10}}
        if ratio is not None:
            parameters["target"]["aspect_ratio"] = ratio
        result = assess_calls(call(parameters), SCHEMA, row)
        assert result["requested_aspect_correct"] is expected
        assert result["all_args_valid"] is expected


def test_task_image_and_extra_parameters_are_separate_failures(tmp_path):
    row = sample(tmp_path, "i2v", constrained=False)
    base = {"task": "fl2va", "prompt": "Wave.", "conditions": [FIRST_FRAME]}
    for changes, metric in [
        ({"task": "t2va"}, "task_correct"),
        ({"conditions": []}, "first_frame_correct"),
        ({"duration": 10}, "schema_valid"),
    ]:
        result = assess_calls(call({**deepcopy(base), **changes}), SCHEMA, row)
        assert result["parsed_sol_h3_call"]
        assert not result[metric]
        assert not result["all_args_valid"]
    Path(row["image"]["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen source image changed"):
        assess_calls(call(base), SCHEMA, row)


def test_real_parser_handles_model_xml_and_rejects_extra_calls(tmp_path):
    parser = ToolParser.get_tool_parser("qwen3_coder", SimpleNamespace(decode=lambda value: value))
    text = (
        "<think>Choose the tool.</think><tool_call><function=sol_h3>"
        "<parameter=task>t2va</parameter><parameter=prompt>A person waves.</parameter>"
        '<parameter=target>{"duration_seconds": 10, "aspect_ratio": "9:16"}</parameter>'
        "</function></tool_call>"
    )
    _, parsed = asyncio.run(parser.extract_tool_calls(text, tools=[OpenAIFunctionToolSchema.model_validate(SCHEMA)]))
    calls = [c.model_dump() for c in parsed]
    result = assess_calls(calls, SCHEMA, sample(tmp_path))
    assert result["all_args_valid"]
    assert not assess_calls(calls + calls, SCHEMA, sample(tmp_path))["parsed_sol_h3_call"]
    assert rates([])["all_args_valid"]["percent"] is None


def test_stream_refills_and_preserves_identity_when_completions_are_out_of_order():
    class Engine:
        def __init__(self):
            self.active = []
            self.next_id = 0
            self.llm_engine = self
            self.high_watermark = 0
            self.output_processor = SimpleNamespace(request_states={})

        def enqueue(self, prompts, sampling, use_tqdm):
            ids = []
            for prompt in prompts:
                request_id = str(self.next_id)
                self.next_id += 1
                self.active.append((request_id, prompt))
                internal_id = f"{request_id}-random-suffix"
                self.output_processor.request_states[internal_id] = SimpleNamespace(external_req_id=request_id)
                ids.append(internal_id)
            self.high_watermark = max(self.high_watermark, len(self.active))
            return ids

        def has_unfinished_requests(self):
            return bool(self.active)

        def step(self):
            request_id, prompt = self.active.pop()
            return [SimpleNamespace(request_id=request_id, finished=True, prompt=prompt)]

    engine = Engine()
    outputs = list(stream_generations(engine, list(range(7)), lambda e: (f"prompt-{e}", e * 10), None, 3))
    assert engine.high_watermark == 3
    assert [entry for entry, _, _ in outputs] == [2, 3, 4, 5, 6, 1, 0]
    for entry, metadata, output in outputs:
        assert metadata == entry * 10 and output.prompt == f"prompt-{entry}"


def test_stream_does_not_silently_drop_lost_requests():
    engine = SimpleNamespace(
        enqueue=lambda *args, **kwargs: ["0"],
        llm_engine=SimpleNamespace(
            has_unfinished_requests=lambda: False,
            output_processor=SimpleNamespace(request_states={"0": SimpleNamespace(external_req_id="0")}),
        ),
    )
    with pytest.raises(RuntimeError, match="lost pending"):
        list(stream_generations(engine, [1], lambda e: (e, e), None, 1))


@pytest.mark.parametrize("arguments", ['{"task":', '["t2va"]', '{"target":{"duration_seconds":NaN}}'])
def test_malformed_arguments_are_scored_and_can_be_saved(tmp_path, arguments):
    result = assess_calls([{"name": "sol_h3", "arguments": arguments}], SCHEMA, sample(tmp_path))
    assert result["parsed_sol_h3_call"] and not result["all_args_valid"]
    assert "arguments" in result["errors"]
    json.dumps(result, allow_nan=False)
