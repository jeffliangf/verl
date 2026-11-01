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
"""Native multi-turn credit masks, hard budgets, and unchanged source data."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_tournament_on_cpu import TemplateTokenizer, prepared_rl  # noqa: F401

from examples.video_agents import inspection_rl
from examples.video_agents.inspection_episode import InvalidAction
from examples.video_agents.observability import ProviderError
from examples.video_agents.storage import read_json, read_jsonl, write_json


def call(name, **parameters):
    return (
        "<think>Reason before the next action.</think><tool_call><function="
        + name
        + ">"
        + "".join(f"<parameter={key}>{value}</parameter>" for key, value in parameters.items())
        + "</function></tool_call>"
    )


G = call("sol_h3", prompt="A clear scene", duration=5, seed=123456789)
INSPECT = call("inspect_video", video_id="video_1")
F = call("finish_video", final_video_id="video_1", decision_summary="The current video is sufficient.")


@pytest.fixture
def prepared_inspection(prepared_rl, tmp_path):  # noqa: F811
    source, _, source_args = prepared_rl
    args = inspection_rl.parse_args(
        [
            "prepare",
            "--reference-run",
            str(source),
            "--run-dir",
            str(tmp_path / "inspection"),
            "--data-dir",
            str(tmp_path / "inspection-data"),
        ]
    )
    inspection_rl.prepare(args)
    return args, source_args


def test_prepare_preserves_user_goals_order_generator_and_template(prepared_inspection):
    args, source = prepared_inspection
    for split in ("train", "val"):
        old = read_jsonl(source.data_dir / f"{split}.jsonl")
        new = read_jsonl(args.data_dir / f"{split}.jsonl")
        for before, after in zip(old, new, strict=True):
            assert after["prompt"][1:] == before["prompt"][1:]
            assert after["extra_info"] == before["extra_info"]
            assert after["tools"][0] == before["tools"][0]
            assert after["chat_template_kwargs"] == before["chat_template_kwargs"]
            assert (
                after["prompt"][0]["content"]
                == before["prompt"][0]["content"] + "\n\n" + inspection_rl.SYSTEM_EXTENSION
            )
    config, _ = inspection_rl.training_config(
        inspection_rl.parse_args(
            [
                "train",
                "--run-dir",
                str(args.run_dir),
                "--data-dir",
                str(args.data_dir),
                "--",
                "actor_rollout_ref.rollout.n=4",
            ]
        )
    )
    assert config.data.max_prompt_length + config.data.max_response_length == 32768
    assert config.inspection_rl.turn_max_tokens == 8192


@pytest.fixture
def native_factory(prepared_inspection, monkeypatch):
    from examples.video_agents import inspection_loop
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.tool_registry import initialize_tools_from_config
    from verl.workers.rollout.replica import TokenOutput

    args, _ = prepared_inspection
    config, _ = inspection_rl.training_config(
        inspection_rl.parse_args(
            [
                "train",
                "--run-dir",
                str(args.run_dir),
                "--data-dir",
                str(args.data_dir),
                "--",
                "actor_rollout_ref.rollout.n=4",
            ]
        )
    )
    row = read_jsonl(args.data_dir / "train.jsonl")[0]
    row["raw_prompt"] = row.pop("prompt")
    generated, inspected, requests = [], [], []
    report = {"status": "ok", "evidence": "full observation " * 100, "sentinel": "END_OF_REPORT"}

    def generate(directory, protocol, request):
        assert protocol["h3_phase_timeouts"] == {"queue": 600, "generation": 150, "download": 120}
        generated.append(deepcopy(request))
        return {
            "status": "succeeded",
            "path": str(directory / "video.mp4"),
            "request": request,
            "prompt": request["prompt"],
            "sha256": "test",
        }

    def inspect(video, original, directory, protocol, settings, output_dir):
        inspected.append((video, original))
        return report

    monkeypatch.setattr(inspection_loop, "generate_video", generate)
    monkeypatch.setattr(inspection_loop, "inspect_video", inspect)
    tools = initialize_tools_from_config(args.run_dir / "tools.json")

    def make(turns, stop_reason="completed"):
        texts = iter(turns)

        async def server(**kwargs):
            requests.append(deepcopy(kwargs))
            text = next(texts)
            tokens = list(text.encode())
            return TokenOutput(token_ids=tokens, log_probs=[-0.25] * len(tokens), stop_reason=stop_reason)

        loop = inspection_loop.InspectionLoop(
            trainer_config=DictConfigWrap(config),
            server_manager=SimpleNamespace(generate=server),
            tokenizer=TemplateTokenizer(),
            processor=None,
            dataset_cls=None,
            data_config=DictConfigWrap(config.data),
            tools=ToolListWrap(tools),
        )
        return loop

    return make, row, generated, inspected, requests, report


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "turns,generations,inspections", [([G, F], 1, 0), ([G, INSPECT, F], 1, 1), ([G, INSPECT, G], 2, 1)]
)
async def test_native_paths_full_feedback_and_exact_policy_masks(native_factory, turns, generations, inspections):
    make, row, generated, inspected, requests, report = native_factory
    loop = make(turns)
    output = await loop.run({}, **row)
    assert len(generated) == generations
    assert len(inspected) == inspections
    assert all(request["duration"] == 5 and request["seed"] == 123456789 for request in generated)
    assert output.extra_fields["h3_result"]["status"] == "succeeded"
    assert output.extra_fields["episode"]["assistant_turns"] == len(turns)
    assert len(output.response_ids) == len(output.response_mask) == len(output.response_logprobs)
    policy_ids = [token for token, mask in zip(output.response_ids, output.response_mask, strict=True) if mask]
    assert bytes(policy_ids).decode() == "".join(turns)
    assert all(
        logprob == (-0.25 if mask else 0.0)
        for mask, logprob in zip(output.response_mask, output.response_logprobs, strict=True)
    )
    assert all(request["sampling_params"]["max_tokens"] == 8192 for request in requests)
    if inspections:
        final_context = bytes(requests[2]["prompt_ids"]).decode()
        assert json.dumps(report) in final_context
        assert G in final_context and INSPECT in final_context
        expected = loop.protocol["samples"][row["extra_info"]["split"]][row["extra_info"]["index"]]["prompt"]
        assert inspected[0][1] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "turns,ng,ni",
    [
        ([INSPECT], 0, 0),
        ([F], 0, 0),
        ([G, G], 1, 0),
        ([G, INSPECT, INSPECT], 1, 1),
        ([G, call("inspect_video", video_id="someone_elses_video")], 1, 0),
        ([G, "Just return the video"], 1, 0),
        ([G + INSPECT], 0, 0),
    ],
)
async def test_illegal_actions_never_reach_backends(native_factory, turns, ng, ni):
    make, row, generated, inspected, _, _ = native_factory
    output = await make(turns).run({}, **row)
    assert output.extra_fields["h3_result"]["status"] == "invalid_call"
    assert (len(generated), len(inspected)) == (ng, ni)


@pytest.mark.asyncio
async def test_length_truncation_is_scored_not_executed_or_dropped(native_factory):
    make, row, generated, _, _, _ = native_factory
    output = await make([G], stop_reason="length").run({}, **row)
    assert output.extra_fields["h3_result"]["status"] == "invalid_call"
    assert generated == []


@pytest.mark.asyncio
async def test_inspection_service_failure_propagates_for_group_refill(native_factory, monkeypatch):
    from examples.video_agents import inspection_loop

    make, row, _, _, _, _ = native_factory

    def failed(*args):
        raise ProviderError("inspection", "outage")

    monkeypatch.setattr(inspection_loop, "inspect_video", failed)
    with pytest.raises(ProviderError, match="outage"):
        await make([G, INSPECT]).run({}, **row)


@pytest.mark.asyncio
@pytest.mark.parametrize("duration,seed", [(5, 0), (10, 9223372036854775807)])
async def test_original_generator_extremes(native_factory, duration, seed):
    make, row, generated, _, _, _ = native_factory
    output = await make([call("sol_h3", prompt="Original constraints", duration=duration, seed=seed), F]).run({}, **row)
    assert output.extra_fields["h3_result"]["status"] == "succeeded"
    assert generated == [{"prompt": "Original constraints", "duration": duration, "seed": seed}]


@pytest.mark.asyncio
async def test_no_third_generation_even_if_called_directly(native_factory):
    make, row, generated, inspected, _, _ = native_factory
    loop = make([G, INSPECT, G])
    await loop.run({}, **row)
    with pytest.raises(InvalidAction):
        await loop.episode.execute("sol_h3", {"prompt": "Forbidden third video"})
    assert len(generated) == 2 and len(inspected) == 1


@pytest.mark.asyncio
async def test_observation_overflow_never_silently_selects_an_earlier_video(native_factory):
    make, row, generated, inspected, _, report = native_factory
    loop = make([G, INSPECT])
    report["evidence"] = "long external observation " * 1500
    output = await loop.run({}, **row)
    assert len(generated) == 1 and len(inspected) == 1
    assert output.extra_fields["h3_result"]["status"] == "invalid_call"
    assert "does not fit" in output.extra_fields["h3_result"]["reason"]
    assert loop.episode.events[-1]["response"]["report"] == report


@pytest.mark.parametrize("uploaded_record", [True, False])
def test_inspection_upload_cleanup_is_scoped_and_idempotent(tmp_path, monkeypatch, uploaded_record):
    from examples.video_agents import inspection_worker

    deleted = []
    fake = SimpleNamespace(
        GeminiRestClient=lambda *args, **kwargs: SimpleNamespace(delete_file=deleted.append),
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(inspection_worker, "load_external", lambda _: fake)
    directory = tmp_path / "one-inspection"
    folder = directory / "identity-hash"
    write_json(folder / "identity.json", {})
    if uploaded_record:
        write_json(folder / "uploaded.json", {"name": "files/own-upload"})
    else:
        write_json(
            folder / "request-1.json",
            {"input": [{"uri": "https://generativelanguage.googleapis.com/v1beta/files/own-upload"}]},
        )
    inspection_worker.cleanup_uploads(directory, tmp_path)
    inspection_worker.cleanup_uploads(directory, tmp_path)
    assert deleted == ["files/own-upload"]
    assert read_json(folder / "cleanup.json")["status"] == "deleted"
