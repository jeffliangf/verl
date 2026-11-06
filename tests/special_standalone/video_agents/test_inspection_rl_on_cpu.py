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

import http.client
import json
from copy import deepcopy
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
from test_gemini_transport_on_cpu import interrupted_gemini  # noqa: F401
from test_tournament_on_cpu import TemplateTokenizer, prepared_rl  # noqa: F401

from examples.video_agents import inspection_rl, rl
from examples.video_agents.h3_deadlines import phase_budgets, worker_timeout
from examples.video_agents.h3_request import video_request
from examples.video_agents.inspection_episode import InvalidAction
from examples.video_agents.observability import ProviderError
from examples.video_agents.storage import file_sha256, read_json, read_jsonl, write_json
from examples.video_agents.tournament import load_protocol, runtime_protocol


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
    assert config.inspection_rl.timeout_seconds == 120


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ([], {"queue": 600, "generation": 150, "download": 120}),
        (
            [
                "video_rl.h3_phase_timeouts.queue=300",
                "video_rl.timeouts.h3=75",
                "video_rl.h3_phase_timeouts.download=60",
            ],
            {"queue": 300, "generation": 75, "download": 60},
        ),
        (["video_rl.h3_phase_timeouts=null"], None),
    ],
)
def test_both_entry_points_share_one_h3_configuration(prepared_inspection, overrides, expected):
    inspection, source = prepared_inspection
    for entry, prepared in ((rl, source), (inspection_rl, inspection)):
        args = entry.parse_args(
            [
                "train",
                "--run-dir",
                str(prepared.run_dir),
                "--data-dir",
                str(prepared.data_dir),
                "--",
                "actor_rollout_ref.rollout.n=4",
                *overrides,
            ]
        )
        config, _ = entry.training_config(args)
        runtime = runtime_protocol(load_protocol(prepared.run_dir / "protocol.json"), config.video_rl)
        assert runtime["h3_phase_timeouts"] == expected
        assert worker_timeout(runtime, config.video_rl.timeouts.h3) == (
            sum(expected.values()) + 60 if expected else config.video_rl.timeouts.h3
        )
        if entry is inspection_rl:
            assert not {"h3_queue_timeout_seconds", "h3_download_timeout_seconds"} & set(config.inspection_rl)


@pytest.mark.parametrize("invalid", ["queue=0", "download=-1", "generation=true"])
def test_both_launchers_reject_invalid_phase_budgets(prepared_inspection, invalid):
    inspection, source = prepared_inspection
    for entry, prepared in ((rl, source), (inspection_rl, inspection)):
        args = entry.parse_args(
            [
                "train",
                "--run-dir",
                str(prepared.run_dir),
                "--data-dir",
                str(prepared.data_dir),
                "--",
                "actor_rollout_ref.rollout.n=4",
                f"video_rl.h3_phase_timeouts.{invalid}",
            ]
        )
        with pytest.raises(ValueError, match="H3 phase budgets"):
            entry.training_config(args)


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
        assert protocol["h3_phase_timeouts"] == phase_budgets(config.video_rl)
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
@pytest.mark.parametrize("phases", [None, {"queue": 300, "generation": 75, "download": 60}])
async def test_inspection_loop_uses_shared_h3_budgets(native_factory, phases):
    make, row, generated, _, _, _ = native_factory
    loop = make([G, F])
    loop.config.video_rl.h3_phase_timeouts = phases
    await loop.run({}, **row)
    assert len(generated) == 1
    assert loop.protocol["h3_phase_timeouts"] == phases


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
    assert all(request["target"]["duration_seconds"] == 5 and request["seed"] == 123456789 for request in generated)
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
async def test_inspection_service_failure_finishes_with_previous_video(native_factory, monkeypatch):
    from examples.video_agents import inspection_loop

    make, row, generated, _, requests, _ = native_factory

    def failed(*args):
        raise ProviderError("inspection", "outage")

    monkeypatch.setattr(inspection_loop, "inspect_video", failed)
    loop = make([G, INSPECT])
    output = await loop.run({}, **row)
    assert len(generated) == 1 and len(requests) == 2
    assert output.extra_fields["h3_result"] == loop.episode.videos["video_1"]
    assert output.extra_fields["h3_result"]["status"] == "succeeded"
    episode = output.extra_fields["episode"]
    assert episode["terminal_reason"] == "inspection_failed"
    assert episode["fallback"] == 1 and episode["revised"] == 0
    assert episode["inspection_calls"] == 1
    saved = read_json(loop.episode.directory / "episode.json")
    event = saved["events"][-1]
    assert event["status"] == "failed" and event["error"] == "outage"
    assert event["seconds"] >= 0
    response = event["response"]
    assert response["status"] == "failed"
    assert response["error"] == "Inspection tool failed. The previous video is final."
    assert response["final_video_id"] == "video_1"
    assert response["environment"]["available_actions"] == []
    assert "report" not in response
    policy = [token for token, mask in zip(output.response_ids, output.response_mask, strict=True) if mask]
    observation = [token for token, mask in zip(output.response_ids, output.response_mask, strict=True) if not mask]
    assert bytes(policy).decode() == G + INSPECT
    assert json.dumps(response) in bytes(observation).decode()
    assert all(
        logprob == (-0.25 if mask else 0.0)
        for mask, logprob in zip(output.response_mask, output.response_logprobs, strict=True)
    )
    with pytest.raises(InvalidAction):
        await loop.episode.execute("sol_h3", {"prompt": "Must not revise after inspection failure"})


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["upload", "interaction"])
async def test_interrupted_inspection_worker_finishes_with_previous_video(
    native_factory,
    interrupted_gemini,  # noqa: F811
    tmp_path,
    monkeypatch,
    operation,
):
    from examples.video_agents import inspection_loop, inspection_worker

    make, row, generated, _, requests, _ = native_factory
    interrupted_gemini.operation = operation
    repo = tmp_path / "fault-evaluator"
    script = repo / "scripts/run_gemini_pairwise_eval.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        dedent(f"""\
        import json
        import urllib.request

        API_BASE = {interrupted_gemini.url!r}
        def read_env_file(path):
            return {{}}
        def safe_error_text(error, key):
            return str(error)
        class GeminiRestClient:
            def __init__(self, *args, **kwargs):
                pass
            def upload_video(self, path):
                if {operation!r} == "upload":
                    return self.request_json("POST", API_BASE + "/content/1")[0]
                return {{"name": "files/1", "uri": API_BASE + "/files/1"}}
            def request_json(self, method, url, **kwargs):
                request = urllib.request.Request(url, data=b"{{}}", method=method)
                with urllib.request.urlopen(request, timeout=2) as response:
                    return json.loads(response.read()), response.headers
            def delete_file(self, name):
                pass
        """)
    )
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-no-provider-calls")
    generate = inspection_loop.generate_video

    def generated_video(*args):
        video = generate(*args)
        path = Path(video["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"local test video")
        return {**video, "sha256": file_sha256(path)}

    def inspect(video, original, directory, protocol, settings, output_dir):
        return inspection_worker.inspect_video(
            video, original, directory, {**protocol, "eval_repo": str(repo)}, settings, output_dir
        )

    monkeypatch.setattr(inspection_loop, "generate_video", generated_video)
    monkeypatch.setattr(inspection_loop, "inspect_video", inspect)
    loop = make([G, INSPECT])
    output = await loop.run({}, **row)
    assert len(generated) == 1 and len(requests) == 2
    assert output.extra_fields["h3_result"] == loop.episode.videos["video_1"]
    assert output.extra_fields["episode"]["fallback"] == 1
    assert loop.episode.terminal_reason == "inspection_failed"
    result = read_json(loop.episode.directory / "inspection/worker-result.json")
    assert result["status"] == "provider_error"
    assert result["error_type"] == "RuntimeError"  # The unchanged inspector wraps the HTTP exception.
    assert "IncompleteRead" in result["error"]
    assert len(interrupted_gemini.calls) == 1


@pytest.mark.parametrize(
    "error,provider",
    [
        (http.client.IncompleteRead(b""), True),
        (http.client.BadStatusLine("invalid status"), True),
        (http.client.LineTooLong("header"), True),
        (http.client.CannotSendRequest("bug"), False),
        (FileNotFoundError("local video"), False),
        (RuntimeError("local bug"), False),
    ],
)
def test_inspection_classifies_wrapped_response_errors_without_hiding_local_errors(tmp_path, error, provider):
    from examples.video_agents.inspection_worker import provider_failure

    try:
        try:
            raise error
        except Exception as exc:
            raise RuntimeError(str(exc)) from None
    except RuntimeError as wrapped:
        assert provider_failure(wrapped, tmp_path) is provider


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("bad local artifact"), ProviderError("h3", "unrelated failure")])
async def test_inspection_fallback_does_not_hide_unrelated_errors(native_factory, monkeypatch, error):
    from examples.video_agents import inspection_loop

    make, row, _, _, _, _ = native_factory

    def failed(*args):
        raise error

    monkeypatch.setattr(inspection_loop, "inspect_video", failed)
    with pytest.raises(type(error), match=str(error)):
        await make([G, INSPECT]).run({}, **row)


@pytest.mark.asyncio
async def test_inspection_worker_deadline_ends_episode_without_waiting_for_cleanup(native_factory, monkeypatch):
    import subprocess

    from examples.video_agents import inspection_loop, inspection_worker

    make, row, generated, _, requests, _ = native_factory
    cleanup = []

    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 120
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(inspection_loop, "inspect_video", inspection_worker.inspect_video)
    monkeypatch.setattr(inspection_worker.subprocess, "run", timeout)
    monkeypatch.setattr(inspection_worker, "schedule_cleanup", lambda *args: cleanup.append(args))
    loop = make([G, INSPECT])
    output = await loop.run({}, **row)
    assert output.extra_fields["h3_result"] == loop.episode.videos["video_1"]
    assert output.extra_fields["episode"]["fallback"] == 1
    assert len(generated) == 1 and len(requests) == 2 and len(cleanup) == 1
    assert "120s" in loop.episode.events[-1]["error"]


def test_report_saved_before_cleanup_timeout_is_kept(tmp_path, monkeypatch):
    import subprocess

    from examples.video_agents import inspection_worker

    report = {"valid": "inspection feedback"}
    directory = tmp_path / "inspection"
    cleanup = []

    def timeout(command, **kwargs):
        write_json(directory / "worker-result.json", {"status": "succeeded", "report": report})
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(inspection_worker.subprocess, "run", timeout)
    monkeypatch.setattr(inspection_worker, "schedule_cleanup", lambda *args: cleanup.append(args))
    result = inspection_worker.inspect_video(
        {},
        "original request",
        directory,
        {"eval_repo": str(tmp_path)},
        {"timeout_seconds": 120, "max_inflight": 2},
        tmp_path,
    )
    assert result == report and len(cleanup) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("duration,seed", [(5, 0), (10, 9223372036854775807)])
async def test_original_generator_extremes(native_factory, duration, seed):
    make, row, generated, _, _, _ = native_factory
    output = await make([call("sol_h3", prompt="Original constraints", duration=duration, seed=seed), F]).run({}, **row)
    assert output.extra_fields["h3_result"]["status"] == "succeeded"
    assert generated == [video_request("Original constraints", duration=duration, seed=seed)]


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
