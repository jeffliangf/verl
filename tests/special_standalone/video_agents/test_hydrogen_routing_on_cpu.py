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
"""Direct Vertex routing with the real evaluator, without credentials or API calls."""

import copy
import json
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from test_judge_recovery_on_cpu import external_runner  # noqa: F401
from test_tournament_on_cpu import candidates, prepared_rl  # noqa: F401

from examples.video_agents import gemini_transport as transport
from examples.video_agents import inspection, judge, judge_worker, streaming_judge
from examples.video_agents.config import JUDGE_SETTINGS
from examples.video_agents.storage import file_sha256, read_jsonl, write_json, write_jsonl


@pytest.fixture(autouse=True)
def vertex_environment(monkeypatch):
    for name in ("VERTEX_PROJECT", "VERTEX_LOCATION", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(params=["gemini-4-argon", "gemini-4-hydrogen"])
def model(request):
    return request.param


@pytest.fixture
def wire(model, external_runner, monkeypatch):  # noqa: F811
    from google.genai import types

    state = SimpleNamespace(calls=[], failures=[], created=0, closed=0, finish_reason="STOP")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(external_runner.GeminiRestClient, "_retry_delay", lambda *args: 0)
    monkeypatch.setattr(
        external_runner.GeminiRestClient,
        "_request_once",
        lambda *args, **kwargs: pytest.fail("No REST or OpenRouter fallback is allowed"),
    )

    def generate_content(**request):
        assert request["model"] == model
        state.calls.append(request)
        if state.failures:
            raise state.failures.pop(0)
        return types.GenerateContentResponse(
            response_id=f"test-{len(state.calls)}",
            model_version=model,
            candidates=[
                types.Candidate(
                    finish_reason=state.finish_reason,
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part(text="Thought summary", thought=True),
                            types.Part(
                                text=json.dumps(
                                    {"preferred_video": "A", "reason": "Visible motion", "confidence": "high"}
                                )
                            ),
                        ],
                    ),
                )
            ],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=20,
                candidates_token_count=10,
                thoughts_token_count=7,
                total_token_count=37,
            ),
        )

    def close():
        state.closed += 1

    def create(**kwargs):
        assert kwargs["project"] == "utopai-labs" and kwargs["location"] == "global"
        state.created += 1
        return SimpleNamespace(models=SimpleNamespace(generate_content=generate_content), close=close)

    monkeypatch.setattr(transport, "create_vertex_client", create)
    return state


def payload(runner, client, tmp_path, model):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    a.write_bytes(b"video-a-with-audio")
    b.write_bytes(b"video-b-with-audio")
    return runner.build_interaction_request(
        pair={"prompt": "Original prompt"},
        instructions="Original rubric",
        file_a=client.upload_video(a),
        file_b=client.upload_video(b),
        model=model,
        thinking_level="high",
        seed=0,
        max_output_tokens=None,
        video_processing="standard",
    )


def test_factory_uses_explicit_adc_and_one_sdk_attempt(monkeypatch):
    import google.auth
    from google import genai

    credentials, received = object(), {}

    def adc(**kwargs):
        assert kwargs["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
        return credentials, "unrelated-default-project"

    def create(**kwargs):
        received.update(kwargs)
        return "client"

    monkeypatch.setenv("GEMINI_API_KEY", "unused-key")
    monkeypatch.setattr(google.auth, "default", adc)
    monkeypatch.setattr(genai, "Client", create)
    assert transport.create_vertex_client(**transport.vertex_settings(), timeout=45) == "client"
    assert received["credentials"] is credentials
    assert received["enterprise"] is True
    assert received["project"] == "utopai-labs" and received["location"] == "global"
    assert received["http_options"].timeout == 45_000
    assert received["http_options"].retry_options.attempts == 1
    assert "api_key" not in received


def test_new_rl_runs_default_to_vertex_hydrogen_and_saved_models_are_preserved(prepared_rl, tmp_path):  # noqa: F811
    from examples.video_agents.storage import fingerprint, read_json
    from examples.video_agents.tournament import load_protocol

    directory, protocol, _ = prepared_rl
    assert protocol["judge"]["judge_model"] == "gemini-4-hydrogen"
    assert transport.uses_vertex(protocol["judge"]["judge_model"])
    command = judge.command(directory, protocol, "original", ["pair"])
    assert command[command.index("--model") + 1] == "gemini-4-hydrogen"

    legacy = read_json(directory / "protocol.json")
    legacy["judge"]["judge_model"] = "gemini-3.8-flash"
    path = tmp_path / "saved-flash-run" / "protocol.json"
    write_json(path, legacy)
    write_json(path.with_name("protocol-sha256.json"), fingerprint(legacy))
    loaded = load_protocol(path)
    assert loaded["judge"]["judge_model"] == "gemini-3.8-flash"
    assert not transport.uses_vertex(loaded["judge"]["judge_model"])


def test_missing_adc_fails_without_api_key_fallback(monkeypatch):
    import google.auth
    from google import genai
    from google.auth.exceptions import DefaultCredentialsError

    def adc(**kwargs):
        raise DefaultCredentialsError("not configured")

    monkeypatch.setattr(google.auth, "default", adc)
    monkeypatch.setattr(genai, "Client", lambda **kwargs: pytest.fail("Must stop before making a client"))
    monkeypatch.setenv("GEMINI_API_KEY", "unused-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "unused-router-key")
    with pytest.raises(RuntimeError, match="gcloud auth application-default login --project utopai-labs"):
        transport.create_vertex_client(**transport.vertex_settings())


def test_real_sdk_sends_adc_to_vertex_and_serializes_video_settings(model, monkeypatch):
    import google.auth
    import httpx
    from google import genai
    from google.oauth2.credentials import Credentials

    received = []

    def send(request):
        assert request.url.host == "aiplatform.googleapis.com"
        assert "/projects/utopai-labs/locations/global/" in request.url.path
        assert request.url.path.endswith(f"/publishers/google/models/{model}:generateContent")
        assert request.headers["authorization"] == "Bearer test-adc-token"
        assert "x-goog-api-key" not in request.headers
        received.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [{"text": "{}"}]}, "finishReason": "STOP"}],
                "modelVersion": model,
            },
        )

    real_client = genai.Client

    def create(**kwargs):
        kwargs["http_options"].httpx_client = httpx.Client(transport=httpx.MockTransport(send))
        return real_client(**kwargs)

    monkeypatch.setattr(google.auth, "default", lambda **kwargs: (Credentials(token="test-adc-token"), "unused"))
    monkeypatch.setattr(genai, "Client", create)
    monkeypatch.setenv("GEMINI_API_KEY", "must-not-be-used")
    body = inspection.build_request(
        video_uri="data:video/mp4;base64,dmlkZW8=",
        original_request="Original",
        generation_prompt="Rewrite",
        settings={"model": model, "fps": 4, "resolution": "high"},
    )
    with transport.create_vertex_client(**transport.vertex_settings()) as client:
        client.models.generate_content(**transport.vertex_request(body))
    sent = received[0]
    video = sent["contents"][0]["parts"][0]
    assert video["inlineData"] == {"data": "dmlkZW8=", "mime_type": "video/mp4"}
    assert video["mediaProcessing"] == "STATIC" and video["videoMetadata"]["fps"] == 4
    assert video["mediaResolution"]["level"] == "MEDIA_RESOLUTION_HIGH"
    assert sent["generationConfig"]["seed"] == 0
    assert sent["generationConfig"]["responseJsonSchema"] == body["response_format"]["schema"]


def test_argon_and_hydrogen_use_adc_and_flash_keeps_its_key(model, external_runner, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="existing-key"\n')
    assert transport.read_api_key(external_runner, model, tmp_path / "missing") is None
    assert transport.client_type(external_runner, model).vertex_genai
    assert transport.read_api_key(external_runner, "gemini-3.8-flash", env) == "existing-key"
    assert transport.client_type(external_runner, "gemini-3.8-flash") is external_runner.GeminiRestClient
    assert transport.transport_identity("gemini-3.8-flash") == {}


def test_adapter_preserves_media_order_schema_and_settings(model, external_runner, wire, tmp_path):  # noqa: F811
    client = transport.make_client(external_runner, model, tmp_path / "missing")
    body = payload(external_runner, client, tmp_path, model)
    original = copy.deepcopy(body)
    judgment, diagnostics = client.evaluate(body, require_agentic_steps=False)
    assert body == original
    sent = wire.calls[0]
    config = sent["config"]
    assert config.seed == 0 and config.thinking_config.thinking_level == "HIGH"
    assert config.max_output_tokens is None
    assert config.response_json_schema == external_runner.JUDGMENT_SCHEMA
    parts = sent["contents"][0].parts
    assert [p.inline_data.data for p in parts if p.inline_data] == [b"video-a-with-audio", b"video-b-with-audio"]
    assert [p.text for p in parts if p.text] == [p["text"] for p in body["input"] if p["type"] == "text"]
    assert judgment["preferred_video"] == "A"
    assert diagnostics["backend"] == transport.BACKEND
    assert diagnostics["vertex_api"] == "generateContent"
    assert diagnostics["served_model"] == model
    assert diagnostics["usage"]["total_input_tokens"] == 20
    body["generation_config"]["max_output_tokens"] = 512
    body["system_instruction"] = "Keep this system instruction"
    translated = transport.vertex_request(body)
    assert translated["config"].max_output_tokens == 512
    assert translated["config"].system_instruction == body["system_instruction"]


def test_client_cannot_silently_switch_models(model, external_runner, wire, tmp_path):  # noqa: F811
    client = transport.make_client(external_runner, model, tmp_path / "missing")
    body = payload(external_runner, client, tmp_path, model)
    body["model"] = next(candidate for candidate in transport.VERTEX_MODELS if candidate != model)
    with pytest.raises(ValueError, match="configured client model"):
        client.evaluate(body, require_agentic_steps=False)
    assert not wire.calls


@pytest.mark.parametrize("status", [429, 503])
def test_retries_stay_on_vertex_and_preserve_request(model, external_runner, wire, tmp_path, status):  # noqa: F811
    from google.genai.errors import APIError

    wire.failures = [APIError(status, {"error": {"message": "Transient failure"}})]
    client = transport.make_client(external_runner, model, tmp_path / "missing", max_attempts=2)
    client.evaluate(payload(external_runner, client, tmp_path, model), require_agentic_steps=False)
    assert len(wire.calls) == 2 and wire.calls[0] == wire.calls[1]


def test_network_recovery_uses_worker_retry_budget(model, external_runner, wire, tmp_path):  # noqa: F811
    import httpx

    wire.failures = [httpx.ReadTimeout("connection interrupted")]
    external_runner.GeminiRestClient = transport.client_type(external_runner, model)
    judge_worker.instrument_client(external_runner, tmp_path / "timings.jsonl")
    client = external_runner.GeminiRestClient(None, max_attempts=2)
    client.evaluate(payload(external_runner, client, tmp_path, model), require_agentic_steps=False)
    assert len(wire.calls) == 2 and wire.calls[0] == wire.calls[1]
    request = next(
        row for row in read_jsonl(tmp_path / "timings.jsonl") if row["stage"] == "gemini_request" and "attempts" in row
    )
    assert request["attempts"] == 2 and request["recovered"] is True


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failure_never_retries_or_reroutes(model, external_runner, wire, tmp_path, status):  # noqa: F811
    from google.genai.errors import APIError

    wire.failures = [APIError(status, {"error": {"message": "Access denied"}})]
    client = transport.make_client(external_runner, model, tmp_path / "missing", max_attempts=3)
    with pytest.raises(external_runner.HttpFailure) as raised:
        client.evaluate(payload(external_runner, client, tmp_path, model), require_agentic_steps=False)
    assert raised.value.status == status and len(wire.calls) == 1


def test_incomplete_response_is_rejected(model, external_runner, wire, tmp_path):  # noqa: F811
    wire.finish_reason = "MAX_TOKENS"
    client = transport.make_client(external_runner, model, tmp_path / "missing")
    with pytest.raises(external_runner.EvalError, match="incomplete"):
        client.evaluate(payload(external_runner, client, tmp_path, model), require_agentic_steps=False)


def test_agentic_processing_fails_before_network(model, external_runner, wire, tmp_path):  # noqa: F811
    client = transport.make_client(external_runner, model, tmp_path / "missing")
    body = payload(external_runner, client, tmp_path, model)
    with pytest.raises(ValueError, match="standard"):
        client.evaluate(body, require_agentic_steps=True)
    body["input"][2]["processing"] = "agentic"
    with pytest.raises(ValueError, match="agentic"):
        transport.vertex_request(body)
    assert not wire.calls


def test_inspection_fps_and_resolution_are_preserved(
    model,
):
    body = inspection.build_request(
        video_uri="data:video/mp4;base64,dmlkZW8=",
        original_request="Original",
        generation_prompt="Rewrite",
        settings={"model": model, "fps": 4, "resolution": "high"},
    )
    translated = transport.vertex_request(body)
    video = translated["contents"][0].parts[0]
    assert video.inline_data.data == b"video"
    assert video.media_processing == "STATIC" and video.video_metadata.fps == 4
    assert video.media_resolution.level == "MEDIA_RESOLUTION_HIGH"
    assert translated["config"].response_json_schema == body["response_format"]["schema"]


def test_backend_identity_checks_project_location_and_api(model, monkeypatch):
    row = transport.transport_identity(model)
    transport.validate_backend(row, model)
    for key, value in (
        ("backend", "openrouter-vertex"),
        ("vertex_project", "another-project"),
        ("vertex_location", "us"),
        ("vertex_api", "interactions"),
    ):
        with pytest.raises(ValueError, match="different backend"):
            transport.validate_backend({**row, key: value}, model)
    monkeypatch.setenv("VERTEX_PROJECT", "custom-project")
    monkeypatch.setenv("VERTEX_LOCATION", "eu")
    assert transport.vertex_settings() == {"project": "custom-project", "location": "eu"}


def test_generated_launch_preflight_checks_both_models_on_vertex(model, tmp_path, monkeypatch):
    import socket

    from examples.video_agents import mixed_group16, storage, tournament

    calls = []
    protocol = {
        "eval_repo": str(tmp_path),
        "judge": {"judge_model": model},
        "eval_judge": {"judge_model": model},
    }

    @contextmanager
    def create(**kwargs):
        assert kwargs == {"project": "utopai-labs", "location": "global", "timeout": 30}
        yield SimpleNamespace(models=SimpleNamespace(get=lambda **request: calls.append(request)))

    monkeypatch.setenv("RL_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(tournament, "load_protocol", lambda path: protocol)
    monkeypatch.setattr(storage, "load_external", lambda path: object())
    monkeypatch.setattr(transport, "create_vertex_client", create)
    exec(mixed_group16.GEMINI_PREFLIGHT, {})
    assert calls == [{"model": model}]
    result = storage.read_json(tmp_path / f"gemini-preflight-{socket.gethostname()}.json")
    assert result["checks"][0]["backend"] == "vertex"


def test_real_batch_worker_includes_reference_image_and_resumes_by_backend(
    model,
    external_runner,  # noqa: F811
    wire,
    tmp_path,
    monkeypatch,  # noqa: F811
):
    from PIL import Image

    image = tmp_path / "reference.png"
    Image.new("RGB", (32, 32), "blue").save(image)
    client = transport.make_client(external_runner, model, tmp_path / ".env")
    payload(external_runner, client, tmp_path, model)
    client.sdk.close()
    settings = {**JUDGE_SETTINGS, "judge_model": model}
    manifest = {
        "schema_version": 1,
        "instructions_version": 4,
        "instructions_how_to_compare": "Same rubric",
        "metadata": {"pair_count": 1},
        "pairs": {
            "pair": {
                "prompt_id": "sample",
                "prompt": "Same original prompt",
                "source": "test",
                "model_A": "video_agent",
                "model_B": "reference",
                "reference": "reference",
                "video_path_A": str(tmp_path / "a.mp4"),
                "video_path_B": str(tmp_path / "b.mp4"),
                "reference_image": {"path": str(image), "sha256": file_sha256(image), "bytes": image.stat().st_size},
            }
        },
    }
    mp, output = tmp_path / "manifest.json", tmp_path / "judgments.jsonl"
    write_json(mp, manifest)
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="unused-gemini-key"\nOPENROUTER_API_KEY="unused-router-key"\n')
    monkeypatch.setattr(judge_worker, "load_external", lambda path: external_runner)
    args = [
        "--runner",
        str(tmp_path / "runner.py"),
        "--timings",
        str(tmp_path / "timings.jsonl"),
        "--parallel-pairs",
        "--manifest",
        str(mp),
        "--output",
        str(output),
        "--env-file",
        str(env),
        "--model",
        model,
        "--thinking-level",
        "high",
        "--video-processing",
        "standard",
        "--seed",
        "0",
        "--all",
        "--execute",
    ]
    assert judge_worker.main(args) == 0
    assert len(wire.calls) == 1
    content = wire.calls[0]["contents"][0].parts
    assert [p.inline_data.mime_type if p.inline_data else "text" for p in content] == [
        "text",
        "text",
        "image/png",
        "text",
        "video/mp4",
        "text",
        "video/mp4",
        "text",
    ]
    assert content[2].inline_data.data == image.read_bytes()
    assert wire.created == wire.closed
    rows, _ = judge.read_judgments(output, manifest, mp, settings)
    assert rows["pair"]["backend"] == transport.BACKEND
    assert judge_worker.main(args) == 0 and len(wire.calls) == 1
    stale = read_jsonl(output)
    stale[0]["backend"] = "ai-studio"
    write_jsonl(output, stale)
    with pytest.raises(ValueError, match="different backend"):
        judge_worker.main(args)
    with pytest.raises(ValueError, match="different backend"):
        judge.read_judgments(output, manifest, mp, settings)
    assert len(wire.calls) == 1
    fallback = judge.neutral_fallback("pair", manifest, mp, settings, "provider_failure")
    assert fallback["backend"] == transport.BACKEND


@pytest.mark.parametrize("validate", [False, True])
def test_real_streaming_entry_point_routes_reward_and_eval_models(
    model,
    prepared_rl,  # noqa: F811
    external_runner,  # noqa: F811
    wire,
    tmp_path,
    monkeypatch,
    validate,  # noqa: F811
):
    directory, protocol, _ = prepared_rl
    protocol = copy.deepcopy(protocol)
    protocol["judge" if not validate else "eval_judge"] = {**JUDGE_SETTINGS, "judge_model": model}
    sample = protocol["samples"]["val" if validate else "train"][0]
    clips = candidates(tmp_path)[:1] if validate else candidates(tmp_path)
    group = directory / "vertex-stream"
    protocol["provider_recovery_root"] = str(group / "provider-recovery")
    # A stale Gemini File API quota latch must not block Vertex's inline media.
    write_json(
        group / "provider-recovery/gemini-upload-quota/state.json",
        {"blocked": True, "retry_at": 10**12, "lease_until": 10**12},
    )
    session = streaming_judge.StreamingJudge(group, protocol, sample, validate=validate)
    write_json(session.root / "context.json", session.context)
    for i, clip in enumerate(clips):
        session.offer(i, clip)
    write_json(session.root / "finished.json", True)
    monkeypatch.setattr(streaming_judge, "load_external", lambda path: external_runner)
    monkeypatch.setattr(sys, "argv", ["streaming_judge", str(group)])
    streaming_judge.main()
    rows = [r for path in group.rglob("judgments-*.jsonl") for r in read_jsonl(path)]
    assert len(rows) == (6 if validate else 12)
    assert all(r["status"] == "completed" and r["backend"] == transport.BACKEND for r in rows)
    assert len(wire.calls) == len(rows)
    assert wire.created == wire.closed
    assert not (group / "judge/gemini-files.jsonl").exists()
