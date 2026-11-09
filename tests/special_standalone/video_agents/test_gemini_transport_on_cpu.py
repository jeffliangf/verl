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
"""Exercise interrupted Gemini HTTP responses without contacting a provider."""

import http.client
import json
import subprocess
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from test_judge_recovery_on_cpu import external_runner  # noqa: F401
from test_tournament_on_cpu import candidates, prepared_rl  # noqa: F401

from examples.video_agents import judge_worker, streaming_judge, tournament
from examples.video_agents.observability import ProviderError
from examples.video_agents.storage import read_json, write_json


@pytest.fixture
def interrupted_gemini():
    """Serve real truncated chunked bodies, including HTTP error response bodies."""
    state = SimpleNamespace(operation="interaction", remaining=None, status=200, calls=[], uploads=0)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self):
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            operation = (
                "cleanup"
                if self.command == "DELETE"
                else "poll"
                if self.command == "GET"
                else "interaction"
                if self.path.endswith("/interactions")
                else "start"
                if self.path == "/upload"
                else "upload"
            )
            state.calls.append((operation, body))
            if operation == state.operation and state.remaining != 0:
                if state.remaining is not None:
                    state.remaining -= 1
                self.send_response(state.status)
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                self.wfile.write(b"8\r\npartial")
                self.wfile.flush()
                self.close_connection = True
                return
            self.send_response(200)
            if operation == "start":
                state.uploads += 1
                self.send_header("X-Goog-Upload-URL", f"{state.url}/content/{state.uploads}")
            name = "files/" + self.path.rsplit("/", 1)[-1]
            file = {"name": name, "uri": f"{state.url}/{name}", "state": "ACTIVE"}
            response = {"file": file} if operation == "upload" else file if operation == "poll" else {}
            payload = json.dumps(response).encode()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_POST = do_GET = do_DELETE = respond

    class Server(ThreadingHTTPServer):
        request_queue_size = 128  # Accommodate the shared default of 100 concurrent comparisons.

    server = Server(("127.0.0.1", 0), Handler)
    state.url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.fixture
def local_runner(external_runner, interrupted_gemini, monkeypatch, tmp_path):  # noqa: F811
    runner = external_runner
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-no-provider-calls")
    monkeypatch.setattr(runner, "API_BASE", interrupted_gemini.url)
    monkeypatch.setattr(runner, "UPLOAD_BASE", interrupted_gemini.url + "/upload")
    # Newer evaluator versions cache this URL on the class rather than reading API_BASE.
    monkeypatch.setattr(
        runner.GeminiRestClient, "interactions_url", interrupted_gemini.url + "/interactions", raising=False
    )
    urlopen = runner.urllib.request.urlopen

    def local_urlopen(request, *args, **kwargs):
        assert request.full_url.startswith(interrupted_gemini.url + "/"), "Tests must only contact the local server"
        return urlopen(request, *args, **kwargs)

    monkeypatch.setattr(runner.urllib.request, "urlopen", local_urlopen)
    monkeypatch.setattr(runner.GeminiRestClient, "_retry_delay", lambda *args: 0)
    monkeypatch.setattr(
        runner,
        "extract_interaction_result",
        lambda *args, **kwargs: ({"preferred_video": "A", "reason": "Test response", "confidence": "high"}, {}),
    )
    judge_worker.instrument_client(runner, tmp_path / "http-timings.jsonl", file_control_timeout=1)
    return runner


@pytest.mark.parametrize("operation", ["interaction", "upload", "poll", "cleanup"])
@pytest.mark.parametrize("permanent", [False, True])
def test_interrupted_http_reuses_bounded_retries(local_runner, interrupted_gemini, tmp_path, operation, permanent):
    runner, server = local_runner, interrupted_gemini
    server.operation, server.remaining = operation, None if permanent else 1
    client = runner.GeminiRestClient("fake-key", max_attempts=3, request_timeout=1, file_timeout=5)
    video = tmp_path / "test.mp4"
    video.write_bytes(b"test video")

    def call():
        if operation == "interaction":
            return client.evaluate({"input": [{"uri": "files/existing"}]}, require_agentic_steps=False)
        if operation == "upload":
            return client.upload_video(video)
        if operation == "poll":
            return client.wait_for_file({"name": "files/test", "state": "PROCESSING"})
        return client.delete_file("files/test")

    if permanent:
        with pytest.raises(runner.NetworkFailure) as raised:
            call()
        assert isinstance(raised.value.__cause__, http.client.IncompleteRead)
    else:
        call()
    attempts = [body for name, body in server.calls if name == operation]
    assert len(attempts) == (3 if permanent else 2)
    assert len(set(attempts)) == 1  # Retrying preserves the request and uploaded-video references.


def test_truncated_http_error_body_is_also_retried(local_runner, interrupted_gemini):
    interrupted_gemini.status, interrupted_gemini.remaining = 503, 1
    client = local_runner.GeminiRestClient("fake-key", max_attempts=3, request_timeout=1)
    assert client.evaluate({}, require_agentic_steps=False)[0]["preferred_video"] == "A"
    assert len(interrupted_gemini.calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403])
def test_truncated_http_error_body_preserves_fatal_status(local_runner, interrupted_gemini, status):
    from examples.video_agents import judge

    interrupted_gemini.status = status
    client = local_runner.GeminiRestClient("fake-key", max_attempts=3, request_timeout=1)
    with pytest.raises(local_runner.HttpFailure) as raised:
        client.evaluate({}, require_agentic_steps=False)
    assert raised.value.status == status
    assert len(interrupted_gemini.calls) == 1
    with pytest.raises(RuntimeError, match="Nonrecoverable"):
        judge.fallback_reason({"stage": "interaction", "error_type": "HttpFailure", "error": str(raised.value)})


@pytest.mark.parametrize("error", [http.client.BadStatusLine("invalid status"), http.client.LineTooLong("header")])
def test_malformed_http_is_recordable_as_a_provider_failure(external_runner, tmp_path, monkeypatch, error):  # noqa: F811
    def request(*args, **kwargs):
        raise error

    monkeypatch.setattr(external_runner.GeminiRestClient, "_request_once", request)
    judge_worker.instrument_client(external_runner, tmp_path / "timings.jsonl")
    client = external_runner.GeminiRestClient("fake-key", max_attempts=1)
    with pytest.raises(external_runner.NetworkFailure) as raised:
        client.evaluate({}, require_agentic_steps=False)
    assert raised.value.__cause__ is error


@pytest.mark.parametrize("error", [RuntimeError("bug"), ValueError("bad input"), http.client.CannotSendRequest("bug")])
def test_judge_transport_does_not_hide_local_errors(external_runner, tmp_path, monkeypatch, error):  # noqa: F811
    def request(*args, **kwargs):
        raise error

    monkeypatch.setattr(external_runner.GeminiRestClient, "_request_once", request)
    judge_worker.instrument_client(external_runner, tmp_path / "timings.jsonl")
    client = external_runner.GeminiRestClient("fake-key", max_attempts=3)
    with pytest.raises(type(error)) as raised:
        client.evaluate({}, require_agentic_steps=False)
    assert raised.value is error


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize("operation", ["upload", "interaction", "cleanup"])
def test_real_judgments_protect_training_and_evaluation(
    prepared_rl,  # noqa: F811
    local_runner,
    interrupted_gemini,
    tmp_path,
    monkeypatch,
    streamed,
    validate,
    operation,
):
    runner, server = local_runner, interrupted_gemini
    server.operation = operation
    directory, protocol, _ = prepared_rl
    sample = protocol["samples"]["val" if validate else "train"][0]
    clips = candidates(tmp_path)[:1] if validate else candidates(tmp_path)
    group = directory / "interrupted-judgments"

    def execute(command, **kwargs):
        _, forwarded = judge_worker.parse_options(command[3:])
        code = runner.main(forwarded)
        if code:
            raise subprocess.CalledProcessError(code, command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tournament.subprocess, "run", execute)

    def score():
        if streamed:
            session = streaming_judge.StreamingJudge(group, protocol, sample, validate=validate)
            write_json(session.root / "context.json", session.context)
            for i, clip in enumerate(clips):
                session.offer(i, clip)
            write_json(session.root / "finished.json", True)
            with judge_worker.parallel_pairs(runner):
                streaming_judge.run_stream(group, runner)
            write_json(session.root / "outcome.json", {"timed_out": False})
        return tournament.score_group(group, protocol, sample, clips, validate=validate, streamed=streamed)

    if validate and operation != "cleanup":
        with pytest.raises(ProviderError) as raised:
            score()
        assert raised.value.stage == "gemini"
        assert not (group / "rewards.json").exists()
        assert not list(group.rglob("fallbacks-*.json"))
    else:
        rewards = score()
        assert all(reward["score"] == 0.5 for reward in rewards)
        fallbacks = [row for path in group.rglob("fallbacks-*.json") for row in read_json(path)]
        if operation == "cleanup":
            assert not fallbacks  # A failed deletion cannot discard completed comparisons.
        else:
            assert len(fallbacks) == 12  # Both presentation orders for all six training pairs.
            assert all(row["error"]["error_type"] == "NetworkFailure" for row in fallbacks)
            assert all(reward["judge_fallback_fraction"] == 1 for reward in rewards)
    assert Counter(name for name, _ in server.calls)[operation] > 0


def test_failed_processing_uploads_remain_owned_until_all_pairs_settle(
    local_runner, interrupted_gemini, tmp_path, monkeypatch
):
    from examples.video_agents.gemini_files import UploadJournal

    runner, server = local_runner, interrupted_gemini
    server.remaining = 0
    video = tmp_path / "test.mp4"
    video.write_bytes(b"test video")

    def fail_processing(self, file_object):
        raise runner.NetworkFailure("Processing poll failed after an upload was accepted")

    monkeypatch.setattr(runner.GeminiRestClient, "wait_for_file", fail_processing)
    journal = UploadJournal(tmp_path / "gemini-files.jsonl")
    with judge_worker.parallel_pairs(runner):
        client = runner.GeminiRestClient("fake-key", max_attempts=3, request_timeout=1, file_timeout=5)
        with pytest.raises(runner.NetworkFailure):
            client.upload_video(video)
        assert len(journal.pending()) == 3
        assert not any(name == "cleanup" for name, _ in server.calls)
    assert not journal.pending()
    assert Counter(name for name, _ in server.calls)["cleanup"] == 3
