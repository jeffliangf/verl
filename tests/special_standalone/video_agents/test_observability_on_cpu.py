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
"""Check concurrent timing records, error visibility, and Gemini stage boundaries."""

import pickle
import socket
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from examples.video_agents import judge_worker
from examples.video_agents.judge_worker import instrument_client
from examples.video_agents.observability import (
    ProviderError,
    failure_record,
    record_event,
    summarize_timings,
    timed_stage,
)
from examples.video_agents.storage import read_jsonl


def test_concurrent_timings_survive_errors_without_logging_credentials(tmp_path):
    path = tmp_path / "timings.jsonl"

    def work(i):
        try:
            with timed_stage(path, "request", request_number=i):
                if i % 2:
                    raise TimeoutError("private-token-must-not-appear")
        except TimeoutError:
            pass

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(work, range(32)))
    rows = read_jsonl(path)
    assert len(rows) == 64
    assert set(Counter(row["stage_id"] for row in rows).values()) == {2}
    starts = {row["stage_id"]: row for row in rows if row["event"] == "start"}
    ends = {row["stage_id"]: row for row in rows if row["event"] == "end"}
    assert len(starts) == 32 and starts.keys() == ends.keys()
    assert Counter(row["status"] for row in ends.values()) == {"succeeded": 16, "error": 16}
    assert all(row["elapsed_seconds"] >= 0 for row in ends.values())
    assert "private-token-must-not-appear" not in path.read_text()


def test_failure_record_keeps_stage_message_and_cause_chain():
    try:
        try:
            raise TimeoutError("read timed out")
        except TimeoutError as exc:
            raise ProviderError("gemini", "Gemini original pass exceeded 300s") from exc
    except ProviderError as exc:
        record = failure_record(exc, sample_id="eval-0")
    assert record["sample_id"] == "eval-0" and record["stage"] == "gemini" and record["type"] == "ProviderError"
    assert record["error"] == "Gemini original pass exceeded 300s"
    assert "TimeoutError: read timed out" in record["traceback"]
    assert failure_record(ValueError("bad artifact"))["stage"] == "internal"


def test_stage_errors_record_the_provider_stage_without_messages(tmp_path):
    path = tmp_path / "timings.jsonl"
    with pytest.raises(ProviderError):
        with timed_stage(path, "group"):
            raise ProviderError("h3", "private message")
    end = read_jsonl(path)[-1]
    assert (end["status"], end["error_type"], end["failure_stage"]) == ("error", "ProviderError", "h3")
    assert "private message" not in path.read_text()


def test_provider_error_retains_classification_across_processes():
    error = pickle.loads(pickle.dumps(ProviderError("h3", "Timed out")))
    assert error.stage == "h3" and str(error) == "Timed out"


def test_new_events_preserve_an_interrupted_utf8_record(tmp_path):
    path = tmp_path / "timings.jsonl"
    fragment = b'{"stage":"\xe2\x82'
    path.write_bytes(fragment)
    with timed_stage(path, "h3_submission"):
        pass
    assert path.read_bytes().startswith(fragment + b"\n")
    summary = summarize_timings(tmp_path)
    assert summary["malformed_lines"] == 1
    assert summary["stages"]["h3_submission"]["successful_observations"] == 1
    assert not summary["unfinished_stages"]


def test_summary_separates_failed_and_unfinished_requests_from_latency(tmp_path):
    path = tmp_path / "timings.jsonl"
    for i, (status, seconds) in enumerate([("succeeded", 10), ("succeeded", 20), ("error", 900)]):
        record_event(path, "gemini_request", "start", stage_id=str(i))
        record_event(path, "gemini_request", "end", stage_id=str(i), status=status, elapsed_seconds=seconds)
    record_event(path, "h3_submission", "start", stage_id="unfinished")
    record_event(path, "group", "start", stage_id="group")
    record_event(path, "group", "end", stage_id="group", status="dropped", failure_stage="h3", elapsed_seconds=300)
    with path.open("a") as handle:
        handle.write('{"event":')
    summary = summarize_timings(tmp_path)
    request = summary["stages"]["gemini_request"]
    assert request["outcomes"] == {"succeeded": 2, "error": 1}
    assert request["p50_seconds"] == 15
    assert request["p99_seconds"] == pytest.approx(19.9)
    assert request["proposed_timeout_seconds"] == 40
    assert summary["stages"]["group"]["outcomes"] == {"dropped": 1} and summary["malformed_lines"] == 1
    assert [row["stage_id"] for row in summary["unfinished_stages"]] == ["unfinished"]


def test_gemini_upload_and_judgment_have_separate_timeouts_and_logs(tmp_path):
    calls = []

    class Client:
        request_timeout = 34
        file_timeout = 300

        def _request_once(self, method, url, **kwargs):
            calls.append((url, kwargs.get("timeout", self.request_timeout)))
            if url.endswith("/interactions"):
                raise TimeoutError("Gemini stopped responding")
            return {}

        def upload_video(self, path):
            self._request_once("POST", "https://example.test/upload")
            self._request_once("POST", "https://example.test/upload/content", timeout=self.file_timeout)
            return self.wait_for_file({})

        def wait_for_file(self, value):
            self._request_once("GET", "https://example.test/files/1")
            return value

        def evaluate(self, payload, *, require_agentic_steps):
            return self._request_once("POST", "https://example.test/interactions")

        def delete_file(self, name):
            return self._request_once("DELETE", "https://example.test/files/1")

    runner = SimpleNamespace(GeminiRestClient=Client)
    timings = tmp_path / "timings.jsonl"
    instrument_client(runner, timings)
    client = runner.GeminiRestClient()
    client.upload_video(tmp_path / "video.mp4")
    with pytest.raises(TimeoutError):
        client.evaluate({}, require_agentic_steps=False)
    client.delete_file("files/1")
    assert [timeout for _, timeout in calls] == [30, 300, 30, 34, 30]
    completed = [row for row in read_jsonl(timings) if row["event"] == "end"]
    assert {row["stage"] for row in completed} == {
        "gemini_upload",
        "gemini_file_processing",
        "gemini_request",
        "gemini_file_delete",
        "gemini_http",
    }
    judgment = next(row for row in completed if row["stage"] == "gemini_request")
    assert judgment["status"] == "error" and judgment["error_type"] == "TimeoutError"


def test_judge_worker_forwards_evaluator_options_without_live_calls(tmp_path, monkeypatch):
    forwarded = []
    runner = SimpleNamespace(GeminiRestClient=type("Client", (), {}), main=lambda args: forwarded.extend(args) or 0)
    monkeypatch.setattr(judge_worker, "load_external", lambda path: runner)
    args = ["--request-timeout", "34", "--file-timeout", "300", "--pair-id", "0_vs_1"]
    assert judge_worker.main(["--runner", "runner.py", "--timings", str(tmp_path / "timings.jsonl"), *args]) == 0
    assert forwarded == args


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_judge_control_timeout_must_be_finite_and_positive(tmp_path, value):
    with pytest.raises(SystemExit, match="2"):
        judge_worker.main(
            ["--runner", "unused.py", "--timings", str(tmp_path / "timings.jsonl"), "--file-control-timeout", value]
        )


def test_control_timeout_preserves_shorter_file_budget_and_explicit_override(tmp_path):
    calls = []

    class Client:
        request_timeout = 120
        file_timeout = 10

        def _request_once(self, method, url, **kwargs):
            calls.append(kwargs["timeout"])

    runner = SimpleNamespace(GeminiRestClient=Client)
    instrument_client(runner, tmp_path / "timings.jsonl", file_control_timeout=30)
    client = runner.GeminiRestClient()
    client._request_once("DELETE", "https://example.test/files/1")
    client._request_once("POST", "https://example.test/upload", timeout=7)
    assert calls == [10, 7]


def test_ipv4_resolution_is_scoped_to_gemini_and_preserves_dns(monkeypatch):
    calls = []
    sentinel = object()

    def resolver(*args):
        calls.append(args)
        return sentinel

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    with judge_worker.gemini_address_family("ipv4"):
        assert socket.getaddrinfo("generativelanguage.googleapis.com", 443, type=socket.SOCK_STREAM) is sentinel
        socket.getaddrinfo(b"generativelanguage.googleapis.com", 443)
        socket.getaddrinfo("generativelanguage.googleapis.com", 443, family=socket.AF_INET6)
        socket.getaddrinfo("another.googleapis.com", 443)
        socket.getaddrinfo("127.0.0.1", 28080)
    assert [call[2] for call in calls] == [socket.AF_INET, socket.AF_INET, socket.AF_INET6, 0, 0]
    assert calls[0] == ("generativelanguage.googleapis.com", 443, socket.AF_INET, socket.SOCK_STREAM, 0, 0)
    assert socket.getaddrinfo is resolver


def test_gemini_resolution_restored_after_evaluator_error():
    original = socket.getaddrinfo
    with pytest.raises(RuntimeError, match="evaluator failed"):
        with judge_worker.gemini_address_family("ipv4"):
            assert socket.getaddrinfo is not original
            raise RuntimeError("evaluator failed")
    assert socket.getaddrinfo is original


def test_auto_address_family_keeps_resolver():
    original = socket.getaddrinfo
    with judge_worker.gemini_address_family("auto"):
        assert socket.getaddrinfo is original


@pytest.mark.parametrize("family", ["ipv4", "auto"])
def test_worker_address_family_applies_during_runner_only(tmp_path, monkeypatch, family):
    original = socket.getaddrinfo
    forwarded = []

    def main(args):
        assert (socket.getaddrinfo is original) == (family == "auto")
        forwarded.extend(args)
        return 0

    runner = SimpleNamespace(GeminiRestClient=type("Client", (), {}), main=main)
    monkeypatch.setattr(judge_worker, "load_external", lambda path: runner)
    timings = tmp_path / "timings.jsonl"
    assert (
        judge_worker.main(
            ["--runner", "runner.py", "--timings", str(timings), "--address-family", family, "--pair-id", "0_vs_1"]
        )
        == 0
    )
    assert forwarded == ["--pair-id", "0_vs_1"]
    assert read_jsonl(timings)[0]["address_family"] == family
    assert socket.getaddrinfo is original
