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
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from examples.video_agents import judge_worker
from examples.video_agents.judge_worker import instrument_client
from examples.video_agents.observability import ProviderError, record_event, summarize_timings, timed_stage
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
    record_event(path, "group", "dropped")
    with path.open("a") as handle:
        handle.write('{"event":')
    summary = summarize_timings(tmp_path)
    request = summary["stages"]["gemini_request"]
    assert request["outcomes"] == {"succeeded": 2, "error": 1}
    assert request["p50_seconds"] == 15
    assert request["p99_seconds"] == pytest.approx(19.9)
    assert request["proposed_timeout_seconds"] == 40
    assert summary["dropped_groups"] == 1 and summary["malformed_lines"] == 1
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
    assert [timeout for _, timeout in calls] == [300, 300, 34, 300]
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
