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
"""Check neutral scoring and real evaluator retry behavior without provider calls."""

from types import SimpleNamespace

import pytest

from examples.video_agents import judge, judge_worker, scoring
from examples.video_agents.config import CANDIDATE, DEFAULT_EVAL_REPO
from examples.video_agents.storage import load_external, read_jsonl


@pytest.mark.parametrize("first,second,expected", [(1, None, 0.75), (0, None, 0.25), (None, None, 0.5)])
def test_neutral_order_preserves_the_observed_vote(first, second, expected):
    pair = {"model_A": CANDIDATE, "model_B": "reference", "reference": "reference", "prompt_id": "p", "source": "s"}
    pair.update(video_path_A="a.mp4", video_path_B="b.mp4")

    def row(value):
        if value is None:
            return {"status": "fallback", "fallback_policy": "neutral-v1", "fallback_score": 0.5}
        return {"status": "completed", "preferred_model": CANDIDATE if value else "reference"}

    result = scoring.score_pair(pair, row(first), row(second), allow_fallback=True)
    reversed_result = scoring.score_pair(judge.swap_pair(pair), row(second), row(first), allow_fallback=True)
    assert result["score"] == reversed_result["score"] == expected
    assert result["candidate_first_win"] == reversed_result["candidate_first_win"]
    assert result["fallback_count"] == (first is None) + (second is None)
    assert result["outcome"] == "fallback"
    assert result["swapped_winner"] is None
    with pytest.raises(ValueError):
        scoring.score_pair(pair, row(first), row(second))


@pytest.mark.parametrize(
    "kind,message,expected",
    [
        ("NetworkFailure", "read timed out", "provider_failure"),
        ("FileProcessingFailure", "video not ACTIVE", "provider_failure"),
        ("HttpFailure", "HTTP 503: unavailable", "provider_failure"),
        ("HttpFailure", 'HTTP 400: {"error":{"code":"prohibited_content"}}', "content_refusal"),
        (
            "HttpFailure",
            'HTTP 400: {"error":{"message":"Request contains an invalid argument.","code":"invalid_request"}}',
            "judge_request_rejected",
        ),
        ("EvalError", "Gemini output is not valid JSON", "invalid_judge_response"),
        ("HttpFailure", "HTTP 401: unauthorized", None),
        ("HttpFailure", "HTTP 403: permission denied", None),
        ("HttpFailure", "HTTP 400: invalid argument", None),
        ("HttpFailure", 'HTTP 400: {"error":{"code":"invalid_request","message":"Unknown field: typo"}}', None),
        ("OSError", "missing local video", None),
        ("UnknownError", "programming error", None),
    ],
)
def test_fallback_only_accepts_unscorable_provider_responses(kind, message, expected):
    row = {"stage": "interaction", "error_type": kind, "error": message}
    if expected:
        assert judge.fallback_reason(row) == expected
    else:
        with pytest.raises(RuntimeError, match="Nonrecoverable"):
            judge.fallback_reason(row)


@pytest.mark.parametrize("stage", ["upload", "file_processing", "cleanup", None])
def test_generic_invalid_argument_outside_interaction_remains_fatal(stage):
    row = {
        "stage": stage,
        "error_type": "HttpFailure",
        "error": 'HTTP 400: {"error":{"code":"invalid_request","message":"Request contains an invalid argument."}}',
    }
    with pytest.raises(RuntimeError, match="Nonrecoverable"):
        judge.fallback_reason(row)


def test_network_retry_opt_in_is_scoped_to_judgment_posts(tmp_path):
    calls = []

    class Client:
        def request_json(self, method, url, **kwargs):
            calls.append((method, url, kwargs))

    runner = SimpleNamespace(GeminiRestClient=Client)
    judge_worker.instrument_client(runner, tmp_path / "timings.jsonl")
    client = runner.GeminiRestClient()
    client.request_json("POST", "https://example.test/interactions", retry_network=False)
    client.request_json("POST", "https://example.test/files", retry_network=False)
    client.request_json("GET", "https://example.test/interactions", retry_network=False)
    assert [call[2]["retry_network"] for call in calls] == [True, False, False]


def test_file_processing_deadline_is_a_provider_failure_but_invalid_local_files_are_not():
    row = {
        "stage": "upload",
        "error_type": "EvalError",
        "error": "Timed out waiting for Gemini file files/1 to become ACTIVE",
    }
    assert judge.fallback_reason(row) == "provider_failure"
    row["error"] = "Video is empty: local.mp4"
    with pytest.raises(RuntimeError):
        judge.fallback_reason(row)


@pytest.fixture
def external_runner():
    path = DEFAULT_EVAL_REPO / "scripts/run_gemini_pairwise_eval.py"
    if not path.exists():
        pytest.skip("Optional local evaluator integration; no downloads or live calls")
    return load_external(path)


@pytest.mark.parametrize(
    "failures,expected_attempts,recovered",
    [
        (["network", "network"], 3, True),
        ([429, "network"], 3, True),
        (["network"] * 3, 3, False),
        ([401], 1, False),
        ([400], 1, False),
    ],
)
def test_real_evaluator_shares_three_attempts_and_reuses_uploaded_files(
    external_runner, tmp_path, monkeypatch, failures, expected_attempts, recovered
):
    runner = external_runner
    requests, sleeps = [], []

    def request(self, method, url, **kwargs):
        requests.append((method, url, kwargs["data"]))
        index = len(requests) - 1
        if index < len(failures):
            failure = failures[index]
            if failure == "network":
                raise runner.NetworkFailure("read timed out")
            raise runner.HttpFailure(failure, "provider error")
        return {}, b"{}"

    monkeypatch.setattr(runner.GeminiRestClient, "_request_once", request)
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)
    monkeypatch.setattr(runner.random, "uniform", lambda a, b: 0)
    monkeypatch.setattr(runner, "extract_interaction_result", lambda *a, **kw: ({"preferred_video": "A"}, {}))
    timings = tmp_path / "timings.jsonl"
    judge_worker.instrument_client(runner, timings)
    client = runner.GeminiRestClient("fake-test-key", max_attempts=3, request_timeout=45)
    payload = {"input": [{"uri": "files/existing-video-a"}, {"uri": "files/existing-video-b"}]}
    if recovered:
        assert client.evaluate(payload, require_agentic_steps=False)[0]["preferred_video"] == "A"
    else:
        with pytest.raises(runner.EvalError):
            client.evaluate(payload, require_agentic_steps=False)
    assert len(requests) == expected_attempts
    assert len(set(requests)) == 1
    assert sleeps == [1, 2][: expected_attempts - 1]
    ends = [r for r in read_jsonl(timings) if r["event"] == "end" and r["stage"] == "gemini_request"]
    assert len(ends) == 1
    assert ends[0]["attempts"] == expected_attempts and ends[0]["recovered"] is recovered
