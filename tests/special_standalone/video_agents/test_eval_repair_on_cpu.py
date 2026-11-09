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
"""Replay timeout/quota incidents with the real evaluator and no provider calls."""
# Imported pytest fixtures intentionally share names with test parameters.
# ruff: noqa: F811

import copy
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_h3_recovery_on_cpu import Clock
from test_judge_recovery_on_cpu import external_runner  # noqa: F401
from test_tournament_on_cpu import candidates, native_worker, prepared_rl  # noqa: F401

from examples.video_agents import eval_recovery, judge_worker, streaming_judge, tournament, upload_admission
from examples.video_agents.config import judging_protocol
from examples.video_agents.storage import file_sha256, read_json, write_json


@pytest.fixture
def evaluation(prepared_rl, external_runner, tmp_path, monkeypatch):
    directory, original, _ = prepared_rl
    protocol = copy.deepcopy(original)
    protocol["eval_judge"] = {**protocol["judge"], "judge_model": "gemini-4-argon"}
    protocol["provider_recovery_root"] = str(directory / "provider-recovery")
    sample = copy.deepcopy(protocol["samples"]["val"][0])
    fourth = tmp_path / "fourth-reference.mp4"
    fourth.write_bytes(b"fourth frozen reference")
    sample["references"]["fourth"] = {"path": str(fourth), "sha256": file_sha256(fourth)}
    protocol["reference_fields"] = list(sample["references"])
    clip = candidates(tmp_path)[:1]
    group = directory / "evaluation"
    session = streaming_judge.StreamingJudge(group, protocol, sample, validate=True)
    write_json(session.root / "context.json", session.context)
    session.offer(0, clip[0])
    write_json(session.root / "finished.json", True)
    runner = external_runner
    clock = Clock()
    monkeypatch.setattr(eval_recovery, "time", clock)
    monkeypatch.setattr(upload_admission, "time", clock)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-no-network")
    seen, uploads, deleted = Counter(), Counter(), []
    faults, upload_faults = {}, {}
    manifests = streaming_judge.ready_manifests(protocol, sample, clip, True)
    first = next(iter(manifests["original"]["pairs"].values()))
    target = (Path(first["video_path_A"]).name, Path(first["video_path_B"]).name)

    class Client(runner.GeminiRestClient):
        def upload_video(self, path):
            uploads[path.name] += 1
            remaining = upload_faults.get(path.name, [])
            if remaining:
                raise remaining.pop(0)
            return {"name": f"files/{path.stem}-{uploads[path.name]}", "uri": path.name, "mime_type": "video/mp4"}

        def evaluate(self, payload, **kwargs):
            assert self.max_attempts == 1 and self.request_timeout == 180
            assert payload["model"] == "gemini-4-argon"
            identity = tuple(part["uri"] for part in payload["input"] if part["type"] == "video")
            seen[identity] += 1
            remaining = faults.get(identity, [])
            if remaining:
                raise remaining.pop(0)
            return {"preferred_video": "A", "reason": "Visible evidence", "confidence": "high"}, {}

        def delete_file(self, name):
            deleted.append(name)

        def _request_once(self, *args, **kwargs):
            raise AssertionError("Tests must not contact a provider")

    monkeypatch.setattr(runner, "GeminiRestClient", Client)

    def run():
        with judge_worker.parallel_pairs(runner):
            streaming_judge.run_stream(group, runner)
        write_json(session.root / "outcome.json", {"timed_out": False})

    def score():
        return tournament.score_group(group, protocol, sample, clip, validate=True, streamed=True)

    return SimpleNamespace(
        directory=group,
        root=session.root,
        protocol=protocol,
        sample=sample,
        clips=clip,
        runner=runner,
        clock=clock,
        seen=seen,
        uploads=uploads,
        deleted=deleted,
        faults=faults,
        upload_faults=upload_faults,
        target=target,
        run=run,
        score=score,
    )


def test_timeouts_repair_only_missing_vote_and_resume_without_rejudging(evaluation):
    e = evaluation
    e.faults[e.target] = [e.runner.NetworkFailure("read timed out")] * 2
    e.run()
    assert len(e.seen) == 8 and e.seen[e.target] == 3
    assert sum(e.seen.values()) == 10  # Seven successful votes were never repeated.
    assert all(count == 1 for count in e.uploads.values())
    assert len(e.deleted) == 5
    state = read_json(e.root / "repair.json")
    assert state["complete"] and not state["exhausted"]
    assert sorted(row["attempts"] for row in state["pairs"].values()) == [1] * 7 + [3]
    assert e.score()[0]["judge_comparisons"] == 8
    original_votes = {p: p.read_bytes() for p in e.root.rglob("judgments-*.jsonl")}
    e.run()
    assert sum(e.seen.values()) == 10 and sum(e.uploads.values()) == 5
    assert all(path.read_bytes() == data for path, data in original_votes.items())
    assert read_json(e.root / "repair.json")["deadline_at"] == state["deadline_at"]


@pytest.mark.parametrize("failure", ["refusal", "timeout", "auth"])
def test_permanent_or_exhausted_errors_do_not_invent_votes(evaluation, failure):
    e = evaluation
    if failure == "timeout":
        error = e.runner.NetworkFailure("read timed out")
    elif failure == "refusal":
        error = e.runner.HttpFailure(400, '{"error":{"code":"prohibited_content"}}', {})
    else:
        error = e.runner.HttpFailure(401, "HTTP 401: unauthorized", {})
    e.faults[e.target] = [error] * 10
    if failure == "auth":
        with pytest.raises(RuntimeError, match="Nonrecoverable"):
            e.run()
        assert e.seen[e.target] == 1
        return
    e.run()
    assert len(e.seen) == 8  # One failure never prevents the other seven comparisons.
    assert e.seen[e.target] == (3 if failure == "timeout" else 1)
    with pytest.raises(tournament.ProviderError):
        e.score()
    assert not list(e.root.rglob("fallbacks-*.json"))
    previous = e.seen.copy()
    e.run()
    assert e.seen == previous


def test_quota_repair_drains_readers_cleans_owned_uploads_and_resumes(evaluation):
    e = evaluation
    e.upload_faults[e.target[0]] = [
        e.runner.HttpFailure(429, "HTTP 429: FileStorageBytesPerProject quota exhausted", {})
    ]
    e.run()
    assert len(e.seen) == 8 and set(e.seen.values()) == {1}
    assert read_json(e.root / "repair.json")["complete"]
    assert not read_json(Path(e.protocol["provider_recovery_root"]) / "gemini-upload-quota/state.json")["blocked"]
    assert e.deleted and e.score()[0]["judge_comparisons"] == 8


def test_interrupted_evaluation_reuses_votes_and_remaining_attempt_budget(evaluation):
    e = evaluation
    e.faults[e.target] = [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        e.run()
    state = read_json(e.root / "repair.json")
    e.run()
    assert len(e.seen) == 8 and sum(e.seen.values()) == 9
    assert e.seen[e.target] == 2
    after = read_json(e.root / "repair.json")
    assert after["deadline_at"] == state["deadline_at"] and after["complete"]


def test_i2v_repair_preserves_and_shares_the_source_image(evaluation, monkeypatch, tmp_path):
    from PIL import Image

    from examples.video_agents.image_assets import image_asset

    e = evaluation
    image = tmp_path / "source.png"
    Image.new("RGB", (64, 64), "red").save(image)
    e.sample["image"] = image_asset(image)
    context = read_json(e.root / "context.json")
    context["sample"] = e.sample
    write_json(e.root / "context.json", context)
    uploaded_images, observed_images = [], []
    client = e.runner.GeminiRestClient
    original = client.evaluate

    def upload_image(self, path):
        uploaded_images.append(path)
        return {"name": "files/source", "uri": "source.png", "mime_type": "image/png"}

    def evaluate(self, payload, **kwargs):
        observed_images.extend(part["uri"] for part in payload["input"] if part["type"] == "image")
        return original(self, payload, **kwargs)

    monkeypatch.setattr(client, "upload_image", upload_image)
    monkeypatch.setattr(client, "evaluate", evaluate)
    e.faults[e.target] = [e.runner.NetworkFailure("read timeout")]
    with judge_worker.reference_images(e.runner):
        e.run()
    assert uploaded_images == [image]
    assert observed_images == ["source.png"] * 9
    assert "files/source" in e.deleted and e.score()[0]["judge_comparisons"] == 8


def test_quota_admission_shares_probe_ownership_and_expires_killed_probes(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(upload_admission, "time", clock)
    first = upload_admission.UploadAdmission(tmp_path, clock.time() + 300)
    second = upload_admission.UploadAdmission(tmp_path, clock.time() + 300)
    first.failed(None, "HTTP 429: generativelanguage.googleapis.com/file_storage_bytes")
    with pytest.raises(TimeoutError, match="paused"):
        second.acquire()
    clock.sleep(15)
    token = first.acquire()
    with pytest.raises(TimeoutError, match="paused"):
        second.acquire()
    clock.sleep(181)
    replacement = second.acquire()
    first.succeeded(token)
    assert first.blocked()  # A stale process cannot clear the replacement's lease.
    second.succeeded(replacement)
    assert first.acquire() is None and not second.blocked()


def test_quota_wait_respects_the_original_evaluation_deadline(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(upload_admission, "time", clock)
    gate = upload_admission.UploadAdmission(tmp_path, clock.time() + 3)
    gate.wait_for_probe = True
    gate.failed(None, "FileStorageBytesPerProject")
    with pytest.raises(TimeoutError, match="deadline"):
        gate.acquire()
    assert clock.elapsed == 3 and gate.blocked()


def test_repair_expiration_survives_restart_without_more_api_calls(evaluation):
    e = evaluation
    write_json(e.root / "repair.json", {"deadline_at": e.clock.time() - 1, "pairs": {}})
    e.run()
    assert not e.seen and not e.uploads
    state = read_json(e.root / "repair.json")
    assert state["exhausted"] and not state["complete"]
    with pytest.raises(tournament.ProviderError):
        e.score()
    e.run()
    assert read_json(e.root / "repair.json")["deadline_at"] == state["deadline_at"]


def test_completed_votes_reject_changed_candidate_or_reference(evaluation):
    e = evaluation
    e.run()
    path = Path(e.clips[0]["path"])
    path.write_bytes(b"different checkpoint output")
    with pytest.raises(ValueError, match="Candidate video changed"):
        e.run()
    assert sum(e.seen.values()) == 8


def test_evaluation_timeouts_do_not_change_training_or_protocol_identity(evaluation):
    protocol = evaluation.protocol
    training = judging_protocol(protocol, validate=False)
    validation = judging_protocol(protocol, validate=True)
    assert training["timeouts"]["judge_request"] == 45 and training["timeouts"]["judge_pass"] == 300
    assert validation["timeouts"]["judge_request"] == 180 and validation["timeouts"]["judge_pass"] == 900
    assert validation["judge"]["judge_model"] == "gemini-4-argon"
    assert training["protocol_sha256"] == validation["protocol_sha256"]
    assert training["judge_concurrency"] == validation["judge_concurrency"]
    assert training["max_judge_fallback_fraction"] == validation["max_judge_fallback_fraction"]


@pytest.mark.asyncio
async def test_native_validation_finishes_repair_before_scoring_and_publishing(native_worker, monkeypatch):
    from examples.video_agents import agent
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    worker, events, published = native_worker
    worker.protocol["judge_streaming"] = False
    worker.protocol["eval_repair"] = {"enabled": True, "request_timeout": 180, "pass_timeout": 900}
    stages = []

    class Stream:
        def __init__(self, directory, protocol, sample, *, validate):
            assert validate and protocol["timeouts"]["judge_request"] == 180

        async def start(self):
            stages.append("start")

        def offer(self, index, candidate):
            stages.append("offer")

        async def finish(self):
            stages.append("repaired")

        async def close(self):
            stages.append("closed")

    async def rollout(*args, **kwargs):
        return AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[2],
            response_mask=[1],
            metrics=AgentLoopMetrics(),
            extra_fields={"h3_result": {"status": "succeeded"}},
        )

    def score(*args, **kwargs):
        assert stages[-1] == "repaired" and kwargs["streamed"]
        stages.append("scored")
        return [{"score": 0.75}]

    async def publish(*args, **kwargs):
        assert stages[-1] == "scored"
        published.append(kwargs)

    monkeypatch.setattr(agent, "StreamingJudge", Stream)
    monkeypatch.setattr(agent, "score_group", score)
    monkeypatch.setattr(agent.AgentLoopWorkerTQBase, "_agent_loop_postprocess", publish)
    worker._run_agent_loop = rollout
    record = await worker._run_prompt(
        {"uid": "test", "global_steps": 75, "extra_info": {"id": "eval-0"}}, {}, {"validate": True}
    )
    assert not record["dropped"] and len(published) == 1
    assert stages == ["start", "offer", "repaired", "scored", "closed"]
