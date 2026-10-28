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
"""Exercise bounded H3 recovery with fault injection and real HTTP transfers."""

import errno
import hashlib
import io
import json
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from examples.video_agents import h3
from examples.video_agents.storage import file_sha256, read_json

MEDIA = {
    "duration_s": 10.125,
    "video": {"width": 1344, "height": 768, "fps": 24.0, "codec": "avc1"},
    "audio": {"sample_rate": 32000, "channels": 2, "codec": "mp4a"},
}
VIDEO = b"the gateway's archived MP4"


def http_error(code, *, body=None, headers=None):
    return urllib.error.HTTPError(
        "http://gateway/v1/videos",
        code,
        "injected failure",
        headers or {},
        io.BytesIO(json.dumps(body or {}).encode()),
    )


class Clock:
    def __init__(self):
        self.elapsed = 0.0
        self.delays = []

    def monotonic(self):
        return self.elapsed

    def time(self):
        return 1793230000 + self.elapsed

    def sleep(self, seconds):
        self.delays.append(seconds)
        self.elapsed += seconds


class Gateway:
    """Model caller IDs, durable jobs, failure codes and archived checksums."""

    def __init__(self, task):
        self.task = task
        self.jobs, self.posts, self.polls, self.downloads = {}, [], [], []
        self.post_errors, self.poll_errors, self.download_results = [], [], []
        self.generation_results = []
        self.ready, self.keyed, self.accept_before_error = True, True, True
        self.health_calls = 0

    def get_json(self, url, **kwargs):
        if url.endswith("/healthz"):
            self.health_calls += 1
            return {
                "service": "Sol-H3 gateway",
                "status": "ready" if self.ready else "unavailable",
                "idempotency_key": self.keyed,
                "queued": 0,
                "queue_capacity": 8,
            }
        self.polls.append((url, kwargs["timeout"]))
        if self.poll_errors:
            raise self.poll_errors.pop(0)
        return dict(self.jobs[url.rsplit("/", 1)[1]])

    def submit(self, origin, request, *, idempotency_key=None, timeout):
        saved = read_json(Path(self.task["state"]))
        assert saved["idempotency_key"] == idempotency_key
        assert saved["request"] == request and saved["worker_url"] == origin
        self.posts.append((origin, dict(request), idempotency_key, timeout))
        error = self.post_errors.pop(0) if self.post_errors else None
        if callable(error):
            error = error(idempotency_key)
        if error and not self.accept_before_error:
            raise error
        identifier = idempotency_key or str(uuid4())
        if identifier not in self.jobs:
            outcome = self.generation_results.pop(0) if self.generation_results else "succeeded"
            job = {"id": identifier, "status": "succeeded", **request}
            if outcome == "succeeded":
                job.update(video_sha256=hashlib.sha256(VIDEO).hexdigest(), video_bytes=len(VIDEO))
            else:
                job.update(status="failed", error_code=outcome, error="injected generation failure")
            self.jobs[identifier] = job
        if error:
            self.ready = False  # Replays must not wait for new-admission readiness.
            raise error
        return dict(self.jobs[identifier])

    def download(self, origin, identifier, output, **kwargs):
        self.downloads.append((origin, identifier, kwargs["timeout"]))
        result = self.download_results.pop(0) if self.download_results else VIDEO
        if isinstance(result, Exception):
            raise result
        output.write_bytes(result)


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    task = {
        "id": "candidate",
        "request": {"prompt": "Preserve this exact prompt", "duration": 10, "seed": 7},
        "state": str(tmp_path / "state.json"),
        "output": str(tmp_path / "video.mp4"),
    }
    protocol = {
        "h3_endpoints": ["http://gateway:18080"],
        "media_profile": h3.media_profile(MEDIA),
        "timeouts": {"h3_request": 30},
    }
    clock, client = Clock(), Gateway(task)
    monkeypatch.setattr(h3, "time", clock)
    monkeypatch.setattr(h3.random, "uniform", lambda *args: 0)
    return SimpleNamespace(
        task=task,
        protocol=protocol,
        clock=clock,
        client=client,
        run=lambda: h3.generate_one(task, protocol, client, lambda path: MEDIA, timeout=150),
        state=lambda: read_json(Path(task["state"])),
    )


def test_submission_replays_one_persisted_key_for_mixed_errors(candidate):
    c = candidate
    c.client.post_errors = [http_error(502), TimeoutError("response lost")]
    assert c.run()
    assert len(c.client.posts) == 3 and len(c.client.jobs) == 1
    keys = {row[2] for row in c.client.posts}
    assert keys == {c.state()["task_id"]} and str(UUID(next(iter(keys)))) == next(iter(keys))
    assert all(row[1] == c.task["request"] for row in c.client.posts)
    assert c.client.health_calls == 1 and c.clock.delays == [1, 2]
    assert c.state()["transport_retries"] == 2
    events = Path(c.task["state"]).with_name("timings.jsonl").read_text()
    assert c.task["request"]["prompt"] not in events
    assert '"event": "retry"' in events and '"event": "regeneration"' not in events


def test_accepted_502_polls_without_replaying(candidate):
    c = candidate
    c.client.post_errors = [lambda key: http_error(502, body={"detail": {"id": key}})]
    assert c.run()
    assert len(c.client.posts) == len(c.client.polls) == len(c.client.jobs) == 1
    assert c.state()["task_id"] == c.client.posts[0][2]


def test_interrupted_submission_reuses_key_after_restart(candidate):
    c = candidate
    c.client.post_errors = [KeyboardInterrupt()]
    with pytest.raises(KeyboardInterrupt):
        c.run()
    before = c.state()
    assert before["status"] == "submitting" and not before.get("task_id")
    assert c.run()
    assert len(c.client.jobs) == 1 and len(c.client.posts) == 2
    assert c.client.health_calls == 1
    assert c.state()["deadline_at"] == before["deadline_at"]


def test_submission_retry_budget_survives_restart(candidate):
    c = candidate
    c.client.post_errors = [TimeoutError("lost response")] * 3
    assert not c.run()
    assert not c.run()
    assert len(c.client.posts) == 3 and len(c.client.jobs) == 1
    assert c.state()["status"] == "submit_unknown"


def test_legacy_ambiguous_submission_keeps_conservative_behavior(candidate):
    c = candidate
    c.client.keyed = False
    c.client.post_errors = [TimeoutError("lost response")]
    assert not c.run() and not c.run()
    assert len(c.client.posts) == 1 and c.client.posts[0][2] is None


@pytest.mark.parametrize("code", sorted(h3.RECOVERABLE_FAILURES))
def test_only_confirmed_failures_create_a_new_generation(candidate, code):
    c = candidate
    c.client.generation_results = [code, "succeeded"]
    assert c.run()
    first, second = c.client.posts
    assert first[2] != second[2] and first[1] == second[1] == c.task["request"]
    assert c.state()["generation_attempt"] == 2
    assert c.state()["previous_attempts"][0]["task_id"] == first[2]
    assert c.state()["previous_attempts"][0]["error_code"] == code
    assert len(c.client.jobs) == 2 and len(c.client.downloads) == 1


def test_generation_budget_cannot_be_reset_by_resuming(candidate):
    c = candidate
    c.client.generation_results = ["worker_restarted"] * 3
    assert not c.run() and not c.run()
    assert len(c.client.posts) == 2 and len(c.client.jobs) == 2
    assert c.state()["status"] == "failed" and c.state()["generation_attempt"] == 2


def test_unknown_terminal_failure_is_not_regenerated(candidate):
    c = candidate
    c.client.generation_results = ["unknown_failure"]
    assert not c.run()
    assert len(c.client.posts) == 1 and c.state()["error_code"] == "unknown_failure"


@pytest.mark.parametrize("code", [400, 401, 403, 409, 422])
def test_request_errors_are_fatal_without_retry(candidate, code):
    c = candidate
    c.client.accept_before_error = False
    c.client.post_errors = [http_error(code)]
    assert not c.run() and not c.run()
    assert len(c.client.posts) == 1 and not c.client.jobs
    assert c.state()["error_stage"] == "configuration"


def test_poll_and_download_retries_preserve_one_generation(candidate):
    c = candidate
    c.client.poll_errors = [http_error(503), ConnectionResetError("poll interrupted")]
    c.client.download_results = [ConnectionResetError("download interrupted"), b"truncated video", VIDEO]
    assert c.run()
    assert len(c.client.posts) == len(c.client.jobs) == 1
    assert len(c.client.polls) == len(c.client.downloads) == 3
    assert c.state()["transport_retries"] == 4
    assert c.state()["download_sha256"] == c.state()["sha256"] == hashlib.sha256(VIDEO).hexdigest()


def test_checksum_is_verified_before_remux(candidate, monkeypatch):
    c = candidate
    probes = iter([{"duration_s": 0, "video": {"frames": 0}}, MEDIA])
    monkeypatch.setattr(h3, "remux_container", lambda path, timeout: path.write_bytes(VIDEO + b"remux"))
    assert h3.generate_one(c.task, c.protocol, c.client, lambda path: next(probes), timeout=150)
    assert c.state()["download_sha256"] == hashlib.sha256(VIDEO).hexdigest()
    assert c.state()["sha256"] == hashlib.sha256(VIDEO + b"remux").hexdigest()
    assert c.state()["container_remuxed"]


def test_persistent_corrupt_download_never_regenerates(candidate):
    c = candidate
    c.client.download_results = [b"corrupt"] * 3
    assert not c.run()
    assert len(c.client.posts) == 1 and len(c.client.downloads) == 3
    assert c.state()["error_stage"] == "artifact" and not Path(c.task["output"]).exists()


def test_local_storage_error_is_not_a_transport_retry(candidate):
    c = candidate
    c.client.download_results = [OSError(errno.ENOSPC, "disk full")]
    assert not c.run()
    assert len(c.client.posts) == len(c.client.downloads) == 1
    assert c.state()["error_stage"] == "artifact"


@pytest.mark.parametrize("format", ["seconds", "date"])
def test_retry_after_is_honored(candidate, format):
    c = candidate
    value = "7" if format == "seconds" else format_datetime(datetime.fromtimestamp(c.clock.time() + 7, timezone.utc))
    c.client.post_errors = [http_error(429, headers={"Retry-After": value})]
    assert c.run() and c.clock.delays == [7]


def test_retry_after_cannot_extend_the_deadline(candidate):
    c = candidate
    c.client.post_errors = [http_error(503, headers={"Retry-After": "151"})]
    assert not c.run()
    assert len(c.client.posts) == 1 and not c.clock.delays
    assert c.state()["error_stage"] == "provider"


def test_regeneration_and_restart_share_the_original_150_seconds(candidate, monkeypatch):
    c = candidate
    original = c.client.get_json

    def status(url, **kwargs):
        response = original(url, **kwargs)
        if not url.endswith("/healthz"):
            response["status"] = "running"
            if len(c.client.posts) == 1 and c.clock.elapsed >= 100:
                response.update(status="failed", error_code="worker_restarted")
        return response

    monkeypatch.setattr(c.client, "get_json", status)
    assert not c.run()
    assert len(c.client.posts) == 2
    assert c.clock.elapsed <= 150 and c.clock.elapsed >= 149
    assert c.state()["deadline_at"] == 1793230150
    assert c.state()["error_stage"] == "provider"
    c.clock.elapsed += 10
    calls = len(c.client.polls)
    assert not c.run() and len(c.client.polls) == calls and len(c.client.posts) == 2


def test_missing_status_does_not_create_another_job(candidate):
    c = candidate
    c.client.poll_errors = [http_error(404)]
    assert not c.run()
    assert len(c.client.posts) == 1 and len(c.client.jobs) == 1


def test_wrong_job_id_in_502_is_not_adopted(candidate):
    c = candidate
    c.client.post_errors = [http_error(502, body={"detail": {"id": str(uuid4())}})]
    assert not c.run()
    assert not c.client.polls and c.state()["error_stage"] == "protocol"


@pytest.mark.parametrize("invalid", [None, {}, {"status": "unexpected"}, {"id": str(uuid4()), "status": "running"}])
def test_malformed_job_status_remains_fatal(candidate, monkeypatch, invalid):
    c = candidate
    original = c.client.get_json
    monkeypatch.setattr(
        c.client, "get_json", lambda url, **kwargs: original(url, **kwargs) if url.endswith("/healthz") else invalid
    )
    assert not c.run() and c.state()["error_stage"] == "protocol"
    assert len(c.client.posts) == 1 and not c.client.downloads


def test_gateway_success_requires_checksum_metadata(candidate, monkeypatch):
    c = candidate
    original = c.client.get_json

    def without_checksum(url, **kwargs):
        response = original(url, **kwargs)
        response.pop("video_sha256", None)
        return response

    monkeypatch.setattr(c.client, "get_json", without_checksum)
    assert not c.run() and c.state()["error_stage"] == "protocol"
    assert not c.client.downloads and len(c.client.posts) == 1


@pytest.mark.parametrize("key", list(h3.RECOVERY_DEFAULTS))
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_recovery_limits_are_rejected(key, invalid):
    with pytest.raises(ValueError, match="positive integers"):
        h3.recovery_settings({key: invalid})


def test_slots_bound_processes_and_release_after_failure(tmp_path):
    # Independent file descriptors model independently running agent executors.
    lock, gate = threading.Lock(), threading.Event()
    entered, maximum, active = [], 0, 0

    def run(index):
        nonlocal maximum, active
        try:
            with h3.generation_slot(tmp_path, 2, tmp_path / "timings.jsonl"):
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                    entered.append(index)
                    if len(entered) == 2:
                        gate.set()
                try:
                    assert gate.wait(5)
                    if index == 0:
                        raise RuntimeError("subprocess failed")
                finally:
                    with lock:
                        active -= 1
        except RuntimeError:
            assert index == 0

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(run, range(6)))
    assert maximum == 2 and sorted(entered) == list(range(6))
    with h3.generation_slot(tmp_path, 1, tmp_path / "timings.jsonl"):
        pass


def test_real_http_recovery_keeps_one_generation(candidate):
    c = candidate
    calls = {"post": 0, "download": 0}
    jobs, keys = {}, []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, data, code=200):
            payload = data if isinstance(data, bytes) else json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            assert self.headers["Authorization"] == "Bearer test-token"
            key = self.headers["Idempotency-Key"]
            assert read_json(Path(c.task["state"]))["idempotency_key"] == key
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body == c.task["request"]
            keys.append(key)
            calls["post"] += 1
            job = jobs.setdefault(
                key,
                {
                    "id": key,
                    "status": "succeeded",
                    "video_bytes": len(VIDEO),
                    "video_sha256": hashlib.sha256(VIDEO).hexdigest(),
                },
            )
            if calls["post"] == 1:
                self.close_connection = True  # Accepted, then lost the HTTP response.
                return
            self.respond(job, 202)

        def do_GET(self):
            if self.path == "/healthz":
                assert not calls["post"]  # Recovery must bypass health/admission gating.
                return self.respond(
                    {
                        "service": "Sol-H3 gateway",
                        "status": "ready",
                        "queued": 0,
                        "queue_capacity": 8,
                        "idempotency_key": True,
                    }
                )
            assert self.headers["Authorization"] == "Bearer test-token"
            key = self.path.split("/")[3]
            if self.path.endswith("/video"):
                calls["download"] += 1
                if calls["download"] == 1:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(VIDEO)))
                    self.end_headers()
                    self.wfile.write(VIDEO[:5])
                    self.close_connection = True
                    return
                return self.respond(VIDEO)
            self.respond(jobs[key])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        c.protocol["h3_endpoints"] = [f"http://127.0.0.1:{server.server_port}"]
        assert h3.generate_one(c.task, c.protocol, h3.Client("test-token"), lambda path: MEDIA, timeout=150)
        assert calls == {"post": 2, "download": 2} and len(jobs) == len(set(keys)) == 1
        assert file_sha256(Path(c.task["output"])) == hashlib.sha256(VIDEO).hexdigest()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
