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
"""Exercise self-hosted job recovery and the real HTTP client without GPUs."""

import json
import subprocess
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from examples.prompt_rewrite import h3
from examples.prompt_rewrite import self_hosted_h3 as hosted
from examples.prompt_rewrite.storage import read_json

MEDIA = {
    "duration_s": 10.125,
    "video": {"width": 1344, "height": 768, "fps": 24.0, "codec": "avc1"},
    "audio": {"sample_rate": 32000, "channels": 2, "codec": "mp4a"},
}
JOB = "338a55c9-c8f3-4dc4-982b-df72221cd893"


@pytest.fixture
def job(tmp_path):
    protocol = {
        "media_profile": h3.media_profile(MEDIA),
        **hosted.configuration(
            hosted.BACKEND,
            "http://worker-a:18080,http://worker-b:18080",
            "https://max.invalid",
            h3.media_profile(MEDIA),
        ),
    }
    task = {
        "id": "sample",
        "request": h3.request_for_protocol("Exact video prompt", protocol),
        "state": str(tmp_path / "state.json"),
        "output": str(tmp_path / "video.mp4"),
    }
    return task, protocol


class FakeClient:
    """Expose one ready worker and record routing and uncertain submissions."""

    def __init__(self):
        self.posts, self.polls, self.downloads = [], [], []
        self.post_error = None
        self.poll_error = None
        self.status = "succeeded"

    def get_json(self, url, **kwargs):
        if url.endswith("/healthz"):
            return {"status": "ready" if "worker-b" in url else "warming", "queued": 0, "queue_capacity": 8}
        self.polls.append(url)
        if self.poll_error:
            raise self.poll_error
        return {"id": JOB, "status": self.status, "error": "inference failed"}

    def submit(self, origin, request, **kwargs):
        self.posts.append((origin, request))
        if self.post_error:
            error, self.post_error = self.post_error, None
            raise error
        return {"id": JOB}

    def download(self, origin, job_id, path, **kwargs):
        self.downloads.append((origin, job_id))
        path.write_bytes(b"test video")


@pytest.mark.parametrize("status", ["ready", "warming"])
def test_public_status_distinguishes_health_from_model_provenance(job, status):
    _, protocol = job
    health = {"status": status, "queued": 0, "queue_capacity": 8, "width": 1344, "transformer_forwards": 4}
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append((self.path, self.headers.get("Authorization")))
            self.send_response(200 if health["status"] == "ready" else 503)
            self.end_headers()
            self.wfile.write(json.dumps(health).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        protocol["h3_endpoints"] = [f"http://127.0.0.1:{server.server_port}"]
        result = hosted.server_status(protocol)
        assert result["all_ready"] is (status == "ready")
        check = result["endpoints"][0]["settings_check"]
        assert check["status"] == "unverified"
        assert "model_revision" in check["missing"] and "adapter_revision" in check["missing"]
        assert "steps" not in check["missing"] and not check["mismatches"]
        health.update(protocol["h3"])
        assert hosted.server_status(protocol)["endpoints"][0]["settings_check"]["status"] == "reported_match"
        health["model_revision"] = "different-model"
        check = hosted.server_status(protocol)["endpoints"][0]["settings_check"]
        assert check["status"] == "mismatch" and check["mismatches"]["model_revision"]["reported"] == "different-model"
        assert requests == [("/healthz", None)] * 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert not hosted.server_status(protocol)["endpoints"][0]["reachable"]


def test_resume_pins_worker_and_does_not_repost(job):
    task, protocol = job
    client = FakeClient()
    client.poll_error = OSError("temporary connection failure")
    assert not hosted.generate_one(task, protocol, client, lambda path: MEDIA)
    assert read_json(Path(task["state"]))["task_id"] == JOB
    client.poll_error = None
    assert hosted.generate_one(task, protocol, client, lambda path: MEDIA)
    assert hosted.generate_one(task, protocol, client, lambda path: MEDIA)
    assert client.posts == [("http://worker-b:18080", {"prompt": "Exact video prompt", "duration": 10, "seed": 42})]
    assert all(url.startswith("http://worker-b:18080/v1/jobs/") for url in client.polls)
    assert client.downloads == [("http://worker-b:18080", JOB)]


def test_single_gateway_allows_multiple_concurrent_requests(job, monkeypatch):
    task, protocol = job
    protocol["h3_endpoints"] = ["http://gateway:18080"]
    barrier = threading.Barrier(3)

    def generate(*args, **kwargs):
        barrier.wait(timeout=5)
        return True

    monkeypatch.setattr(hosted, "generate_one", generate)
    tasks = [{**task, "id": str(index)} for index in range(3)]
    results = hosted.generate_tasks(tasks, protocol, lambda path: MEDIA, concurrency=3, client=FakeClient())
    assert results == {"0": True, "1": True, "2": True}


def test_ambiguous_submission_is_never_retried(job):
    task, protocol = job
    client = FakeClient()
    client.post_error = TimeoutError("response lost after acceptance")
    assert not hosted.generate_one(task, protocol, client, lambda path: MEDIA)
    assert not hosted.generate_one(task, protocol, client, lambda path: MEDIA)
    assert len(client.posts) == 1


@pytest.mark.parametrize("code", [429, 503])
def test_explicit_backpressure_can_be_retried(job, code):
    task, protocol = job
    client = FakeClient()
    client.post_error = urllib.error.HTTPError("http://worker-b:18080/v1/videos", code, "busy", {}, None)
    assert hosted.generate_one(task, protocol, client, lambda path: MEDIA, poll_interval=0.001)
    assert len(client.posts) == 2


def test_failed_generation_is_not_success_and_does_not_repost(job):
    task, protocol = job
    client = FakeClient()
    client.status = "failed"
    assert not hosted.generate_one(task, protocol, client, lambda path: MEDIA)
    assert not hosted.generate_one(task, protocol, client, lambda path: MEDIA)
    assert len(client.posts) == 1 and not client.downloads


def test_changed_pool_and_corrupted_media_cannot_resume(job):
    task, protocol = job
    client = FakeClient()
    assert hosted.generate_one(task, protocol, client, lambda path: MEDIA)
    changed = {**protocol, "h3_endpoints": ["http://worker-c:18080"]}
    with pytest.raises(ValueError, match="different request or worker pool"):
        hosted.generate_one(task, changed, client, lambda path: MEDIA)
    Path(task["output"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        hosted.generate_one(task, protocol, client, lambda path: MEDIA)


def test_invalid_media_remains_an_artifact_error(job):
    task, protocol = job
    assert not hosted.generate_one(task, protocol, FakeClient(), lambda path: {**MEDIA, "duration_s": 5})
    assert read_json(Path(task["state"]))["error_stage"] == "artifact"


@pytest.mark.parametrize("duration", [5, 10])
def test_generated_media_must_match_the_requested_duration(job, duration):
    task, protocol = job
    task["request"].update(duration=duration, seed=7)
    client = FakeClient()
    media = {**MEDIA, "duration_s": duration + 0.125}
    assert hosted.generate_one(task, protocol, client, lambda path: media)
    assert client.posts[0][1] == {"prompt": "Exact video prompt", "duration": duration, "seed": 7}
    assert read_json(Path(task["state"]))["media"] == media


def test_http_client_authenticates_download_and_bypasses_proxies(job, monkeypatch):
    task, protocol = job
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload):
            requests.append((self.command, self.path, self.headers.get("Authorization")))
            assert self.headers.get("Authorization") == "Bearer test-token"
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body == task["request"] and self.path == "/v1/videos"
            self.respond({"id": JOB})

        def do_GET(self):
            if self.path == "/healthz":
                self.respond({"status": "ready", "queued": 0, "queue_capacity": 8})
            elif self.path.endswith("/video"):
                self.respond(b"test MP4")
            else:
                self.respond({"id": JOB, "status": "succeeded"})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
        monkeypatch.setenv("NO_PROXY", "")
        protocol["h3_endpoints"] = [f"http://127.0.0.1:{server.server_port}"]
        assert hosted.generate_one(task, protocol, hosted.Client("test-token"), lambda path: MEDIA)
        assert requests[-1] == ("GET", f"/v1/jobs/{JOB}/video", "Bearer test-token")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_backend_configuration_preserves_reference_provenance(job):
    _, protocol = job
    assert protocol["h3"]["model"] == "MiniMax-H3-Base-FL2VA"
    assert protocol["reference_h3"]["settings"]["model"] == "MiniMax-H3-Max"
    with pytest.raises(ValueError, match="at most 8000"):
        h3.request_for_protocol("x" * 8001, protocol)
    with pytest.raises(ValueError, match="origins"):
        hosted.configuration(hosted.BACKEND, "http://secret@worker:18080", "max", protocol["media_profile"])


def test_max_api_is_disabled_for_cli_and_direct_callers(job, monkeypatch, tmp_path):
    from examples.prompt_rewrite.__main__ import parse_args
    from examples.prompt_rewrite.evaluation import EvaluationRun

    task, protocol = job
    assert parse_args(["prepare", "--run-dir", str(tmp_path)]).h3_backend == "self-hosted"
    with pytest.raises(SystemExit):
        parse_args(["prepare", "--run-dir", str(tmp_path), "--h3-backend", "max"])
    monkeypatch.setattr(
        hosted.Client, "from_environment", classmethod(lambda *a: pytest.fail("Read credentials for a disabled API"))
    )
    for call in (
        lambda: hosted.configuration("max", None, "max", protocol["media_profile"]),
        lambda: EvaluationRun.prepare(tmp_path / "disabled", h3_backend="max"),
        lambda: h3.generate_one(task),
        lambda: h3.generate_tasks([task]),
        lambda: hosted.generate_one(task, {**protocol, "h3_backend": "max"}, FakeClient(), lambda _: MEDIA),
        lambda: hosted.generate_tasks([task], {**protocol, "h3_backend": "max"}, lambda _: MEDIA),
        lambda: h3.run_with_fallbacks(tmp_path, {"h3_backend": "max"}, [task], 1, 1),
    ):
        with pytest.raises(ValueError, match="H3 Max API is disabled"):
            call()
    assert not (tmp_path / "videos/failure-policy.json").exists()
    assert not Path(task["state"]).exists()


def test_missing_self_hosted_connection_does_not_use_max_environment(job, monkeypatch):
    _, protocol = job
    monkeypatch.delenv("H3_SELF_HOSTED_ENDPOINTS", raising=False)
    monkeypatch.delenv("H3_SELF_HOSTED_TOKEN", raising=False)
    monkeypatch.delenv("H3_SELF_HOSTED_TOKEN_FILE", raising=False)
    monkeypatch.setenv("H3_HOST", "https://api.minimax.io")
    monkeypatch.setenv("H3_TOKEN", "paid-token-must-not-be-used")
    with pytest.raises(ValueError, match="H3_SELF_HOSTED_ENDPOINTS"):
        hosted.configuration("self-hosted", None, "max", protocol["media_profile"])
    with pytest.raises(ValueError, match="H3_SELF_HOSTED_TOKEN"):
        hosted.Client.from_environment(protocol)


def test_shared_token_path_works_without_driver_environment(job, tmp_path, monkeypatch):
    _, protocol = job
    monkeypatch.delenv("H3_SELF_HOSTED_TOKEN", raising=False)
    monkeypatch.delenv("H3_SELF_HOSTED_TOKEN_FILE", raising=False)
    token_file = tmp_path / "token"
    token_file.write_text("private-test-token\n")
    config = hosted.configuration("self-hosted", "http://worker:18080", "max", protocol["media_profile"], token_file)
    assert "private-test-token" not in json.dumps(config)
    assert hosted.Client.from_environment(config).headers == {"Authorization": "Bearer private-test-token"}


def test_fragmented_mp4_remux_preserves_encoded_video_and_audio(tmp_path):
    ffmpeg = pytest.importorskip("imageio_ffmpeg").get_ffmpeg_exe()
    output = tmp_path / "fragmented.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=size=32x32:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=sample_rate=32000",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-ac",
            "2",
            "-movflags",
            "empty_moov+frag_keyframe",
            str(output),
        ],
        check=True,
        timeout=30,
    )

    def packet_hashes():
        return subprocess.check_output(
            [
                ffmpeg,
                "-v",
                "error",
                "-i",
                str(output),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c",
                "copy",
                "-f",
                "streamhash",
                "-hash",
                "sha256",
                "-",
            ],
            text=True,
            timeout=30,
        )

    before = packet_hashes()
    original = output.read_bytes()
    hosted.remux_container(output, 30)
    assert packet_hashes() == before
    assert output.read_bytes() != original
