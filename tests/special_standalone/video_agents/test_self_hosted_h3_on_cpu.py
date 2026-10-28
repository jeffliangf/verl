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

from examples.video_agents import h3
from examples.video_agents.storage import read_json, write_json

MEDIA = {
    "duration_s": 10.125,
    "video": {"width": 1344, "height": 768, "fps": 24.0, "codec": "avc1"},
    "audio": {"sample_rate": 32000, "channels": 2, "codec": "mp4a"},
}
JOB = "338a55c9-c8f3-4dc4-982b-df72221cd893"
REFERENCE_MANIFEST = {"settings": h3.SETTINGS, "endpoint": h3.ENDPOINT, "h3_endpoints": ["http://reference:18080"]}


@pytest.fixture
def job(tmp_path):
    protocol = {
        "media_profile": h3.media_profile(MEDIA),
        **h3.configuration("http://worker-a:18080,http://worker-b:18080", h3.media_profile(MEDIA), REFERENCE_MANIFEST),
    }
    task = {
        "id": "sample",
        "request": h3.request_body("Exact video prompt", protocol["h3"]),
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
        result = h3.server_status(protocol)
        assert result["all_ready"] is (status == "ready")
        check = result["endpoints"][0]["settings_check"]
        assert check["status"] == "unverified"
        assert "model_revision" in check["missing"] and "adapter_revision" in check["missing"]
        assert "steps" not in check["missing"] and not check["mismatches"]
        health.update(protocol["h3"])
        assert h3.server_status(protocol)["endpoints"][0]["settings_check"]["status"] == "reported_match"
        health["model_revision"] = "different-model"
        check = h3.server_status(protocol)["endpoints"][0]["settings_check"]
        assert check["status"] == "mismatch" and check["mismatches"]["model_revision"]["reported"] == "different-model"
        assert requests == [("/healthz", None)] * 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert not h3.server_status(protocol)["endpoints"][0]["reachable"]


def test_resume_pins_worker_and_does_not_repost(job):
    task, protocol = job
    client = FakeClient()
    client.poll_error = OSError("temporary connection failure")
    assert not h3.generate_one(task, protocol, client, lambda path: MEDIA)
    assert read_json(Path(task["state"]))["task_id"] == JOB
    client.poll_error = None
    assert h3.generate_one(task, protocol, client, lambda path: MEDIA)
    assert h3.generate_one(task, protocol, client, lambda path: MEDIA)
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

    monkeypatch.setattr(h3, "generate_one", generate)
    tasks = [{**task, "id": str(index)} for index in range(3)]
    results = h3.generate_tasks(tasks, protocol, lambda path: MEDIA, concurrency=3, client=FakeClient())
    assert results == {"0": True, "1": True, "2": True}


def test_ambiguous_submission_is_never_retried(job):
    task, protocol = job
    client = FakeClient()
    client.post_error = TimeoutError("response lost after acceptance")
    assert not h3.generate_one(task, protocol, client, lambda path: MEDIA)
    assert not h3.generate_one(task, protocol, client, lambda path: MEDIA)
    assert len(client.posts) == 1
    timing_path = Path(task["state"]).parent / "timings.jsonl"
    events = [json.loads(line) for line in timing_path.read_text().splitlines()]
    requests = [row for row in events if row["stage"] == "h3_submission" and row["event"] == "end"]
    assert len(requests) == 1 and requests[0]["status"] == "error"
    assert requests[0]["error_type"] == "TimeoutError"
    assert requests[0]["timeout_seconds"] == 30


def test_interrupted_submission_is_never_retried(job, monkeypatch):
    task, protocol = job
    client = FakeClient()

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "submit", interrupt)
    with pytest.raises(KeyboardInterrupt):
        h3.generate_one(task, protocol, client, lambda path: MEDIA)
    assert read_json(Path(task["state"]))["status"] == "submitting"
    assert not h3.generate_one(task, protocol, client, lambda path: MEDIA)


@pytest.mark.parametrize("code", [429, 503])
def test_explicit_backpressure_can_be_retried(job, code):
    task, protocol = job
    client = FakeClient()
    client.post_error = urllib.error.HTTPError("http://worker-b:18080/v1/videos", code, "busy", {}, None)
    assert h3.generate_one(task, protocol, client, lambda path: MEDIA, poll_interval=0.001)
    assert len(client.posts) == 2


def test_failed_generation_is_not_success_and_does_not_repost(job):
    task, protocol = job
    client = FakeClient()
    client.status = "failed"
    assert not h3.generate_one(task, protocol, client, lambda path: MEDIA)
    assert not h3.generate_one(task, protocol, client, lambda path: MEDIA)
    assert len(client.posts) == 1 and not client.downloads


def test_changed_pool_and_corrupted_media_cannot_resume(job):
    task, protocol = job
    client = FakeClient()
    assert h3.generate_one(task, protocol, client, lambda path: MEDIA)
    changed = {**protocol, "h3_endpoints": ["http://worker-c:18080"]}
    with pytest.raises(ValueError, match="different request or worker pool"):
        h3.generate_one(task, changed, client, lambda path: MEDIA)
    Path(task["output"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        h3.generate_one(task, protocol, client, lambda path: MEDIA)


def test_invalid_media_remains_an_artifact_error(job):
    task, protocol = job
    assert not h3.generate_one(task, protocol, FakeClient(), lambda path: {**MEDIA, "duration_s": 5})
    assert read_json(Path(task["state"]))["error_stage"] == "artifact"


@pytest.mark.parametrize("duration", [5, 10])
def test_generated_media_must_match_the_requested_duration(job, duration):
    task, protocol = job
    task["request"].update(duration=duration, seed=7)
    client = FakeClient()
    media = {**MEDIA, "duration_s": duration + 0.125}
    assert h3.generate_one(task, protocol, client, lambda path: media)
    assert client.posts[0][1] == {"prompt": "Exact video prompt", "duration": duration, "seed": 7}
    assert read_json(Path(task["state"]))["media"] == media


@pytest.mark.parametrize("execution", ["client", "worker"])
def test_http_generation_authenticates_and_resumes_without_reposting(job, monkeypatch, execution):
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
        if execution == "worker":
            # Exercise the renamed module in an actual subprocess, including
            # worker recovery from a missing download without a second POST.
            directory = Path(task["state"]).parent
            repo = directory / "eval-repo"
            (repo / "bestofn").mkdir(parents=True)
            (repo / "bestofn/normalize.py").write_text(f"def probe(path): return {MEDIA!r}\n")
            protocol["eval_repo"] = str(repo)
            write_json(directory / "protocol.json", protocol)
            write_json(directory / "generation.json", [task])
            monkeypatch.setenv("H3_SELF_HOSTED_TOKEN", "test-token")

            def generate():
                subprocess.run(h3.worker_command(directory, 0.001, 10), check=True, timeout=15)

        else:

            def generate():
                assert h3.generate_one(task, protocol, h3.Client("test-token"), lambda path: MEDIA)

        generate()
        generate()
        Path(task["output"]).unlink()
        generate()
        assert read_json(Path(task["state"]))["status"] == "succeeded"
        assert sum(method == "POST" for method, _, _ in requests) == 1
        assert requests[-1] == ("GET", f"/v1/jobs/{JOB}/video", "Bearer test-token")
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_backend_configuration_preserves_reference_provenance(job):
    _, protocol = job
    assert protocol["h3_backend"] == "self-hosted"
    assert protocol["h3"]["model"] == "MiniMax-H3-Base-FL2VA"
    assert protocol["reference_h3"] == {
        "backend": "self-hosted",
        "settings": h3.SETTINGS,
        "endpoint": "/v1/videos",
        "endpoints": ["http://reference:18080"],
    }
    with pytest.raises(ValueError, match="at most 8000"):
        h3.request_body("x" * 8001, protocol["h3"])
    with pytest.raises(ValueError, match="origins"):
        h3.configuration("http://secret@worker:18080", protocol["media_profile"], REFERENCE_MANIFEST)


def test_cli_has_no_max_backend_option(tmp_path):
    from examples.video_agents import rl

    with pytest.raises(SystemExit):
        rl.parse_args(["prepare", "--run-dir", str(tmp_path), "--h3-backend", "max"])


def test_missing_self_hosted_connection_does_not_use_max_environment(job, monkeypatch):
    _, protocol = job
    monkeypatch.delenv("H3_SELF_HOSTED_ENDPOINTS", raising=False)
    monkeypatch.delenv("H3_SELF_HOSTED_TOKEN", raising=False)
    monkeypatch.delenv("H3_SELF_HOSTED_TOKEN_FILE", raising=False)
    monkeypatch.setenv("H3_HOST", "https://api.minimax.io")
    monkeypatch.setenv("H3_TOKEN", "paid-token-must-not-be-used")
    with pytest.raises(ValueError, match="H3_SELF_HOSTED_ENDPOINTS"):
        h3.configuration(None, protocol["media_profile"], REFERENCE_MANIFEST)
    with pytest.raises(ValueError, match="H3_SELF_HOSTED_TOKEN"):
        h3.Client.from_environment(protocol)


def test_shared_token_path_works_without_driver_environment(job, tmp_path, monkeypatch):
    _, protocol = job
    monkeypatch.delenv("H3_SELF_HOSTED_TOKEN", raising=False)
    monkeypatch.delenv("H3_SELF_HOSTED_TOKEN_FILE", raising=False)
    token_file = tmp_path / "token"
    token_file.write_text("private-test-token\n")
    config = h3.configuration("http://worker:18080", protocol["media_profile"], REFERENCE_MANIFEST, token_file)
    assert "private-test-token" not in json.dumps(config)
    assert h3.Client.from_environment(config).headers == {"Authorization": "Bearer private-test-token"}


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
    h3.remux_container(output, 30)
    assert packet_hashes() == before
    assert output.read_bytes() != original
