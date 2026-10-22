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
"""Durable client for the Sol-H3 gateway or workers, with authenticated downloads."""

import json
import os
import random
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from math import isfinite
from pathlib import Path
from uuid import UUID

from .config import H3_ENDPOINT, H3_SETTINGS
from .storage import file_sha256, fingerprint, read_json, write_json

BACKEND = "self-hosted"
ENDPOINT = "/v1/videos"
DEFAULT_CONCURRENCY = 8
DISABLED_API_MESSAGE = (
    "The H3 Max API is disabled. Prepare a new run with --h3-backend self-hosted "
    "and --h3-endpoints pointing to the Sol-H3 gateway."
)
SETTINGS = {
    "model": "MiniMax-H3-Base-FL2VA",
    "model_revision": "42ed227ee7df40d41602854ae760620d6eb651fe",
    "adapter": "FastVideo/FastVideo-FastH3-4-step-Preview-v1-LoRA/dense-datafree",
    "adapter_revision": "f509e629374cac104e7f62daecce6d1488a3041d",
    "steps": 4,
    "duration": 10,
    "seed": 42,
    "width": 1344,
    "height": 768,
    "fps": 24,
}


def add_arguments(parser):
    """Expose the self-hosted service; the paid Max API is disabled."""
    parser.add_argument("--h3-backend", choices=(BACKEND,), default=BACKEND)
    parser.add_argument(
        "--h3-endpoints", help="Sol-H3 gateway URL or comma-separated worker origins; or set H3_SELF_HOSTED_ENDPOINTS"
    )
    parser.add_argument(
        "--h3-token-file", type=Path, help="Bearer-token file readable at this path by every Ray worker"
    )


def require_backend(backend):
    """Reject legacy or unknown providers before any credentials or network work."""
    if backend != BACKEND:
        raise ValueError(DISABLED_API_MESSAGE)


def configuration(backend, endpoints, reference_host, profile, token_file=None, *, reference_manifest=None):
    """Freeze the self-hosted generator and the references' actual provenance."""
    require_backend(backend)
    endpoints = endpoints or os.environ.get("H3_SELF_HOSTED_ENDPOINTS")
    if not endpoints:
        raise ValueError("Set --h3-endpoints or H3_SELF_HOSTED_ENDPOINTS to the Sol-H3 gateway URL")
    origins = [url.strip().rstrip("/") for url in endpoints.split(",")]
    for origin in origins:
        parsed = urllib.parse.urlsplit(origin)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Supply --h3-endpoints as comma-separated http(s) origins without credentials or paths")
        _ = parsed.port  # Validate malformed port numbers before freezing a run.
    if len(set(origins)) != len(origins):
        raise ValueError("Duplicate self-hosted H3 endpoint")
    expected = {"width": 1344, "height": 768, "fps": 24, "audio_sample_rate": 32000, "audio_channels": 2}
    if any(profile.get(key) != value for key, value in expected.items()):
        raise ValueError("Self-hosted H3 requires 1344x768/24-fps references with 32-kHz stereo audio")
    result = {
        "h3_backend": BACKEND,
        "h3_endpoints": origins,
        "h3": deepcopy(SETTINGS),
        "reference_h3": {"settings": deepcopy(H3_SETTINGS), "host": reference_host, "endpoint": H3_ENDPOINT},
    }
    if reference_manifest is not None:
        result["reference_h3"] = {
            "backend": reference_manifest.get("h3_backend", "max"),
            "settings": deepcopy(reference_manifest["settings"]),
            "endpoint": reference_manifest["endpoint"],
        }
        if reference_manifest.get("h3_backend") == BACKEND:
            result["reference_h3"]["endpoints"] = list(reference_manifest["h3_endpoints"])
        else:
            result["reference_h3"]["host"] = reference_host
    token_file = token_file or os.environ.get("H3_SELF_HOSTED_TOKEN_FILE")
    if token_file:
        # Freeze only the filename, never its contents. Ray actors can then read
        # the shared file without relying on driver environment propagation.
        result["h3_token_file"] = str(Path(token_file).expanduser().resolve())
    return result


def request_body(prompt, settings):
    """Build the fixed-duration standalone rewrite-evaluation request."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError("Self-hosted H3 needs a nonempty prompt of at most 8000 characters")
    if settings["duration"] != 10:
        raise ValueError("The prompt-rewrite evaluator requires 10-second clips")
    return {"prompt": prompt, "duration": settings["duration"], "seed": settings["seed"]}


def identity(task, protocol):
    """Bind recovery to the exact request and worker pool, independent of routing."""
    return {
        "request_sha256": fingerprint(task["request"]),
        "backend": BACKEND,
        "endpoints": protocol["h3_endpoints"],
        "endpoint": ENDPOINT,
    }


class Client:
    """Use only the standard library; the caller's environment needs no GPU packages."""

    def __init__(self, token):
        if not token:
            raise ValueError("Missing self-hosted H3 bearer token")
        self.headers = {"Authorization": f"Bearer {token}"}

    @classmethod
    def from_environment(cls, protocol=None):
        """Keep the service credential separate from the paid H3-Max credential."""
        token = os.environ.get("H3_SELF_HOSTED_TOKEN", "").strip()
        token_file = os.environ.get("H3_SELF_HOSTED_TOKEN_FILE") or (protocol or {}).get("h3_token_file")
        if not token and token_file:
            token = Path(token_file).expanduser().read_text().strip()
        if not token:
            raise ValueError("Set H3_SELF_HOSTED_TOKEN_FILE or H3_SELF_HOSTED_TOKEN on every generation worker")
        return cls(token)

    def open(self, url, *, body=None, timeout=30):
        """Contact private worker origins directly, without environment HTTP proxies."""
        headers = dict(self.headers)
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, headers=headers)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        return opener.open(request, timeout=timeout)

    def get_json(self, url, *, timeout=30):
        """Read health or status from a known worker origin."""
        with self.open(url, timeout=timeout) as response:
            return json.load(response)

    def submit(self, origin, body, *, timeout=30):
        """Submit once; ambiguous failures must not trigger another POST."""
        with self.open(origin + ENDPOINT, body=body, timeout=timeout) as response:
            return json.load(response)

    def download(self, origin, job_id, output, *, timeout=30):
        """Authenticate the MP4 request and publish the local file atomically."""
        temporary = output.with_suffix(".mp4.part")
        try:
            with self.open(f"{origin}/v1/jobs/{job_id}/video", timeout=timeout) as source:
                with temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)


def available_workers(client, endpoints):
    """Prefer the shortest visible queue and randomize ties across RL processes."""

    def inspect(origin):
        try:
            health = client.get_json(origin + "/healthz", timeout=2)
            if health.get("status") != "ready" or health["queued"] >= health["queue_capacity"]:
                return None
            return origin, health["queued"] + int(health.get("active_job") is not None)
        except (OSError, ValueError, KeyError):
            return None

    with ThreadPoolExecutor(max_workers=min(16, len(endpoints))) as pool:
        ready = [result for result in pool.map(inspect, endpoints) if result is not None]
    random.shuffle(ready)
    return [origin for origin, _ in sorted(ready, key=lambda row: row[1])]


def server_status(protocol, *, timeout=5):
    """Read public health endpoints without credentials or submitting any jobs.

    Readiness does not attest to model identity. Compare only settings actually
    reported by the service, and explicitly list missing provenance fields.
    """
    require_backend(protocol.get("h3_backend"))
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("Status timeout must be positive and finite")
    expected = {key: value for key, value in protocol["h3"].items() if key not in {"duration", "seed"}}

    def inspect(origin):
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                response = opener.open(origin + "/healthz", timeout=timeout)
            except urllib.error.HTTPError as exc:
                if exc.code != 503:
                    raise
                response = exc  # Warming/unavailable workers still return useful health JSON.
            with response:
                health = json.load(response)
            if not isinstance(health, dict):
                raise ValueError("Expected a health JSON object")
            observed = {key: health[key] for key in expected if key in health}
            if "steps" not in observed and "transformer_forwards" in health:
                observed["steps"] = health["transformer_forwards"]
            missing = sorted(expected.keys() - observed.keys())
            mismatches = {
                key: {"expected": expected[key], "reported": value}
                for key, value in observed.items()
                if value != expected[key]
            }
            return {
                "endpoint": origin,
                "reachable": True,
                "health": health,
                "settings_check": {
                    "status": "mismatch" if mismatches else "unverified" if missing else "reported_match",
                    "missing": missing,
                    "mismatches": mismatches,
                },
            }
        except (OSError, ValueError) as exc:
            return {"endpoint": origin, "reachable": False, "error": str(exc)}

    endpoints = protocol["h3_endpoints"]
    with ThreadPoolExecutor(max_workers=min(16, max(1, len(endpoints)))) as pool:
        workers = list(pool.map(inspect, endpoints))
    return {
        "expected_settings": expected,
        "all_ready": bool(workers) and all(row.get("health", {}).get("status") == "ready" for row in workers),
        "endpoints": workers,
    }


def remux_container(path, timeout):
    """Copy encoded streams into a regular MP4 for the evaluator's box parser."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
        except ImportError as exc:
            raise RuntimeError("Install ffmpeg or imageio-ffmpeg in the H3 client environment") from exc
        executable = get_ffmpeg_exe()
    temporary = path.with_suffix(".remux.mp4")
    try:
        result = subprocess.run(
            [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                "-y",
                str(temporary),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        if result.returncode:
            raise ValueError(f"H3 MP4 container remux failed: {result.stderr[-1000:]}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def generate_one(task, protocol, client, probe, poll_interval=1, timeout=1800):
    """Persist the worker and job ID, resume polling, and verify the downloaded media."""
    from .h3 import TERMINAL_FAILURES, media_profile

    require_backend(protocol.get("h3_backend"))
    path, output = Path(task["state"]), Path(task["output"])
    expected = identity(task, protocol)
    state = read_json(path) if path.exists() else {**expected, "status": "pending"}
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Saved H3 task belongs to a different request or worker pool: {path}")

    def save(**updates):
        state.update(updates)
        write_json(path, state)

    if state["status"] == "succeeded" and output.exists():
        if file_sha256(output) != state.get("sha256"):
            raise ValueError(f"Completed H3 video changed: {output}")
        return True
    if state["status"] in TERMINAL_FAILURES:
        return False
    deadline, error_stage = time.monotonic() + timeout, "provider"

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("H3 polling/admission timed out; resume uses the saved worker and task ID")
        return min(30, seconds)

    try:
        if state.get("task_id"):
            if state.get("worker_url") not in protocol["h3_endpoints"]:
                raise ValueError("Saved job's worker is outside the frozen endpoint pool")
            if str(UUID(state["task_id"])) != state["task_id"]:
                raise ValueError("Invalid saved H3 job ID")
        while not state.get("task_id"):
            remaining()
            for origin in available_workers(client, protocol["h3_endpoints"]):
                save(status="submitting", worker_url=origin, request=task["request"])
                try:
                    response = client.submit(origin, task["request"], timeout=remaining())
                    job_id = str(UUID(response["id"]))
                except urllib.error.HTTPError as exc:
                    if exc.code in {429, 503}:
                        # These service responses explicitly reject admission.
                        save(status="pending", error=f"Worker rejected admission: HTTP {exc.code}")
                        continue
                    save(
                        status="rejected" if 400 <= exc.code < 500 else "submit_unknown", error=f"POST HTTP {exc.code}"
                    )
                    return False
                except Exception as exc:
                    save(status="submit_unknown", error=str(exc))
                    return False
                save(status="submitted", task_id=job_id, error=None)
                break
            if not state.get("task_id"):
                time.sleep(min(poll_interval, remaining()))
        origin, job_id = state["worker_url"], state["task_id"]
        while True:
            response = client.get_json(f"{origin}/v1/jobs/{job_id}", timeout=remaining())
            if response.get("id") != job_id:
                raise ValueError("H3 status returned a different job ID")
            status = response["status"]
            if status == "failed":
                save(status="failed", error=response.get("error"))
                return False
            if status == "succeeded":
                save(status="downloading")
                output.parent.mkdir(parents=True, exist_ok=True)
                client.download(origin, job_id, output, timeout=remaining())
                error_stage = "artifact"
                media = probe(output)
                remuxed = False
                # Sol-H3 writes fragmented MP4s. The evaluator's minimal box
                # parser reads only moov/stts, which are empty in that format.
                # Repackage the encoded streams; do not decode or re-encode.
                if media.get("duration_s") == 0 and (media.get("video") or {}).get("frames") == 0:
                    remux_container(output, remaining())
                    media, remuxed = probe(output), True
                if media_profile(media, duration=task["request"]["duration"]) != protocol["media_profile"]:
                    raise ValueError(f"H3 media format differs from the reference videos: {output}")
                save(
                    status="succeeded",
                    sha256=file_sha256(output),
                    media=media,
                    bytes=output.stat().st_size,
                    container_remuxed=remuxed,
                    error=None,
                )
                return True
            if status not in {"queued", "running"}:
                raise ValueError(f"Unknown self-hosted H3 job status: {status}")
            save(status=status)
            time.sleep(min(poll_interval, remaining()))
    except Exception as exc:
        save(status="error", error=str(exc), error_type=type(exc).__name__, error_stage=error_stage)
        return False


def generate_tasks(
    tasks, protocol, probe, concurrency=DEFAULT_CONCURRENCY, poll_interval=1, timeout=1800, *, client=None
):
    """Bound request concurrency; a single gateway origin may dispatch to many GPU workers."""
    require_backend(protocol.get("h3_backend"))
    if concurrency < 1 or any(not isfinite(value) or value <= 0 for value in (poll_interval, timeout)):
        raise ValueError("Concurrency, polling interval, and timeout must be positive and finite")
    client = client or Client.from_environment(protocol)

    def run(task):
        ok = generate_one(task, protocol, client, probe, poll_interval, timeout)
        print(f"H3 {task['id']}: {'succeeded' if ok else 'incomplete; inspect its state file'}", flush=True)
        return task["id"], ok

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return dict(pool.map(run, tasks))
