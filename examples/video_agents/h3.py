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
"""Sol-H3 client and generation worker: durable jobs, downloads, and media checks."""

import argparse
import errno
import fcntl
import http.client
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from email.utils import parsedate_to_datetime
from math import isfinite
from pathlib import Path
from uuid import UUID, uuid4

from .observability import record_event, timed_stage
from .storage import file_sha256, fingerprint, load_external, read_json, write_json

BACKEND = "self-hosted"
ENDPOINT = "/v1/videos"
DEFAULT_CONCURRENCY = 8
TERMINAL_FAILURES = {"failed", "cancelled", "expired", "rejected", "submit_unknown", "submitting"}
RECOVERABLE_FAILURES = {"worker_restarted", "inference_failed", "storage_failure", "worker_lost"}
RETRYABLE_HTTP = {408, 429, 500, 502, 503, 504, 507}
RECOVERY_DEFAULTS = {"h3_max_request_attempts": 3, "h3_max_generation_attempts": 2, "h3_max_inflight": 2}
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
    """Expose the self-hosted service endpoints and credential file."""
    parser.add_argument(
        "--h3-endpoints", help="Sol-H3 gateway URL or comma-separated worker origins; or set H3_SELF_HOSTED_ENDPOINTS"
    )
    parser.add_argument(
        "--h3-token-file", type=Path, help="Bearer-token file readable at this path by every Ray worker"
    )


def configuration(endpoints, profile, reference_manifest, token_file=None):
    """Freeze the self-hosted generator and the references' recorded provenance."""
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
        "reference_h3": {
            "backend": BACKEND,
            "settings": deepcopy(reference_manifest["settings"]),
            "endpoint": reference_manifest["endpoint"],
            "endpoints": list(reference_manifest["h3_endpoints"]),
        },
    }
    token_file = token_file or os.environ.get("H3_SELF_HOSTED_TOKEN_FILE")
    if token_file:
        # Freeze only the filename, never its contents. Ray actors can then read
        # the shared file without relying on driver environment propagation.
        result["h3_token_file"] = str(Path(token_file).expanduser().resolve())
    return result


def request_body(prompt, settings):
    """Reconstruct the ten-second requests used to generate reference videos."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError("Self-hosted H3 needs a nonempty prompt of at most 8000 characters")
    if settings["duration"] != 10:
        raise ValueError("Reference videos must be 10-second clips")
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

    def open(self, url, *, body=None, timeout=30, idempotency_key=None):
        """Contact private worker origins directly, without environment HTTP proxies."""
        headers = dict(self.headers)
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
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

    def submit(self, origin, body, *, timeout=30, idempotency_key=None):
        """Reuse a persisted caller ID when the server advertises support."""
        with self.open(origin + ENDPOINT, body=body, timeout=timeout, idempotency_key=idempotency_key) as response:
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


def available_workers(client, endpoints, *, timeout=2):
    """Prefer the shortest visible queue and randomize ties across RL processes."""

    def inspect(origin):
        try:
            health = client.get_json(origin + "/healthz", timeout=timeout)
            if health.get("status") != "ready" or health["queued"] >= health["queue_capacity"]:
                return None
            return origin, health["queued"] + int(health.get("active_job") is not None), health
        except (OSError, ValueError, KeyError):
            return None

    with ThreadPoolExecutor(max_workers=min(16, len(endpoints))) as pool:
        ready = [result for result in pool.map(inspect, endpoints) if result is not None]
    random.shuffle(ready)
    return [(origin, health) for origin, _, health in sorted(ready, key=lambda row: row[1])]


def server_status(protocol, *, timeout=5):
    """Read public health endpoints without credentials or submitting any jobs.

    Readiness does not attest to model identity. Compare only settings actually
    reported by the service, and explicitly list missing provenance fields.
    """
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


def media_profile(media: dict, *, duration: int = 10) -> dict:
    """Compare the native visual/audio formats without transcoding either side."""
    video, audio = media.get("video") or {}, media.get("audio") or {}
    if not video or not audio or not duration - 1 <= media.get("duration_s", 0) <= duration + 1:
        raise ValueError(f"Expected an approximately {duration}-second video with audio")
    return {
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": video.get("fps"),
        "video_codec": video.get("codec"),
        "audio_codec": audio.get("codec"),
        "audio_sample_rate": audio.get("sample_rate"),
        "audio_channels": audio.get("channels"),
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


class DownloadIntegrityError(ValueError):
    """A transfer did not match the gateway's archived artifact."""


def recovery_settings(settings):
    """Validate launch-time recovery budgets independently of frozen inputs."""
    result = {key: settings.get(key, default) for key, default in RECOVERY_DEFAULTS.items()}
    if any(type(value) is not int or value < 1 for value in result.values()):
        raise ValueError("H3 attempt and concurrency limits must be positive integers")
    return result


def retryable(error):
    """Retry transport failures without treating local filesystem errors as network errors."""
    if isinstance(error, DownloadIntegrityError):
        return True
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRYABLE_HTTP
    if isinstance(error, urllib.error.URLError):
        return retryable(error.reason)
    if isinstance(error, http.client.HTTPException):
        return True
    return isinstance(error, OSError) and error.errno in {
        None,
        errno.ETIMEDOUT,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.EPIPE,
        errno.EHOSTUNREACH,
        errno.ENETUNREACH,
        errno.ENETDOWN,
    }


def retry_delay(error, attempt):
    """Honor either Retry-After format, with exponential backoff and jitter."""
    delay = 2 ** (attempt - 1) + random.uniform(0, 0.25)
    if isinstance(error, urllib.error.HTTPError):
        value = (error.headers or {}).get("Retry-After")
        if value:
            try:
                seconds = float(value)
            except ValueError:
                try:
                    seconds = parsedate_to_datetime(value).timestamp() - time.time()
                except (ValueError, TypeError, OverflowError):
                    seconds = 0
            if isfinite(seconds):
                delay = max(delay, seconds)
    return delay


@contextmanager
def generation_slot(run_dir, limit, timings):
    """Bound H3 lifecycles across processes sharing a run directory.

    Advisory locks release on process exit. Waiting happens before the generation
    subprocess starts, so it does not consume that candidate's in-flight budget.
    """
    slots = Path(run_dir) / ".h3-slots"
    slots.mkdir(parents=True, exist_ok=True)
    acquired = None
    try:
        with timed_stage(timings, "h3_global_slot_wait", limit=limit):
            while acquired is None:
                indices = list(range(limit))
                random.shuffle(indices)
                for index in indices:
                    handle = (slots / str(index)).open("a+")
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                    except BaseException:
                        handle.close()
                        raise
                    else:
                        acquired = handle
                        break
                if acquired is None:
                    time.sleep(0.1)
        yield
    finally:
        if acquired is not None:
            acquired.close()


class _Generation:
    """Recover one candidate without changing its request or resetting its deadline."""

    def __init__(self, task, protocol, client, probe, poll_interval, timeout):
        self.task, self.protocol, self.client, self.probe = task, protocol, client, probe
        self.path, self.output = Path(task["state"]), Path(task["output"])
        self.timings = self.path.parent / "timings.jsonl"
        expected = identity(task, protocol)
        self.state = read_json(self.path) if self.path.exists() else {**expected, "status": "pending"}
        if any(self.state.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Saved H3 task belongs to a different request or worker pool: {self.path}")
        self.limits = recovery_settings(protocol)
        self.poll_interval, self.timeout = poll_interval, timeout
        self.request_timeout = protocol.get("timeouts", {}).get("h3_request", 30)
        if any(not isfinite(value) or value <= 0 for value in (poll_interval, timeout, self.request_timeout)):
            raise ValueError("H3 polling and timeouts must be positive and finite")
        self.error_stage = "provider"

    def save(self, **updates):
        previous = self.state.get("status")
        self.state.update(updates)
        write_json(self.path, self.state)
        if "status" in updates and updates["status"] != previous:
            record_event(
                self.timings,
                "h3",
                "status",
                status=self.state["status"],
                job_id=self.state.get("task_id"),
                idempotency_key=self.state.get("idempotency_key"),
                generation_attempt=self.state.get("generation_attempt", 1),
            )

    def remaining(self):
        seconds = self.deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("H3 in-flight deadline exhausted; the saved job is not automatically regenerated")
        return min(self.request_timeout, seconds)

    def pause(self, seconds):
        if seconds >= self.deadline - time.monotonic():
            raise TimeoutError("H3 retry delay exceeds the remaining in-flight deadline")
        time.sleep(seconds)
        self.remaining()

    def request(self, stage, operation, *, submission=False):
        """Share a persisted retry budget across network/HTTP errors and client restarts."""
        counts = self.state.setdefault("request_attempts", {})
        while counts.get(stage, 0) < self.limits["h3_max_request_attempts"]:
            limit = self.remaining()
            attempt = counts.get(stage, 0) + 1
            counts[stage] = attempt
            retries = self.state.get("transport_retries", 0) + int(attempt > 1)
            self.save(request_attempts=counts, transport_retries=retries)
            try:
                with timed_stage(
                    self.timings,
                    stage,
                    timeout_seconds=limit,
                    request_attempt=attempt,
                    generation_attempt=self.state.get("generation_attempt", 1),
                    job_id=self.state.get("task_id"),
                    idempotency_key=self.state.get("idempotency_key"),
                ) as timing:
                    response = operation(limit)
                    if isinstance(response, dict):
                        timing["job_status"] = response.get("status")
                        timing["response_job_id"] = response.get("id")
                self.remaining()
            except Exception as error:
                if isinstance(error, urllib.error.HTTPError) and error.code in {400, 401, 403, 409, 422}:
                    self.error_stage = "configuration"
                elif isinstance(error, DownloadIntegrityError):
                    self.error_stage = "artifact"
                elif not isinstance(error, OSError | http.client.HTTPException):
                    self.error_stage = "protocol"
                elif not retryable(error):
                    self.error_stage = "artifact" if stage == "h3_download" else "provider"
                safe = (
                    not submission
                    or self.state.get("idempotency_key")
                    or (isinstance(error, urllib.error.HTTPError) and error.code in {429, 503, 507})
                )
                if not safe or not retryable(error) or attempt >= self.limits["h3_max_request_attempts"]:
                    raise
                delay = retry_delay(error, attempt)
                record_event(
                    self.timings,
                    "h3_recovery",
                    "retry",
                    operation=stage,
                    next_attempt=attempt + 1,
                    job_id=self.state.get("task_id"),
                    idempotency_key=self.state.get("idempotency_key"),
                    error_type=type(error).__name__,
                    http_status=getattr(error, "code", None),
                    backoff_seconds=delay,
                )
                self.pause(delay)
            else:
                counts.pop(stage, None)
                self.save(request_attempts=counts)
                self.error_stage = "provider"
                return response
        raise OSError(f"H3 {stage} request attempt budget exhausted")

    def submit_once(self, timeout):
        key = self.state.get("idempotency_key")
        try:
            response = self.client.submit(
                self.state["worker_url"],
                self.task["request"],
                timeout=timeout,
                idempotency_key=key,
            )
        except urllib.error.HTTPError as error:
            # The gateway can acknowledge admission in a 502 when route persistence fails.
            accepted = None
            if error.code == 502:
                try:
                    detail = json.loads(error.read(32768)).get("detail")
                    if isinstance(detail, dict):
                        accepted = detail.get("id")
                except (ValueError, OSError, AttributeError):
                    pass
            if accepted is None:
                if not key and error.code in {429, 503, 507}:
                    self.save(status="pending")
                raise
            response = {"id": accepted}
        identifier = response["id"]
        if str(UUID(identifier)) != identifier or (key and identifier != key):
            raise ValueError("H3 submission returned a different or invalid job ID")
        # Persist acceptance before the request counter is reset.
        self.save(status="submitted", task_id=identifier, error=None)
        return response

    def submit(self):
        if not self.state.get("worker_url"):
            while True:
                with timed_stage(self.timings, "h3_health"):
                    workers = available_workers(
                        self.client,
                        self.protocol["h3_endpoints"],
                        timeout=min(2, self.remaining()),
                    )
                if workers:
                    origin, health = workers[0]
                    keyed = health.get("idempotency_key") is True
                    self.save(
                        status="submitting",
                        worker_url=origin,
                        request=self.task["request"],
                        idempotency_key=str(uuid4()) if keyed else None,
                        checksums_required=keyed and health.get("service") == "Sol-H3 gateway",
                    )
                    break
                self.pause(min(self.poll_interval, self.remaining()))
        # Saved attempts bypass readiness/capacity checks: replay and lookup can
        # still recover existing work when the gateway admits no new jobs.
        self.request("h3_submission", self.submit_once, submission=True)

    def regenerate(self):
        attempt = self.state.get("generation_attempt", 1)
        if (
            not self.state.get("idempotency_key")
            or self.state.get("error_code") not in RECOVERABLE_FAILURES
            or attempt >= self.limits["h3_max_generation_attempts"]
        ):
            return False
        self.remaining()
        history = self.state.get("previous_attempts", []) + [
            {key: self.state.get(key) for key in ("task_id", "idempotency_key", "worker_url", "error_code", "error")}
        ]
        record_event(
            self.timings,
            "h3_recovery",
            "regeneration",
            previous_job_id=self.state.get("task_id"),
            error_code=self.state["error_code"],
            generation_attempt=attempt + 1,
        )
        self.save(
            status="pending",
            previous_attempts=history,
            generation_attempt=attempt + 1,
            task_id=None,
            idempotency_key=None,
            worker_url=None,
            request_attempts={},
            error=None,
            error_code=None,
        )
        return True

    def poll_once(self, timeout):
        identifier = self.state["task_id"]
        response = self.client.get_json(f"{self.state['worker_url']}/v1/jobs/{identifier}", timeout=timeout)
        if (
            not isinstance(response, dict)
            or response.get("id") != identifier
            or response.get("status") not in {"queued", "running", "failed", "succeeded"}
        ):
            raise ValueError("H3 status returned a different job ID or an invalid job state")
        return response

    def download(self, response):
        expected_hash, expected_bytes = response.get("video_sha256"), response.get("video_bytes")
        verify = self.state.get("checksums_required") or expected_hash is not None or expected_bytes is not None
        if verify and (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(c not in "0123456789abcdef" for c in expected_hash)
            or type(expected_bytes) is not int
            or expected_bytes <= 0
        ):
            self.error_stage = "protocol"
            raise ValueError("H3 succeeded without valid video_sha256 and video_bytes")
        self.save(status="downloading")
        self.output.parent.mkdir(parents=True, exist_ok=True)

        def fetch(timeout):
            self.client.download(self.state["worker_url"], self.state["task_id"], self.output, timeout=timeout)
            digest, size = file_sha256(self.output), self.output.stat().st_size
            if verify and (digest != expected_hash or size != expected_bytes):
                self.output.unlink(missing_ok=True)
                raise DownloadIntegrityError("H3 download differs from the gateway checksum or size")
            self.save(download_sha256=digest, download_bytes=size)

        self.request("h3_download", fetch)
        self.error_stage = "artifact"
        with timed_stage(self.timings, "h3_media_validation", job_id=self.state["task_id"]):
            media = self.probe(self.output)
            remuxed = False
            if media.get("duration_s") == 0 and (media.get("video") or {}).get("frames") == 0:
                remux_container(self.output, self.remaining())
                media, remuxed = self.probe(self.output), True
            if media_profile(media, duration=self.task["request"]["duration"]) != self.protocol["media_profile"]:
                raise ValueError(f"H3 media format differs from the reference videos: {self.output}")
        provider_timings = {
            key: response[key]
            for key in ("created_at", "started_at", "finished_at", "pipeline_s", "processing_s")
            if key in response
        }
        if all(key in provider_timings for key in ("created_at", "started_at", "finished_at")):
            provider_timings.update(
                queued_seconds=response["started_at"] - response["created_at"],
                running_seconds=response["finished_at"] - response["started_at"],
                total_seconds=response["finished_at"] - response["created_at"],
            )
        self.remaining()
        record_event(self.timings, "h3_provider", "completed", job_id=self.state["task_id"], **provider_timings)
        self.save(
            status="succeeded",
            sha256=file_sha256(self.output),
            bytes=self.output.stat().st_size,
            media=media,
            container_remuxed=remuxed,
            provider_timings=provider_timings,
            error=None,
        )
        record_event(
            self.timings,
            "h3_recovery",
            "completed",
            job_id=self.state["task_id"],
            generation_attempts=self.state.get("generation_attempt", 1),
            transport_retries=self.state.get("transport_retries", 0),
        )
        return True

    def run(self):
        if self.state["status"] == "succeeded" and self.output.exists():
            if file_sha256(self.output) != self.state.get("sha256"):
                raise ValueError(f"Completed H3 video changed: {self.output}")
            return True
        if self.state.get("error_stage") in {"artifact", "configuration", "protocol"}:
            return False
        if self.state["status"] in {"cancelled", "expired", "rejected"}:
            return False
        if self.state["status"] in {"submitting", "submit_unknown"} and not self.state.get("idempotency_key"):
            return False
        for name in ("task_id", "idempotency_key"):
            value = self.state.get(name)
            if value is not None and str(UUID(value)) != value:
                raise ValueError(f"Invalid saved H3 {name}")
        if self.state.get("idempotency_key") and self.state.get("task_id") not in {
            None,
            self.state["idempotency_key"],
        }:
            raise ValueError("Saved H3 job ID differs from its idempotency key")
        if self.state.get("worker_url") and self.state["worker_url"] not in self.protocol["h3_endpoints"]:
            raise ValueError("Saved job's worker is outside the frozen endpoint pool")
        expires = self.state.get("deadline_at", time.time() + self.timeout)
        self.deadline = time.monotonic() + min(self.timeout, expires - time.time())
        # A resumed reader starts a new bounded GET operation. Submission counts
        # survive restarts so replaying an unknown admission cannot multiply retries.
        counts = self.state.setdefault("request_attempts", {})
        counts.pop("h3_poll", None)
        counts.pop("h3_download", None)
        self.save(deadline_at=expires, generation_attempt=self.state.get("generation_attempt", 1))
        try:
            while True:
                self.remaining()
                if self.state["status"] == "failed" and not self.regenerate():
                    return False
                if not self.state.get("task_id"):
                    self.submit()
                response = self.request("h3_poll", self.poll_once)
                status = response["status"]
                if status == "failed":
                    self.save(status="failed", error=response.get("error"), error_code=response.get("error_code"))
                elif status == "succeeded":
                    return self.download(response)
                else:
                    self.save(status=status)
                    self.pause(min(self.poll_interval, self.remaining()))
        except Exception as error:
            if isinstance(error, TimeoutError):
                self.error_stage = "provider"
            status = "error"
            if not self.state.get("task_id") and self.state.get("worker_url"):
                status = "submit_unknown"
            self.save(status=status, error=str(error), error_type=type(error).__name__, error_stage=self.error_stage)
            return False


def generate_one(task, protocol, client, probe, poll_interval=1, timeout=1800):
    """Recover one video using durable attempt IDs and one total in-flight deadline."""
    return _Generation(task, protocol, client, probe, poll_interval, timeout).run()


def generate_tasks(
    tasks, protocol, probe, concurrency=DEFAULT_CONCURRENCY, poll_interval=1, timeout=1800, *, client=None
):
    """Bound request concurrency; a single gateway origin may dispatch to many GPU workers."""
    if concurrency < 1 or any(not isfinite(value) or value <= 0 for value in (poll_interval, timeout)):
        raise ValueError("Concurrency, polling interval, and timeout must be positive and finite")
    client = client or Client.from_environment(protocol)

    def run(task):
        ok = generate_one(task, protocol, client, probe, poll_interval, timeout)
        print(f"H3 {task['id']}: {'succeeded' if ok else 'incomplete; inspect its state file'}", flush=True)
        return task["id"], ok

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return dict(pool.map(run, tasks))


def worker_command(
    run_dir: Path, poll_interval: float, timeout: float, *, max_request_attempts=3, max_generation_attempts=2
) -> list[str]:
    """Run provider threads in a process that can be stopped at a hard deadline."""
    return [
        sys.executable,
        "-m",
        "examples.video_agents.h3",
        "--run-dir",
        str(run_dir),
        "--poll-interval",
        str(poll_interval),
        "--timeout",
        str(timeout),
        "--max-request-attempts",
        str(max_request_attempts),
        "--max-generation-attempts",
        str(max_generation_attempts),
    ]


def main():
    """Internal generation worker; the parent holds the run lock and hard deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-interval", type=float, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--max-request-attempts", type=int, default=3)
    parser.add_argument("--max-generation-attempts", type=int, default=2)
    args = parser.parse_args()
    protocol = read_json(args.run_dir / "protocol.json")
    protocol.update(
        h3_max_request_attempts=args.max_request_attempts, h3_max_generation_attempts=args.max_generation_attempts
    )
    tasks = read_json(args.run_dir / "generation.json")
    probe = load_external(Path(protocol["eval_repo"]) / "bestofn/normalize.py").probe
    results = generate_tasks(
        tasks, protocol, probe, min(DEFAULT_CONCURRENCY, len(tasks)), args.poll_interval, args.timeout
    )
    return int(not all(results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
