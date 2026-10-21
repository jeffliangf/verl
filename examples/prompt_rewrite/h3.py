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
"""Direct H3 generation with persistent task IDs and no Context-IR call."""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from math import isfinite
from pathlib import Path

from .config import H3_CONCURRENCY, H3_ENDPOINT, H3_SETTINGS
from .storage import file_sha256, fingerprint, freeze_json, load_external, read_json, write_json

TERMINAL_FAILURES = {"failed", "cancelled", "expired", "rejected", "submit_unknown", "submitting"}
PASS_TIMEOUT_SECONDS = 1800


def request_body(rewritten_prompt: str, settings: dict | None = None) -> dict:
    """Use current defaults for new runs and frozen settings when resuming."""
    if not isinstance(rewritten_prompt, str) or not rewritten_prompt.strip():
        raise ValueError("H3 needs a nonempty rewritten prompt")
    # Send this exact text to generation; Max defaults explicitly disable expansion.
    body = deepcopy(H3_SETTINGS if settings is None else settings)
    return {**body, "content": [{"type": "text", "text": rewritten_prompt}]}


def media_profile(media: dict) -> dict:
    """Compare the native visual/audio formats without transcoding either side."""
    video, audio = media.get("video") or {}, media.get("audio") or {}
    if not video or not audio or not 9 <= media.get("duration_s", 0) <= 11:
        raise ValueError("Expected an approximately 10-second video with audio")
    return {
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": video.get("fps"),
        "video_codec": video.get("codec"),
        "audio_codec": audio.get("codec"),
        "audio_sample_rate": audio.get("sample_rate"),
        "audio_channels": audio.get("channels"),
    }


def generate_one(
    task: dict,
    client,
    host: str,
    headers: dict,
    probe,
    profile: dict,
    poll_interval: float = 20,
    timeout: float = PASS_TIMEOUT_SECONDS,
) -> bool:
    """Resume an accepted task; never automatically retry an uncertain submission.

    ``client`` supplies the existing eval repo's HTTP/download helpers. Passing
    a fake client makes crash recovery testable without any API requests.
    """
    state_path, output = Path(task["state"]), Path(task["output"])
    identity = {"request_sha256": fingerprint(task["request"]), "host": host, "endpoint": H3_ENDPOINT}
    state = read_json(state_path) if state_path.exists() else {**identity, "status": "pending"}
    if any(state.get(key) != value for key, value in identity.items()):
        raise ValueError(f"Saved H3 task belongs to a different request or host: {state_path}")

    def save(**updates):
        state.update(updates)
        write_json(state_path, state)

    if state["status"] == "succeeded" and output.exists():
        if file_sha256(output) != state.get("sha256"):
            raise ValueError(f"Completed H3 video changed: {output}")
        return True
    if state["status"] in TERMINAL_FAILURES:
        return False
    error_stage = "provider"
    try:
        if not state.get("task_id"):
            # Persist before POST: process death must not cause a duplicate billable job.
            save(status="submitting", request=task["request"])
            try:
                response = client.post_json(host + H3_ENDPOINT, headers, task["request"])
                if not response.get("task_id"):
                    raise RuntimeError("H3 submission returned no task_id")
            except Exception as exc:
                status = "rejected" if str(exc).startswith("POST HTTP ") else "submit_unknown"
                save(status=status, error=str(exc))
                return False
            save(status="submitted", task_id=str(response["task_id"]))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            response = client.get_json(f"{host}/v2/query/video_generation/{state['task_id']}", headers)
            base = response.get("base_resp") or {}
            if base.get("status_code") == 502:
                time.sleep(poll_interval)
                continue
            if base.get("status_code") not in (None, 0):
                raise RuntimeError(f"H3 query failed: {base}")
            result = response.get("task") or response
            status = str(result.get("status", "unknown")).lower()
            if status in {"failed", "cancelled", "expired", "rejected"}:
                save(status=status, error=result.get("error"))
                return False
            if status in {"succeeded", "success", "completed"}:
                url = (result.get("content") or {}).get("url")
                if not url:
                    raise RuntimeError("Completed H3 task has no content.url")
                save(status="downloading")
                client.download_video(url, output)
                error_stage = "artifact"
                media = probe(output)
                if media_profile(media) != profile:
                    raise ValueError(f"H3 media format differs from the reference videos: {output}")
                save(
                    status="succeeded", sha256=file_sha256(output), media=media, bytes=output.stat().st_size, error=None
                )
                return True
            save(status=status)
            time.sleep(poll_interval)
        raise TimeoutError("H3 polling timed out; resume will query the saved task ID")
    except Exception as exc:
        save(status="error", error=str(exc), error_type=type(exc).__name__, error_stage=error_stage)
        return False


def generate_tasks(
    tasks: list[dict],
    client,
    host: str,
    headers: dict,
    probe,
    profile: dict,
    concurrency: int = H3_CONCURRENCY,
    poll_interval: float = 20,
    timeout: float = PASS_TIMEOUT_SECONDS,
) -> dict[str, bool]:
    """Generate a cohort with bounded concurrency; return success by dataset ID."""
    if concurrency < 1 or any(not isfinite(v) or v <= 0 for v in (poll_interval, timeout)):
        raise ValueError("Concurrency, polling interval, and timeout must be positive and finite")

    def run(task):
        ok = generate_one(task, client, host, headers, probe, profile, poll_interval, timeout)
        print(f"H3 {task['id']}: {'succeeded' if ok else 'incomplete; inspect its state file'}", flush=True)
        return task["id"], ok

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return dict(pool.map(run, tasks))


def task_identity(task: dict, protocol: dict) -> dict:
    """Bind a generation decision to the exact submitted request and provider."""
    return {"request_sha256": fingerprint(task["request"]), "host": protocol["h3_host"], "endpoint": H3_ENDPOINT}


def read_fallbacks(run_dir: Path, protocol: dict, tasks: list[dict] | None = None) -> dict:
    """Validate explicit neutral decisions without treating missing videos as successes."""
    path = run_dir / "videos/fallbacks.json"
    if not path.exists():
        return {}
    policy = read_json(run_dir / "videos/failure-policy.json")
    if policy.get("protocol_sha256") != fingerprint(protocol) or policy.get("comparison_score_on_failure") != 0.5:
        raise ValueError("H3 fallback policy does not match this evaluation")
    tasks = read_json(run_dir / "generation.json") if tasks is None else tasks
    by_id = {task["id"]: task for task in tasks}
    records = read_json(path)
    for sample_id, row in records.items():
        task = by_id.get(sample_id)
        if task is None or row.get("id") != sample_id or row.get("score") != 0.5 or row.get("stage") != "h3":
            raise ValueError(f"Invalid H3 fallback: {sample_id}")
        if any(row.get(key) != value for key, value in task_identity(task, protocol).items()):
            raise ValueError(f"H3 fallback belongs to a different request: {sample_id}")
        state_path = Path(task["state"])
        state = read_json(state_path) if state_path.exists() else None
        if row.get("state_sha256") != fingerprint(state):
            raise ValueError(f"H3 state changed after the neutral decision: {sample_id}")
    return records


def record_fallbacks(run_dir: Path, protocol: dict, tasks: list[dict], *, timed_out: bool = False) -> dict:
    """Preserve failed task states and record their neutral scores in a sidecar."""
    records = read_fallbacks(run_dir, protocol, tasks)
    for task in tasks:
        if task["id"] in records:
            continue
        path = Path(task["state"])
        state = read_json(path) if path.exists() else None
        identity = task_identity(task, protocol)
        if state is not None and any(state.get(key) != value for key, value in identity.items()):
            raise ValueError(f"Saved H3 task belongs to a different request: {task['id']}")
        status = (state or {}).get("status", "pending")
        if status == "succeeded":
            continue
        if status == "error" and (state or {}).get("error_stage") == "artifact":
            raise ValueError(f"Invalid H3 video artifact: {task['id']}: {state.get('error')}")
        provider_error = status == "error" and (
            state.get("error_stage") == "provider"
            or state.get("error_type") == "TimeoutError"
            or str(state.get("error", "")).startswith("H3 polling timed out")
        )
        if status not in TERMINAL_FAILURES and not provider_error and not timed_out:
            continue
        records[task["id"]] = {
            "id": task["id"],
            "stage": "h3",
            "score": 0.5,
            **identity,
            "state_sha256": fingerprint(state),
            "task_id": (state or {}).get("task_id"),
            "status": status,
            "reason": (state or {}).get("error")
            or ("H3 generation exceeded its hard timeout" if timed_out else f"H3 generation ended with {status}"),
        }
    if records:
        write_json(run_dir / "videos/fallbacks.json", records)
    return records


def worker_command(run_dir: Path, concurrency: int, poll_interval: float, timeout: float) -> list[str]:
    """Run provider threads in a process that can be stopped at a hard deadline."""
    return [
        sys.executable,
        "-m",
        "examples.prompt_rewrite.h3",
        "--run-dir",
        str(run_dir),
        "--concurrency",
        str(concurrency),
        "--poll-interval",
        str(poll_interval),
        "--timeout",
        str(timeout),
    ]


def run_with_fallbacks(
    run_dir: Path,
    protocol: dict,
    tasks: list[dict],
    concurrency: int,
    poll_interval: float,
    timeout: float = PASS_TIMEOUT_SECONDS,
):
    """Bound the entire generation pass, including submission, polling, and downloads."""
    if concurrency < 1 or any(not isfinite(v) or v <= 0 for v in (poll_interval, timeout)):
        raise ValueError("H3 concurrency, polling interval, and timeout must be positive and finite")
    freeze_json(
        run_dir / "videos/failure-policy.json",
        {
            "protocol_sha256": fingerprint(protocol),
            "pass_timeout_seconds": timeout,
            "comparison_score_on_failure": 0.5,
        },
    )
    records = record_fallbacks(run_dir, protocol, tasks)

    def pending():
        return [
            task
            for task in tasks
            if task["id"] not in records
            and (
                not Path(task["state"]).exists()
                or read_json(Path(task["state"])).get("status") != "succeeded"
                or not Path(task["output"]).exists()
            )
        ]

    if pending():
        timed_out = False
        try:
            # The worker uses threads, so terminating it also stops hung API calls.
            subprocess.run(worker_command(run_dir, concurrency, poll_interval, timeout), check=False, timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
        records = record_fallbacks(run_dir, protocol, tasks, timed_out=timed_out)
    if pending():
        raise RuntimeError(
            "H3 generation incomplete without a recorded provider failure or timeout; inspect worker logs"
        )


def main():
    """Internal generation worker; the parent holds the evaluation run lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=H3_CONCURRENCY)
    parser.add_argument("--poll-interval", type=float, default=20)
    parser.add_argument("--timeout", type=float, default=PASS_TIMEOUT_SECONDS)
    args = parser.parse_args()
    protocol = read_json(args.run_dir / "protocol.json")
    tasks = read_json(args.run_dir / "generation.json")
    fallbacks = read_fallbacks(args.run_dir, protocol, tasks)
    repo = Path(protocol["eval_repo"])
    client = load_external(repo / "scripts/generate_h3_repeats.py")
    client.load_env()
    host, token = os.environ.get("H3_HOST", "").rstrip("/"), os.environ.get("H3_TOKEN", "")
    if not token or host != protocol["h3_host"]:
        raise ValueError("H3 credentials are missing or the host differs from the frozen evaluation")
    results = generate_tasks(
        [task for task in tasks if task["id"] not in fallbacks],
        client,
        host,
        client.auth_headers(token),
        load_external(repo / "bestofn/normalize.py").probe,
        protocol["media_profile"],
        args.concurrency,
        args.poll_interval,
        args.timeout,
    )
    return int(not all(results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
