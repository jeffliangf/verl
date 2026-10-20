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

import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

from .config import H3_CONCURRENCY, H3_ENDPOINT, H3_SETTINGS
from .storage import file_sha256, fingerprint, read_json, write_json

TERMINAL_FAILURES = {"failed", "cancelled", "expired", "rejected", "submit_unknown", "submitting"}


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
    task: dict, client, host: str, headers: dict, probe, profile: dict, poll_interval: float = 20, timeout: float = 7200
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
            if status in {"failed", "cancelled", "expired"}:
                save(status=status, error=result.get("error"))
                return False
            if status in {"succeeded", "success", "completed"}:
                url = (result.get("content") or {}).get("url")
                if not url:
                    raise RuntimeError("Completed H3 task has no content.url")
                save(status="downloading")
                client.download_video(url, output)
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
        save(status="error", error=str(exc))
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
) -> dict[str, bool]:
    """Generate a cohort with bounded concurrency; return success by dataset ID."""
    if concurrency < 1 or poll_interval <= 0:
        raise ValueError("Concurrency and polling interval must be positive")

    def run(task):
        ok = generate_one(task, client, host, headers, probe, profile, poll_interval)
        print(f"H3 {task['id']}: {'succeeded' if ok else 'incomplete; inspect its state file'}", flush=True)
        return task["id"], ok

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return dict(pool.map(run, tasks))
