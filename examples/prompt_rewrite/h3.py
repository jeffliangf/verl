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
import subprocess
import sys
from copy import deepcopy
from math import isfinite
from pathlib import Path

from . import self_hosted_h3
from .config import H3_ENDPOINT, H3_SETTINGS
from .storage import fingerprint, freeze_json, load_external, read_json, write_json

TERMINAL_FAILURES = {"failed", "cancelled", "expired", "rejected", "submit_unknown", "submitting"}
PASS_TIMEOUT_SECONDS = 1800


def request_body(rewritten_prompt: str, settings: dict | None = None) -> dict:
    """Reconstruct historical Max requests for offline artifact validation only."""
    if not isinstance(rewritten_prompt, str) or not rewritten_prompt.strip():
        raise ValueError("H3 needs a nonempty rewritten prompt")
    body = deepcopy(H3_SETTINGS if settings is None else settings)
    return {**body, "content": [{"type": "text", "text": rewritten_prompt}]}


def request_for_protocol(prompt: str, protocol: dict) -> dict:
    """Build a standalone rewrite-evaluation request from frozen settings."""
    if protocol.get("h3_backend") == self_hosted_h3.BACKEND:
        return self_hosted_h3.request_body(prompt, protocol["h3"])
    return request_body(prompt, protocol["h3"])


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


def generate_one(*args, **kwargs):
    """Reject calls to the retired Max client, including direct Python callers."""
    raise ValueError(self_hosted_h3.DISABLED_API_MESSAGE)


def generate_tasks(*args, **kwargs):
    """Reject calls to the retired Max batch client without reading credentials."""
    raise ValueError(self_hosted_h3.DISABLED_API_MESSAGE)


def task_identity(task: dict, protocol: dict) -> dict:
    """Bind a generation decision to the exact submitted request and provider."""
    if protocol.get("h3_backend") == self_hosted_h3.BACKEND:
        return self_hosted_h3.identity(task, protocol)
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
    self_hosted_h3.require_backend(protocol.get("h3_backend"))
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
    parser.add_argument("--concurrency", type=int, default=self_hosted_h3.DEFAULT_CONCURRENCY)
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=PASS_TIMEOUT_SECONDS)
    args = parser.parse_args()
    protocol = read_json(args.run_dir / "protocol.json")
    self_hosted_h3.require_backend(protocol.get("h3_backend"))
    tasks = read_json(args.run_dir / "generation.json")
    fallbacks = read_fallbacks(args.run_dir, protocol, tasks)
    repo = Path(protocol["eval_repo"])
    results = self_hosted_h3.generate_tasks(
        [task for task in tasks if task["id"] not in fallbacks],
        protocol,
        load_external(repo / "bestofn/normalize.py").probe,
        args.concurrency,
        args.poll_interval,
        args.timeout,
    )
    return int(not all(results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
