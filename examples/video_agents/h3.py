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
"""Sol-H3 subprocess worker and shared media checks."""

import argparse
import sys
from pathlib import Path

from . import self_hosted_h3
from .storage import load_external, read_json

TERMINAL_FAILURES = {"failed", "cancelled", "expired", "rejected", "submit_unknown", "submitting"}
PASS_TIMEOUT_SECONDS = 1800


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


def worker_command(run_dir: Path, concurrency: int, poll_interval: float, timeout: float) -> list[str]:
    """Run provider threads in a process that can be stopped at a hard deadline."""
    return [
        sys.executable,
        "-m",
        "examples.video_agents.h3",
        "--run-dir",
        str(run_dir),
        "--concurrency",
        str(concurrency),
        "--poll-interval",
        str(poll_interval),
        "--timeout",
        str(timeout),
    ]


def main():
    """Internal generation worker; the parent holds the run lock and hard deadline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=self_hosted_h3.DEFAULT_CONCURRENCY)
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument("--timeout", type=float, default=PASS_TIMEOUT_SECONDS)
    args = parser.parse_args()
    protocol = read_json(args.run_dir / "protocol.json")
    self_hosted_h3.require_backend(protocol.get("h3_backend"))
    tasks = read_json(args.run_dir / "generation.json")
    repo = Path(protocol["eval_repo"])
    results = self_hosted_h3.generate_tasks(
        tasks,
        protocol,
        load_external(repo / "bestofn/normalize.py").probe,
        args.concurrency,
        args.poll_interval,
        args.timeout,
    )
    return int(not all(results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
