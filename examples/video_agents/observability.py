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
"""Durable stage timings without prompts, credentials, or provider response bodies."""

import fcntl
import json
import os
import time
import traceback
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
from uuid import uuid4


class ProviderError(RuntimeError):
    """A service failure whose prompt group must not contribute a training reward."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage

    def __reduce__(self):
        return type(self), (self.stage, str(self))


def failure_record(exc: BaseException, **fields) -> dict:
    """Everything needed to locate a failure later, including its cause chain."""
    return {
        **fields,
        "type": type(exc).__name__,
        "stage": getattr(exc, "stage", "internal"),
        "error": str(exc),
        "traceback": "".join(traceback.format_exception(exc)),
    }


def record_event(path: Path, stage: str, event: str, **fields):
    """Append one complete record, including when sibling rollouts share a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "stage": stage,
        "event": event,
        **fields,
    }
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")  # Preserve an interrupted row without damaging the next event.
        handle.write((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def timed_stage(path: Path, stage: str, **fields):
    """Pair start/end events by ID; unmatched starts identify interrupted work."""
    fields = {**fields, "stage_id": uuid4().hex}
    record_event(path, stage, "start", **fields)
    started = time.monotonic()
    result = {"status": "succeeded"}
    try:
        yield result
    except BaseException as exc:
        result.update(status="error", error_type=type(exc).__name__)
        if isinstance(exc, ProviderError):
            result["failure_stage"] = exc.stage
        raise
    finally:
        record_event(path, stage, "end", **fields, **result, elapsed_seconds=time.monotonic() - started)


def summarize_timings(run_dir: Path):
    """Report successful-stage percentiles separately from failures and unfinished work.

    The run-level ``group`` stage counts group outcomes (succeeded, dropped, error).
    """
    elapsed, outcomes = defaultdict(list), defaultdict(Counter)
    starts, ends, malformed = {}, set(), 0
    for path in sorted(Path(run_dir).rglob("timings*.jsonl")):
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    malformed += 1
                    continue
                stage, event = row["stage"], row["event"]
                if event == "start":
                    starts[row["stage_id"]] = {"path": str(path), **row}
                elif event == "end":
                    ends.add(row["stage_id"])
                    outcomes[stage][row["status"]] += 1
                    if row["status"] == "succeeded":
                        elapsed[stage].append(row["elapsed_seconds"])
                elif stage == "h3_provider" and event == "completed":
                    for key in ("queued_seconds", "running_seconds", "total_seconds"):
                        if key in row:
                            name = f"h3_{key.removesuffix('_seconds')}"
                            outcomes[name]["succeeded"] += 1
                            elapsed[name].append(row[key])
    stages = {}
    for name, counts in sorted(outcomes.items()):
        values = sorted(elapsed[name])
        entry = {"outcomes": dict(counts), "successful_observations": len(values)}
        if values:

            def percentile(q, observed=values):
                position = (len(observed) - 1) * q
                index = int(position)
                return observed[index] + (observed[min(index + 1, len(observed) - 1)] - observed[index]) * (
                    position - index
                )

            entry.update(
                p50_seconds=percentile(0.5),
                p99_seconds=percentile(0.99),
                max_seconds=values[-1],
                proposed_timeout_seconds=ceil(2 * percentile(0.99)),
            )
        stages[name] = entry
    return {
        "run_dir": str(Path(run_dir).resolve()),
        "percentile_method": "linear interpolation at (n - 1) * q; successful observations only",
        "stages": stages,
        "unfinished_stages": [row for key, row in starts.items() if key not in ends],
        "malformed_lines": malformed,
    }
