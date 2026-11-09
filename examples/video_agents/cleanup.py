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
"""Durable background cleanup after a group's last judge reader has exited."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .gemini_files import UploadJournal, reap
from .provider_limits import provider_limits
from .storage import item_key, load_external, read_json, write_json


def cleanup_root(protocol):
    return Path(protocol["provider_recovery_root"]) / "cleanup"


def enqueue(directory, protocol, *, active=False):
    root = cleanup_root(protocol)
    directory = Path(directory).resolve()
    if not directory.is_relative_to(Path(protocol["provider_recovery_root"]).resolve().parent):
        raise ValueError("Cleanup directory must belong to this run")
    path = root / ("active" if active else "pending") / f"{item_key(str(directory))}.json"
    write_json(path, {"directory": str(directory)})
    return path


def ensure_running(protocol):
    """One leader per shared run; another start is harmless after a crash."""
    root = cleanup_root(protocol)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "leader.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
    with (root / "reaper.log").open("ab") as log:
        subprocess.Popen(
            [sys.executable, "-m", "examples.video_agents.cleanup", str(root)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


@contextmanager
def readers(directory, protocol):
    """Register while holding the lease, before any upload can be created.

    The reaper takes this same exclusive lock. Process death releases it, so
    both successful exit and SIGKILL are recoverable without a fragile PID lease.
    """
    path = Path(directory) / "judge/readers.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        task = enqueue(directory, protocol, active=True)
        (cleanup_root(protocol) / "pending" / task.name).unlink(missing_ok=True)
        yield


def clean_task(path):
    directory = Path(read_json(path)["directory"])
    root = path.parent.parent
    if not directory.resolve().is_relative_to(root.parent.parent.resolve()):
        raise ValueError("Cleanup task escaped its run directory")
    with (directory / "judge/readers.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        journal = UploadJournal(directory / "judge/gemini-files.jsonl")
        if journal.pending():
            # Active reader registrations are durable crash recovery records,
            # not cleanup backlog. Only exited readers exert upload backpressure.
            if path.parent.name == "active":
                pending = root / "pending" / path.name
                pending.parent.mkdir(parents=True, exist_ok=True)
                path.replace(pending)
                path = pending
            protocol = read_json(directory / "judge/stream/context.json")["protocol"]
            runner = load_external(Path(protocol["eval_repo"]) / "scripts/run_gemini_pairwise_eval.py")
            env = runner.read_env_file(Path(protocol["eval_repo"]) / "arena/.env")
            key = os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY", "")
            client = runner.GeminiRestClient(key, max_attempts=1, request_timeout=5, file_timeout=5)
            from .judge_worker import gemini_address_family

            with gemini_address_family("ipv4"):
                result = reap(journal, runner, client, limits=provider_limits(protocol))
            if result["pending_after"]:
                return False
        for category in ("pending", "active"):
            (root / category / path.name).unlink(missing_ok=True)
        return True


def run(root, *, idle_seconds=60):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "leader.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        last_work, retry_at = time.monotonic(), {}
        while True:
            tasks = sorted((root / "pending").glob("*.json")) + sorted((root / "active").glob("*.json"))
            retry_at = {task: deadline for task, deadline in retry_at.items() if task in tasks}
            if tasks:
                last_work = time.monotonic()
            elif time.monotonic() - last_work >= idle_seconds:
                return
            for task in tasks:
                if not task.exists():
                    continue  # Cleaning one entry can also remove its active registration.
                if time.monotonic() < retry_at.get(task, 0):
                    continue
                try:
                    done = clean_task(task)
                except Exception as exc:
                    # Keep the durable task, including on identity/code errors.
                    print(f"Cleanup deferred: {task.name}: {type(exc).__name__}", flush=True)
                    done = False
                if done:
                    retry_at.pop(task, None)
                else:
                    retry_at[task] = time.monotonic() + 5
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    run(args.root)


if __name__ == "__main__":
    main()
