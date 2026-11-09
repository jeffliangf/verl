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
"""Real process death, reader exclusion, shared admission and nonblocking cleanup."""

import asyncio
import json
import os
import select
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from examples.video_agents import cleanup
from examples.video_agents.gemini_files import UploadJournal
from examples.video_agents.provider_limits import ProviderLimits
from examples.video_agents.storage import write_json
from examples.video_agents.streaming_judge import StreamingJudge

LIMITS = {"comparisons": 2, "uploads": 1, "deletions": 1, "cleanup_backlog": 4}


def test_provider_client_exits_when_its_worker_is_killed():
    child_code = """
import os, time
from examples.video_agents.storage import guard_parent
guard_parent(os.getppid())
print(os.getpid(), flush=True)
time.sleep(60)
"""
    parent_code = """
import subprocess, sys, time
subprocess.Popen([sys.executable, '-c', sys.argv[1]])
time.sleep(60)
"""
    parent = subprocess.Popen([sys.executable, "-c", parent_code, child_code], stdout=subprocess.PIPE, text=True)
    child_pid = None
    try:
        assert select.select([parent.stdout], [], [], 10)[0], "Judge did not start"
        child_pid = int(parent.stdout.readline())
        parent.kill()
        parent.wait(timeout=5)
        # The grandchild inherited the pipe. EOF proves it also exited rather
        # than holding a reader lease indefinitely after the worker disappeared.
        assert select.select([parent.stdout], [], [], 5)[0], "Judge survived worker death"
        assert parent.stdout.read() == ""
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        parent.stdout.close()


def test_provider_limit_is_shared_and_a_killed_holder_releases_capacity(tmp_path):
    program = """
import sys, time, json
from examples.video_agents.provider_limits import ProviderLimits
with ProviderLimits(sys.argv[1], json.loads(sys.argv[2]), timeout=2).slot('uploads'):
    print('held', flush=True)
    time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(tmp_path / "slots"), json.dumps(LIMITS)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "held"
        with pytest.raises(TimeoutError):
            with ProviderLimits(tmp_path / "slots", LIMITS, timeout=0.03).slot("uploads"):
                pytest.fail("A second process acquired the occupied slot")
    finally:
        child.kill()
        child.wait(timeout=5)
        child.stdout.close()
    with ProviderLimits(tmp_path / "slots", LIMITS, timeout=0.03).slot("uploads"):
        pass


def test_shared_comparison_cap_bounds_independent_clients(tmp_path):
    active, maximum = 0, 0
    lock = threading.Lock()

    def compare(_):
        nonlocal active, maximum
        with ProviderLimits(tmp_path / "slots", LIMITS, timeout=5).slot("comparisons"):
            with lock:
                active += 1
                maximum = max(active, maximum)
            time.sleep(0.01)
            with lock:
                active -= 1

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(compare, range(16)))
    assert maximum == 2 and active == 0


def test_cleanup_backlog_pauses_uploads_but_not_deletions(tmp_path):
    for i in range(4):
        write_json(tmp_path / "cleanup/pending" / f"{i}.json", {})
    limits = ProviderLimits(tmp_path / "slots", LIMITS, timeout=0.02)
    with pytest.raises(TimeoutError):
        with limits.slot("uploads"):
            pytest.fail("Upload passed a full cleanup backlog")
    with limits.slot("deletions"):
        pass
    (tmp_path / "cleanup/pending/0.json").unlink()
    with limits.slot("uploads"):
        pass


def test_reaper_waits_for_real_reader_process_then_recovers_after_kill(tmp_path, monkeypatch):
    protocol = {"provider_recovery_root": str(tmp_path / "provider-recovery"), "eval_repo": str(tmp_path)}
    directory = tmp_path / "train/group"
    write_json(directory / "judge/stream/context.json", {"protocol": protocol})
    program = """
import sys, json, time
from pathlib import Path
from examples.video_agents.cleanup import readers
from examples.video_agents.gemini_files import UploadJournal
directory = Path(sys.argv[1])
with readers(directory, json.loads(sys.argv[2])):
    UploadJournal(directory / 'judge/gemini-files.jsonl').created('files/owned', directory / 'video.mp4')
    print('reading', flush=True)
    time.sleep(60)
"""
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(directory), json.dumps(protocol)], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "reading"
        (task,) = (tmp_path / "provider-recovery/cleanup/active").glob("*.json")
        assert cleanup.clean_task(task) is False
        # Live readers must not fill the deletion backlog and block one another's uploads.
        with ProviderLimits(tmp_path / "provider-recovery/slots", {**LIMITS, "cleanup_backlog": 1}).slot("uploads"):
            pass
    finally:
        child.kill()
        child.wait(timeout=5)
        child.stdout.close()
    journal = UploadJournal(directory / "judge/gemini-files.jsonl")
    display = journal.pending()["files/owned"]
    deleted = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def request_json(self, *args, **kwargs):
            return {"name": "files/owned", "displayName": display}, {}

        def delete_file(self, name):
            deleted.append(name)

    runner = SimpleNamespace(
        GeminiRestClient=Client,
        read_env_file=lambda path: {},
        API_BASE="https://fake.invalid",
        HttpFailure=RuntimeError,
        EvalError=RuntimeError,
    )
    monkeypatch.setattr(cleanup, "load_external", lambda path: runner)
    assert cleanup.clean_task(task) is True
    assert deleted == ["files/owned"] and not journal.pending() and not task.exists()


@pytest.mark.asyncio
async def test_reward_path_enqueues_cleanup_without_waiting_for_delete(tmp_path, monkeypatch):
    directory = tmp_path / "train/group"
    protocol = {
        "provider_recovery_root": str(tmp_path / "provider-recovery"),
        "judge_background_cleanup": True,
        "timeouts": {"judge_pass": 1},
    }
    journal = UploadJournal(directory / "judge/gemini-files.jsonl")
    journal.created("files/owned", directory / "video.mp4")
    stream = object.__new__(StreamingJudge)
    stream.directory, stream.root, stream.protocol = directory, directory / "judge/stream", protocol
    stream.log = None

    async def wait():
        return 0

    stream.process = SimpleNamespace(returncode=0, wait=wait)
    monkeypatch.setattr(cleanup, "ensure_running", lambda protocol: None)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *a, **k: pytest.fail("Inline cleanup subprocess"))
    await asyncio.wait_for(stream.finish(), timeout=1)
    await asyncio.wait_for(stream.close(), timeout=1)
    assert journal.pending() and (stream.root / "outcome.json").exists()
    assert len(list((tmp_path / "provider-recovery/cleanup/pending").glob("*.json"))) == 1
