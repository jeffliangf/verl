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
"""Verify ownership and recovery after an evaluator is killed mid-group."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from examples.video_agents.gemini_files import UploadJournal, reap


class HttpFailure(RuntimeError):
    def __init__(self, status):
        self.status = status


@pytest.mark.parametrize("state", ["present", "absent", "transient", "foreign"])
def test_killed_judge_upload_journal_preserves_ownership(tmp_path, state):
    path = tmp_path / "gemini-files.jsonl"
    # Kill a real process: cleanup cannot rely on its Python finally blocks.
    program = """
import sys, time
from pathlib import Path
from examples.video_agents.gemini_files import UploadJournal
try:
    UploadJournal(Path(sys.argv[1])).created('files/owned', Path('/candidate.mp4'))
    print('ready', flush=True)
    time.sleep(60)
finally:
    Path(sys.argv[1]).unlink()
"""
    child = subprocess.Popen([sys.executable, "-c", program, str(path)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "ready"
    finally:
        child.kill()
        child.wait(timeout=5)
        child.stdout.close()
    journal = UploadJournal(path)
    display = journal.pending()["files/owned"]
    deleted = []

    class Client:
        def request_json(self, method, url, **kwargs):
            assert method == "GET" and url == "https://test.invalid/files/owned"
            if state == "absent":
                raise HttpFailure(404)
            if state == "transient":
                raise HttpFailure(503)
            return {"name": "files/owned", "displayName": "another-owner" if state == "foreign" else display}, {}

        def delete_file(self, name):
            deleted.append(name)

    runner = SimpleNamespace(API_BASE="https://test.invalid", HttpFailure=HttpFailure, EvalError=RuntimeError)
    if state == "foreign":
        with pytest.raises(ValueError, match="ownership mismatch"):
            reap(journal, runner, Client())
    else:
        result = reap(journal, runner, Client())
        assert result["pending_after"] == (1 if state == "transient" else 0)
    assert deleted == (["files/owned"] if state == "present" else [])
    assert bool(journal.pending()) == (state in {"transient", "foreign"})
