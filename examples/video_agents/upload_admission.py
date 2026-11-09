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
"""Pause run-owned Gemini uploads after a storage-quota error; probe one at a time."""

import fcntl
import time
from contextlib import contextmanager
from uuid import uuid4

from .storage import read_json, write_json


def storage_quota(error):
    message = str(error).lower()
    return any(part in message for part in ("filestoragebytesperproject", "file_storage_bytes", "file storage quota"))


class UploadAdmission:
    """Shared by agent processes through the run directory, never by credentials.

    Existing comparisons may drain and release files while new uploads wait.
    One successful upload/processing probe reopens admission. A killed probe's
    lease expires; waiting never extends the caller's saved evaluation deadline.
    """

    def __init__(self, directory, deadline):
        self.directory, self.deadline = directory, deadline
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "state.json"
        self.wait_for_probe = False

    def blocked(self):
        with self.locked() as state:
            return state.get("blocked", False)

    @contextmanager
    def locked(self):
        with (self.directory / "lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            state = read_json(self.path) if self.path.exists() else {}
            yield state

    def acquire(self):
        while time.time() < self.deadline:
            with self.locked() as state:
                if not state.get("blocked"):
                    return None
                now = time.time()
                if now >= max(state.get("retry_at", 0), state.get("lease_until", 0)):
                    token = str(uuid4())
                    state.update(probe=token, lease_until=min(self.deadline, now + 180))
                    write_json(self.path, state)
                    return token
            if not self.wait_for_probe:
                # Do not hold a comparison reader while waiting for cleanup of
                # that same prompt's files. The scheduler drains readers first.
                raise TimeoutError("Gemini uploads paused for storage recovery")
            time.sleep(max(0, min(1, self.deadline - time.time())))
        raise TimeoutError("Evaluation deadline exhausted while waiting for Gemini storage capacity")

    def succeeded(self, token):
        if token is None:
            return
        with self.locked() as state:
            if state.get("probe") == token:
                write_json(self.path, {"blocked": False})

    def failed(self, token, error):
        with self.locked() as state:
            if storage_quota(error) or (token is not None and state.get("probe") == token):
                write_json(self.path, {"blocked": True, "retry_at": time.time() + 15})
