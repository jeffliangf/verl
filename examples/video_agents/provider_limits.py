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
"""Run-wide provider slots, released by the kernel when a judge dies."""

import fcntl
import random
import time
from contextlib import contextmanager
from pathlib import Path


class ProviderLimits:
    def __init__(self, root, limits, *, timeout=300):
        self.root, self.limits, self.timeout = Path(root), dict(limits), timeout
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def slot(self, kind):
        capacity = self.limits[kind]
        deadline = time.monotonic() + self.timeout
        acquired = None
        cursor = random.randrange(capacity)
        try:
            while acquired is None:
                # Cleanup has its own slots and can progress when admission is full.
                blocked = kind == "uploads" and (
                    len(list((self.root.parent / "cleanup/pending").glob("*.json"))) >= self.limits["cleanup_backlog"]
                )
                if not blocked:
                    # Avoid a full shared-directory scan from every waiting
                    # thread. Rotate a small probe window with randomized starts.
                    for _ in range(min(capacity, 8)):
                        index, cursor = cursor, (cursor + 1) % capacity
                        handle = (self.root / f"{kind}-{index}.lock").open("a+")
                        try:
                            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            handle.close()
                        else:
                            acquired = handle
                            break
                if acquired is None:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for shared Gemini {kind} admission")
                    time.sleep(min(random.uniform(0.1, 0.3), max(0, deadline - time.monotonic())))
            yield
        finally:
            if acquired is not None:
                acquired.close()


def provider_limits(protocol):
    settings = protocol.get("judge_limits")
    root = protocol.get("provider_recovery_root")
    return (
        ProviderLimits(Path(root) / "slots", settings, timeout=protocol["timeouts"]["judge_pass"])
        if settings and root
        else None
    )
