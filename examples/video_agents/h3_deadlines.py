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
"""Separate gateway queueing, GPU execution, and artifact-transfer budgets."""

from collections.abc import Mapping
from math import isfinite

from . import h3


def phase_budgets(protocol):
    budgets = protocol.get("h3_phase_timeouts")
    if budgets is None:
        return None
    if (
        not isinstance(budgets, Mapping)
        or set(budgets) != {"queue", "generation", "download"}
        or any(
            isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value) or value <= 0
            for value in budgets.values()
        )
    ):
        raise ValueError("H3 phase budgets require positive finite queue, generation and download seconds")
    return dict(budgets)


def worker_timeout(protocol, legacy_timeout):
    budgets = phase_budgets(protocol)
    # A final process guard also bounds blocked local I/O and worker startup.
    # It is separate from the generation budget and is not reset by retries.
    return sum(budgets.values()) + 60 if budgets else legacy_timeout


def timestamp(response, key, *, required=False):
    value = response.get(key)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"H3 phase accounting requires a finite {key} timestamp")
    return value


class PhaseTimedGeneration(h3._Generation):
    """Use gateway timestamps for compute time, including across retry attempts.

    Polls retain their own bounded HTTP retries. A late observation of an already
    completed job is checked against its server execution interval, so queueing
    and client transfer delays cannot turn a short render into a render timeout.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.budgets = phase_budgets(self.protocol)

    def poll_once(self, timeout):
        response = super().poll_once(timeout)
        status = response["status"]
        created = timestamp(response, "created_at", required=True)
        started = timestamp(response, "started_at", required=status in {"running", "succeeded"})
        finished = timestamp(response, "finished_at", required=status == "succeeded")
        if (started is not None and started < created) or (
            finished is not None and finished < (started if started is not None else created)
        ):
            raise ValueError("H3 returned inconsistent job timestamps")
        now = h3.time.time()
        queue_end = started if started is not None else finished if finished is not None else now
        usage = {
            "queue": max(0, queue_end - created),
            "generation": max(0, (finished if finished is not None else now) - started) if started is not None else 0,
        }
        # Terminal server timestamps replace provisional observations. The ledger
        # survives subprocess recovery and regeneration with a new durable job ID.
        ledger = self.state.setdefault("phase_usage_by_job", {})
        ledger[response["id"]] = usage
        for phase in ("queue", "generation"):
            elapsed = sum(row[phase] for row in ledger.values())
            if elapsed > self.budgets[phase]:
                self.state["timeout_phase"] = phase
                raise TimeoutError(f"H3 {phase} exceeded {self.budgets[phase]}s ({elapsed:.3f}s observed)")
        return response

    def download(self, response):
        expires = self.state.get("download_deadline_at", h3.time.time() + self.budgets["download"])
        self.state["download_deadline_at"] = expires
        self.deadline = min(self.deadline, h3.time.monotonic() + expires - h3.time.time())
        try:
            self.remaining()
            return super().download(response)
        except TimeoutError:
            self.state["timeout_phase"] = "download"
            raise
