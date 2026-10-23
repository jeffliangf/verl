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
"""Instrument the frozen external evaluator without modifying its judging logic."""

import argparse
import math
import socket
from contextlib import contextmanager
from pathlib import Path

from .observability import record_event, timed_stage
from .storage import load_external


@contextmanager
def gemini_address_family(address_family: str):
    """Avoid unreachable Gemini IPv6 routes in this evaluator process only."""
    if address_family == "auto":
        yield
        return
    if address_family != "ipv4":
        raise ValueError("Gemini address family must be auto or ipv4")
    original = socket.getaddrinfo

    def resolve(host, port, family=socket.AF_UNSPEC, type=0, proto=0, flags=0):
        # Resolve the original hostname normally, preserving TLS verification
        # and DNS rotation. Other services and explicit families are unchanged.
        if host in ("generativelanguage.googleapis.com", b"generativelanguage.googleapis.com"):
            if family == socket.AF_UNSPEC:
                family = socket.AF_INET
        return original(host, port, family, type, proto, flags)

    socket.getaddrinfo = resolve
    try:
        yield
    finally:
        socket.getaddrinfo = original


def instrument_client(runner, timings: Path, file_control_timeout: float = 30.0):
    """Measure uploads, file processing, interactions, and cleanup independently."""

    class TimedClient(runner.GeminiRestClient):
        def _request_once(self, method, url, **kwargs):
            # Payload uploads supply their own longer timeout. Small metadata,
            # polling, and cleanup requests must not consume that entire budget.
            if not url.endswith("/interactions") and kwargs.get("timeout") is None:
                kwargs["timeout"] = min(self.file_timeout, file_control_timeout)
            with timed_stage(
                timings,
                "gemini_http",
                method=method,
                operation="interaction" if url.endswith("/interactions") else "file",
                timeout_seconds=kwargs.get("timeout") or self.request_timeout,
            ):
                return super()._request_once(method, url, **kwargs)

        def upload_video(self, path):
            with timed_stage(timings, "gemini_upload", video_path=str(path), timeout_seconds=self.file_timeout):
                return super().upload_video(path)

        def wait_for_file(self, file_object):
            with timed_stage(timings, "gemini_file_processing", timeout_seconds=self.file_timeout):
                return super().wait_for_file(file_object)

        def evaluate(self, request_payload, *, require_agentic_steps):
            with timed_stage(timings, "gemini_request", timeout_seconds=self.request_timeout):
                return super().evaluate(request_payload, require_agentic_steps=require_agentic_steps)

        def delete_file(self, name):
            with timed_stage(
                timings, "gemini_file_delete", timeout_seconds=min(self.file_timeout, file_control_timeout)
            ):
                return super().delete_file(name)

    runner.GeminiRestClient = TimedClient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--file-control-timeout", type=float, default=30.0)
    parser.add_argument("--address-family", choices=("ipv4", "auto"), default="ipv4")
    args, forwarded = parser.parse_known_args(argv)
    if not math.isfinite(args.file_control_timeout) or args.file_control_timeout <= 0:
        parser.error("--file-control-timeout must be finite and positive")
    runner = load_external(args.runner)
    instrument_client(runner, args.timings, args.file_control_timeout)
    record_event(args.timings, "gemini_network", "configured", address_family=args.address_family)
    with gemini_address_family(args.address_family):
        return runner.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
