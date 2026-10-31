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
"""Instrument the frozen evaluator and retry transient judgment transport failures."""

import argparse
import math
import socket
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
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
    interaction = ContextVar("gemini_interaction", default=None)

    class TimedClient(runner.GeminiRestClient):
        def request_json(self, method, url, **kwargs):
            # Reuse the uploaded files and the upstream HTTP retry budget. Do
            # not wrap evaluate() in another loop, which would multiply retries.
            if method == "POST" and url.endswith("/interactions"):
                kwargs["retry_network"] = True
            return super().request_json(method, url, **kwargs)

        def _request_once(self, method, url, **kwargs):
            # Payload uploads supply their own longer timeout. Small metadata,
            # polling, and cleanup requests must not consume that entire budget.
            if not url.endswith("/interactions") and kwargs.get("timeout") is None:
                kwargs["timeout"] = min(self.file_timeout, file_control_timeout)
            trace = interaction.get()
            if url.endswith("/interactions") and trace is not None:
                trace["attempts"] += 1
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
            with timed_stage(timings, "gemini_request", timeout_seconds=self.request_timeout) as trace:
                trace.update(attempts=0, recovered=False)
                token = interaction.set(trace)
                try:
                    result = super().evaluate(request_payload, require_agentic_steps=require_agentic_steps)
                    trace["recovered"] = trace["attempts"] > 1
                    return result
                finally:
                    interaction.reset(token)

        def delete_file(self, name):
            with timed_stage(
                timings, "gemini_file_delete", timeout_seconds=min(self.file_timeout, file_control_timeout)
            ):
                return super().delete_file(name)

    runner.GeminiRestClient = TimedClient


@contextmanager
def parallel_pairs(runner, *, keep_uploaded_files=False):
    """Schedule individual pairs, sharing uploads until all comparisons have settled."""
    original_client, original_batches = runner.GeminiRestClient, runner.group_into_prompt_batches
    uploads, lock = {}, threading.Lock()

    class SharedUploads(original_client):
        def upload_video(self, path):
            key = str(Path(path).resolve())
            with lock:
                owner = key not in uploads
                if owner:
                    uploads[key] = Future()
                future = uploads[key]
            if owner:
                try:
                    future.set_result((self, super().upload_video(path)))
                except BaseException as exc:
                    future.set_exception(exc)
            return future.result()[1]

        def delete_file(self, name):
            # The runner calls this after each batch. Other pairs still need the
            # file, so cleanup belongs after its thread pool has joined.
            return None

    def batches(pairs, batch_size):
        return [([pair[1]["prompt_id"]], [pair]) for pair in pairs]

    runner.GeminiRestClient, runner.group_into_prompt_batches = SharedUploads, batches
    try:
        yield
    finally:
        runner.GeminiRestClient, runner.group_into_prompt_batches = original_client, original_batches
        if not keep_uploaded_files:
            for future in uploads.values():
                if future.done() and future.exception() is None:
                    client, file_object = future.result()
                    try:
                        original_client.delete_file(client, file_object["name"])
                    except runner.EvalError:
                        print("Warning: Gemini file cleanup failed after the comparisons settled.", flush=True)


def parse_options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    parser.add_argument("--file-control-timeout", type=float, default=30.0)
    parser.add_argument("--address-family", choices=("ipv4", "auto"), default="ipv4")
    parser.add_argument("--parallel-pairs", action="store_true")
    args, forwarded = parser.parse_known_args(argv)
    if not math.isfinite(args.file_control_timeout) or args.file_control_timeout <= 0:
        parser.error("--file-control-timeout must be finite and positive")
    return args, forwarded


def main(argv=None):
    args, forwarded = parse_options(argv)
    runner = load_external(args.runner)
    instrument_client(runner, args.timings, args.file_control_timeout)
    record_event(args.timings, "gemini_network", "configured", address_family=args.address_family)
    with gemini_address_family(args.address_family):
        if args.parallel_pairs:
            with parallel_pairs(runner, keep_uploaded_files="--keep-uploaded-files" in forwarded):
                return runner.main(forwarded)
        return runner.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
