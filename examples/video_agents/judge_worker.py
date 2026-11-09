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
import http.client
import inspect
import math
import socket
import threading
import urllib.error
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .gemini_files import UploadJournal, valid_name
from .image_assets import image_bytes
from .observability import record_event, timed_stage
from .storage import load_external


def supports_reference_images(runner):
    """Old frozen T2V runners remain usable; mixed preparation requires the image API."""
    builder = getattr(runner, "build_interaction_request", None)
    return (
        callable(builder)
        and "reference_image" in inspect.signature(builder).parameters
        and hasattr(getattr(runner, "GeminiRestClient", None), "upload_image")
        and callable(getattr(runner, "image_mime_type", None))
    )


@contextmanager
def reference_images(runner):
    """Connect the upstream image helper to its batch runner without changing its rubric.

    A private marker carries builder arguments only inside this process. Before
    evaluate(), replace it with the upstream request built with an uploaded image.
    T2V requests pass through unchanged. SharedUploads owns cleanup when enabled.
    """
    if not supports_reference_images(runner):
        yield
        return
    original_client = runner.GeminiRestClient
    original_build = runner.build_interaction_request
    original_mime = runner.image_mime_type
    marker = "_verl_reference_image_request"

    def build(**kwargs):
        payload = original_build(**kwargs)
        if kwargs["pair"].get("reference_image") is not None:
            payload[marker] = kwargs
        return payload

    def image_mime(path):
        from .h3_request import MAX_IMAGE_BYTES, image_info

        with Path(path).open("rb") as handle:
            data = handle.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Reference image exceeds the frozen H3 image limit")
        return image_info(data)[0]

    class ReferenceImageClient(original_client):
        def evaluate(self, request_payload, *, require_agentic_steps):
            arguments = request_payload.get(marker)
            if arguments is None:
                return super().evaluate(request_payload, require_agentic_steps=require_agentic_steps)
            asset = arguments["pair"]["reference_image"]
            image_bytes(asset)
            uploaded = self.upload_image(Path(asset["path"]))
            try:
                payload = original_build(**arguments, reference_image=uploaded)
                return super().evaluate(payload, require_agentic_steps=require_agentic_steps)
            finally:
                if uploaded.get("name"):
                    try:
                        self.delete_file(uploaded["name"])
                    except runner.EvalError:
                        print("Warning: Gemini reference image cleanup failed.", flush=True)

    runner.GeminiRestClient = ReferenceImageClient
    runner.build_interaction_request = build
    runner.image_mime_type = image_mime
    try:
        yield
    finally:
        runner.GeminiRestClient = original_client
        runner.build_interaction_request = original_build
        runner.image_mime_type = original_mime


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
    journal = UploadJournal(timings.parent / "gemini-files.jsonl")
    owners, owners_lock = {}, threading.Lock()

    class TimedClient(runner.GeminiRestClient):
        def _upload_content_once(self, upload_url, path, mime_type):
            file_object = super()._upload_content_once(upload_url, path, mime_type)
            name = file_object.get("name")
            if not valid_name(name):
                raise runner.EvalError("Uploaded file metadata has no valid resource name")
            journal.created(name, path)
            with owners_lock:
                owners[name] = self
            return file_object

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
                try:
                    return super()._request_once(method, url, **kwargs)
                except (http.client.IncompleteRead, http.client.BadStatusLine, http.client.LineTooLong) as exc:
                    # The frozen client misses these response-read failures.
                    # Preserve an HTTP status even if reading its error body
                    # failed: an interrupted 401/403 must still remain fatal.
                    error, seen = exc, set()
                    while error is not None and id(error) not in seen:
                        seen.add(id(error))
                        if isinstance(error, urllib.error.HTTPError):
                            raise runner.HttpFailure(
                                error.code, "Interrupted HTTP error response", error.headers
                            ) from exc
                        error = error.__cause__ or error.__context__
                    # Use the existing retry budget and per-pair error ledger.
                    raise runner.NetworkFailure(f"Network response failed: {exc}") from exc

        def upload_video(self, path):
            with timed_stage(timings, "gemini_upload", video_path=str(path), timeout_seconds=self.file_timeout):
                return super().upload_video(path)

        def upload_image(self, path):
            with timed_stage(timings, "gemini_image_upload", image_path=str(path), timeout_seconds=self.file_timeout):
                return super().upload_image(path)

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
                result = super().delete_file(name)
                journal.deleted(name)
                return result

    def cleanup_uploads():
        # Include uploads whose polling/retry failed before the shared Future
        # could return a handle. Per-pair deletion still waits for all readers.
        for name in journal.pending():
            client = owners.get(name)
            if client is not None:
                try:
                    TimedClient.delete_file(client, name)
                except runner.EvalError:
                    print("Warning: Gemini upload remains in the cleanup journal.", flush=True)

    runner.GeminiRestClient = TimedClient
    runner._verl_cleanup_uploads = cleanup_uploads


@contextmanager
def parallel_pairs(runner, *, keep_uploaded_files=False):
    """Schedule individual pairs, sharing uploads until all comparisons have settled."""
    original_client, original_batches = runner.GeminiRestClient, runner.group_into_prompt_batches
    uploads, lock = {}, threading.Lock()

    class SharedUploads(original_client):
        def upload_video(self, path):
            return self.shared_upload("video", path, super().upload_video)

        def upload_image(self, path):
            return self.shared_upload("image", path, super().upload_image)

        def shared_upload(self, kind, path, upload):
            key = (kind, str(Path(path).resolve()))
            with lock:
                owner = key not in uploads
                if owner:
                    uploads[key] = Future()
                future = uploads[key]
            if owner:
                try:
                    future.set_result((self, upload(path)))
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
            cleanup = getattr(runner, "_verl_cleanup_uploads", None)
            if cleanup is not None:
                cleanup()


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
    with gemini_address_family(args.address_family), reference_images(runner):
        if args.parallel_pairs:
            with parallel_pairs(runner, keep_uploaded_files="--keep-uploaded-files" in forwarded):
                return runner.main(forwarded)
        try:
            return runner.main(forwarded)
        finally:
            if "--keep-uploaded-files" not in forwarded:
                runner._verl_cleanup_uploads()


if __name__ == "__main__":
    raise SystemExit(main())
