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
from pathlib import Path

from .observability import timed_stage
from .storage import load_external


def instrument_client(runner, timings: Path):
    """Measure uploads, file processing, interactions, and cleanup independently."""

    class TimedClient(runner.GeminiRestClient):
        def _request_once(self, method, url, **kwargs):
            # Only judgment interactions use the calibrated request limit.
            # Upload initialization, polling, and cleanup remain at the file limit.
            if not url.endswith("/interactions") and kwargs.get("timeout") is None:
                kwargs["timeout"] = self.file_timeout
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
            with timed_stage(timings, "gemini_file_delete", timeout_seconds=self.file_timeout):
                return super().delete_file(name)

    runner.GeminiRestClient = TimedClient


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--timings", type=Path, required=True)
    args, forwarded = parser.parse_known_args(argv)
    runner = load_external(args.runner)
    instrument_client(runner, args.timings)
    return runner.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
