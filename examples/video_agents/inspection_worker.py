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
"""Bounded subprocess adapter around the unchanged V1 Gemini inspector."""

import argparse
import http.client
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from urllib.parse import urlparse

from .h3 import generation_slot
from .h3_request import frame_count
from .inspection_agent import feedback
from .observability import ProviderError, timed_stage
from .storage import load_external, read_json, write_json

_CLEANUP = ThreadPoolExecutor(max_workers=2, thread_name_prefix="inspection-cleanup")
# Uploaded media and their cleanup markers. Only I2V inspections upload an image.
UPLOADS = {"video": ("uploaded.json", "cleanup.json"), "image": ("uploaded-image.json", "cleanup-image.json")}


def inspect_generation(video, original_request, directory, eval_repo, source_image=None):
    """Supply new clip lengths and I2V first frames without changing legacy T2V requests."""
    inspector = feedback
    if "task" in video["request"]:
        # Use a separate module namespace so concurrent/legacy callers retain
        # the original validator. The rubric is unchanged. Rollout records keep a
        # summary without first-frame bytes, so read the length from the target.
        inspector = load_external(Path(feedback.__file__))
        duration = frame_count(video["request"]["target"]["duration_seconds"]) / 24
        inspector.validate_feedback = partial(inspector.validate_feedback, duration=duration)
    return inspector.inspect(video, original_request, [], directory, eval_repo, source_image=source_image)


def schedule_cleanup(directory, eval_repo):
    """Keep owned-media cleanup outside the expired inspection deadline."""

    def cleanup():
        try:
            cleanup_uploads(directory, eval_repo)
        except Exception as exc:
            # Cleanup must not replace the tool failure or invalidate a ready report.
            write_json(Path(directory) / "cleanup-error.json", {"error_type": type(exc).__name__})

    _CLEANUP.submit(cleanup)


def inspect_video(video, original_request, directory, protocol, settings, output_dir, source_image=None):
    directory = Path(directory)
    request_path = directory / "input.json"
    write_json(
        request_path,
        {
            "video": video,
            "original_request": original_request,
            "eval_repo": protocol["eval_repo"],
            "directory": str(directory),
            **({"source_image": source_image} if source_image else {}),
        },
    )
    timings = directory / "timings.jsonl"
    # A distinct lock namespace; this does not occupy an H3 generation slot.
    with generation_slot(Path(output_dir) / "inspection-quota", int(settings["max_inflight"]), timings):
        with timed_stage(timings, "inspection", timeout_seconds=settings["timeout_seconds"]):
            try:
                with (directory / "worker.log").open("a") as log:
                    process = subprocess.run(
                        [sys.executable, "-m", __name__, "--input", str(request_path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=settings["timeout_seconds"],
                        check=False,
                    )
            except subprocess.TimeoutExpired as exc:
                schedule_cleanup(directory, protocol["eval_repo"])
                path = directory / "worker-result.json"
                if path.exists():
                    result = read_json(path)
                    if result["status"] == "succeeded":
                        # The report can be ready while the child is deleting its upload.
                        return result["report"]
                raise ProviderError(
                    "inspection", f"Gemini inspection exceeded its {settings['timeout_seconds']}s service deadline"
                ) from exc
            path = directory / "worker-result.json"
            if not path.exists():
                raise RuntimeError(f"Inspection worker exited {process.returncode} without a result: {directory}")
            result = read_json(path)
            if result["status"] == "provider_error":
                raise ProviderError("inspection", result["error"])
            if process.returncode or result["status"] != "succeeded":
                raise RuntimeError(f"Inspection worker failed: {result}")
            return result["report"]


def provider_failure(exc, directory):
    """Separate service/response failures from local artifact and programming errors."""
    chain, error = [], exc
    while error is not None and id(error) not in {id(e) for e in chain}:
        chain.append(error)
        error = error.__cause__ or error.__context__
    provider_types = {"EvalError", "HttpFailure", "NetworkFailure", "FileProcessingFailure"}
    if any(type(error).__name__ in provider_types for error in chain):
        return True
    # The frozen inspector wraps exceptions in RuntimeError, retaining their
    # context. Truncated/malformed HTTP responses can escape its client's
    # network-error wrapper before any response JSON is saved.
    response_errors = (http.client.IncompleteRead, http.client.BadStatusLine, http.client.LineTooLong)
    if any(isinstance(error, response_errors) for error in chain):
        return True
    # V1 validates an external response after persisting it. Bad JSON/schema or
    # an incomplete/wrong-model response is a failed service observation.
    if any(Path(directory).glob("*/response-*.json")):
        return any(isinstance(error, ValueError | feedback.ValidationError) for error in chain)
    return False


def cleanup_uploads(directory, eval_repo):
    """Delete only media uploaded for this inspection, without changing its feedback."""
    gemini = load_external(Path(eval_repo) / "scripts/run_gemini_pairwise_eval.py")
    key = os.environ.get("GEMINI_API_KEY") or gemini.read_env_file(Path(eval_repo) / "arena/.env").get("GEMINI_API_KEY")
    if not key:
        return
    client = gemini.GeminiRestClient(key, max_attempts=1, request_timeout=15, file_timeout=15)
    for identity in Path(directory).glob("*/identity.json"):
        folder = identity.parent
        request = folder / "request-1.json"
        parts = read_json(request)["input"] if request.exists() else []
        for kind, (record, marker_name) in UPLOADS.items():
            marker = folder / marker_name
            if marker.exists() and read_json(marker).get("status") == "deleted":
                continue
            uploaded = folder / record
            name = read_json(uploaded).get("name") if uploaded.exists() else None
            if name is None:
                # A timed-out child may have persisted its request but not reached
                # V1's finally block that records the upload.
                uris = [part["uri"] for part in parts if "uri" in part and part.get("type", "video") == kind]
                parsed = urlparse(uris[0]) if uris else None
                if (
                    parsed
                    and parsed.hostname == "generativelanguage.googleapis.com"
                    and parsed.path.startswith("/v1beta/files/")
                ):
                    name = parsed.path.removeprefix("/v1beta/")
            if not name:
                continue
            try:
                client.delete_file(name)
                result = {"status": "deleted", "name": name}
            except Exception as error:
                result = {"status": "failed", "name": name, "error": gemini.safe_error_text(error, key)}
            write_json(marker, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    data = read_json(args.input)
    directory = Path(data["directory"])
    try:
        report = inspect_generation(
            data["video"], data["original_request"], directory, data["eval_repo"], data.get("source_image")
        )
        result = {"status": "succeeded", "report": report}
    except Exception as exc:
        result = {
            "status": "provider_error" if provider_failure(exc, directory) else "execution_error",
            "error": str(exc),
            "error_type": type(exc).__name__,
        }
    write_json(directory / "worker-result.json", result)
    cleanup_uploads(directory, data["eval_repo"])
    return 0 if result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
