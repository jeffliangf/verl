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
import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from .h3 import generation_slot
from .inspection_agent import feedback
from .observability import ProviderError, timed_stage
from .storage import load_external, read_json, write_json


def inspect_video(video, original_request, directory, protocol, settings, output_dir):
    directory = Path(directory)
    request_path = directory / "input.json"
    write_json(
        request_path,
        {
            "video": video,
            "original_request": original_request,
            "eval_repo": protocol["eval_repo"],
            "directory": str(directory),
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
                cleanup_uploads(directory, protocol["eval_repo"])
                raise ProviderError("inspection", "Gemini inspection exceeded its service deadline") from exc
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
        marker = folder / "cleanup.json"
        if marker.exists() and read_json(marker).get("status") == "deleted":
            continue
        uploaded = folder / "uploaded.json"
        name = read_json(uploaded).get("name") if uploaded.exists() else None
        if name is None and (folder / "request-1.json").exists():
            # A timed-out child may have persisted its request but not reached
            # V1's finally block that records uploaded.json.
            uri = read_json(folder / "request-1.json")["input"][0]["uri"]
            parsed = urlparse(uri)
            if parsed.hostname == "generativelanguage.googleapis.com" and parsed.path.startswith("/v1beta/files/"):
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
        report = feedback.inspect(data["video"], data["original_request"], [], directory, data["eval_repo"])
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
