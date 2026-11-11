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
"""Evidence-first video inspection; dimension ratings are derived from grounded findings."""

import argparse
import json
import math
import os
import time
from pathlib import Path

from jsonschema import Draft202012Validator

from . import inspection
from .storage import file_sha256, fingerprint, load_external, read_json, run_lock, write_json

ASSET_DIR = Path(__file__).with_name("inspection_v4")
SYSTEM_PROMPT_PATH = ASSET_DIR / "system_prompt.txt"
RESPONSE_SCHEMA_PATH = ASSET_DIR / "response_schema.json"
RESPONSE_SCHEMA = read_json(RESPONSE_SCHEMA_PATH)
DIMENSIONS = inspection.DIMENSIONS
DEFAULT_SETTINGS = inspection.DEFAULT_SETTINGS
SEVERITY = {"minor": 1, "major": 2, "critical": 3}


def build_request(*, video_uri, original_request, generation_prompt, settings=None, system_prompt=None):
    """Only the video, original request and actual generation prompt enter the assessment."""
    request = inspection.build_request(
        video_uri=video_uri,
        original_request=original_request,
        generation_prompt=generation_prompt,
        settings=settings,
        system_prompt=SYSTEM_PROMPT_PATH.read_text() if system_prompt is None else system_prompt,
    )
    task = json.loads(request["input"][1]["text"])
    task["task"] = (
        "Inspect this one video and its audio according to the system rubric. Return only the specified evidence JSON."
    )
    request["input"][1]["text"] = json.dumps(task, ensure_ascii=False)
    request["response_format"]["schema"] = RESPONSE_SCHEMA
    return request


def validate_evidence(result, duration=None):
    """Provider errors and inconsistent evidence remain errors, never quality ratings."""
    Draft202012Validator(RESPONSE_SCHEMA).validate(result)
    json.dumps(result, allow_nan=False)
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    assessed = set(result["assessed_dimensions"])
    if result["status"] == "unassessable":
        if assessed or result["findings"]:
            raise ValueError("unassessable requires no assessed dimensions or findings")
        return result
    if not assessed:
        raise ValueError("ok requires at least one assessed dimension")
    ids = set()
    for finding in result["findings"]:
        for field in ("id", "description", "impact", "repair_goal"):
            if not finding[field].strip():
                raise ValueError(f"Finding {field} must be nonempty")
        if finding["id"] in ids:
            raise ValueError("Finding IDs must be unique")
        ids.add(finding["id"])
        if finding["dimension"] not in assessed:
            raise ValueError("A finding must belong to an assessed dimension")
        start, end = finding["start_s"], finding["end_s"]
        if (start is None) != (end is None):
            raise ValueError("Both timestamps must be numeric or both null")
        if start is not None and (end < start or (duration is not None and end > duration)):
            raise ValueError("Finding timestamps must be ordered and within the clip")
    return result


def normalize(result, duration=None):
    """Derive good/fair/bad/n/a, retaining every original finding and its severity/confidence."""
    validate_evidence(result, duration)
    grades = dict.fromkeys(DIMENSIONS, "n/a")
    for dimension in result["assessed_dimensions"]:
        severity = max((SEVERITY[f["severity"]] for f in result["findings"] if f["dimension"] == dimension), default=0)
        grades[dimension] = "bad" if severity >= 2 else "fair" if severity == 1 else "good"
    return {
        "schema_version": "inspection-v4",
        "status": result["status"],
        "dimensions": grades,
        "findings": result["findings"],
    }


def inspect_video(
    video_path,
    original_request,
    generation_prompt,
    *,
    output_dir=None,
    eval_repo=None,
    settings=None,
    system_prompt_file=None,
    duration=None,
):
    """Upload and inspect one video; return categorical dimensions and the observed evidence.

    The caller applies the separately versioned acceptance policy. No N, tournament information,
    target label or expected answer is an input to this tool.
    """
    inputs = {
        "video_path": str(video_path),
        "original_request": original_request,
        "generation_prompt": generation_prompt,
    }
    Draft202012Validator(inspection.INPUT_SCHEMA).validate(inputs)
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    video = Path(video_path).expanduser().resolve(strict=True)
    if not video.is_file():
        raise ValueError("video_path must identify a file")
    policy = Path(system_prompt_file or SYSTEM_PROMPT_PATH).read_text()
    template = build_request(
        video_uri=inspection.PREVIEW_URI,
        original_request=original_request,
        generation_prompt=generation_prompt,
        settings=settings,
        system_prompt=policy,
    )
    digest = file_sha256(video)
    identity = {"video_sha256": digest, "request_template": template, "duration": duration}
    directory = Path(output_dir or Path(__file__).resolve().parents[2] / "outputs/video_inspection_v4") / fingerprint(
        identity
    )
    with run_lock(directory):
        cached = directory / "evidence.json"
        if cached.exists():
            return normalize(read_json(cached), duration)
        write_json(directory / "input.json", {**inputs, "video_path": str(video)})
        write_json(directory / "identity.json", identity)
        repo = Path(eval_repo or Path.home() / "video-generation-eval")
        gemini = load_external(repo / "scripts/run_gemini_pairwise_eval.py")
        key = os.environ.get("GEMINI_API_KEY") or gemini.read_env_file(repo / "arena/.env").get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Configured Gemini credential is unavailable")
        began = time.monotonic()
        try:
            client = gemini.GeminiRestClient(key, max_attempts=4, request_timeout=300, file_timeout=900)
            uploaded = client.upload_video(video)
            upload_seconds = time.monotonic() - began
            if uploaded["mime_type"] != "video/mp4" or file_sha256(video) != digest:
                raise ValueError("Uploaded media type or video checksum changed")
            request = {**template, "input": [{**template["input"][0], "uri": uploaded["uri"]}, template["input"][1]]}
            write_json(directory / "request.json", request)
            response, _ = client.request_json(
                "POST",
                gemini.API_BASE + "/interactions",
                payload=request,
                retry_network=False,
                context="Evidence video inspection",
            )
            write_json(directory / "response.json", response)
            if response.get("model") != request["model"] or response.get("status") != "completed":
                raise ValueError("Gemini returned a different model or incomplete interaction")
            evidence = validate_evidence(json.loads(gemini.interaction_output_text(response)), duration)
            result = normalize(evidence, duration)
            write_json(directory / "evidence.json", evidence)
            write_json(directory / "result.json", result)
            write_json(
                directory / "timing.json", {"upload_seconds": upload_seconds, "total_seconds": time.monotonic() - began}
            )
            return result
        except Exception as error:
            raise RuntimeError(gemini.safe_error_text(error, key)) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preview-request", type=Path)
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-repo", type=Path)
    parser.add_argument("--system-prompt-file", type=Path, default=SYSTEM_PROMPT_PATH)
    parser.add_argument("--model", default=DEFAULT_SETTINGS["model"])
    parser.add_argument("--duration", type=float)
    args = parser.parse_args()
    inputs = read_json(args.input_json)
    Draft202012Validator(inspection.INPUT_SCHEMA).validate(inputs)
    if args.preview_request:
        request = build_request(
            video_uri=inspection.PREVIEW_URI,
            original_request=inputs["original_request"],
            generation_prompt=inputs["generation_prompt"],
            settings={"model": args.model},
            system_prompt=args.system_prompt_file.read_text(),
        )
        write_json(args.preview_request, request)
        print(str(args.preview_request))
    else:
        result = inspect_video(
            **inputs,
            output_dir=args.output_dir,
            eval_repo=args.eval_repo,
            system_prompt_file=args.system_prompt_file,
            settings={"model": args.model},
            duration=args.duration,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
