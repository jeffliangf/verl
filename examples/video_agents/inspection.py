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
"""Standalone inspection-v2: categorical assessment without scoring or acceptance."""

import argparse
import json
import math
import os
import time
from pathlib import Path

from jsonschema import Draft202012Validator

from .storage import file_sha256, fingerprint, load_external, read_json, run_lock, write_json

ASSET_DIR = Path(__file__).with_name("inspection_v2")
SYSTEM_PROMPT_PATH = ASSET_DIR / "system_prompt.txt"
RESPONSE_SCHEMA_PATH = ASSET_DIR / "response_schema.json"
TOOL_SCHEMA_PATH = ASSET_DIR / "tool_schema.json"
RESPONSE_SCHEMA = read_json(RESPONSE_SCHEMA_PATH)
INPUT_SCHEMA = read_json(TOOL_SCHEMA_PATH)["function"]["parameters"]
DIMENSIONS = tuple(RESPONSE_SCHEMA["properties"]["dimensions"]["required"])
DEFAULT_SETTINGS = {
    "model": "gemini-3.8-flash",
    "fps": 4,
    "resolution": "high",
    "thinking_level": "medium",
    "seed": 0,
    "max_output_tokens": 12288,
}
PREVIEW_URI = "UPLOAD_URI_REPLACED_AFTER_UPLOADING_THE_SUPPLIED_VIDEO"
TASK = (
    "Inspect this one video and its audio using the system rubric. "
    "Return only schema_version, status, all eight dimension ratings, and grounded findings. "
    "Do not return a numerical score, pass/fail decision, ranking, or rewritten prompt."
)


def build_request(*, video_uri, original_request, generation_prompt, settings=None, system_prompt=None):
    """Build the exact Gemini Interactions request body, without credentials or network access."""
    config = {**DEFAULT_SETTINGS, **(settings or {})}
    if set(config) != set(DEFAULT_SETTINGS):
        raise ValueError("Unsupported inspection setting")
    for name, value in (("original_request", original_request), ("generation_prompt", generation_prompt)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")
    if not isinstance(video_uri, str) or not video_uri.strip():
        raise ValueError("A video URI is required")
    if not isinstance(config["fps"], int | float) or not math.isfinite(config["fps"]) or config["fps"] <= 0:
        raise ValueError("fps must be finite and positive")
    prompt = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8") if system_prompt is None else system_prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The system prompt must be a nonempty string")
    task = {
        "original_request": original_request,
        "rewritten_generation_prompt": generation_prompt,
        "task": TASK,
    }
    return {
        "model": config["model"],
        "system_instruction": prompt,
        "store": False,
        "background": False,
        "stream": False,
        "input": [
            {
                "type": "video",
                "uri": video_uri,
                "mime_type": "video/mp4",
                "processing": {"type": "static", "fps": config["fps"]},
                "resolution": config["resolution"],
            },
            {"type": "text", "text": json.dumps(task, ensure_ascii=False)},
        ],
        "response_format": {"type": "text", "mime_type": "application/json", "schema": RESPONSE_SCHEMA},
        "generation_config": {name: config[name] for name in ("thinking_level", "seed", "max_output_tokens")},
    }


def validate_result(result, duration=None):
    """Enforce the output contract, including cross-field consistency; never compute acceptance."""
    Draft202012Validator(RESPONSE_SCHEMA).validate(result)
    json.dumps(result, allow_nan=False)
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    ratings, findings = result["dimensions"], result["findings"]
    if result["status"] == "unassessable":
        if any(rating != "n/a" for rating in ratings.values()) or findings:
            raise ValueError("unassessable requires all dimensions n/a and no findings")
        return result
    if not any(rating in {"good", "fair", "bad"} for rating in ratings.values()):
        raise ValueError("status ok requires at least one assessed dimension")
    ids = set()
    for finding in findings:
        for field in ("id", "description", "impact", "repair_goal"):
            if not finding[field].strip():
                raise ValueError(f"Finding {field} must be nonempty")
        if finding["id"] in ids:
            raise ValueError("Finding IDs must be unique")
        ids.add(finding["id"])
        if ratings[finding["dimension"]] not in {"fair", "bad"}:
            raise ValueError("Findings must support a fair or bad dimension")
        start, end = finding["start_s"], finding["end_s"]
        if (start is None) != (end is None):
            raise ValueError("Both timestamps must be numeric or both null")
        if start is not None and (end < start or (duration is not None and end > duration)):
            raise ValueError("Finding timestamps must be ordered and within the clip")
    for dimension, rating in ratings.items():
        supporting = [finding for finding in findings if finding["dimension"] == dimension]
        if rating in {"fair", "bad"} and not supporting:
            raise ValueError(f"{dimension}: {rating} requires a supporting finding")
        if rating == "bad" and not any(finding["confidence"] == "high" for finding in supporting):
            raise ValueError(f"{dimension}: bad requires high-confidence evidence")
    return result


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
    """Upload one MP4, inspect it, save the exact request/response, and return only the v2 assessment.

    The first three arguments are agent inputs. Remaining arguments are server-owned configuration.
    Uses the existing local Gemini REST helper; does not register a tool in the training agent loop.
    """
    inputs = {
        "video_path": str(video_path),
        "original_request": original_request,
        "generation_prompt": generation_prompt,
    }
    Draft202012Validator(INPUT_SCHEMA).validate(inputs)
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("duration must be finite and positive")
    video = Path(video_path).expanduser().resolve(strict=True)
    if not video.is_file():
        raise ValueError("video_path must identify a file")
    prompt = Path(system_prompt_file or SYSTEM_PROMPT_PATH).read_text(encoding="utf-8")
    template = build_request(
        video_uri=PREVIEW_URI,
        original_request=original_request,
        generation_prompt=generation_prompt,
        settings=settings,
        system_prompt=prompt,
    )
    digest = file_sha256(video)
    identity = {"video_sha256": digest, "request_template": template, "duration": duration}
    request_id = fingerprint(identity)
    root = Path(output_dir or Path(__file__).resolve().parents[2] / "outputs/video_inspection_v2")
    directory = root / request_id
    with run_lock(directory):
        result_path = directory / "result.json"
        if result_path.exists():
            return validate_result(read_json(result_path), duration)
        write_json(directory / "input.json", {**inputs, "video_path": str(video)})
        write_json(directory / "identity.json", identity)
        repo = Path(eval_repo or Path.home() / "video-generation-eval")
        gemini = load_external(repo / "scripts/run_gemini_pairwise_eval.py")
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            key = gemini.read_env_file(repo / "arena/.env").get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Configured Gemini credential is unavailable")
        started = time.monotonic()
        try:
            client = gemini.GeminiRestClient(key, max_attempts=4, request_timeout=300, file_timeout=900)
            uploaded = client.upload_video(video)
            upload_seconds = time.monotonic() - started
            if uploaded["mime_type"] != "video/mp4":
                raise ValueError("The inspection input must be an MP4 video")
            if file_sha256(video) != digest:
                raise ValueError("The video changed during upload")
            request = {**template, "input": [{**template["input"][0], "uri": uploaded["uri"]}, template["input"][1]]}
            write_json(directory / "request.json", request)
            response, _ = client.request_json(
                "POST",
                gemini.API_BASE + "/interactions",
                payload=request,
                retry_network=False,
                context="Video inspection v2",
            )
            write_json(directory / "response.json", response)
            if response.get("status") != "completed":
                raise ValueError("Gemini inspection did not complete; see the saved response.json")
            result = validate_result(json.loads(gemini.interaction_output_text(response) or ""), duration)
            write_json(
                directory / "timing.json",
                {"upload_seconds": upload_seconds, "total_seconds": time.monotonic() - started},
            )
            write_json(result_path, result)
            return result
        except Exception as exc:
            raise RuntimeError(gemini.safe_error_text(exc, key)) from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, required=True, help="Exactly the three agent tool inputs")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preview-request", type=Path, help="Write the request body offline, with a placeholder upload URI"
    )
    mode.add_argument("--execute", action="store_true", help="Upload the supplied video and call Gemini")
    parser.add_argument("--model", default=DEFAULT_SETTINGS["model"])
    parser.add_argument("--fps", type=float, default=DEFAULT_SETTINGS["fps"])
    parser.add_argument("--system-prompt-file", type=Path, default=SYSTEM_PROMPT_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--eval-repo", type=Path)
    parser.add_argument("--duration", type=float, help="Optional known clip duration for validating finding timestamps")
    args = parser.parse_args()
    inputs = read_json(args.input_json)
    Draft202012Validator(INPUT_SCHEMA).validate(inputs)
    settings = {"model": args.model, "fps": args.fps}
    if args.preview_request:
        request = build_request(
            video_uri=PREVIEW_URI,
            original_request=inputs["original_request"],
            generation_prompt=inputs["generation_prompt"],
            settings=settings,
            system_prompt=args.system_prompt_file.read_text(encoding="utf-8"),
        )
        write_json(args.preview_request, request)
        print(f"Offline request preview: {args.preview_request}. Only the uploaded video URI is a placeholder.")
    else:
        result = inspect_video(
            **inputs,
            settings=settings,
            system_prompt_file=args.system_prompt_file,
            output_dir=args.output_dir,
            eval_repo=args.eval_repo,
            duration=args.duration,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
