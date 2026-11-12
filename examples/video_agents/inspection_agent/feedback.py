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
"""Evidence and repair feedback without acceptance or stopping decisions."""

import json
import math
import time
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from examples.video_agents.storage import file_sha256, fingerprint, load_external, read_json, run_lock, write_json

PROMPT_PATH = Path(__file__).with_name("inspector_prompt.txt")
SETTINGS = {
    "model": "gemini-3.8-flash",
    "fps": 4,
    "resolution": "high",
    "thinking_level": "medium",
    "seed": 0,
    "max_output_tokens": 12288,
}
CATEGORIES = [
    "visual_artifacts",
    "temporal_consistency",
    "motion",
    "physics",
    "audio_quality",
    "audio_visual_alignment",
    "prompt_alignment",
    "framing_camera",
    "style",
    "other",
]


def obj(properties):
    return {"type": "object", "additionalProperties": False, "properties": properties, "required": list(properties)}


def string(**kwargs):
    return {"type": "string", "minLength": 1, **kwargs}


def array(items, **kwargs):
    return {"type": "array", "items": items, **kwargs}


TIMESTAMP = {"type": ["number", "null"], "minimum": 0}
FINDING = obj(
    {
        "id": string(),
        "category": string(enum=CATEGORIES),
        "severity": string(enum=["minor", "major"]),
        "description": string(),
        "evidence": string(),
        "impact": string(),
        "repair_goal": string(),
        "start_s": TIMESTAMP,
        "end_s": TIMESTAMP,
        "confidence": string(enum=["medium", "high"]),
    }
)
SCHEMA = obj(
    {
        "schema_version": string(enum=["native-inspection-feedback-v1"]),
        "status": string(enum=["ok", "unassessable"]),
        "assessment_note": string(),
        "audio_assessed": {"type": "boolean"},
        "observed_sequence": array(
            obj(
                {
                    "phase": string(enum=["beginning", "main_event", "ending"]),
                    "observation": string(),
                    "start_s": TIMESTAMP,
                    "end_s": TIMESTAMP,
                }
            ),
            maxItems=3,
        ),
        "successful_elements_to_preserve": array(string(), maxItems=4),
        "findings": array(FINDING),
        "primary_repair_targets": array(string(), maxItems=2, uniqueItems=True),
        "prior_issue_checks": array(
            obj(
                {
                    "prior_issue_id": string(),
                    "state": string(enum=["resolved", "persistent", "unassessable"]),
                    "current_finding_id": {"type": ["string", "null"]},
                    "evidence": string(),
                }
            )
        ),
        "uncertain_observations": array(string()),
    }
)


def validate_feedback(value, prior_findings=(), duration=10.125):
    Draft202012Validator(SCHEMA).validate(value)
    json.dumps(value, allow_nan=False)
    ids = [finding["id"] for finding in value["findings"]]
    if len(ids) != len(set(ids)) or not set(value["primary_repair_targets"]).issubset(ids):
        raise ValueError("Finding IDs must be unique and repair targets must refer to them")
    checks = [item["prior_issue_id"] for item in value["prior_issue_checks"]]
    if len(checks) != len(set(checks)) or set(checks) != {item["id"] for item in prior_findings}:
        raise ValueError("Explicitly recheck every supplied prior issue exactly once")
    for check in value["prior_issue_checks"]:
        if check["state"] == "persistent":
            if check["current_finding_id"] not in ids:
                raise ValueError("A persistent prior issue must map to an explicit current finding")
        elif check["current_finding_id"] is not None:
            raise ValueError("Resolved or unassessable rechecks have no confirmed current finding")
    if value["status"] == "unassessable":
        if value["findings"] or value["successful_elements_to_preserve"]:
            raise ValueError("Unassessable media cannot supply confirmed defects or strengths")
        if any(item["state"] != "unassessable" for item in value["prior_issue_checks"]):
            raise ValueError("An unassessable report cannot confirm a prior issue state")
    for item in value["findings"] + value["observed_sequence"]:
        start, end = item["start_s"], item["end_s"]
        if (start is None) != (end is None):
            raise ValueError("Timestamps must both be null or numeric")
        if start is not None and (not math.isfinite(start) or not math.isfinite(end) or end < start or end > duration):
            raise ValueError("Timestamps must be finite, ordered, and within the media")
    return value


def build_request(video_uri, original_request, generation_prompt, prior_findings=(), settings=None):
    config = {**SETTINGS, **(settings or {})}
    task = {
        "original_request": original_request,
        "generation_prompt": generation_prompt,
        "prior_findings_to_recheck": list(prior_findings),
        "task": "Return observed evidence and repair feedback for this video only.",
    }
    return {
        "model": config["model"],
        "system_instruction": PROMPT_PATH.read_text(),
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
        "response_format": {"type": "text", "mime_type": "application/json", "schema": SCHEMA},
        "generation_config": {k: config[k] for k in ("thinking_level", "seed", "max_output_tokens")},
    }


def inspect(video, original_request, prior_findings, directory, eval_repo):
    """Inspect one registered artifact; durable cache includes media, policy, schema and rechecks."""
    path = Path(video["path"])
    if file_sha256(path) != video["sha256"]:
        raise ValueError("Registered video bytes changed")
    template = build_request("MEDIA_URI", original_request, video["request"]["prompt"], prior_findings)
    identity = {"video_sha256": video["sha256"], "request": template}
    directory = Path(directory) / fingerprint(identity)
    with run_lock(directory):
        write_json(directory / "identity.json", identity)
        if (directory / "result.json").exists():
            return validate_feedback(read_json(directory / "result.json"), prior_findings)
        gemini = load_external(Path(eval_repo) / "scripts/run_gemini_pairwise_eval.py")
        import os

        key = os.environ.get("GEMINI_API_KEY") or gemini.read_env_file(Path(eval_repo) / "arena/.env").get(
            "GEMINI_API_KEY"
        )
        if not key:
            raise RuntimeError("Configured Gemini credential is unavailable")
        client = gemini.GeminiRestClient(key, max_attempts=4, request_timeout=300, file_timeout=900)
        start = time.monotonic()
        uploaded = None
        try:
            uploaded = client.upload_video(path)
            request = {**template, "input": [{**template["input"][0], "uri": uploaded["uri"]}, template["input"][1]]}
            for attempt in range(1, 4):
                write_json(directory / f"request-{attempt}.json", request)
                response, _ = client.request_json(
                    "POST",
                    gemini.API_BASE + "/interactions",
                    payload=request,
                    retry_network=False,
                    context="Native video inspection",
                )
                write_json(directory / f"response-{attempt}.json", response)
                if response.get("model") != SETTINGS["model"] or response.get("status") != "completed":
                    raise ValueError("Unexpected model or incomplete inspection")
                try:
                    result = validate_feedback(json.loads(gemini.interaction_output_text(response)), prior_findings)
                except (ValueError, ValidationError) as error:
                    if attempt == 3:
                        raise
                    request = {
                        **request,
                        "input": request["input"][:2]
                        + [{"type": "text", "text": "Correct the response format: " + str(error)[:800]}],
                    }
                    continue
                write_json(directory / "result.json", result)
                write_json(directory / "timing.json", {"total_seconds": time.monotonic() - start, "attempts": attempt})
                return result
        except Exception as error:
            raise RuntimeError(gemini.safe_error_text(error, key)) from None
        finally:
            # Let the service expire uploaded media. Avoid serial cleanup latency on the policy path.
            if uploaded:
                write_json(directory / "uploaded.json", uploaded)
