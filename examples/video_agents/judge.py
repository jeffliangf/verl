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
"""Adapt frozen video pairs to the existing Gemini evaluation runner."""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from .config import CANDIDATE, REFERENCE_FIELDS
from .storage import file_sha256, freeze_json, item_key, read_json

ORDERS = ("original", "swapped")
EXTERNAL_FILES = (
    "scripts/run_gemini_pairwise_eval.py",
    "scripts/build_pairwise_manifest.py",
    "bestofn/normalize.py",
)


def swap_pair(pair: dict) -> dict:
    swapped = dict(pair)
    for field in ("model", "video_path"):
        swapped[field + "_A"], swapped[field + "_B"] = pair[field + "_B"], pair[field + "_A"]
    return swapped


def build_manifests(protocol: dict, candidate_videos: dict[str, dict]) -> dict[str, dict]:
    """Each matchup uses the original prompt and the same candidate video."""
    pairs = {}
    for sample in protocol["samples"]:
        if sample["id"] not in candidate_videos:
            continue
        for reference in REFERENCE_FIELDS:
            pair_id = item_key(sample["id"]) + "__vs_" + reference
            pair = {
                "prompt_id": sample["id"],
                "prompt": sample["prompt"],
                "source": sample["source"],
                "reference": reference,
                "model_A": CANDIDATE,
                "model_B": reference,
                "video_path_A": candidate_videos[sample["id"]]["path"],
                "video_path_B": sample["references"][reference]["path"],
            }
            if hashlib.sha256(pair_id.encode()).digest()[0] % 2:
                pair = swap_pair(pair)
            pairs[pair_id] = pair
    base = {
        "schema_version": 1,
        "instructions_version": protocol["judge"]["instructions_version"],
        "instructions_how_to_compare": protocol["rubric"],
        "metadata": {
            "pair_count": len(pairs),
            "prompt_count": len(candidate_videos),
            "excluded_models": [],
            "candidate_label": protocol["name"],
        },
    }
    return {
        "original": {**base, "pairs": pairs},
        "swapped": {**base, "pairs": {key: swap_pair(pair) for key, pair in pairs.items()}},
    }


def freeze_manifests(run_dir: Path, manifests: dict[str, dict]):
    for order, manifest in manifests.items():
        freeze_json(run_dir / "judge" / f"manifest-{order}.json", manifest)


def command(run_dir: Path, protocol: dict, order: str, pair_ids, *, allow_fallback: bool = False) -> list[str]:
    """Explicitly override the runner defaults to match the established comparison runs."""
    if order not in ORDERS:
        raise ValueError("Unknown presentation order")
    settings, timeouts = protocol["judge"], protocol["timeouts"]
    repo = Path(protocol["eval_repo"])
    args = [
        sys.executable,
        "-m",
        "examples.video_agents.judge_worker",
        "--runner",
        str(repo / "scripts/run_gemini_pairwise_eval.py"),
        "--timings",
        str(run_dir / "judge" / f"timings-{order}.jsonl"),
        "--manifest",
        str(run_dir / "judge" / f"manifest-{order}.json"),
        "--output",
        str(run_dir / "judge" / f"judgments-{order}.jsonl"),
        "--env-file",
        str(repo / "arena/.env"),
        "--model",
        settings["judge_model"],
        "--thinking-level",
        settings["thinking_level"],
        "--video-processing",
        settings["video_processing"],
        "--seed",
        str(settings["seed"]),
        "--concurrency",
        "1",
        "--request-timeout",
        str(timeouts["judge_request"]),
        "--file-timeout",
        str(timeouts["judge_file"]),
        "--file-control-timeout",
        str(timeouts["judge_control"]),
        "--max-attempts",
        str(protocol["judge_max_attempts"]),
        # Training can keep judging other pairs after an exhausted comparison.
        "--max-errors",
        str(max(1, len(pair_ids)) if allow_fallback else 1),
        "--all",
        "--execute",
    ]
    if settings["max_output_tokens"] is not None:
        args.extend(["--max-output-tokens", str(settings["max_output_tokens"])])
    for pair_id in sorted(pair_ids):
        args.extend(["--pair-id", pair_id])
    return args


def evaluation_signature(pair: dict, instructions: str) -> str:
    """The upstream runner's signature format (also checked in the integration test)."""
    payload = json.dumps(
        {"pair": pair, "instructions_how_to_compare": instructions},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _ledger_rows(path: Path, counts: Counter):
    """Preserve interrupted lines and match the upstream runner's resume behavior."""
    if not path.exists():
        return
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                counts["malformed_lines"] += 1
                continue
            if not isinstance(row, dict):
                raise ValueError(f"Expected a judgment object at {path}:{line_number}")
            yield row


def validate_identity(row: dict, manifest: dict, manifest_hash: str, settings: dict) -> dict:
    pair_id = row.get("pair_id")
    pair = manifest["pairs"].get(pair_id)
    if pair is None:
        raise ValueError(f"Unknown judgment pair: {pair_id}")
    if row.get("manifest_sha256") != manifest_hash or row.get("evaluation_signature") != evaluation_signature(
        pair, manifest["instructions_how_to_compare"]
    ):
        raise ValueError(f"Judgment does not match the frozen manifest: {pair_id}")
    if any(key not in row or row[key] != value for key, value in settings.items()):
        raise ValueError(f"Judgment uses different Gemini settings: {pair_id}")
    if (row.get("model_A"), row.get("model_B")) != (pair["model_A"], pair["model_B"]):
        raise ValueError(f"Judgment has different A/B identities: {pair_id}")
    if row.get("prompt_id") != pair["prompt_id"]:
        raise ValueError(f"Judgment has a different prompt ID: {pair_id}")
    return pair


def read_judgments(path: Path, manifest: dict, manifest_path: Path, settings: dict) -> tuple[dict, dict]:
    """Read real completed judgments; missing/error rows remain incomplete."""
    completed, counts = {}, Counter()
    manifest_hash = file_sha256(manifest_path)
    for row in _ledger_rows(path, counts):
        counts[row.get("status", "unknown")] += 1
        if row.get("status") != "completed":
            continue
        pair_id = row.get("pair_id")
        pair = validate_identity(row, manifest, manifest_hash, settings)
        if pair_id in completed:
            raise ValueError(f"Unknown or duplicate completed pair in {path}: {pair_id}")
        preferred = row.get("preferred_video")
        if preferred not in ("A", "B") or row.get("preferred_model") != pair[f"model_{preferred}"]:
            raise ValueError(f"Invalid preferred video/model in {pair_id}")
        completed[pair_id] = row
    return completed, dict(counts)


def judgment_errors(path: Path, pair_ids) -> dict[str, str]:
    """The latest recorded error for each listed pair, for failure reports."""
    errors = {}
    for row in _ledger_rows(path, Counter()):
        if row.get("status") == "error" and row.get("pair_id") in pair_ids:
            errors[row["pair_id"]] = f"{row.get('stage')}: {row.get('error')}"
    return errors


def fallback_reason(row: dict) -> str:
    """Allow unscorable provider responses; explicit input, auth, and code errors stay fatal."""
    error_type, message = row.get("error_type"), row.get("error", "")
    if error_type in {"NetworkFailure", "FileProcessingFailure"}:
        return "provider_failure"
    if error_type == "HttpFailure":
        match = re.match(r"HTTP (\d{3}):\s*(.*)", message, re.DOTALL)
        if match:
            status, detail = int(match[1]), match[2]
            if status in {408, 429, 500, 502, 503, 504}:
                return "provider_failure"
            if status == 400:
                try:
                    error = json.loads(detail).get("error", {})
                    if isinstance(error, dict) and error.get("code") == "prohibited_content":
                        return "content_refusal"
                    # Individual interactions can fail with this generic response
                    # while the other comparisons of the same videos succeed.
                    if row.get("stage") == "interaction" and error == {
                        "code": "invalid_request",
                        "message": "Request contains an invalid argument.",
                    }:
                        return "judge_request_rejected"
                except (ValueError, AttributeError):
                    pass
    if error_type == "EvalError" and row.get("stage") == "interaction":
        return "invalid_judge_response"
    if error_type == "EvalError" and message.startswith("Timed out waiting for Gemini file "):
        return "provider_failure"
    raise RuntimeError(f"Nonrecoverable Gemini error ({error_type}): {message}")


def read_failure_rows(path: Path, manifest: dict, manifest_path: Path, settings: dict) -> dict:
    """Read identity-checked errors before deciding whether fallback is allowed."""
    errors = {}
    manifest_hash = file_sha256(manifest_path)
    for row in _ledger_rows(path, Counter()):
        if row.get("status") == "error":
            validate_identity(row, manifest, manifest_hash, settings)
            errors[row["pair_id"]] = row
    return errors


def read_fallbacks(path: Path, manifest: dict, manifest_path: Path, settings: dict) -> dict:
    """Recover explicit neutral decisions without treating them as Gemini votes."""
    if not path.exists():
        return {}
    rows = read_json(path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected a fallback ledger at {path}")
    result = {}
    manifest_hash = file_sha256(manifest_path)
    for row in rows:
        validate_identity(row, manifest, manifest_hash, settings)
        key = row["pair_id"]
        if (
            key in result
            or row.get("status") != "fallback"
            or row.get("fallback_policy") != "neutral-v1"
            or row.get("fallback_score") != 0.5
            or row.get("preferred_model") is not None
            or not row.get("fallback_reason")
        ):
            raise ValueError(f"Invalid or duplicate neutral fallback: {path}: {key}")
        result[key] = row
    return result


def neutral_fallback(key: str, manifest: dict, manifest_path: Path, settings: dict, reason: str, error=None) -> dict:
    """Bind an imputed half vote to the same identity as the missing real judgment."""
    pair = manifest["pairs"][key]
    return {
        **settings,
        **{name: pair[name] for name in ("prompt_id", "model_A", "model_B")},
        "pair_id": key,
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_signature": evaluation_signature(pair, manifest["instructions_how_to_compare"]),
        "status": "fallback",
        "fallback_policy": "neutral-v1",
        "fallback_score": 0.5,
        "fallback_reason": reason,
        "error": error,
    }
