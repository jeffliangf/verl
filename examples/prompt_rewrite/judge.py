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
import subprocess
import sys
from collections import Counter
from math import isfinite
from pathlib import Path

from .config import CANDIDATE, JUDGE_CONCURRENCY, REFERENCE_FIELDS
from .storage import file_sha256, freeze_json, item_key, read_json, write_json

ORDERS = ("original", "swapped")
REQUEST_TIMEOUT_SECONDS = 300
PASS_TIMEOUT_SECONDS = 1800


def swap_pair(pair: dict) -> dict:
    swapped = dict(pair)
    for field in ("model", "video_path"):
        swapped[field + "_A"], swapped[field + "_B"] = pair[field + "_B"], pair[field + "_A"]
    return swapped


def build_manifests(protocol: dict, candidate_videos: dict[str, dict]) -> dict[str, dict]:
    """Each matchup uses the original prompt and the same candidate video."""
    candidate = protocol.get("candidate_field", CANDIDATE)
    pairs = {}
    for sample in protocol["samples"]:
        for reference in protocol.get("reference_fields", REFERENCE_FIELDS):
            pair_id = item_key(sample["id"]) + "__vs_" + reference
            pair = {
                "prompt_id": sample["id"],
                "prompt": sample["prompt"],
                "source": sample["source"],
                "reference": reference,
                "model_A": candidate,
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
            "prompt_count": len(protocol["samples"]),
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


def command(run_dir: Path, protocol: dict, order: str, execute: bool, concurrency: int) -> list[str]:
    """Explicitly override the runner defaults to match the established comparison runs."""
    if order not in ORDERS or concurrency < 1:
        raise ValueError("Unknown order or invalid concurrency")
    settings = protocol["judge"]
    repo = Path(protocol["eval_repo"])
    args = [
        sys.executable,
        str(repo / "scripts/run_gemini_pairwise_eval.py"),
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
        str(concurrency),
        "--all",
        "--execute" if execute else "--dry-run",
    ]
    if settings["max_output_tokens"] is not None:
        args.extend(["--max-output-tokens", str(settings["max_output_tokens"])])
    return args


def run_judge(
    run_dir: Path,
    protocol: dict,
    execute: bool = False,
    concurrency: int = JUDGE_CONCURRENCY,
    *,
    neutral_on_failure: bool = False,
    request_timeout_seconds: float = REQUEST_TIMEOUT_SECONDS,
    pass_timeout_seconds: float = PASS_TIMEOUT_SECONDS,
):
    # Sequential orientations keep the stated concurrency bound across the whole run.
    # The upstream runner resumes by pair signature and records every successful call.
    failed = []
    if any(not isfinite(value) or value <= 0 for value in (request_timeout_seconds, pass_timeout_seconds)):
        raise ValueError("Gemini timeouts must be positive")
    if execute and neutral_on_failure:
        # Keep the original protocol and actual Gemini ledgers unchanged when
        # enabling the explicitly requested recovery policy on an existing run.
        freeze_json(
            run_dir / "judge/failure-policy.json",
            {
                "request_timeout_seconds": request_timeout_seconds,
                "pass_timeout_seconds": pass_timeout_seconds,
                "comparison_score_if_either_order_fails": 0.5,
            },
        )
    for order in ORDERS:
        args = command(run_dir, protocol, order, execute, concurrency)
        if not (execute and neutral_on_failure):
            result = subprocess.run(args, check=False)
            if result.returncode:
                failed.append(order)
            continue
        manifest_path = run_dir / "judge" / f"manifest-{order}.json"
        path = run_dir / "judge" / f"judgments-{order}.jsonl"
        manifest = read_json(manifest_path)
        record_fallbacks(path, manifest, manifest_path, protocol["judge"])
        completed, _ = read_judgments(path, manifest, manifest_path, protocol["judge"])
        pending = [key for key in manifest["pairs"] if key not in completed]
        if not pending:
            continue
        args.extend(
            [
                "--request-timeout",
                str(request_timeout_seconds),
                "--file-timeout",
                str(request_timeout_seconds),
                "--max-attempts",
                "1",
                "--max-errors",
                str(len(pending) + 1),
            ]
        )
        for pair_id in pending:
            args.extend(["--pair-id", pair_id])
        timed_out = False
        try:
            # The upstream runner uses threads. Killing and waiting for its
            # process also stops hung workers before we record missing results.
            subprocess.run(args, check=False, timeout=pass_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        record_fallbacks(path, manifest, manifest_path, protocol["judge"], timed_out=timed_out)
        completed, _ = read_judgments(path, manifest, manifest_path, protocol["judge"])
        if set(completed) != set(manifest["pairs"]):
            # A fatal setup/import/credential error is not a timed-out judgment.
            failed.append(order)
    if failed:
        raise RuntimeError(
            f"Incomplete judge passes: {', '.join(failed)}. Both orders were attempted; inspect the result ledgers "
            "and use report --allow-partial to report explicit coverage."
        )


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


def fallback_path(path: Path) -> Path:
    return path.with_name(path.name.replace("judgments-", "fallbacks-")).with_suffix(".json")


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


def record_fallbacks(path: Path, manifest: dict, manifest_path: Path, settings: dict, *, timed_out=False):
    """Persist explicit neutral decisions separately from real Gemini judgments."""
    completed, _ = read_judgments(path, manifest, manifest_path, settings)
    errors = {}
    digest = file_sha256(manifest_path)
    for row in _ledger_rows(path, Counter()):
        if row.get("status") == "error":
            validate_identity(row, manifest, digest, settings)
            errors[row["pair_id"]] = row
    target = fallback_path(path)
    records = read_json(target) if target.exists() else {}
    for pair_id, pair in manifest["pairs"].items():
        if pair_id in completed or (pair_id not in errors and not timed_out):
            continue
        error = errors.get(pair_id, {})
        records[pair_id] = {
            "status": "fallback",
            "pair_id": pair_id,
            "prompt_id": pair["prompt_id"],
            "model_A": pair["model_A"],
            "model_B": pair["model_B"],
            "manifest_sha256": digest,
            "evaluation_signature": evaluation_signature(pair, manifest["instructions_how_to_compare"]),
            **settings,
            "fallback_score": 0.5,
            "reason": error.get("error", "Judging pass exceeded its hard timeout before returning a judgment"),
            "error_type": error.get("error_type", "TimeoutExpired"),
        }
    if records:
        write_json(target, records)


def read_judgments(path: Path, manifest: dict, manifest_path: Path, settings: dict) -> tuple[dict, dict]:
    """Reject stale, mixed, duplicate, or misoriented completed judgments.

    Only explicitly recorded fallback decisions can supply neutral scores.
    Missing/error rows alone remain incomplete. Real Gemini judgments are never
    fabricated or overwritten; a later valid judgment takes precedence.
    """
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
    target = fallback_path(path)
    if target.exists():
        policy = read_json(path.parent / "failure-policy.json")
        if policy.get("comparison_score_if_either_order_fails") != 0.5:
            raise ValueError("Neutral fallback policy is missing or incompatible")
        for pair_id, row in read_json(target).items():
            validate_identity(row, manifest, manifest_hash, settings)
            if row.get("pair_id") != pair_id or row.get("status") != "fallback" or row.get("fallback_score") != 0.5:
                raise ValueError(f"Invalid neutral fallback: {pair_id}")
            if pair_id not in completed:
                completed[pair_id] = row
                counts["fallback"] += 1
    return completed, dict(counts)
