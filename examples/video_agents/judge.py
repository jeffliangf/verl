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
import sys
from collections import Counter
from pathlib import Path

from .config import CANDIDATE, REFERENCE_FIELDS
from .storage import file_sha256, freeze_json, item_key

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


def command(run_dir: Path, protocol: dict, order: str) -> list[str]:
    """Explicitly override the runner defaults to match the established comparison runs."""
    if order not in ORDERS:
        raise ValueError("Unknown presentation order")
    settings = protocol["judge"]
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
        "--all",
        "--execute",
    ]
    if settings["max_output_tokens"] is not None:
        args.extend(["--max-output-tokens", str(settings["max_output_tokens"])])
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
    fallback = path.with_name(path.name.replace("judgments-", "fallbacks-")).with_suffix(".json")
    if fallback.exists():
        raise ValueError("Legacy fallback ledgers cannot be used as video-agent rewards")
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


def read_policy_refusals(path: Path, manifest: dict, manifest_path: Path, settings: dict, completed: dict) -> list:
    """Identify explicit policy refusals, without interpreting other failures as ties."""
    refusals = []
    manifest_hash = file_sha256(manifest_path)
    for row in _ledger_rows(path, Counter()):
        if (row.get("status"), row.get("stage"), row.get("error_type")) != ("error", "interaction", "HttpFailure"):
            continue
        error = row.get("error")
        if not isinstance(error, str):
            continue
        status, _, body = error.partition(": ")
        if status not in ("HTTP 400", "HTTP 403"):
            continue
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("error"), dict):
            continue
        if payload["error"].get("code") != "prohibited_content":
            continue
        # Error rows need the same identity checks as successful votes. A stale
        # refusal superseded by a real completed judgment is no longer pending.
        validate_identity(row, manifest, manifest_hash, settings)
        if row["pair_id"] not in completed:
            refusals.append(row)
    return refusals
