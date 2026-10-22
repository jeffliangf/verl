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
"""Read, import, and validate legacy rewrite artifacts; no policy inference."""

from dataclasses import dataclass
from pathlib import Path

from .config import build_messages
from .storage import fingerprint, freeze_json, item_key, read_json, write_jsonl


@dataclass(frozen=True)
class RewriteResult:
    text: str
    finish_reason: str = "stop"
    output_tokens: int | None = None


def validate_rewrite(result: RewriteResult):
    if not isinstance(result.text, str) or not result.text.strip():
        raise ValueError("Model produced an empty rewrite")
    if result.finish_reason == "length":
        raise ValueError("Rewrite was truncated; increase max_new_tokens in a new run")
    if result.finish_reason not in {"stop", "eos", "provided"}:
        raise ValueError(f"Rewrite did not complete normally: {result.finish_reason}")
    if "<think>" in result.text or "</think>" in result.text:
        raise ValueError("Rewrite includes reasoning tags; check that thinking is disabled")


def rewrite_identity(sample: dict, protocol: dict) -> str:
    return fingerprint(
        {
            "id": sample["id"],
            "messages": build_messages(
                sample["prompt"], protocol["system_prompt"], protocol.get("rewrite_instruction", "")
            ),
            "rewriter": protocol["rewriter"],
        }
    )


def record_path(run_dir: Path, item_id: str) -> Path:
    return run_dir / "rewrites" / (item_key(item_id) + ".json")


def save_rewrite(run_dir: Path, protocol: dict, sample: dict, result: RewriteResult):
    validate_rewrite(result)
    record = {
        "id": sample["id"],
        "prompt": sample["prompt"],
        "source": sample["source"],
        "rewritten_prompt": result.text,
        "finish_reason": result.finish_reason,
        "output_tokens": result.output_tokens,
        "request_sha256": rewrite_identity(sample, protocol),
        "text_sha256": fingerprint(result.text),
    }
    freeze_json(record_path(run_dir, sample["id"]), record)


def load_rewrites(run_dir: Path, protocol: dict, require_complete: bool = True) -> dict[str, dict]:
    records = {}
    for sample in protocol["samples"]:
        path = record_path(run_dir, sample["id"])
        if not path.exists():
            continue
        record = read_json(path)
        if (
            record.get("id") != sample["id"]
            or record.get("prompt") != sample["prompt"]
            or record.get("request_sha256") != rewrite_identity(sample, protocol)
            or record.get("text_sha256") != fingerprint(record.get("rewritten_prompt"))
        ):
            raise ValueError(f"Rewrite record differs from the frozen input: {path}")
        validate_rewrite(RewriteResult(record["rewritten_prompt"], record["finish_reason"]))
        records[sample["id"]] = record
    if require_complete and len(records) != len(protocol["samples"]):
        raise ValueError(f"Only {len(records)}/{len(protocol['samples'])} rewrites are complete")
    return records


def export_rewrites(run_dir: Path, protocol: dict) -> dict[str, dict]:
    records = load_rewrites(run_dir, protocol)
    write_jsonl(run_dir / "rewrites.jsonl", records.values())
    return records


def import_rewrites(run_dir: Path, protocol: dict, rows: list[dict]):
    """Import ``id, rewritten_prompt`` rows from an existing SFT/RL rollout.

    IDs must cover this exact evaluation cohort. An optional prompt field is
    checked against the original prompt; supplied truncation markers are honored.
    """
    samples = {sample["id"]: sample for sample in protocol["samples"]}
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item_id, str) for item_id in ids) or len(set(ids)) != len(ids) or set(ids) != set(samples):
        raise ValueError("Imported rewrites must contain each eval ID exactly once")
    results = []
    for row in rows:
        if "prompt" in row and row["prompt"] != samples[row["id"]]["prompt"]:
            raise ValueError(f"Imported prompt differs from the eval prompt: {row['id']}")
        result = RewriteResult(
            row.get("rewritten_prompt"), row.get("finish_reason", "provided"), row.get("output_tokens")
        )
        validate_rewrite(result)
        results.append(result)
    freeze_json(run_dir / "rewrite-source.json", {"mode": "import", "input_sha256": fingerprint(rows)})
    for row, result in zip(rows, results, strict=True):
        save_rewrite(run_dir, protocol, samples[row["id"]], result)
    return export_rewrites(run_dir, protocol)
