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
"""Read evaluation samples and verify the provenance of both reference videos."""

from pathlib import Path

from .config import H3_ENDPOINT, H3_SETTINGS, REFERENCE_FIELDS, REFERENCE_VIDEO_FIELDS
from .h3 import media_profile, request_body
from .storage import file_sha256, fingerprint, read_json, read_jsonl


def load_samples(path: Path, limit: int | None = None) -> list[dict]:
    """Validate IDs/text without modifying the user's dataset."""
    rows, seen = read_jsonl(path), set()
    if not rows or (limit is not None and not 0 < limit <= len(rows)):
        raise ValueError("Dataset must be nonempty and limit must be within its row count")
    for row in rows:
        for field in ("id", "prompt", "h3_context_ir"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing nonempty {field} in eval row {row.get('id')}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate eval ID: {row['id']}")
        if "source" in row and (not isinstance(row["source"], str) or not row["source"].strip()):
            raise ValueError(f"Invalid source in eval row {row['id']}")
        seen.add(row["id"])
    return rows[:limit]


def reference_inventory(rows: list[dict], reference_dir: Path, probe) -> tuple[list[dict], dict, str]:
    """Verify request texts, completed state, bytes, checksums, and media formats.

    The source file hash may change when video paths are attached to JSONL rows;
    actual request equality is the authority for matching existing videos.
    """
    manifest = read_json(reference_dir / "manifest.json")
    if (
        manifest.get("endpoint") != H3_ENDPOINT
        or manifest.get("settings") != H3_SETTINGS
        or manifest.get("context_ir_rewriter_enabled") is not False
    ):
        raise ValueError("Reference videos use different H3 generation settings")
    tasks = {}
    for task in manifest["tasks"]:
        key = (task["item_id"], task["field"])
        if key in tasks:
            raise ValueError(f"Duplicate reference task: {key}")
        tasks[key] = task
    samples, common_profile, common_host = [], None, None
    for row in rows:
        references = {}
        for field in REFERENCE_FIELDS:
            task = tasks.get((row["id"], field))
            if task is None or task["request"] != request_body(row[field]):
                raise ValueError(f"Reference request differs from dataset: {row['id']}/{field}")
            video_path = (reference_dir / task["output"]).resolve()
            video_field = REFERENCE_VIDEO_FIELDS[field]
            if not isinstance(row.get(video_field), str) or not row[video_field].strip():
                raise ValueError(f"Missing reference video field: {row['id']}/{video_field}")
            if Path(row[video_field]).expanduser().resolve() != video_path:
                raise ValueError(f"Dataset video path differs from reference manifest: {row['id']}/{field}")
            state = read_json(reference_dir / task["state"])
            expected_request = fingerprint(task["request"])
            if (
                state.get("status") != "succeeded"
                or state.get("request_sha256") != expected_request
                or task.get("request_sha256") != expected_request
                or state.get("endpoint") != H3_ENDPOINT
            ):
                raise ValueError(f"Reference state is incomplete or mismatched: {row['id']}/{field}")
            digest = file_sha256(video_path)
            if state.get("sha256") != digest or state.get("bytes") != video_path.stat().st_size:
                raise ValueError(f"Reference video has changed: {video_path}")
            profile = media_profile(probe(video_path))
            host = state.get("host")
            if not isinstance(host, str) or not host:
                raise ValueError("Reference H3 host was not recorded")
            common_profile, common_host = common_profile or profile, common_host or host
            if profile != common_profile or host != common_host:
                raise ValueError("Reference videos have mixed native formats or H3 hosts")
            references[field] = {"path": str(video_path), "sha256": digest}
        samples.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "h3_context_ir": row["h3_context_ir"],
                "source": row.get("source", "unknown"),
                "references": references,
            }
        )
    if common_profile["height"] != 768 or common_profile["fps"] != 24:
        raise ValueError("Expected the existing 768p, 24fps H3 reference profile")
    return samples, common_profile, common_host


def verify_references(samples: list[dict]):
    """Check frozen reference bytes before generation, judging, or scoring."""
    for sample in samples:
        for reference in sample["references"].values():
            if file_sha256(Path(reference["path"])) != reference["sha256"]:
                raise ValueError(f"Frozen reference video changed: {reference['path']}")
