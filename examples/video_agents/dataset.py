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

from . import h3
from .config import REFERENCE_FIELDS, REFERENCE_VIDEO_FIELDS
from .storage import file_sha256, read_json, read_jsonl


def load_samples(path: Path) -> list[dict]:
    """Validate IDs/text without modifying the user's dataset."""
    rows, seen = read_jsonl(path), set()
    if not rows:
        raise ValueError("Evaluation dataset must be nonempty")
    for row in rows:
        for field in ("id", "prompt", "h3_context_ir"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing nonempty {field} in eval row {row.get('id')}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate eval ID: {row['id']}")
        if "source" in row and (not isinstance(row["source"], str) or not row["source"].strip()):
            raise ValueError(f"Invalid source in eval row {row['id']}")
        seen.add(row["id"])
    return rows


def best_of_4_reference(row: dict, profile: dict, probe) -> dict:
    """Verify the winner exported by run_candidate41_best_of_4.py.

    These exports use videos/<item>/best_of_4.mp4 and items/<item>.json,
    rather than the fixed-seed Context-IR reference manifest.
    """
    field = REFERENCE_VIDEO_FIELDS["best_of_4"]
    if not isinstance(row.get(field), str) or not row[field].strip():
        raise ValueError(f"Missing reference video field: {row['id']}/{field}")
    path = Path(row[field]).expanduser().resolve()
    record = read_json(path.parent.parent.parent / "items" / f"{path.parent.name}.json")
    if record.get("id") != row["id"] or record.get("prompt") != row["prompt"] or record.get("status") != "succeeded":
        raise ValueError(f"Best-of-4 job is incomplete or differs from dataset: {row['id']}")
    lanes = record.get("lanes", [])
    model_ids = {lane.get("model_id") for lane in lanes}
    winner = record.get("winner_model_id")
    if (
        len(lanes) != 4
        or len(model_ids) != 4
        or winner is None
        or winner not in model_ids
        or (record.get("selection") or {}).get("selected_model") != winner
        or any(
            lane.get("status") != "completed" or lane.get("processing_path") != "context-ir-sol-h3-native"
            for lane in lanes
        )
    ):
        raise ValueError(f"Best-of-4 reference requires four completed self-hosted H3 lanes and a winner: {row['id']}")
    video = record.get("video") or {}
    if Path(video.get("path", "")).expanduser().resolve() != path:
        raise ValueError(f"Dataset video path differs from best-of-4 record: {row['id']}")
    digest = file_sha256(path)
    if video.get("sha256") != digest or video.get("bytes") != path.stat().st_size:
        raise ValueError(f"Reference video has changed: {path}")
    if h3.media_profile(probe(path)) != profile or h3.media_profile(video.get("media", {})) != profile:
        raise ValueError(f"Reference media differs from its best-of-4 record or Context-IR reference: {path}")
    return {"path": str(path), "sha256": digest}


def reference_inventory(rows: list[dict], reference_dir: Path, probe) -> tuple[list[dict], dict]:
    """Verify request texts, completed state, bytes, checksums, and media formats.

    The source file hash may change when video paths are attached to JSONL rows;
    actual request equality is the authority for matching existing videos.
    """
    manifest = read_json(reference_dir / "manifest.json")
    endpoints = manifest.get("h3_endpoints")
    if manifest.get("h3_backend") != h3.BACKEND:
        raise ValueError("Reference videos must come from the self-hosted H3 service")
    if (
        manifest.get("endpoint") != h3.ENDPOINT
        or manifest.get("settings") != h3.SETTINGS
        or manifest.get("h3") != h3.SETTINGS
        or manifest.get("context_ir_rewriter_enabled") is not False
        or not isinstance(endpoints, list)
        or not endpoints
        or any(not isinstance(origin, str) or not origin for origin in endpoints)
        or len(set(endpoints)) != len(endpoints)
    ):
        raise ValueError("Reference videos use different H3 settings or an invalid endpoint pool")
    tasks = {}
    for task in manifest["tasks"]:
        key = (task["item_id"], task["field"])
        if key in tasks:
            raise ValueError(f"Duplicate reference task: {key}")
        tasks[key] = task
    samples = []
    for row in rows:
        references = {}
        for field in REFERENCE_FIELDS:
            if field == "best_of_4":
                references[field] = best_of_4_reference(row, manifest["media_profile"], probe)
                continue
            task = tasks.get((row["id"], field))
            if task is None or task["request"] != h3.request_body(row[field], manifest["settings"]):
                raise ValueError(f"Reference request differs from dataset: {row['id']}/{field}")
            video_path = (reference_dir / task["output"]).resolve()
            video_field = REFERENCE_VIDEO_FIELDS[field]
            if not isinstance(row.get(video_field), str) or not row[video_field].strip():
                raise ValueError(f"Missing reference video field: {row['id']}/{video_field}")
            if Path(row[video_field]).expanduser().resolve() != video_path:
                raise ValueError(f"Dataset video path differs from reference manifest: {row['id']}/{field}")
            state = read_json(reference_dir / task["state"])
            if (
                state.get("status") != "succeeded"
                or any(state.get(key) != value for key, value in h3.identity(task, manifest).items())
                or state.get("worker_url") not in endpoints
                or not state.get("task_id")
            ):
                raise ValueError(f"Reference job is incomplete or from a different request/pool: {row['id']}/{field}")
            digest = file_sha256(video_path)
            if state.get("sha256") != digest or state.get("bytes") != video_path.stat().st_size:
                raise ValueError(f"Reference video has changed: {video_path}")
            profile = h3.media_profile(probe(video_path))
            if profile != manifest.get("media_profile") or h3.media_profile(state.get("media", {})) != profile:
                raise ValueError(f"Reference media differs from its manifest or state: {video_path}")
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
    return samples, manifest["media_profile"]


def verify_references(samples: list[dict]):
    """Check frozen reference bytes before generation, judging, or scoring."""
    for sample in samples:
        for reference in sample["references"].values():
            if file_sha256(Path(reference["path"])) != reference["sha256"]:
                raise ValueError(f"Frozen reference video changed: {reference['path']}")
