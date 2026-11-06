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
"""Verify saved best-of-N winners and their corresponding 50-step regenerations."""

import json
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import h3
from .dataset import best_of_4_reference, best_of_16_reference
from .mixed_data import REFERENCE_COLUMNS
from .storage import file_sha256, read_json


def provenance(paths):
    return {str(p.resolve()): file_sha256(p) for p in paths}


def check_video(path, record, media):
    """Check the export identity independently of the new candidate's recipe."""
    digest = file_sha256(path)
    if (
        Path(record.get("path", "")).resolve() != path
        or record.get("sha256") != digest
        or record.get("bytes", record.get("size")) != path.stat().st_size
    ):
        raise ValueError(f"Reference differs from its recorded video: {path}")
    return {"path": str(path), "sha256": digest, "media": media}


def i2v_reference(sample, path, count, probe):
    root, item = path.parent.parent.parent, path.parent.name
    selection_path = path.with_suffix(".selection.json")
    selection = read_json(selection_path)
    models = selection.get("candidate_model_ids", [])
    winner = selection.get("winner")
    available = len(models)
    cancelled = selection.get("cancelled_repeats", [])
    if (
        selection.get("best_of") != count
        or not 1 <= available <= count
        or len(set(models)) != available
        or winner not in models
        or selection.get("judgments_completed") != available * (available - 1)
        or selection.get("judgments_expected") != available * (available - 1)
    ):
        raise ValueError(f"Incomplete I2V best-of-{count} selection: {sample['id']}")
    if available != count and (
        selection.get("available_samples") != available
        or len(set(cancelled)) != count - available
        or any(type(r) is not int or not 1 <= r <= count for r in cancelled)
        or set(models) != {f"minimax-h3-sol-i2v-repeat{r}" for r in range(1, count + 1) if r not in cancelled}
    ):
        raise ValueError(f"Undocumented missing I2V selection candidates: {sample['id']}")
    # Do not allow winner labels to escape the recorded artifact directory.
    if not isinstance(winner, str) or Path(winner).name != winner:
        raise ValueError("Invalid reference winner label")
    generation_path = root / "raw" / item / f"{winner}.generation.json"
    generation = read_json(generation_path)
    if (
        generation.get("id") != sample["id"]
        or generation.get("prompt") != sample["prompt"]
        or generation.get("image_sha256") != sample["image"]["sha256"]
        or generation.get("repeat") != selection.get("selected_repeat")
    ):
        raise ValueError(f"I2V reference has a different prompt/image: {sample['id']}")
    media = probe(path)
    result = check_video(path, selection.get("video", {}), media)
    original = generation.get("video", {})
    if original.get("sha256") != result["sha256"] or file_sha256(Path(original["path"])) != result["sha256"]:
        raise ValueError(f"I2V reference differs from the selected generation: {sample['id']}")
    if h3.media_profile(media) != h3.media_profile(selection["video"]["media"]):
        raise ValueError(f"I2V reference media changed: {sample['id']}")
    result["provenance"] = provenance([selection_path, generation_path])
    result["recipe"] = {
        "steps": 4,
        "best_of": count,
        "available_samples": available,
        "cancelled_repeats": cancelled,
        "selection": "pairwise tournament",
    }
    return result


def regenerated_reference(sample, path, source, probe):
    """The 50-step columns regenerate a saved winner; they do not reselect N new clips."""
    completion_path, request_path = path.parent / "completion.json", path.parent / "request.json"
    completion, request = read_json(completion_path), read_json(request_path)
    request_hash = sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        completion.get("generation_id") != source["sha256"]
        or path.parent.name != source["sha256"]
        or completion.get("request_sha256") != request_hash
        or request.get("num_inference_steps") != 50
        or request.get("task") != ("fl2va" if sample["task"] == "i2v" else "t2va")
    ):
        raise ValueError(f"50-step reference does not match its selected source: {sample['id']}")
    conditions = request.get("conditions", [])
    extra_paths = []
    if sample["task"] == "i2v":
        if len(conditions) != 1 or any(
            conditions[0].get(k) != v for k, v in {"type": "image", "role": "keyframe", "frame_index": 0}.items()
        ):
            raise ValueError("50-step I2V reference requires one first frame")
        uri = urlsplit(conditions[0]["uri"])
        if uri.scheme != "file" or uri.netloc:
            raise ValueError("Expected the recorded local 50-step first frame")
        image_path = Path(unquote(uri.path))
        if file_sha256(image_path) != sample["image"]["sha256"]:
            raise ValueError(f"50-step reference uses a different source image: {sample['id']}")
        extra_paths.append(image_path)
    elif conditions:
        raise ValueError("T2V reference unexpectedly has image conditions")
    media = probe(path)
    result = check_video(path, completion, media)
    recorded = completion.get("media", {})
    output = {
        "width": media["video"]["width"],
        "height": media["video"]["height"],
        "fps": media["video"]["fps"],
        "sample_rate": media["audio"]["sample_rate"],
        "channels": media["audio"]["channels"],
    }
    if any(recorded.get(k) != v for k, v in output.items()) or recorded.get("full_decode_passed") is not True:
        raise ValueError(f"50-step reference media differs from its completion: {sample['id']}")
    # Keep the actual recipe (including seed, duration, aspect ratio, and shifts).
    result["recipe"] = {k: v for k, v in request.items() if k not in {"prompt", "conditions"}}
    result["selected_source_sha256"] = source["sha256"]
    result["selection_recipe"] = source["recipe"]
    result["provenance"] = provenance([completion_path, request_path, *extra_paths])
    return result


def reference_inventory(samples, rows, probe):
    """Validate each task's four exports without requiring a shared generation profile."""
    result = []
    for sample, row in zip(samples, rows, strict=True):
        references = {}
        for count, field in zip((4, 16), REFERENCE_COLUMNS[:2], strict=True):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing reference: {sample['id']}/{field}")
            path = Path(row[field]).expanduser().resolve()
            if sample["task"] == "i2v":
                reference = i2v_reference(sample, path, count, probe)
            else:
                media = probe(path)
                validate = best_of_4_reference if count == 4 else best_of_16_reference
                reference = validate(row, h3.media_profile(media), probe)
                root, item = path.parent.parent.parent, path.parent.name
                records = [root / "items" / f"{item}.json"]
                if count == 16:
                    records += [root / "finals" / item / "best_of_16.json"]
                    records += [root / "groups" / item / f"g{g}.json" for g in (2, 3, 4)]
                reference.update(media=media, provenance=provenance(records), recipe={"steps": 4, "best_of": count})
            references[field] = reference
        for field, source_field in zip(REFERENCE_COLUMNS[2:], REFERENCE_COLUMNS[:2], strict=True):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing reference: {sample['id']}/{field}")
            references[field] = regenerated_reference(
                sample, Path(row[field]).expanduser().resolve(), references[source_field], probe
            )
        result.append({**sample, "references": references})
    return result
