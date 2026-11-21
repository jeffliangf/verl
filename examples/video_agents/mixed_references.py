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
"""Verify saved best-of-N winners, their 50-step regenerations, and Context-IR references."""

import json
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote, urlsplit

from . import h3
from .dataset import best_of_4_reference, best_of_16_reference
from .h3_request import canvas_size, frame_count
from .mixed_data import (
    CONTEXT_IR_COLUMNS,
    CONTEXT_IR_PROVENANCE,
    EVAL_DURATION_SECONDS,
    EVAL_T2V_CANVAS,
    REFERENCE_COLUMNS,
    keeps_own_request,
)
from .storage import file_sha256, read_json

# The provenance record behind each Context-IR column.
CONTEXT_IR_RECORDS = dict(zip(CONTEXT_IR_COLUMNS, ("repeat1", "best_of_4"), strict=True))
OUTPUT_FORMAT = {
    "fps": 24,
    "video_codec": "avc1",
    "audio_codec": "mp4a",
    "audio_sample_rate": 32000,
    "audio_channels": 2,
}


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
    # Rows evaluated as written were generated from their natural request; benchmark rows from the bare prompt.
    text = sample["request"] if keeps_own_request(sample) else sample["prompt"]
    if (
        generation.get("id") != sample["id"]
        or generation.get("prompt") != text
        or generation.get("image_sha256") != sample["image"]["sha256"]
        or generation.get("repeat") != selection.get("selected_repeat")
    ):
        raise ValueError(f"I2V reference has a different prompt/image: {sample['id']}")
    media = probe(path)
    result = check_video(path, selection.get("video", {}), media)
    original = generation.get("video", {})
    if original.get("sha256") != result["sha256"] or file_sha256(Path(original["path"])) != result["sha256"]:
        raise ValueError(f"I2V reference differs from the selected generation: {sample['id']}")
    duration = 10
    if keeps_own_request(sample):
        # The reference must be a valid answer to the row's own request: allowed length and canvas.
        rules = sample["allowed_video_gen_args"]["target"]
        video = media.get("video") or {}
        canvas = {"width": video.get("width"), "height": video.get("height")}
        if video.get("frames") not in rules["duration_seconds"]["allowed_num_frames"] or (
            canvas not in rules["aspect_ratio"]["allowed_canvases"]
        ):
            raise ValueError(f"I2V reference violates its row's requested controls: {sample['id']}")
        duration = round(video["frames"] / 24)
    if h3.media_profile(media, duration=duration) != h3.media_profile(selection["video"]["media"], duration=duration):
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


def context_ir_render(sample, path, digest):
    """Check a render's own job record when it sits next to the video; stored submission lanes have none."""
    completion_path, request_path = path.parent / "completion.json", path.parent / "request.json"
    if not completion_path.exists() and not request_path.exists():
        return []
    completion, request = read_json(completion_path), read_json(request_path)
    request_hash = sha256(
        json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        completion.get("sha256") != digest
        or Path(completion.get("path", "")).expanduser().resolve() != path
        or completion.get("request_sha256") != request_hash
        or completion.get("generation_id") != path.parent.name
        or (completion.get("media") or {}).get("full_decode_passed") is not True
        or request.get("num_inference_steps") != 50
        or request.get("task") != ("fl2va" if sample["task"] == "i2v" else "t2va")
    ):
        raise ValueError(f"Context-IR render record does not match its video: {sample['id']}")
    conditions = request.get("conditions", [])
    if sample["task"] != "i2v":
        if conditions:
            raise ValueError(f"T2V Context-IR reference unexpectedly has image conditions: {sample['id']}")
        return [completion_path, request_path]
    if len(conditions) != 1 or any(
        conditions[0].get(k) != v for k, v in {"type": "image", "role": "keyframe", "frame_index": 0}.items()
    ):
        raise ValueError(f"Context-IR I2V reference requires one first frame: {sample['id']}")
    uri = urlsplit(conditions[0].get("uri", ""))
    if uri.scheme != "file" or uri.netloc:
        raise ValueError(f"Expected a recorded local first frame: {sample['id']}")
    image_path = Path(unquote(uri.path))
    if file_sha256(image_path) != sample["image"]["sha256"]:
        raise ValueError(f"Context-IR reference uses a different source image: {sample['id']}")
    return [completion_path, request_path, image_path]


def context_ir_reference(sample, row, field, probe):
    """One 50-step render of a Context-IR expansion, or the judged best of four, as recorded in the row's provenance.

    The video must also answer the request the policy is evaluated on: the row's own contract when it is evaluated as
    written, otherwise the 10-second benchmark clip (16:9 for T2V, the image's own canvas for I2V).
    """
    information = row.get(CONTEXT_IR_PROVENANCE)
    record = information.get(CONTEXT_IR_RECORDS[field]) if isinstance(information, dict) else None
    if not isinstance(row.get(field), str) or not row[field].strip() or not isinstance(record, dict):
        raise ValueError(f"Missing reference: {sample['id']}/{field}")
    path = Path(row[field]).expanduser().resolve()
    digest = file_sha256(path)
    if Path(record.get("path", "")).expanduser().resolve() != path or record.get("sha256") != digest:
        raise ValueError(f"Reference differs from its recorded video: {sample['id']}/{field}")
    media = probe(path)
    video = media.get("video") or {}
    if any((record.get("media") or {}).get(k) != video.get(k) for k in ("width", "height", "fps", "frames")):
        raise ValueError(f"Reference media differs from its provenance: {sample['id']}/{field}")
    canvas = {"width": video.get("width"), "height": video.get("height")}
    if keeps_own_request(sample):
        rules = sample["allowed_video_gen_args"]["target"]
        if video.get("frames") not in rules["duration_seconds"]["allowed_num_frames"] or (
            canvas not in rules["aspect_ratio"]["allowed_canvases"]
        ):
            raise ValueError(f"Reference violates its row's requested controls: {sample['id']}/{field}")
        duration = round(video["frames"] / 24)
    else:
        if sample["task"] == "t2v":
            expected = EVAL_T2V_CANVAS
        else:
            size = canvas_size(sample["image"]["width"], sample["image"]["height"])
            expected = dict(zip(("width", "height"), size, strict=True))
        if video.get("frames") != frame_count(EVAL_DURATION_SECONDS) or canvas != expected:
            raise ValueError(
                f"Reference is not the {EVAL_DURATION_SECONDS}-second benchmark clip: {sample['id']}/{field}"
            )
        duration = EVAL_DURATION_SECONDS
    output = h3.media_profile(media, duration=duration)
    if any(output.get(k) != v for k, v in OUTPUT_FORMAT.items()):
        raise ValueError(f"Reference media format differs from H3 output: {sample['id']}/{field}")
    records = context_ir_render(sample, path, digest)
    records += [
        Path(p) for p in (information.get("selection_record"), information.get("references_file")) if isinstance(p, str)
    ]
    best_of = 4 if CONTEXT_IR_RECORDS[field] == "best_of_4" else 1
    scores = record.get("scores") or (information.get("selection_audit") or {}).get("scores") or {}
    return {
        "path": str(path),
        "sha256": digest,
        "media": media,
        "recipe": {
            "best_of": best_of,
            "available_samples": len(scores) if best_of > 1 else 1,
            **{k: information[k] for k in ("pipeline", "model", "selection") if k in information},
        },
        "provenance": provenance(records),
    }


def reference_inventory(samples, rows, probe, columns=None):
    """Validate each row's best-of-4/16 exports and, when the row has them, their 50-step regenerations.

    With ``columns`` (Context-IR columns only), every row is evaluated against exactly those references instead.
    """
    if columns is not None:
        columns = tuple(columns)
        if not columns or len(set(columns)) != len(columns) or not set(columns) <= set(CONTEXT_IR_COLUMNS):
            raise ValueError(f"Reference columns must be distinct Context-IR columns: {', '.join(CONTEXT_IR_COLUMNS)}")
        return [
            {**sample, "references": {field: context_ir_reference(sample, row, field, probe) for field in columns}}
            for sample, row in zip(samples, rows, strict=True)
        ]
    result = []
    for sample, row in zip(samples, rows, strict=True):
        regenerated = REFERENCE_COLUMNS[2:] if any(row.get(f) is not None for f in REFERENCE_COLUMNS[2:]) else ()
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
        for field, source_field in zip(regenerated, REFERENCE_COLUMNS[: len(regenerated)], strict=True):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing reference: {sample['id']}/{field}")
            references[field] = regenerated_reference(
                sample, Path(row[field]).expanduser().resolve(), references[source_field], probe
            )
        result.append({**sample, "references": references})
    return result
