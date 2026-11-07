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
"""Annotate a reproducible T2V/I2V pilot and materialize natural policy inputs."""

import argparse
import base64
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from itertools import zip_longest
from math import isfinite
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from .config import DEFAULT_EVAL_REPO, ROOT
from .control_constraints import FRAME_COUNTS, RATIOS, allowed_arguments, ratio_canvas
from .h3_request import ASPECT_RATIOS, frame_count, tool_request
from .image_assets import image_bytes
from .judge_worker import gemini_address_family
from .storage import (
    atomic_text,
    file_sha256,
    fingerprint,
    freeze_json,
    load_external,
    read_json,
    read_jsonl,
    run_lock,
    write_json,
    write_jsonl,
)

POLICY_PATH = Path(__file__).with_name("configs") / "control_annotation.txt"
MODES = ("free", "free", "free", "free", "free", "duration", "duration", "aspect", "aspect", "both")


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


EVIDENCE = {"type": "array", "items": {"type": "string"}}
CONFIDENCE = {"type": "string", "enum": ["high", "medium", "low"]}
NUMBER = {"type": ["number", "null"]}
ANNOTATION_SCHEMA = object_schema(
    {
        "duration": object_schema(
            {
                "kind": {"type": "string", "enum": ["unspecified", "exact", "minimum", "range", "qualitative"]},
                "seconds": NUMBER,
                "max_seconds": NUMBER,
                "evidence": EVIDENCE,
                "confidence": CONFIDENCE,
            }
        ),
        "aspect_ratio": object_schema(
            {
                "kind": {"type": "string", "enum": ["unspecified", "ratio", "orientation", "preserve_source"]},
                "value": {"type": ["string", "null"]},
                "evidence": EVIDENCE,
                "confidence": CONFIDENCE,
            }
        ),
        "recommended": object_schema(
            {
                "duration_seconds": {"type": "integer", "enum": [5, 8, 10, 12, 14]},
                "aspect_ratio": {"type": "string", "enum": list(RATIOS)},
                "reason": {"type": "string"},
            }
        ),
        "feasibility_notes": EVIDENCE,
    }
)


def validate_annotation(value, sample):
    """Validate evidence against the original text before using inferred labels."""
    Draft202012Validator(ANNOTATION_SCHEMA).validate(value)
    for name in ("duration", "aspect_ratio"):
        rule = value[name]
        if any(not quote or quote not in sample["prompt"] for quote in rule["evidence"]):
            raise ValueError(f"{name} evidence must quote the original prompt exactly")
        if rule["kind"] == "unspecified":
            if rule["evidence"]:
                raise ValueError("Unspecified controls must not claim supporting evidence")
        elif not rule["evidence"]:
            raise ValueError(f"{name} needs textual evidence")
    duration = value["duration"]
    if duration["kind"] in {"unspecified", "qualitative"}:
        if duration["seconds"] is not None or duration["max_seconds"] is not None:
            raise ValueError("A qualitative or absent duration cannot specify a numeric target")
    else:
        if (
            type(duration["seconds"]) not in (int, float)
            or not isfinite(duration["seconds"])
            or duration["seconds"] <= 0
        ):
            raise ValueError("Duration must be finite and positive")
        if duration["kind"] == "range":
            high = duration["max_seconds"]
            if type(high) not in (int, float) or not isfinite(high) or high < duration["seconds"]:
                raise ValueError("Invalid duration range")
        elif duration["max_seconds"] is not None:
            raise ValueError("Only a range has an upper endpoint")
    aspect = value["aspect_ratio"]
    options = {
        "unspecified": (None,),
        "ratio": ASPECT_RATIOS,
        "orientation": ("portrait", "landscape"),
        "preserve_source": ("auto",) if sample["image"] else (),
    }
    if aspect["value"] not in options[aspect["kind"]]:
        raise ValueError("Aspect constraint does not match its declared meaning")
    if sample["task"] == "t2v" and value["recommended"]["aspect_ratio"] == "auto":
        raise ValueError("T2V recommendations must select a numeric ratio")
    return value


def select_samples(rows, per_task, seed):
    """Select a balanced pilot, or retain all rows while sharing duplicate annotations."""
    selected = []
    for task in ("t2v", "i2v"):
        unique = {}
        duplicates = {}
        for sample in rows:
            if sample["task"] == task:
                identity = (sample["prompt"], sample["image"]["sha256"] if sample["image"] else None)
                unique.setdefault(identity, sample)
                duplicates.setdefault(unique[identity]["id"], []).append(sample)
        choices = sorted(unique.values(), key=lambda sample: fingerprint([seed, sample["id"]]))
        if per_task is not None and len(choices) < per_task:
            raise ValueError(f"Not enough unique {task} samples")
        for index, sample in enumerate(choices[:per_task]):
            siblings = [sample] if per_task is not None else duplicates[sample["id"]]
            selected.extend({**sibling, "pilot_mode": MODES[index % len(MODES)]} for sibling in siblings)
    return selected


def annotation_identity(sample, model, policy):
    return {
        "model": model,
        "policy": policy,
        "schema": ANNOTATION_SCHEMA,
        "prompt": sample["prompt"],
        "task": sample["task"],
        "image_sha256": sample["image"]["sha256"] if sample["image"] else None,
        "thinking_level": "high",
        "seed": 0,
    }


def annotate(sample, directory, model, policy, runner, key, *, cache_only=False):
    """Resume validated annotations and retain each real provider response."""
    identity = annotation_identity(sample, model, policy)
    root = directory / "annotations" / fingerprint(identity)
    with run_lock(root):
        freeze_json(root / "identity.json", identity)
        if (root / "annotation.json").exists():
            return validate_annotation(read_json(root / "annotation.json"), sample), root
        if cache_only:
            raise ValueError("No validated cached annotation is available")
        client = runner.GeminiRestClient(key, max_attempts=3, request_timeout=180, file_timeout=60)
        inputs = [{"type": "text", "text": "Original scene text:\n" + sample["prompt"]}]
        if sample["image"]:
            asset = sample["image"]
            inputs += [
                {"type": "text", "text": f"Attached first frame ({asset['width']}x{asset['height']}, EXIF-upright):"},
                {
                    "type": "image",
                    "data": base64.b64encode(image_bytes(asset)).decode(),
                    "mime_type": asset["mime_type"],
                },
            ]
        request = {
            "model": model,
            "system_instruction": policy,
            "store": False,
            "background": False,
            "stream": False,
            "input": inputs,
            "response_format": {"type": "text", "mime_type": "application/json", "schema": ANNOTATION_SCHEMA},
            "generation_config": {"thinking_level": "high", "seed": 0},
        }
        # The immutable identity plus original image hash reconstructs this input without duplicate base64 files.
        write_json(root / "source.json", {"id": sample["id"], "image": sample["image"]})
        started = time.monotonic()
        previous = max((int(path.stem.split("-")[-1]) for path in root.glob("response-*.json")), default=0)
        for attempt in range(1, 3):
            response, _ = client.request_json(
                "POST",
                runner.API_BASE + "/interactions",
                payload=request,
                retry_network=True,
                context="Control annotation",
            )
            write_json(root / f"response-{previous + attempt}.json", response)
            if response.get("status") != "completed" or response.get("model", model) != model:
                raise ValueError("Unexpected annotation model or incomplete response")
            try:
                value = validate_annotation(json.loads(runner.interaction_output_text(response) or ""), sample)
            except (ValueError, TypeError, ValidationError) as error:
                # One format/evidence correction, with the original scene and image unchanged.
                if attempt == 2:
                    raise
                request["input"] = inputs + [{"type": "text", "text": "Correct the annotation: " + str(error)[:1000]}]
                continue
            write_json(root / "annotation.json", value)
            write_json(
                root / "timing.json", {"seconds": time.monotonic() - started, "response_attempts": previous + attempt}
            )
            return value, root


def duration_frames(rule):
    if rule["kind"] in {"unspecified", "qualitative"}:
        return list(FRAME_COUNTS)
    if rule["kind"] == "exact":
        try:
            return [frame_count(rule["seconds"])]
        except ValueError:
            return []
    low = rule["seconds"]
    high = rule["max_seconds"] if rule["kind"] == "range" else float("inf")
    # Bounds use nominal worker rounding where possible, retaining unsupported bounds for feasibility checks.
    try:
        lower = frame_count(low)
    except ValueError:
        lower = low * 24
    try:
        upper = frame_count(high)
    except ValueError:
        upper = high * 24
    return [frames for frames in FRAME_COUNTS if lower <= frames <= upper]


def aspect_ratios(sample, rule):
    if rule["kind"] == "unspecified":
        return list(RATIOS)
    if rule["kind"] == "orientation":
        portrait = rule["value"] == "portrait"
        selected = []
        for ratio in RATIOS:
            width, height = ratio_canvas(sample, ratio)
            if (portrait and width < height) or (not portrait and width > height):
                selected.append(ratio)
        return selected
    expected = ratio_canvas(sample, rule["value"])
    return [ratio for ratio in RATIOS if ratio_canvas(sample, ratio) == expected]


def materialize(sample, annotation, annotation_dir, model):
    """Keep scene text intact and derive allowed arguments from the visible augmentation."""
    duration, aspect = deepcopy(annotation["duration"]), deepcopy(annotation["aspect_ratio"])
    uncertain_duration = duration["confidence"] == "low"
    uncertain_aspect = aspect["confidence"] == "low"
    if uncertain_duration:
        duration = {"kind": "unspecified", "seconds": None, "max_seconds": None}
    if uncertain_aspect:
        aspect = {"kind": "unspecified", "value": None}
    desired = annotation["recommended"]
    mode = sample["pilot_mode"]
    additions, origins = [], {"duration_seconds": "original_request", "aspect_ratio": "original_request"}
    retimed = not duration_frames(duration)
    if not uncertain_duration and (
        retimed or (mode in {"duration", "both"} and duration["kind"] in {"unspecified", "qualitative"})
    ):
        duration = {"kind": "exact", "seconds": desired["duration_seconds"], "max_seconds": None}
        additions.append(f"Keep it about {desired['duration_seconds']} seconds long.")
        origins["duration_seconds"] = "explicit_retiming" if retimed else "synthetic_request"
    if not uncertain_aspect and mode in {"aspect", "both"} and aspect["kind"] == "unspecified":
        ratio = desired["aspect_ratio"]
        if sample["image"] and ratio == "auto":
            aspect = {"kind": "preserve_source", "value": "auto"}
            additions.append("Keep the image's original aspect ratio.")
        elif ratio == "1:1":
            aspect = {"kind": "ratio", "value": ratio}
            additions.append("Make it square.")
        elif int(fingerprint(sample["id"])[0], 16) % 2:
            orientation = "portrait" if ratio_canvas(sample, ratio)[0] < ratio_canvas(sample, ratio)[1] else "landscape"
            aspect = {"kind": "orientation", "value": orientation}
            additions.append("Use a vertical frame." if orientation == "portrait" else "Use a landscape frame.")
        else:
            aspect = {"kind": "ratio", "value": ratio}
            additions.append(f"Use a {ratio} frame.")
        origins["aspect_ratio"] = "synthetic_request"
    request = sample["request"]
    if retimed:
        request = (
            f"Adapt the following scene into a video lasting {desired['duration_seconds']} seconds, "
            "adjusting its original timeline while preserving event order:\n\n" + request
        )
        additions = [text for text in additions if not text.startswith("Keep it about")]
    if additions:
        extra = " ".join(additions)
        request = extra + "\n\n" + request if int(fingerprint(sample["id"])[1], 16) % 2 else request + "\n\n" + extra
    target = {}
    if duration["kind"] == "exact":
        target["duration_seconds"] = duration["seconds"]
    if aspect["kind"] in {"ratio", "preserve_source"}:
        target["aspect_ratio"] = aspect["value"]
    output = {
        **sample,
        "request": request,
        "target": target,
        "preferred_target": {k: desired[k] for k in ("duration_seconds", "aspect_ratio")},
    }
    output["control_metadata"] = {
        "annotation_model": model,
        "annotation_file": str(annotation_dir / "annotation.json"),
        "annotation_sha256": file_sha256(annotation_dir / "annotation.json"),
        "original_constraints": {k: annotation[k] for k in ("duration", "aspect_ratio")},
        "effective_constraints": {"duration": duration, "aspect_ratio": aspect},
        "sources": origins,
        "retimed": retimed,
        "review_required": retimed or any(annotation[k]["confidence"] == "low" for k in ("duration", "aspect_ratio")),
    }
    frames = duration_frames(duration)
    output["allowed_video_gen_args"] = allowed_arguments(
        output,
        frames=frames,
        ratios=aspect_ratios(output, aspect),
        # A two-second sub-action imposes no extra choice on a worker whose
        # shortest output is already over five seconds. Defaults satisfy it too.
        require_duration=frames != list(FRAME_COUNTS),
        require_aspect=aspect["kind"] != "unspecified",
    )
    return output


def save_outputs(directory, samples):
    """Write ingestible samples and the exact native messages/tools used by Qwen."""
    from .rl import CONFIG_DIR, dataset_records

    schema = read_json(CONFIG_DIR / "sol_h3_unified_tool.json")
    for sample in samples:
        contract = sample["allowed_video_gen_args"]
        target = contract["target"]
        desired = sample["preferred_target"]
        duration = desired["duration_seconds"]
        if frame_count(duration) not in target["duration_seconds"]["allowed_num_frames"]:
            duration = target["duration_seconds"]["allowed_num_frames"][0] / 24
        ratio = desired["aspect_ratio"]
        if ratio not in target["aspect_ratio"]["allowed_values"]:
            ratio = target["aspect_ratio"]["allowed_values"][0]
        arguments = {
            "task": contract["task"],
            "conditions": contract["conditions"],
            "target": {"duration_seconds": duration, "aspect_ratio": ratio},
            "seed": 42,
        }
        # This checks controls only, without pretending to generate a policy rewrite or a video.
        tool_request({**arguments, "prompt": "Control validation only."}, schema, sample=sample)
        sample["example_allowed_args"] = arguments
    raw = [{**sample, "image": sample["image"]["path"] if sample["image"] else None} for sample in samples]
    write_jsonl(directory / "samples.jsonl", raw)
    records = dataset_records(samples, "train", [schema])
    write_jsonl(directory / "train.jsonl", records)
    examples = {}
    for task in ("t2v", "i2v"):
        choices = [s for s in samples if s["task"] == task and s["pilot_mode"] == "both"]
        if not choices:
            continue
        sample = min(choices, key=lambda s: len(s["request"]))
        record = next(r for r in records if r["extra_info"]["id"] == sample["id"])
        write_json(
            directory / f"{task}.input.json",
            {k: record[k] for k in ("prompt", "images", "tools", "chat_template_kwargs")},
        )
        write_json(directory / f"{task}.allowed-args.json", sample["allowed_video_gen_args"])
        atomic_text(directory / f"{task}.user.txt", record["prompt"][1]["content"] + "\n")
        examples[task] = sample["id"]
    summary = {
        "samples": len(samples),
        "by_task": dict(Counter(s["task"] for s in samples)),
        "augmentation_modes": dict(Counter(s["pilot_mode"] for s in samples)),
        "duration_required": sum(
            s["allowed_video_gen_args"]["target"]["duration_seconds"]["required"] for s in samples
        ),
        "aspect_required": sum(s["allowed_video_gen_args"]["target"]["aspect_ratio"]["required"] for s in samples),
        "retimed": sum(s["control_metadata"]["retimed"] for s in samples),
        "review_required": sum(s["control_metadata"]["review_required"] for s in samples),
        "examples": examples,
        "renderer_sha256": file_sha256(Path(__file__)),
    }
    write_json(directory / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/prompt_rewrite_mixed_rl_v3.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/prompt_rewrite_control_pilot_100")
    parser.add_argument("--eval-repo", type=Path, default=DEFAULT_EVAL_REPO)
    parser.add_argument("--model", default="gemini-4-argon")
    parser.add_argument("--per-task", type=int, default=50)
    parser.add_argument("--all", action="store_true", help="Retain every input row, including repeated prompts")
    parser.add_argument("--annotation-cache", type=Path, help="Reuse annotations under this directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--render-only", action="store_true", help="Rebuild outputs from cache without provider calls")
    args = parser.parse_args(argv)
    if args.per_task < 1 or args.concurrency < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("Sample and concurrency limits must be positive")
    rows = read_jsonl(args.input)
    selected = select_samples(rows, None if args.all else args.per_task, args.seed)
    policy = POLICY_PATH.read_text()
    directory = args.output_dir.resolve()
    cache = args.annotation_cache.resolve() if args.annotation_cache else directory
    config = {
        "input": str(args.input.resolve()),
        "input_sha256": file_sha256(args.input),
        "model": args.model,
        "policy_sha256": fingerprint(policy),
        "schema": ANNOTATION_SCHEMA,
        "seed": args.seed,
        "samples": selected,
    }
    if args.all or args.annotation_cache:
        config.update(all_rows=args.all, annotation_cache=str(cache))
    if not args.execute and not args.render_only:
        print(
            json.dumps({"samples": len(selected), "model": args.model, "output_dir": str(directory), "execute": False})
        )
        return 0
    runner, key = None, None
    if not args.render_only:
        runner = load_external(args.eval_repo / "scripts/run_gemini_pairwise_eval.py")
        key = os.environ.get("GEMINI_API_KEY") or runner.read_env_file(args.eval_repo / "arena/.env").get(
            "GEMINI_API_KEY"
        )
        if not key:
            raise RuntimeError("Configured Gemini credential is unavailable")
    with run_lock(directory), gemini_address_family("ipv4"):
        freeze_json(directory / "manifest.json", config)
        previous_errors = (
            {row["id"]: row["error"] for row in read_json(directory / "errors.json")}
            if (directory / "errors.json").exists()
            else {}
        )
        tasks = [[s for s in selected if s["task"] == task] for task in ("t2v", "i2v")]
        interleaved = [s for pair in zip_longest(*tasks) for s in pair if s is not None]
        pending = interleaved if args.limit is None else interleaved[: args.limit]
        completed, errors = {}, []
        unique = {}
        for sample in pending:
            unique.setdefault(fingerprint(annotation_identity(sample, args.model, policy)), []).append(sample)
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            jobs = {
                pool.submit(
                    annotate, siblings[0], cache, args.model, policy, runner, key, cache_only=args.render_only
                ): siblings
                for siblings in unique.values()
            }
            for future in as_completed(jobs):
                siblings = jobs[future]
                try:
                    annotation, root = future.result()
                    for sample in siblings:
                        completed[sample["id"]] = materialize(sample, annotation, root, args.model)
                        print(
                            json.dumps(
                                {
                                    "completed": len(completed),
                                    "selected": len(pending),
                                    "task": sample["task"],
                                    "id": sample["id"],
                                }
                            ),
                            flush=True,
                        )
                except Exception as error:
                    message = str(error) if args.render_only else runner.safe_error_text(error, key)
                    errors.extend(
                        {"id": s["id"], "error": previous_errors.get(s["id"], message) if args.render_only else message}
                        for s in siblings
                    )
                    write_json(directory / "errors.json", errors)
                    print(
                        json.dumps({"failed": [s["id"] for s in siblings], "error": errors[-1]["error"][:400]}),
                        flush=True,
                    )
        samples = [completed[s["id"]] for s in selected if s["id"] in completed]
        write_json(directory / "errors.json", errors)
        summary = save_outputs(directory, samples)
        summary.update(
            model=args.model, selected=len(selected), complete=len(samples) == len(selected), errors=len(errors)
        )
        write_json(directory / "summary.json", summary)
        print(json.dumps(summary, indent=2))
        return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
