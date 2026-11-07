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
"""Prepare one ordinary shuffled RL dataset from text and first-frame requests."""

import json
from collections import Counter
from copy import deepcopy
from math import isfinite
from pathlib import Path

from . import h3
from .config import DEFAULT_EVAL_REPO, JUDGE_SETTINGS, ROOT
from .h3_request import ASPECT_RATIOS, canvas_size, frame_count
from .image_assets import image_asset
from .judge import EXTERNAL_FILES
from .storage import atomic_text, file_sha256, load_external, read_json, read_jsonl, write_json
from .tournament import validate_group_size

TRAIN_FILES = (
    ROOT / "data/prompt_rewrite_i2v_rl_dedup.jsonl",
    ROOT / "data/prompt_rewrite_rl.jsonl",
    ROOT / "data/prompt_rewrite_sft.jsonl",
)
EVAL_FILES = (ROOT / "data/prompt_rewrite_aa_moviegen_eval_200.jsonl", ROOT / "data/aa_i2v_eval_200.jsonl")
REFERENCE_COLUMNS = (
    "best_of_4_h3_self_hosted_video",
    "best_of_16_h3_self_hosted_video",
    "best_of_4_h3_50step",
    "best_of_16_h3_50step",
)
DATA_DIR = ROOT / "data/prompt_rewrite_mixed_rl_v3"
OBSERVATION_MAX_PIXELS = 768 * 768
EVAL_DURATION_SECONDS = 10
EVAL_T2V_ASPECT_RATIO = "16:9"
EVAL_T2V_CANVAS = dict(zip(("width", "height"), canvas_size(16, 9), strict=True))


def natural_request(prompt, *, image, evaluation=False):
    """Express the user's video request without exposing backend task labels."""
    if evaluation:
        if image:
            instruction = f"Animate this image into a {EVAL_DURATION_SECONDS}-second video"
            return f"{instruction}, adapting the following scene to fit that duration:\n\n{prompt}"
        return (
            f"Create a {EVAL_DURATION_SECONDS}-second video in {EVAL_T2V_ASPECT_RATIO} landscape format, "
            f"adapting the following scene to fit that duration and format:\n\n{prompt}"
        )
    instruction = "Animate this image" if image else "Make a video of the following scene"
    return f"{instruction}:\n\n{prompt}"


def normalize_sources(paths):
    """Keep original requests and assets; teacher rewrites never become policy inputs."""
    samples, rows, seen = [], [], set()
    for path in paths:
        path = Path(path).expanduser().resolve()
        source_rows = read_jsonl(path)
        if not source_rows:
            raise ValueError(f"Empty source dataset: {path}")
        for line, row in enumerate(source_rows, 1):
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("id", "prompt")):
                raise ValueError(f"Missing nonempty ID/prompt at {path}:{line}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate sample ID: {row['id']}")
            seen.add(row["id"])
            source = row.get("source", "unknown")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"Invalid source: {row['id']}")
            image = None
            if row.get("image") is not None:
                if not isinstance(row["image"], str) or not row["image"].strip():
                    raise ValueError(f"Expected an image path: {row['id']}")
                image_path = Path(row["image"]).expanduser()
                image = image_asset(image_path if image_path.is_absolute() else path.parent / image_path)
            task = "i2v" if image else "t2v"
            if row.get("task", task) != task:
                raise ValueError(f"Task does not match the supplied image: {row['id']}")
            request = row.get("request", natural_request(row["prompt"], image=bool(image)))
            if not isinstance(request, str) or not request.strip():
                raise ValueError(f"Expected a nonempty natural request: {row['id']}")
            target = row.get("target", {})
            if not isinstance(target, dict) or set(target) - {"duration_seconds", "aspect_ratio", "short_edge"}:
                raise ValueError(f"Invalid output settings: {row['id']}")
            frame_count(target.get("duration_seconds", 10))
            ratio = target.get("aspect_ratio", "auto" if image else "16:9")
            if ratio not in (*ASPECT_RATIOS, "auto") or type(target.get("short_edge", 768)) is not int:
                raise ValueError(f"Invalid output settings: {row['id']}")
            if target.get("short_edge", 768) != 768:
                raise ValueError(f"Only short_edge 768 is supported: {row['id']}")
            if image and ratio == "auto":
                canvas_size(image["width"], image["height"])
            samples.append(
                {
                    "id": row["id"],
                    "task": task,
                    "prompt": row["prompt"],
                    "request": request,
                    "source": source,
                    "image": image,
                    "target": target,
                    "provenance": {"file": str(path), "line": line, "original_id": row["id"]},
                }
            )
            for field in ("preferred_target", "control_metadata", "allowed_video_gen_args"):
                if field in row:
                    samples[-1][field] = deepcopy(row[field])
            if "allowed_video_gen_args" in samples[-1]:
                from .control_constraints import validate_contract

                validate_contract(samples[-1], samples[-1]["allowed_video_gen_args"])
            rows.append(row)
    return samples, rows


def check_splits(train, validation):
    """Exclude exact ID, prompt, and source-image leakage across held-out splits."""
    for name, key in (
        ("ID", lambda r: r["id"]),
        ("prompt", lambda r: r["prompt"].strip()),
        ("image", lambda r: r["image"]["sha256"] if r.get("image") else None),
    ):
        overlap = {key(r) for r in train} & {key(r) for r in validation} - {None}
        if overlap:
            raise ValueError(f"Train/evaluation overlap by {name}: {len(overlap)}")


def original_request(sample):
    """Give the judge the visible request; retain old frozen-run semantics."""
    if "request" in sample:
        return sample["request"]
    text = sample["prompt"]
    if sample.get("target"):
        text += "\nRequested output settings: " + json.dumps(sample["target"], sort_keys=True)
    return text


def observation(sample):
    """Use native image placeholders; the executor binds the same asset separately."""
    if "request" in sample:
        return ("<image>\n" if sample["image"] else "") + sample["request"]
    # Frozen v3 rows retain their old rendering when explicitly reproduced.
    task = "fl2va" if sample["task"] == "i2v" else "t2va"
    text = f"Generate a video with sol_h3 using task {task}. Preserve the user's request.\n"
    if sample["task"] == "i2v":
        text += "Required first frame, image_ref=input_image:\n<image>\n"
    text += "Choose suitable duration and aspect ratio unless specified.\n\nUser request:\n"
    return text + original_request(sample)


def prepare(args):
    """Freeze the approved mixed experiment offline, including all four baselines."""
    from .judge_worker import supports_reference_images
    from .mixed_references import reference_inventory
    from .rl import CONFIG_DIR, local_path, save_prepared

    train_paths = [local_path(p) for p in args.train_files]
    eval_paths = [local_path(p) for p in args.eval_datasets]
    combined, data_dir, run_dir = map(local_path, (args.combined_file, args.data_dir, args.run_dir))
    reserved = {
        data_dir / f"{split}{suffix}"
        for split in ("train", "val")
        for suffix in (".jsonl", ".preview.json", ".preview.txt")
    }
    if combined in (*train_paths, *eval_paths):
        raise ValueError("The combined output must not overwrite a source dataset")
    if combined in reserved or combined.is_relative_to(run_dir):
        raise ValueError("The combined output must be separate from prepared run artifacts")
    train, _ = normalize_sources(train_paths)
    validation, rows = normalize_sources(eval_paths)
    check_splits(train, validation)
    if not isfinite(args.poll_interval) or args.poll_interval <= 0:
        raise ValueError("Polling interval must be positive and finite")
    if min(args.groups_per_worker, args.h3_concurrency_per_worker) < 1:
        raise ValueError("Worker concurrency must be positive")
    repo = local_path(args.eval_repo)
    dependencies = {relative: file_sha256(repo / relative) for relative in EXTERNAL_FILES}
    if not supports_reference_images(load_external(repo / "scripts/run_gemini_pairwise_eval.py")):
        raise ValueError("Mixed RL requires an evaluator with reference_image and image upload support")
    rubric = load_external(repo / "scripts/build_pairwise_manifest.py")
    if rubric.INSTRUCTIONS_VERSION != JUDGE_SETTINGS["instructions_version"]:
        raise ValueError("Expected the existing v4 Gemini rubric")
    validation = reference_inventory(validation, rows, load_external(repo / "bestofn/normalize.py").probe)
    validation = [
        {
            **sample,
            "request": natural_request(
                row.get("request", sample["prompt"]), image=bool(sample["image"]), evaluation=True
            ),
            "target": {
                **sample["target"],
                "duration_seconds": EVAL_DURATION_SECONDS,
                **({"aspect_ratio": EVAL_T2V_ASPECT_RATIO} if sample["task"] == "t2v" else {}),
            },
            "control_metadata": {
                "duration_seconds": {
                    "source": "benchmark_request",
                    "effective_num_frames": frame_count(EVAL_DURATION_SECONDS),
                },
                **(
                    {"aspect_ratio": {"source": "benchmark_request", "effective_canvas": EVAL_T2V_CANVAS}}
                    if sample["task"] == "t2v"
                    else {}
                ),
            },
        }
        for sample, row in zip(validation, rows, strict=True)
    ]
    reference_notes = [
        {"sample_id": sample["id"], "task": sample["task"], "reference": field, "selection": recipe}
        for sample in validation
        for field, reference in sample["references"].items()
        if (recipe := reference.get("selection_recipe", reference.get("recipe", {})))
        and recipe.get("available_samples", recipe.get("best_of")) != recipe.get("best_of")
    ]
    profile = {
        "width": 1344,
        "height": 768,
        "fps": 24,
        "video_codec": "avc1",
        "audio_codec": "mp4a",
        "audio_sample_rate": 32000,
        "audio_channels": 2,
    }
    service = h3.configuration(args.h3_endpoints, profile, None, args.h3_token_file)
    if len(service["h3_endpoints"]) != 1:
        raise ValueError("Mixed runs require one Sol-H3 gateway, not individual worker addresses")
    for key in ("duration", "seed", "width", "height"):
        service["h3"].pop(key)
    protocol = {
        "schema_version": 4,
        "task": "h3_tournament",
        "tool_contract": "urn:verl:sol-h3:3",
        "h3_api_version": 2,
        "group_size": validate_group_size(args.group_size),
        "samples": {"train": train, "val": validation},
        "source_files": {str(p): file_sha256(p) for p in (*train_paths, *eval_paths)},
        "reference_fields": list(REFERENCE_COLUMNS),
        "reference_notes": reference_notes,
        "media_profile": profile,
        "judge": dict(JUDGE_SETTINGS),
        "eval_judge": {**JUDGE_SETTINGS, "judge_model": args.eval_judge_model},
        "rubric": rubric.INSTRUCTIONS,
        "eval_repo": str(repo),
        "dependencies": dependencies,
        "poll_interval": args.poll_interval,
        "groups_per_worker": args.groups_per_worker,
        "h3_concurrency_per_worker": args.h3_concurrency_per_worker,
        "observation_max_pixels": OBSERVATION_MAX_PIXELS,
        "eval_duration_seconds": EVAL_DURATION_SECONDS,
        "eval_t2v_aspect_ratio": EVAL_T2V_ASPECT_RATIO,
        **service,
    }
    contents = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in train)
    if combined.exists() and combined.read_text() != contents:
        raise ValueError(f"Combined dataset changed; choose a new output: {combined}")
    summary = save_prepared(args, protocol, read_json(CONFIG_DIR / "sol_h3_unified_tool.json"))
    atomic_text(combined, contents)
    summary.update(
        combined_file=str(combined),
        train_by_task=dict(Counter(s["task"] for s in train)),
        eval_by_task=dict(Counter(s["task"] for s in validation)),
        repeated_training_prompt_rows=len(train) - len({(s["task"], s["prompt"].strip()) for s in train}),
        reference_fields=list(REFERENCE_COLUMNS),
        training_judge=protocol["judge"]["judge_model"],
        evaluation_judge=protocol["eval_judge"]["judge_model"],
        evaluation_duration_seconds=EVAL_DURATION_SECONDS,
        evaluation_t2v_aspect_ratio=EVAL_T2V_ASPECT_RATIO,
        evaluation_t2v_canvas=EVAL_T2V_CANVAS,
        videos_per_evaluation=len(validation),
        judgments_per_evaluation=len(validation) * len(REFERENCE_COLUMNS) * 2,
        reference_notes=reference_notes,
    )
    write_json(local_path(args.run_dir) / "preparation-summary.json", summary)
    return summary


def add_parser(commands):
    parser = commands.add_parser("prepare-mixed", help="Prepare unified T2V/I2V RL and four-baseline evaluation")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--combined-file", type=Path, default=ROOT / "data/prompt_rewrite_mixed_rl_v3.jsonl")
    parser.add_argument("--train-files", nargs="+", type=Path, default=TRAIN_FILES)
    parser.add_argument("--eval-datasets", nargs="+", type=Path, default=EVAL_FILES)
    parser.add_argument("--eval-repo", type=Path, default=DEFAULT_EVAL_REPO)
    parser.add_argument("--eval-judge-model", default="gemini-4-argon")
    h3.add_arguments(parser)
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--groups-per-worker", type=int, default=5)
    parser.add_argument("--h3-concurrency-per-worker", type=int, default=5)
