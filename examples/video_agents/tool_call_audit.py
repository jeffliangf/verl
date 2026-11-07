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
"""Run a frozen dataset through Qwen and the native tool validator without generating video."""

import argparse
import asyncio
import importlib.metadata
import json
import os
import socket
import time
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
from types import SimpleNamespace

from jsonschema import Draft202012Validator, ValidationError

from .control_constraints import FIRST_FRAME, ratio_canvas, validate_contract
from .h3_request import frame_count, request_output, request_summary, tool_request
from .image_assets import image_bytes, upright_observation
from .storage import file_sha256, fingerprint, freeze_json, item_key, read_json, read_jsonl, run_lock, write_json

ROOT = Path(__file__).resolve().parents[2]
METRICS = (
    "parsed_sol_h3_call",
    "schema_valid",
    "task_correct",
    "first_frame_correct",
    "duration_supported",
    "aspect_supported",
    "requested_duration_correct",
    "requested_aspect_correct",
    "worker_compatible",
    "all_args_valid",
)


def control_rules(sample):
    """Read annotated training constraints or the existing fixed evaluation targets."""
    contract = sample.get("allowed_video_gen_args")
    if contract is not None:
        validate_contract(sample, contract)
        return deepcopy(contract["target"])
    target = sample.get("target", {})
    duration, ratio = target.get("duration_seconds"), target.get("aspect_ratio")
    return {
        "duration_seconds": {
            "required": duration is not None,
            "allowed_num_frames": [frame_count(duration)] if duration is not None else None,
        },
        "aspect_ratio": {
            "required": ratio is not None,
            # Legacy evaluation validates the resolved canvas, allowing equivalent auto.
            "allowed_values": None,
            "allowed_canvases": [dict(zip(("width", "height"), ratio_canvas(sample, ratio), strict=False))]
            if ratio is not None
            else None,
        },
    }


def assess_calls(calls, schema, sample):
    """Separate invocation, schema, worker support, and instruction matching failures."""
    rules = control_rules(sample)
    result = {key: False for key in METRICS}
    for field, key in (
        ("duration_seconds", "requested_duration_correct"),
        ("aspect_ratio", "requested_aspect_correct"),
    ):
        if not rules[field]["required"]:
            result[key] = None
    result.update(calls=calls, arguments=None, expected_controls=rules, errors={})
    if len(calls) != 1 or calls[0]["name"] != "sol_h3":
        result["errors"]["invocation"] = f"Expected one sol_h3 call; got {[c['name'] for c in calls]}"
        return result
    result["parsed_sol_h3_call"] = True
    try:
        parameters = json.loads(calls[0]["arguments"])
        if not isinstance(parameters, dict):
            raise ValueError("Tool arguments must be a JSON object")
        json.dumps(parameters, allow_nan=False)
    except (ValueError, TypeError) as exc:
        result["errors"]["arguments"] = str(exc)
        return result
    result["arguments"] = parameters
    spec = schema["function"]["parameters"]
    errors = sorted(Draft202012Validator(spec).iter_errors(parameters), key=lambda e: str(e.path))
    result["schema_valid"] = not errors
    if errors:
        result["errors"]["schema"] = [f"{list(e.path)}: {e.message}" for e in errors]
    result["task_correct"] = parameters.get("task") == ("fl2va" if sample["task"] == "i2v" else "t2va")
    result["first_frame_correct"] = parameters.get("conditions", []) == ([FIRST_FRAME] if sample["image"] else [])
    supplied = parameters.get("target", {})
    if isinstance(supplied, dict):
        target_spec = spec["properties"]["target"]["properties"]
        duration = supplied.get("duration_seconds", target_spec["duration_seconds"]["default"])
        ratio = supplied.get("aspect_ratio", "auto" if sample["image"] else "16:9")
        try:
            frames = frame_count(duration)
            result["duration_supported"] = True
            if rules["duration_seconds"]["required"]:
                result["requested_duration_correct"] = (
                    "duration_seconds" in supplied and frames in rules["duration_seconds"]["allowed_num_frames"]
                )
        except (ValueError, TypeError, OverflowError) as exc:
            result["errors"]["duration"] = str(exc)
        try:
            Draft202012Validator(target_spec["aspect_ratio"]).validate(ratio)
            canvas = dict(zip(("width", "height"), ratio_canvas(sample, ratio), strict=False))
            result["aspect_supported"] = True
            if rules["aspect_ratio"]["required"]:
                aspect = rules["aspect_ratio"]
                matches = (
                    ratio in aspect["allowed_values"]
                    if aspect["allowed_values"] is not None
                    else (canvas in aspect["allowed_canvases"])
                )
                result["requested_aspect_correct"] = "aspect_ratio" in supplied and matches
        except (ValueError, TypeError, ValidationError) as exc:
            result["errors"]["aspect"] = exc.message if isinstance(exc, ValidationError) else str(exc)
    # Exercise the real builder twice: first worker capability, then all requested constraints.
    unconstrained = {k: v for k, v in sample.items() if k != "allowed_video_gen_args"}
    unconstrained["target"] = {}
    for key, bound in (("worker_compatible", unconstrained), ("all_args_valid", sample)):
        try:
            request = tool_request(parameters, schema, sample=bound)
            result[key] = True
            if key == "worker_compatible":
                result["worker_request"] = request_summary(request)
                result["resolved_output"] = request_output(request)
        except (ValueError, TypeError, ValidationError) as exc:
            result["errors"][key] = exc.message if isinstance(exc, ValidationError) else str(exc)
    return result


def rates(rows):
    """Use explicit denominators; unconstrained controls are excluded, not counted correct."""
    result = {"samples": len(rows)}
    for key in METRICS:
        applicable = [row for row in rows if row[key] is not None]
        passed = sum(row[key] for row in applicable)
        result[key] = {
            "passed": passed,
            "total": len(applicable),
            "percent": 100 * passed / len(applicable) if applicable else None,
        }
    result["finish_reasons"] = dict(Counter(row["finish_reason"] for row in rows))
    return result


def summarize(rows, expected):
    """Report per-task and per-source accuracy with both unconditional and conditional rates."""
    groups = defaultdict(list)
    for row in rows:
        for key in (row["split"], f"{row['split']}/{row['task']}", f"{row['split']}/{row['task']}/{row['source']}"):
            groups[key].append(row)
    return {
        "complete": len(rows) == expected,
        "expected": expected,
        "completed": len(rows),
        "video_generation_calls": 0,
        "judge_calls": 0,
        "overall": rates(rows),
        "groups": {key: rates(value) for key, value in sorted(groups.items())},
        "among_parsed_calls": {
            key: rates([row for row in value if row["parsed_sol_h3_call"]]) for key, value in sorted(groups.items())
        },
        "metric_notes": {
            "parsed_sol_h3_call": "Exactly one sol_h3 call extracted by the native qwen3_coder parser.",
            "worker_compatible": "Schema, task, first frame and supported values pass; ignores desired controls.",
            "all_args_valid": "The unchanged native tool_request accepts the call against the real bound sample.",
            "requested_controls": "Denominator is only requests with that constraint; missing calls count as failures.",
            "free_controls": "No match penalty; worker-supported values are still required.",
            "length": "Length-limited outputs are retained and parsed, matching the native RL behavior.",
        },
    }


def load_inputs(protocol_path, supplemental=None):
    """Verify frozen dataset identities and pair each observation with its hidden labels."""
    protocol = read_json(protocol_path)
    if fingerprint(protocol) != read_json(protocol_path.with_name("protocol-sha256.json")):
        raise ValueError("Frozen protocol checksum mismatch")
    entries = []
    # Evaluate the held-out set first, so its results are available during the longer training-set pass.
    for split in ("val", "train"):
        spec = protocol["datasets"][split]
        path = Path(spec["path"])
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"Frozen dataset changed: {path}")
        rows = read_jsonl(path)
        samples = protocol["samples"][split]
        task_entries = defaultdict(list)
        for row, sample in zip(rows, samples, strict=True):
            if row["extra_info"]["id"] != sample["id"] or row["extra_info"]["split"] != split:
                raise ValueError("Dataset and protocol sample order disagree")
            control_rules(sample)
            task_entries[sample["task"]].append({"split": split, "record": row, "sample": sample})
        for pair in zip_longest(task_entries["t2v"], task_entries["i2v"]):
            entries.extend(entry for entry in pair if entry is not None)
    if supplemental:
        entries.extend(read_jsonl(supplemental))
    keys = [(e["split"], e["sample"]["id"]) for e in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate audit sample key")
    return entries


def render_input(entry, tokenizer, processor):
    """Use the same placeholder expansion, image orientation and resizing as the native agent."""
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.utils.tokenizer import build_multimodal_processor_inputs

    sample, row = entry["sample"], entry["record"]
    if sample.get("image"):
        image_bytes(sample["image"])
    context = SimpleNamespace(processor=processor, image_key="images", video_key="videos", audio_key="audios")
    messages = upright_observation(RLHFDataset._build_messages(context, deepcopy(row), key="prompt"))
    rendered = processor.apply_chat_template(
        messages, **row["chat_template_kwargs"], tools=row["tools"], add_generation_prompt=True, tokenize=False
    )
    patch_size = getattr(processor.image_processor, "patch_size", 14)
    images, videos, audios = RLHFDataset._process_multi_modal_info(messages, patch_size, {})
    if videos is not None or audios is not None:
        raise ValueError("This audit accepts text and first-frame images only")
    prompt = {"prompt": rendered}
    if images:
        prompt["multi_modal_data"] = {"image": images}
        expected_ids = build_multimodal_processor_inputs(processor, text=[rendered], images=images)["input_ids"][0]
        expected_ids = expected_ids.tolist()
    else:
        expected_ids = tokenizer(rendered, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    if len(expected_ids) > 4096:
        raise ValueError(f"Native input exceeds 4096 tokens: {sample['id']}")
    metadata = {
        "rendered_prompt": rendered,
        "image_sha256": sample["image"]["sha256"] if sample.get("image") else None,
        "observation_image_sizes": [list(image.size) for image in images or []],
        "native_prompt_token_ids": expected_ids,
    }
    return prompt, metadata


def result_path(directory, entry):
    """Use split and content-safe keys without conflating repeated prompt text."""
    return directory / "results" / entry["split"] / (item_key(entry["sample"]["id"]) + ".json")


def stream_generations(llm, entries, prepare, sampling, max_inflight):
    """Refill the engine as requests finish and yield each durable result immediately."""
    remaining = iter(entries)
    active = {}
    exhausted = False
    last_progress = time.monotonic()
    while active or not exhausted:
        incoming = []
        while not exhausted and len(active) + len(incoming) < max_inflight:
            entry = next(remaining, None)
            if entry is None:
                exhausted = True
            else:
                prompt, metadata = prepare(entry)
                incoming.append((entry, prompt, metadata))
        if incoming:
            ids = llm.enqueue([item[1] for item in incoming], sampling, use_tqdm=False)
            for internal_id, (entry, _, metadata) in zip(ids, incoming, strict=True):
                # vLLM enqueue returns randomized internal IDs; RequestOutput uses
                # external IDs. Read the explicit mapping before stepping the engine.
                request_id = llm.llm_engine.output_processor.request_states[internal_id].external_req_id
                if request_id in active:
                    raise RuntimeError("vLLM reused an active request ID")
                active[request_id] = (entry, metadata)
        if not active:
            break
        if not llm.llm_engine.has_unfinished_requests():
            raise RuntimeError("vLLM lost pending audit requests")
        for output in llm.llm_engine.step():
            if output.finished:
                entry, metadata = active.pop(output.request_id)
                yield entry, metadata, output
        if time.monotonic() - last_progress >= 30:
            print(json.dumps({"inflight": len(active), "all_inputs_enqueued": exhausted}), flush=True)
            last_progress = time.monotonic()


def run(args):
    """Run local vLLM inference, persist raw generations, and validate without an executor."""
    from vllm import LLM, SamplingParams

    from verl.experimental.agent_loop.tool_parser import ToolParser
    from verl.tools.schemas import OpenAIFunctionToolSchema
    from verl.utils import hf_processor, hf_tokenizer

    entries = load_inputs(args.protocol, args.supplemental)
    directory = args.output_dir.resolve()
    model = args.model.resolve()
    source_paths = [
        Path(__file__),
        ROOT / "examples/video_agents/h3_request.py",
        ROOT / "examples/video_agents/control_constraints.py",
        ROOT / "examples/video_agents/image_assets.py",
        ROOT / "verl/experimental/agent_loop/tool_parser.py",
        ROOT / "verl/utils/dataset/rl_dataset.py",
        ROOT / "verl/utils/tokenizer/tokenizer.py",
    ]
    settings = {
        "model": str(model),
        "model_files": {p.name: file_sha256(p) for p in model.iterdir() if p.suffix in {".json", ".jinja"}},
        "protocol_sha256": file_sha256(args.protocol),
        "inputs_sha256": fingerprint(entries),
        "sources": {str(p.relative_to(ROOT)): file_sha256(p) for p in source_paths},
        "packages": {
            key: importlib.metadata.version(key) for key in ("torch", "vllm", "transformers", "qwen-vl-utils")
        },
        "sampling": {"temperature": args.temperature, "top_p": 1.0, "top_k": -1, "max_tokens": 8192, "seed": 42},
        "engine": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": 12288,
            "max_num_seqs": args.batch_size,
            "max_num_batched_tokens": 16384,
            "gpu_memory_utilization": 0.85,
            "enforce_eager": True,
            "limit_mm_per_prompt": {"image": 1, "video": 0},
            "enable_prefix_caching": True,
        },
        "expected": len(entries),
        "tool_execution": False,
        "max_inflight": args.batch_size * 2,
    }
    with run_lock(directory):
        freeze_json(directory / "manifest.json", settings)
        freeze_json(directory / "inputs.json", entries)
        results, pending = [], []
        for entry in entries:
            path = result_path(directory, entry)
            if path.exists():
                result = read_json(path)
                if result["input_sha256"] != fingerprint(entry):
                    raise ValueError(f"Stale cached inference: {path}")
                results.append(result)
            else:
                pending.append(entry)
        write_json(directory / "summary.json", summarize(results, len(entries)))
        if not pending:
            return
        if args.limit:
            # A limit changes only this invocation's work budget, never the frozen experiment.
            pending = pending[: args.limit]
        write_json(
            directory / "runtime.json",
            {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "slurm_job_id": os.getenv("SLURM_JOB_ID"),
                "started_unix": time.time(),
                "pending_this_process": len(pending),
            },
        )
        tokenizer, processor = hf_tokenizer(str(model)), hf_processor(str(model))
        parser = ToolParser.get_tool_parser("qwen3_coder", tokenizer)
        llm = LLM(model=str(model), trust_remote_code=False, seed=42, **settings["engine"])
        sampling = SamplingParams(n=1, skip_special_tokens=False, **settings["sampling"])
        loop = asyncio.new_event_loop()
        started = time.monotonic()
        try:
            stream = stream_generations(
                llm, pending, lambda e: render_input(e, tokenizer, processor), sampling, settings["max_inflight"]
            )
            for entry, metadata, output in stream:
                if list(output.prompt_token_ids) != metadata["native_prompt_token_ids"]:
                    raise RuntimeError(f"vLLM and native verl prompt tokens disagree: {entry['sample']['id']}")
                if len(output.outputs) != 1 or output.outputs[0].finish_reason not in {"stop", "length"}:
                    raise RuntimeError(f"Inference did not complete: {output.request_id}")
                generation = output.outputs[0]
                schema = entry["record"]["tools"][0]
                _, calls = loop.run_until_complete(
                    parser.extract_tool_calls(
                        list(generation.token_ids), tools=[OpenAIFunctionToolSchema.model_validate(schema)]
                    )
                )
                sample = entry["sample"]
                result = {
                    "id": sample["id"],
                    "split": entry["split"],
                    "task": sample["task"],
                    "source": sample["source"],
                    "input_sha256": fingerprint(entry),
                    "finish_reason": generation.finish_reason,
                    "stop_reason": generation.stop_reason,
                    "response": tokenizer.decode(generation.token_ids),
                    "response_token_ids": list(generation.token_ids),
                    "response_tokens": len(generation.token_ids),
                    "prompt_tokens": len(output.prompt_token_ids),
                    "native_prompt_tokens_verified": True,
                    **metadata,
                    **assess_calls([c.model_dump() for c in calls], schema, sample),
                }
                write_json(result_path(directory, entry), result)
                results.append(result)
                if len(results) % 16 == 0:
                    summary = summarize(results, len(entries))
                    write_json(directory / "summary.json", summary)
                    print(
                        json.dumps(
                            {
                                "completed": len(results),
                                "expected": len(entries),
                                "inference_seconds": round(time.monotonic() - started, 2),
                                "all_args_valid": summary["overall"]["all_args_valid"],
                            }
                        ),
                        flush=True,
                    )
        finally:
            write_json(directory / "summary.json", summarize(results, len(entries)))
            loop.close()


def main():
    """Parse reproducible local inference settings; no H3 endpoint is accepted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.tensor_parallel_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Batch size, tensor parallel size and limit must be positive")
    run(args)


if __name__ == "__main__":
    main()
