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
"""Shared message construction, injectable inference, and durable rewrite records."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import RewriteConfig, build_messages
from .storage import fingerprint, freeze_json, item_key, read_json, write_jsonl


@dataclass(frozen=True)
class RewriteResult:
    text: str
    finish_reason: str = "stop"
    output_tokens: int | None = None


class VLLMRewriter:
    """Lazy offline inference for the original model or exported HF checkpoints.

    Importing this module and preparing/evaluating runs never loads CUDA or vLLM.
    Trainers can instead pass their own callable to ``run_rewrites``.
    """

    def __init__(self, config: RewriteConfig):
        if config.backend != "vllm":
            raise ValueError("This run uses resident FSDP2 inference; resume it through the SFT launcher")
        self.config = config
        self.engine = None

    def __call__(self, messages: list[list[dict[str, str]]]) -> list[RewriteResult]:
        from vllm import LLM, SamplingParams

        config = self.config
        if self.engine is None:
            self.engine = LLM(
                model=config.model,
                revision=config.revision,
                tokenizer_revision=config.revision,
                tensor_parallel_size=config.tensor_parallel_size,
                dtype="bfloat16",
                max_model_len=config.max_model_len,
                # Bound hybrid-attention state for offline eval on a single H100.
                # Larger input batches are scheduled in chunks by vLLM.
                max_num_seqs=32,
                gpu_memory_utilization=config.gpu_memory_utilization,
                seed=config.seed,
                generation_config="vllm",
                limit_mm_per_prompt={"image": 0, "video": 0},
            )
        tokenizer = self.engine.get_tokenizer()
        prompts = []
        for conversation in messages:
            tokens = tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
                raise ValueError("Chat template must return a flat list of token IDs")
            if len(tokens) + config.max_new_tokens > config.max_model_len:
                raise ValueError("Prompt and output budget exceed max_model_len; use a new run with a larger context")
            prompts.append({"prompt_token_ids": tokens})
        sampling = SamplingParams(
            n=1,
            temperature=config.temperature,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            max_tokens=config.max_new_tokens,
            seed=config.seed,
        )
        results = self.engine.generate(prompts, sampling, use_tqdm=False)
        return [
            RewriteResult(row.outputs[0].text, row.outputs[0].finish_reason, len(row.outputs[0].token_ids))
            for row in results
        ]


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


def run_rewrites(
    run_dir: Path,
    protocol: dict,
    rewriter: Callable[[list[list[dict[str, str]]]], list[RewriteResult]],
    batch_size: int = 8,
):
    """Evaluate any policy callable, saving each completed batch for restart.

    The caller holds the run lock. The callable sees system/user messages only.
    Use this seam for SFT/RL inference that already owns a loaded policy.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    freeze_json(run_dir / "rewrite-source.json", {"mode": "policy", "rewriter": protocol["rewriter"]})
    completed = load_rewrites(run_dir, protocol, require_complete=False)
    pending = [sample for sample in protocol["samples"] if sample["id"] not in completed]
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        messages = [
            build_messages(sample["prompt"], protocol["system_prompt"], protocol.get("rewrite_instruction", ""))
            for sample in batch
        ]
        results = rewriter(messages)
        if len(results) != len(batch):
            raise ValueError("Rewriter returned a different number of outputs than inputs")
        for sample, result in zip(batch, results, strict=True):
            save_rewrite(run_dir, protocol, sample, result)
        print(f"Rewrites: {len(completed) + offset + len(batch)}/{len(protocol['samples'])}", flush=True)
    return export_rewrites(run_dir, protocol)


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
