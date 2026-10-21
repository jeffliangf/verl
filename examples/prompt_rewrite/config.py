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
"""Explicit settings shared by zero-shot, SFT, and RL checkpoint evaluation."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data/prompt_rewrite_aa_moviegen_eval_200.jsonl"
DEFAULT_REFERENCES = ROOT / "data/prompt_rewrite_h3_eval/outputs_h3max_disabled_768p"
DEFAULT_EVAL_REPO = Path.home() / "video-generation-eval"
DEFAULT_SYSTEM_PROMPT = Path(__file__).with_name("system_prompt.txt").read_text(encoding="utf-8").strip()
DEFAULT_REWRITE_INSTRUCTION = Path(__file__).with_name("rewrite_instruction.txt").read_text(encoding="utf-8").strip()

H3_ENDPOINT = "/v2/video_generation"
H3_CONCURRENCY = 500
JUDGE_CONCURRENCY = 500
H3_SETTINGS = {
    "model": "MiniMax-H3-Max",
    "resolution": "768P",
    "duration": 10,
    "ratio": "16:9",
    "seed": 42,
    "extra": {"prompt_expansion_mode": "disabled"},
}
JUDGE_SETTINGS = {
    "judge_model": "gemini-3.7-flash",
    "thinking_level": "high",
    "video_processing": "standard",
    "seed": 0,
    "max_output_tokens": None,
    "instructions_version": 4,
}
REFERENCE_FIELDS = ("prompt", "h3_context_ir")
REFERENCE_VIDEO_FIELDS = {
    "prompt": "prompt_h3_max_video",
    "h3_context_ir": "h3_context_ir_h3_max_video",
}
CANDIDATE = "rewritten_prompt"
SCORE_SOURCES = {"moviegen": "movie_gen_video_bench", "aa_eval": "aa_eval"}


@dataclass(frozen=True)
class RewriteConfig:
    """Use an HF model ID or an exported, immutable SFT/RL checkpoint directory."""

    model: str = "Qwen/Qwen3.8-27B"
    revision: str | None = None
    max_new_tokens: int = 2048
    temperature: float = 0.0
    seed: int = 0
    tensor_parallel_size: int = 1
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.85

    backend: str = "vllm"

    def __post_init__(self):
        if not self.model.strip():
            raise ValueError("A model ID or checkpoint path is required")
        if self.max_new_tokens < 1 or self.max_model_len <= self.max_new_tokens:
            raise ValueError("Require 0 < max_new_tokens < max_model_len")
        if not isfinite(self.temperature) or self.temperature < 0 or self.tensor_parallel_size < 1:
            raise ValueError("Temperature must be nonnegative and tensor parallel size positive")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("GPU memory utilization must be between 0 and 1")
        if self.backend not in {"vllm", "hf_fsdp2"}:
            raise ValueError("Unknown rewrite backend")


def build_messages(
    prompt: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    rewrite_instruction: str = DEFAULT_REWRITE_INSTRUCTION,
) -> list[dict[str, str]]:
    """Share a general system role and a rewrite-specific user task across SFT/eval.

    Only task instructions and the original request belong here; never teacher text.
    For SFT, append an assistant message containing the saved Context-IR target.
    An empty instruction preserves the bare user message of older evaluation runs.
    """
    if not prompt.strip() or not system_prompt.strip():
        raise ValueError("Prompt and system prompt must be nonempty")
    content = f"{rewrite_instruction}\n\nUser request:\n{prompt}" if rewrite_instruction else prompt
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}]
