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
"""Dataset locations, prompts, and judge settings for SFT and video-agent RL."""

import json
from math import ceil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data/prompt_rewrite_aa_moviegen_eval_200.jsonl"
DEFAULT_SELF_HOSTED_REFERENCES = ROOT / "data/prompt_rewrite_h3_eval/outputs_self_hosted_gateway_768p"
DEFAULT_EVAL_REPO = Path.home() / "video-generation-eval"
DEFAULT_SYSTEM_PROMPT = Path(__file__).with_name("system_prompt.txt").read_text(encoding="utf-8").strip()
DEFAULT_REWRITE_INSTRUCTION = Path(__file__).with_name("rewrite_instruction.txt").read_text(encoding="utf-8").strip()
LATENCY_BASELINE = json.loads(Path(__file__).with_name("configs").joinpath("latency_baseline.json").read_text())
DEFAULT_TIMEOUTS = {
    "llm": 900,
    "h3_request": 30,
    "judge_file": 300,
    "judge_pass": 1800,
    **{
        stage: ceil(LATENCY_BASELINE["timeout_multiplier"] * LATENCY_BASELINE[stage]["p99_seconds"])
        for stage in ("h3", "judge_request")
    },
}

# Read-only metadata for existing reference videos; all new generation uses Sol-H3.
MAX_REFERENCE_ENDPOINT = "/v2/video_generation"
MAX_REFERENCE_SETTINGS = {
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
MAX_REFERENCE_VIDEO_FIELDS = {
    "prompt": "prompt_h3_max_video",
    "h3_context_ir": "h3_context_ir_h3_max_video",
}
SELF_HOSTED_REFERENCE_VIDEO_FIELDS = {
    "prompt": "prompt_h3_self_hosted_video",
    "h3_context_ir": "h3_context_ir_h3_self_hosted_video",
}
CANDIDATE = "video_agent"


def build_sft_messages(prompt: str) -> list[dict[str, str]]:
    """Build the prompt-rewrite SFT input from the original request.

    Only task instructions and the original request belong here; never teacher text.
    For SFT, append an assistant message containing the saved Context-IR target.
    """
    if not prompt.strip() or not DEFAULT_SYSTEM_PROMPT or not DEFAULT_REWRITE_INSTRUCTION:
        raise ValueError("Prompt, system prompt, and rewrite instruction must be nonempty")
    content = f"{DEFAULT_REWRITE_INSTRUCTION}\n\nUser request:\n{prompt}"
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}, {"role": "user", "content": content}]
