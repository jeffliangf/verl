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
"""The step-300 rewriter policy's exact training-time input and its tool-call parser.

The system prompt, user wording, sol_h3 tool schema and chat-template kwargs are frozen in policy_spec.json (copied
from line 1 of the RL run's val.jsonl, whose sha256 is recorded there). The tokenizer and chat template come from the
merged checkpoint, whose files are byte-identical to the checkpoint's actor/huggingface copy. The eval's
rewriter/policy_input.py built the same token ids (verified on all 1,002 AA prompts, tests/check_policy_equivalence.py).
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "policy_spec.json").read_text())
MODEL_DIR = "/home/jeff/rl-benchmarks/rl-rewriter-vs-context-ir-candidate8-20261112/model/global_step_300-hf"
SYSTEM, TOOLS, TEMPLATE_KWARGS, USER_PREFIX = (
    SPEC["system"],
    SPEC["tools"],
    SPEC["chat_template_kwargs"],
    SPEC["user_prefix"],
)
# Training sampling (RL rollouts): temperature 0.7, top-p 0.9, top-k off, no penalties.
SAMPLING = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": -1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

_tokenizer = None


def tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    return _tokenizer


def messages(raw_prompt):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER_PREFIX + raw_prompt}]


def prompt_ids(raw_prompt):
    ids = tokenizer().apply_chat_template(
        messages(raw_prompt), tools=TOOLS, add_generation_prompt=True, tokenize=True, **TEMPLATE_KWARGS
    )
    if isinstance(ids, dict) or hasattr(ids, "keys"):  # transformers 5 may return a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


_CALL = re.compile(r"<tool_call>\s*<function=([^>\n]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([^>\n]+)>\n?(.*?)\n?</parameter>", re.S)


def parse(text):
    """Return (request, error) from the decoded response, like the RL loop: exactly one sol_h3 call,
    nonblank prompt of 1-8000 characters after stripping, integer duration/seed when present."""
    body = text.split("</think>", 1)[1] if "</think>" in text else text
    calls = _CALL.findall(body)
    if len(calls) != 1:
        return None, f"expected one tool call, got {len(calls)}"
    name, inner = calls[0]
    if name.strip() != "sol_h3":
        return None, f"unknown tool {name.strip()!r}"
    params = {k.strip(): v for k, v in _PARAM.findall(inner)}
    prompt = params.get("prompt", "").strip()
    if not prompt or len(prompt) > 8000:
        return None, f"invalid prompt length {len(prompt)}"
    request = {"prompt": prompt, "duration": 10, "seed": 42}
    for key in ("duration", "seed"):
        if key in params:
            try:
                request[key] = int(params[key].strip())
            except ValueError:
                return None, f"non-integer {key}"
    return request, None
