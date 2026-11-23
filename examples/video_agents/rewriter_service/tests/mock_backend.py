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
"""Fake vLLM backend for gateway tests: seeds 7, 9, 1009 and 2009 give an output with no tool call, seeds 100 and
101 give the same rewrite text (to exercise the duplicate check), everything else gives a seed-specific rewrite."""

import asyncio
import sys

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()
INVALID = {7, 9, 1009, 2009}
DELAY = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3


@app.get("/health")
async def health():
    return {}


@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    seed = body["seed"]
    await asyncio.sleep(DELAY)
    if seed in INVALID:
        text = "<think>\nthinking forever"
    else:
        rewrite = (
            "Shared rewrite" if seed in (100, 101) else f"Rewrite for seed {seed}, prompt tokens {len(body['prompt'])}"
        )
        text = (
            "<think>\nplan\n</think>\n\n<tool_call>\n<function=sol_h3>\n<parameter=prompt>\n"
            + rewrite
            + "\n</parameter>\n<parameter=duration>\n10\n</parameter>\n</function>\n</tool_call>"
        )
    return {
        "choices": [{"text": text, "finish_reason": "length" if seed in INVALID else "stop"}],
        "usage": {"completion_tokens": 123},
    }


uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
