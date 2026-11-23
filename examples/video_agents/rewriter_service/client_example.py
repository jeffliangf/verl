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
"""Minimal client for the step-300 rewriter service (Python standard library only).

  QWEN_REWRITER_TOKEN=... python client_example.py "A red fox runs through fresh snow at dawn."

It rewrites one prompt for the four candidate-11 lanes (sampling seeds 42..45, the same seeds as the SMD-E renders),
through the same submit + poll protocol as MiniMax's H3 Context-IR. candidate-10 is lane 0 only (seed 42).
"""

import concurrent.futures
import json
import os
import sys
import time
import urllib.error
import urllib.request

HOST = os.environ.get("QWEN_REWRITER_HOST", "http://<login-node-ip>:18400")
TOKEN = os.environ["QWEN_REWRITER_TOKEN"]


def http_json(method, path, body=None, timeout=60, tries=5):
    """Retries network errors and 5xx. That is safe for submits too: the same (prompt, seed) returns the same task."""
    req = urllib.request.Request(
        HOST + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN},
    )
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == tries - 1:
                raise
        except OSError:
            if attempt == tries - 1:
                raise
        time.sleep(2 * (attempt + 1))


def rewrite(prompt, seed, budget_s=450):
    """One lane: returns the rewritten prompt, or None when the rewriter failed (render the raw prompt then, as
    candidate-8 does when Context-IR fails)."""
    deadline = time.time() + budget_s
    task_id = http_json("POST", "/v2/h3_context_ir", {"content": [{"type": "text", "text": prompt}], "seed": seed})[
        "task_id"
    ]
    while time.time() < deadline:
        task = http_json("GET", f"/v2/query/video_generation/{task_id}")["task"]
        if task["status"] == "succeeded":
            return task["content"]["prompt"]
        if task["status"] == "failed":
            print(f"seed {seed}: rewriter failed: {task['error']}", file=sys.stderr)
            return None
        time.sleep(5)
    print(f"seed {seed}: no rewrite within {budget_s}s", file=sys.stderr)
    return None


if __name__ == "__main__":
    raw = sys.argv[1]
    lane_seeds = [42, 43, 44, 45]  # AA seed 42 + seed_offset 0..3
    with concurrent.futures.ThreadPoolExecutor(len(lane_seeds)) as pool:
        rewrites = list(pool.map(lambda s: rewrite(raw, s), lane_seeds))
    for seed, text in zip(lane_seeds, rewrites, strict=False):
        print(f"--- lane seed {seed}: {len(text or '')} chars ---\n{(text or raw)[:400]}...\n")
