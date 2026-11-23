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
"""Latency under concurrency, through the public Context-IR-compatible API (submit + poll), on AA batch-2 prompts.

Levels (all tasks of a level are submitted at once, levels run one after another):
  c1   : 4 prompts, one at a time                    (an idle server)
  c8   : 8 prompts x 1 lane                          (candidate-8's peak: 2 jobs x 4 lanes)
  c32  : 8 prompts x 4 lanes, seeds 42-45            (8 best-of-4 jobs at once)
  c128 : 32 prompts x 4 lanes                        (fills both backends' 64 slots)
  c256 : 64 prompts x 4 lanes                        (twice the slots: half wait in the gateway queue)
Timings are the gateway's own (submit -> finish, and time waiting for a slot). Results: tests/loadtest-results.json.
  python loadtest.py http://HOST:PORT [levels...]
"""

import concurrent.futures
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = sys.argv[1].rstrip("/")
TOKEN = (HERE.parent / "secrets/api_token").read_text().strip()
ITEMS = Path("/home/jeff/rl-benchmarks/rl-rewriter-vs-context-ir-candidate8-20261112/rewriter/items.jsonl")
LEVELS = {"c1": (4, 1, True), "c8": (8, 1, False), "c32": (8, 4, False), "c128": (32, 4, False), "c256": (64, 4, False)}
OUT = HERE / "loadtest-results.json"


def call(method, path, body=None, tries=6):
    """Network errors and 5xx are retried; resubmitting the same (prompt, seed) returns the same task."""
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN},
    )
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == tries - 1:
                raise
        except OSError:
            if attempt == tries - 1:
                raise
        time.sleep(2 * (attempt + 1))


def run_one(prompt_id, prompt, seed):
    task_id = call(
        "POST",
        "/v2/h3_context_ir",
        {
            "model": "MiniMax-H3",
            "duration": 10,
            "ratio": "16:9",
            "content": [{"type": "text", "text": prompt}],
            "seed": seed,
        },
    )["task_id"]
    while True:
        task = call("GET", f"/v2/query/video_generation/{task_id}")["task"]
        if task["status"] in ("succeeded", "failed"):
            return {
                "prompt_id": prompt_id,
                "seed": seed,
                "task_id": task_id,
                "status": task["status"],
                "chars": len((task.get("content") or {}).get("prompt") or ""),
                **task["rewriter"],
            }
        time.sleep(2)


def main():
    levels = sys.argv[2:] or list(LEVELS)
    rows = {}
    for line in ITEMS.open():
        it = json.loads(line)
        rows.setdefault(it["prompt_id"], it["prompt"])
    pool = sorted(rows)
    random.Random(20261113).shuffle(pool)  # disjoint prompts per level, so no task is a cached repeat
    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    cursor = 0
    for name in LEVELS:
        n_prompts, lanes, sequential = LEVELS[name]
        prompts = pool[cursor : cursor + n_prompts]
        cursor += n_prompts
        if name not in levels:
            continue
        jobs = [(pid, rows[pid], 42 + k) for pid in prompts for k in range(lanes)]
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(1 if sequential else len(jobs)) as ex:
            out = list(ex.map(lambda j: run_one(*j), jobs))
        wall = time.time() - t0
        tokens = sum(o["total_completion_tokens"] for o in out)
        results[name] = {
            "tasks": len(out),
            "concurrency": 1 if sequential else len(jobs),
            "wall_s": round(wall, 1),
            "aggregate_tokens_per_s": round(tokens / wall, 1),
            "rows": out,
        }
        el = sorted(o["elapsed_s"] for o in out)
        print(
            f"{name}: {len(out)} tasks wall {wall:.0f}s, elapsed p50 {el[len(el) // 2]:.0f}s max {el[-1]:.0f}s, "
            f"{tokens / wall:.0f} tok/s aggregate, failed {sum(o['status'] != 'succeeded' for o in out)}",
            flush=True,
        )
        OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
