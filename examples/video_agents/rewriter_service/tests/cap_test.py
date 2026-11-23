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
"""Rerun every eval attempt that hit the 8,192-token cap (tests/capped_attempts.jsonl: same AA prompt, same sampling
seed) with max_tokens 32768 and a single attempt, to see how many finish with a valid sol_h3 call once the cap is
lifted, and how long they take. Results: tests/cap-test-results.json.
  python cap_test.py http://HOST:PORT
"""

import concurrent.futures
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = sys.argv[1].rstrip("/")
TOKEN = (HERE.parent / "secrets/api_token").read_text().strip()
MAX_TOKENS = 32768


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


def run_one(item):
    task_id = call(
        "POST",
        "/v2/h3_context_ir",
        {
            "content": [{"type": "text", "text": item["prompt"]}],
            "seed": item["seed"],
            "max_tokens": MAX_TOKENS,
            "attempts": 1,
        },
    )["task_id"]
    while True:
        task = call("GET", f"/v2/query/video_generation/{task_id}")["task"]
        if task["status"] in ("succeeded", "failed"):
            return {k: item[k] for k in ("prompt_id", "lane_index", "attempt", "seed", "lane_valid_in_eval")} | {
                "task_id": task_id,
                "status": task["status"],
                "error": task.get("error"),
                **task["rewriter"],
            }
        time.sleep(5)


items = [json.loads(line) for line in (HERE / "capped_attempts.jsonl").open()]
t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(len(items)) as ex:
    rows = list(ex.map(run_one, items))
wall = time.time() - t0
ok = [r for r in rows if r["status"] == "succeeded"]
tok = sorted(r["completion_tokens"] for r in ok)
summary = {
    "attempts": len(rows),
    "max_tokens": MAX_TOKENS,
    "wall_s": round(wall),
    "valid": len(ok),
    "hit_new_cap": sum(r["finish_reason"] == "length" for r in rows),
    "valid_tokens_median": tok[len(tok) // 2] if tok else None,
    "valid_tokens_max": tok[-1] if tok else None,
    "valid_by_band": {
        band: sum(lo <= t < hi for t in tok)
        for band, (lo, hi) in {
            "8-12k": (8192, 12288),
            "12-16k": (12288, 16384),
            "16-24k": (16384, 24576),
            "24-32k": (24576, 32769),
        }.items()
    },
    "elapsed_s_valid_median": sorted(r["elapsed_s"] for r in ok)[len(ok) // 2] if ok else None,
}
(HERE / "cap-test-results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
print(json.dumps(summary, indent=1))
