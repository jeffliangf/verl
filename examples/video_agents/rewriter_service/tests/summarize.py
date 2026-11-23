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
"""Summarize tests/loadtest-results.json (and tests/cap-test-results.json if present) into tests/summary.json and a
printed table; also compares the service's outputs with the eval's for the same prompt and seed."""

import glob
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL = Path("/home/jeff/rl-benchmarks/rl-rewriter-vs-context-ir-candidate8-20261112")


def q(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


load = json.loads((HERE / "loadtest-results.json").read_text())
table = {}
for name, level in load.items():
    rows = level["rows"]
    el = [r["elapsed_s"] for r in rows]
    qs = [r["queue_s"] or 0 for r in rows]
    speed = [r["total_completion_tokens"] / max(1e-6, r["elapsed_s"] - (r["queue_s"] or 0)) for r in rows]
    table[name] = {
        "tasks": len(rows),
        "concurrency": level["concurrency"],
        "wall_s": level["wall_s"],
        "latency_p50_s": round(q(el, 0.5), 1),
        "latency_p90_s": round(q(el, 0.9), 1),
        "latency_max_s": round(max(el), 1),
        "queue_p50_s": round(q(qs, 0.5), 1),
        "queue_max_s": round(max(qs), 1),
        "tokens_p50": q([r["total_completion_tokens"] for r in rows], 0.5),
        "tokens_per_s_per_request_p50": round(q(speed, 0.5), 1),
        "aggregate_tokens_per_s": level["aggregate_tokens_per_s"],
        "rewrites_per_hour_at_this_load": round(len(rows) / level["wall_s"] * 3600),
        "retried": sum(r["attempts"] > 1 for r in rows),
        "failed": sum(r["status"] != "succeeded" for r in rows),
        "over_8192_tokens": sum((r["completion_tokens"] or 0) > 8192 for r in rows),
    }
hdr = list(next(iter(table.values())))
print("level " + " ".join(h for h in hdr))
for name, row in table.items():
    print(name, " ".join(str(row[h]) for h in hdr))

# Same prompt and seed as the eval: token counts and how long the first attempt's text stays identical.
logs = {}
for path in glob.glob(str(HERE.parent / "logs/tasks-*.jsonl")):
    for line in open(path):
        rec = json.loads(line)
        logs[rec["task_id"]] = rec
pairs = []
for level in load.values():
    for r in level["rows"]:
        rec = logs.get(r["task_id"])
        lane = r["seed"] - 42
        ev_path = EVAL / "rewrites" / r["prompt_id"] / f"sampled-{lane}.json"
        if not rec or not ev_path.exists():
            continue
        ev = json.loads(ev_path.read_text())
        a, b = rec["attempts"][0]["text"], ev["attempts"][0]["text"]
        common = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
        pairs.append(
            {
                "service_tokens": rec["attempts"][0]["completion_tokens"],
                "eval_tokens": ev["attempts"][0]["completion_tokens"],
                "common_prefix_chars": common,
                "service_chars": len(rec["rewrite"] or ""),
                "eval_chars": len(ev["rewrite"] or ""),
            }
        )
compare = {"n": len(pairs)}
if pairs:
    compare.update(
        {
            "service_tokens_median": st.median(p["service_tokens"] for p in pairs),
            "eval_tokens_median_same_items": st.median(p["eval_tokens"] for p in pairs),
            "service_rewrite_chars_median": st.median(p["service_chars"] for p in pairs),
            "eval_rewrite_chars_median_same_items": st.median(p["eval_chars"] for p in pairs),
            "common_prefix_chars_median": st.median(p["common_prefix_chars"] for p in pairs),
            "identical_first_200_chars": sum(p["common_prefix_chars"] >= 200 for p in pairs),
        }
    )
print(json.dumps(compare, indent=1))
summary = {"load": table, "same_seed_vs_eval": compare}
cap = HERE / "cap-test-results.json"
if cap.exists():
    summary["cap_test"] = json.loads(cap.read_text())["summary"]
(HERE / "summary.json").write_text(json.dumps(summary, indent=1))
