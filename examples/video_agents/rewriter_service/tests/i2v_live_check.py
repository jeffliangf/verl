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
"""Image to video through the live gateway.

  python i2v_live_check.py http://HOST:18400 [N=25]

1. bestofn's own Context-IR client (adapters.BestOfNMinimaxH3.context_ir) with the first-frame item exactly as
   h3_selfhost._context_ir_task builds it.
2. N AA batch-2 I2V items (the joint RL runs' eval rows: prompt + first frame) x 4 lanes (seeds 42..45), all
   submitted at once through POST /v2/h3_context_ir and polled.
Reports success, invalid-output resamples, latency, tokens and rewrite length, and how often the policy's
reasoning talks about the image. Results: tests/i2v-live-check.json (task ids, so the logs give the full texts).
"""

import concurrent.futures
import json
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import image_input  # noqa: E402

BASE = sys.argv[1].rstrip("/")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 25
TOKEN = (HERE.parent / "secrets/api_token").read_text().strip()
ROWS = "/home/jeff/rl-benchmarks/grpo-mixed-hydrogen-group8-16h100-20261121-s150-smd-cir/data/val.jsonl"
SEEDS = (42, 43, 44, 45)
IMAGE_WORDS = re.compile(r"\b(image|picture|photo|photograph|first frame|frame shows|shown|depicted|depicts)\b", re.I)


def call(method, path, body=None):
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + TOKEN},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def items():
    rows = [json.loads(line) for line in open(ROWS)]
    rows = [r for r in rows if r["data_source"] == "i2v/aa-i2v-batch2"][:N]
    return [(r["prompt"][1]["content"].split("\n\n", 1)[1], r["images"][0]["image"]) for r in rows]


def main():
    sys.path[:0] = ["/home/jeff/video-generation-eval/bestofn", "/home/jeff/video-generation-eval/scripts"]
    import adapters

    work = items()
    report = {"base": BASE, "n_items": len(work), "lanes": list(SEEDS)}
    # 1. bestofn's client, first item, unchanged except host and key
    cloud = adapters.BestOfNMinimaxH3()
    cloud.host, cloud.api_key = BASE, TOKEN
    prompt, path = work[0]
    extra = [
        {"type": "image_url", "image_url": {"url": adapters.image_data_url(path, "image/png")}, "role": "first_frame"}
    ]
    t0 = time.time()
    text = cloud.context_ir(prompt, extra_content=extra, duration=10, ratio="16:9", timeout_s=450)
    report["bestofn_client"] = {
        "elapsed_s": round(time.time() - t0, 1),
        "chars": len(text),
        "starts_with_first_frame_line": text.startswith(image_input.FL2VA_START + "\n\n"),
        "head": text[:600],
    }
    print("bestofn client:", json.dumps(report["bestofn_client"], ensure_ascii=False)[:900])
    # 2. all items x lanes at once
    t0 = time.time()

    def submit(job):
        (prompt, path), seed = job
        body = {
            "model": "MiniMax-H3",
            "duration": 10,
            "ratio": "16:9",
            "seed": seed,
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "role": "first_frame", "image_url": {"url": adapters.image_data_url(path)}},
            ],
        }
        return call("POST", "/v2/h3_context_ir", body)["task_id"]

    jobs = [(item, seed) for item in work for seed in SEEDS]
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        ids = list(pool.map(submit, jobs))
    print(f"submitted {len(ids)} tasks in {time.time() - t0:.1f}s")
    done = {}
    while len(done) < len(ids) and time.time() - t0 < 3600:
        for tid in ids:
            if tid not in done:
                task = call("GET", f"/v2/query/video_generation/{tid}")["task"]
                if task["status"] in ("succeeded", "failed"):
                    done[tid] = task
        time.sleep(5)
    tasks = [done.get(tid) for tid in ids]
    ok = [t for t in tasks if t and t["status"] == "succeeded"]
    summ = [t["rewriter"] for t in ok]
    # Reasoning texts from the gateway's task log
    texts = {}
    for log in sorted((HERE.parent / "logs").glob("tasks-*.jsonl")):
        for line in log.open():
            rec = json.loads(line)
            if rec["task_id"] in done and rec["attempts"]:
                texts[rec["task_id"]] = rec["attempts"][-1]["text"]
    reasoning = [texts[t["task_id"]].split("</think>")[0] for t in ok if t["task_id"] in texts]
    per_lane = {}
    for (item, seed), t in zip(jobs, tasks, strict=False):
        per_lane.setdefault(item[0], set()).add(t["content"]["prompt"] if t and t["status"] == "succeeded" else None)
    report["batch"] = {
        "tasks": len(ids),
        "succeeded": len(ok),
        "failed": len(ids) - len(ok),
        "wall_s": round(time.time() - t0, 1),
        "first_attempt_valid": sum(1 for s in summ if s["attempts"] == 1),
        "attempts_total": sum(s["attempts"] for s in summ),
        "elapsed_s_p50": statistics.median(s["elapsed_s"] for s in summ) if summ else None,
        "elapsed_s_max": max(s["elapsed_s"] for s in summ) if summ else None,
        "completion_tokens_p50": statistics.median(s["completion_tokens"] for s in summ) if summ else None,
        "rewrite_chars_p50": statistics.median(len(t["content"]["prompt"]) for t in ok) if ok else None,
        "all_start_with_first_frame_line": all(
            t["content"]["prompt"].startswith(image_input.FL2VA_START + "\n\n") for t in ok
        ),
        "image_tokens": sorted({s["image"]["image_tokens"] for s in summ}),
        "reasoning_mentions_image": f"{sum(bool(IMAGE_WORDS.search(r)) for r in reasoning)}/{len(reasoning)}",
        "items_with_4_distinct_rewrites": sum(1 for v in per_lane.values() if len(v) == 4 and None not in v),
        "errors": [t["error"] for t in tasks if t and t["status"] == "failed"][:5],
    }
    report["task_ids"] = ids
    report["example"] = {
        "prompt": work[0][0],
        "image": work[0][1],
        "reasoning_head": reasoning[0][:1500] if reasoning else None,
        "rewrite": ok[0]["content"]["prompt"] if ok else None,
    }
    (HERE / "i2v-live-check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report["batch"], indent=1))


if __name__ == "__main__":
    main()
