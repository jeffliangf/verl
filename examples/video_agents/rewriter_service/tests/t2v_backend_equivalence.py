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
"""Does a vLLM backend still give s300's text-to-video input the same model? Run it once on the old text-only server
and once on the image-enabled one, then compare.

  python t2v_backend_equivalence.py score http://127.0.0.1:18201 OUT.json     (on the serving node)
  python t2v_backend_equivalence.py compare OLD.json NEW.json

score:
  - Teacher-forced: 21 logged T2V rewrites (3 per log day, earliest task ids first). Each is the service's own policy
    input (policy.prompt_ids) followed by the first 2,048 tokens of the logged answer. prompt_logprobs=1 returns every
    answer token's log-probability and the backend's top-1 token there.
  - Greedy: 8 of those prompts, 384 new tokens at temperature 0, one request at a time.
compare: log-probability differences, top-1 agreement, and how many greedy tokens match before the first difference.
"""

import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import policy  # noqa: E402

PER_DAY, ANSWER_TOKENS, GREEDY_PROMPTS, GREEDY_TOKENS = 3, 2048, 8, 384


def post(base, path, body):
    req = urllib.request.Request(
        base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def logged_tasks():
    picked = []
    for path in sorted((HERE.parent / "logs").glob("tasks-*.jsonl")):
        day = []
        for line in path.open():
            rec = json.loads(line)
            if rec.get("task", "t2v") == "t2v" and rec["status"] == "succeeded" and rec["attempts"]:
                day.append(rec)
        day.sort(key=lambda r: r["task_id"])
        picked += day[:PER_DAY]
    return picked


def score(base, out):
    tok = policy.tokenizer()
    records, t0 = [], time.time()
    for rec in logged_tasks():
        ids = policy.prompt_ids(rec["prompt"])
        answer = tok.encode(rec["attempts"][-1]["text"], add_special_tokens=False)[:ANSWER_TOKENS]
        resp = post(
            base,
            "/v1/completions",
            {"model": "rewriter-s300", "prompt": ids + answer, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1},
        )
        plp = resp["choices"][0]["prompt_logprobs"][len(ids) :]
        logprobs, top1 = [], []
        for token, entry in zip(answer, plp, strict=False):
            logprobs.append(entry[str(token)]["logprob"])
            top1.append(int(next(k for k, v in entry.items() if v.get("rank") == 1)))
        records.append(
            {"task_id": rec["task_id"], "prompt_tokens": len(ids), "answer": answer, "logprobs": logprobs, "top1": top1}
        )
    greedy = []
    for rec in logged_tasks()[:GREEDY_PROMPTS]:
        ids = policy.prompt_ids(rec["prompt"])
        resp = post(
            base,
            "/v1/completions",
            {
                "model": "rewriter-s300",
                "prompt": ids,
                "max_tokens": GREEDY_TOKENS,
                "temperature": 0,
                "seed": 0,
                "return_token_ids": True,
                "skip_special_tokens": False,
            },
        )
        choice = resp["choices"][0]
        greedy.append({"task_id": rec["task_id"], "text": choice["text"], "token_ids": choice.get("token_ids")})
    Path(out).write_text(
        json.dumps(
            {
                "backend": base,
                "scored_at": time.time(),
                "elapsed_s": time.time() - t0,
                "teacher_forced": records,
                "greedy": greedy,
            }
        )
    )
    print(f"scored {len(records)} sequences and {len(greedy)} greedy prompts in {time.time() - t0:.0f} s -> {out}")


def compare(a_path, b_path):
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    diffs, top1_same, top1_n = [], 0, 0
    for ra, rb in zip(a["teacher_forced"], b["teacher_forced"], strict=True):
        assert ra["task_id"] == rb["task_id"] and ra["answer"] == rb["answer"], "different sequences"
        diffs += [abs(x - y) for x, y in zip(ra["logprobs"], rb["logprobs"], strict=True)]
        top1_same += sum(x == y for x, y in zip(ra["top1"], rb["top1"], strict=True))
        top1_n += len(ra["top1"])
    diffs.sort()
    prefixes = []
    for ga, gb in zip(a["greedy"], b["greedy"], strict=True):
        ta, tb = ga["token_ids"] or ga["text"], gb["token_ids"] or gb["text"]
        n = next((i for i, (x, y) in enumerate(zip(ta, tb, strict=False)) if x != y), min(len(ta), len(tb)))
        prefixes.append(n)
    report = {
        "tokens_scored": len(diffs),
        "sequences": len(a["teacher_forced"]),
        "abs_logprob_diff": {
            "mean": sum(diffs) / len(diffs),
            "p50": diffs[len(diffs) // 2],
            "p99": diffs[int(0.99 * len(diffs))],
            "max": diffs[-1],
            "exactly_equal_share": sum(d == 0 for d in diffs) / len(diffs),
        },
        "top1_agreement": top1_same / top1_n,
        "greedy_identical_prefix_tokens": prefixes,
        "greedy_tokens_each": GREEDY_TOKENS,
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    {"score": score, "compare": compare}[sys.argv[1]](*sys.argv[2:])
