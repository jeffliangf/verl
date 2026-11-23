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
"""Noise reference for t2v_backend_equivalence.py: score the same sequences on ONE backend while other requests share
the batch (as bestofn's four concurrent lanes always do), to compare with the old-vs-new backend difference.

  python t2v_batch_noise.py http://127.0.0.1:18201 OUT.json [CONCURRENT=32]
Background load: CONCURRENT text-to-video generations (logged prompts, random seeds, 1,536 tokens each) kept running
for the whole scoring pass.
"""

import json
import random
import sys
import threading
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t2v_backend_equivalence as eq  # noqa: E402

base, out = sys.argv[1], sys.argv[2]
concurrent = int(sys.argv[3]) if len(sys.argv) > 3 else 32
prompts = [eq.policy.prompt_ids(r["prompt"]) for r in eq.logged_tasks()]
stop = threading.Event()


def load(i):
    rng = random.Random(i)
    while not stop.is_set():
        body = {
            "model": "rewriter-s300",
            "prompt": rng.choice(prompts),
            "max_tokens": 1536,
            "temperature": 0.7,
            "top_p": 0.9,
            "seed": rng.randrange(1 << 30),
        }
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    base + "/v1/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=900,
            )
        except Exception as exc:  # noqa: BLE001
            print("load request failed:", exc)


threads = [threading.Thread(target=load, args=(i,), daemon=True) for i in range(concurrent)]
for t in threads:
    t.start()
try:
    eq.score(base, out)
finally:
    stop.set()
