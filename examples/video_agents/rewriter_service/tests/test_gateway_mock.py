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
"""End-to-end gateway test against two fake backends, through bestofn's real Context-IR client code."""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SVC = Path(__file__).resolve().parents[1]
PY = sys.executable
TMP = Path(os.environ["TEST_TMP"])
TOKEN = "test-token-123"
(TMP / "token").write_text(TOKEN)
PORT = 18990
procs = []


def start_gateway():
    env = dict(
        os.environ,
        REWRITER_BACKENDS="http://127.0.0.1:18991,http://127.0.0.1:18992",
        REWRITER_TOKEN_FILE=str(TMP / "token"),
        REWRITER_LOG_DIR=str(TMP / "logs"),
        REWRITER_SLOTS_PER_BACKEND="2",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    p = subprocess.Popen(
        [
            PY,
            "-m",
            "uvicorn",
            "--app-dir",
            str(SVC),
            "gateway:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
            "--no-access-log",
        ],
        env=env,
    )
    for _ in range(120):
        try:
            if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read())["ok"]:
                return p
        except Exception:
            pass
        time.sleep(1)
    raise SystemExit("gateway did not come up")


def call(method, path, body=None, token=TOKEN):
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + token} if token else {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


backends = [subprocess.Popen([PY, str(SVC / "tests/mock_backend.py"), str(p)]) for p in (18991, 18992)]
procs += backends
gw = start_gateway()
procs.append(gw)
try:
    # 1. auth
    assert call("POST", "/v1/rewrite", {"prompt": "x"}, token=None)[0] == 401
    assert call("POST", "/v1/rewrite", {"prompt": "x"}, token="wrong")[0] == 401
    assert call("POST", "/v2/h3_context_ir", {"content": [{"type": "image_url"}]})[0] == 400
    # 2. bestofn's own Context-IR client, unchanged except host and key
    sys.path[:0] = ["/home/jeff/video-generation-eval/bestofn", "/home/jeff/video-generation-eval/scripts"]
    import adapters  # noqa: E402

    cloud = adapters.BestOfNMinimaxH3()
    cloud.host, cloud.api_key = f"http://127.0.0.1:{PORT}", TOKEN
    t0 = time.time()
    text = cloud.context_ir("A red fox runs through snow.", duration=10, ratio="16:9", timeout_s=60)
    print("adapter context_ir ->", repr(text), f"{time.time() - t0:.1f}s")
    assert text.startswith("Rewrite for seed")
    # 3. best-of-4 lanes are distinct and use seeds 42..45
    status, r = call("POST", "/v1/rewrite", {"prompt": "A red fox runs through snow.", "n": 4, "seed": 42})
    assert status == 200, r
    seeds = [x["used_seed"] for x in r["rewrites"]]
    assert seeds == [42, 43, 44, 45], seeds
    assert len({x["prompt"] for x in r["rewrites"]}) == 4
    # 4. invalid output retried with seed + 1000
    status, r = call("POST", "/v1/rewrite", {"prompt": "retry case", "seed": 7})
    assert r["rewrites"][0]["used_seed"] == 1007 and r["rewrites"][0]["attempts"] == 2, r
    # 5. all attempts invalid -> failed task with the Context-IR error shape
    status, sub = call("POST", "/v2/h3_context_ir", {"content": [{"type": "text", "text": "fail case"}], "seed": 9})
    for _ in range(50):
        status, q = call("GET", f"/v2/query/video_generation/{sub['task_id']}")
        if q["task"]["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)
    assert q["task"]["status"] == "failed" and q["task"]["error"]["code"] == "rewriter_no_valid_call", q
    try:
        cloud.context_ir("fail case", timeout_s=30)  # random seed: almost surely valid
    except Exception as exc:
        raise AssertionError(exc) from exc
    # 6. duplicate text across lanes of one prompt is resampled (seeds 100 and 101 collide in the mock)
    status, r = call("POST", "/v1/rewrite", {"prompt": "dup case", "n": 2, "seed": 100})
    texts = [x["prompt"] for x in r["rewrites"]]
    assert len(set(texts)) == 2, r
    assert any(x["duplicate_resamples"] == 1 for x in r["rewrites"]), r
    # 7. same (prompt, seed) is reused, not resampled
    status, a = call("POST", "/v2/h3_context_ir", {"content": [{"type": "text", "text": "reuse"}], "seed": 5})
    status, b = call("POST", "/v2/h3_context_ir", {"content": [{"type": "text", "text": "reuse"}], "seed": 5})
    assert a["task_id"] == b["task_id"]
    # 8. one backend down: everything still completes on the other
    backends[0].terminate()
    backends[0].wait()
    time.sleep(6)
    status, r = call("POST", "/v1/rewrite", {"prompt": "backend down", "n": 6, "seed": 300})
    assert all(x["status"] == "succeeded" for x in r["rewrites"]), r
    status, s = call("GET", "/v1/stats")
    print("stats", json.dumps(s)[:600])
    # 9. finished tasks survive a gateway restart
    gw.terminate()
    gw.wait()
    gw = start_gateway()
    procs.append(gw)
    status, q = call("GET", f"/v2/query/video_generation/{a['task_id']}")
    assert status == 200 and q["task"]["status"] == "succeeded", q
    status, c = call("POST", "/v2/h3_context_ir", {"content": [{"type": "text", "text": "reuse"}], "seed": 5})
    assert c["task_id"] == a["task_id"], (c, a)
    print("ALL GATEWAY TESTS PASSED")
finally:
    for p in procs:
        p.terminate()
