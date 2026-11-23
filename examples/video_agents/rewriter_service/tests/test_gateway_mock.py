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
"""End-to-end gateway test against two fake backends, through bestofn's real Context-IR client code.
Text to video (1-9, unchanged) and image to video (10-15). By default image-to-video work has its own backend
(port 18993, as serve.sh runs it); I2V_SHARED=1 tests the single-pool layout instead.
  TEST_TMP=$(mktemp -d) python tests/test_gateway_mock.py      [GATEWAY_APP_DIR=dir to test another copy]"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SVC = Path(__file__).resolve().parents[1]
APP_DIR = Path(os.environ.get("GATEWAY_APP_DIR", SVC))
sys.path.insert(0, str(APP_DIR))
import image_input  # noqa: E402

PY = sys.executable
TMP = Path(os.environ["TEST_TMP"])
TOKEN = "test-token-123"
(TMP / "token").write_text(TOKEN)
# A synthetic first frame the size of the AA I2V frames (1920 x 1080).
FIRST_FRAME = str(TMP / "first-frame.png")
from PIL import Image  # noqa: E402

Image.linear_gradient("L").resize((1920, 1080)).convert("RGB").save(FIRST_FRAME)
PORT = 18990
SHARED = os.environ.get("I2V_SHARED") == "1"
I2V_PORTS = ("18991", "18992") if SHARED else ("18993",)
procs = []


def start_gateway():
    env = dict(
        os.environ,
        REWRITER_BACKENDS="http://127.0.0.1:18991,http://127.0.0.1:18992",
        REWRITER_TOKEN_FILE=str(TMP / "token"),
        REWRITER_LOG_DIR=str(TMP / "logs"),
        **({} if SHARED else {"REWRITER_I2V_BACKENDS": "http://127.0.0.1:18993"}),
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
            str(APP_DIR),
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
if not SHARED:
    procs.append(subprocess.Popen([PY, str(SVC / "tests/mock_backend.py"), "18993"]))
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
    assert text.startswith("Rewrite for seed") and text.rsplit("port ", 1)[1] in ("18991", "18992"), text
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
    # 10. image to video with bestofn's I2V body: h3_selfhost._context_ir_task's first_frame item, verbatim
    import base64
    import io

    from PIL import Image

    extra = [
        {
            "type": "image_url",
            "image_url": {"url": adapters.image_data_url(FIRST_FRAME, "image/png")},
            "role": "first_frame",
        }
    ]
    text = cloud.context_ir("A hand turns the knob.", extra_content=extra, duration=10, ratio="16:9", timeout_s=60)
    print("adapter context_ir (i2v) ->", repr(text[:160]))
    assert text.startswith(image_input.FL2VA_START + "\n\nI2V rewrite for seed") and "image tokens 576" in text, text
    assert text.rsplit("port ", 1)[1] in I2V_PORTS, text  # image work went to the image backend
    # 11. I2V lanes: seeds 42..45, all first-frame prompts, distinct; the summary says i2v with the image's sha
    url = extra[0]["image_url"]["url"]
    status, r = call("POST", "/v1/rewrite", {"prompt": "fox i2v", "n": 4, "seed": 42, "image": url})
    assert status == 200, r
    assert [x["used_seed"] for x in r["rewrites"]] == [42, 43, 44, 45], r
    assert all(x["prompt"].startswith(image_input.FL2VA_START + "\n\n") and x["task"] == "i2v" for x in r["rewrites"])
    assert len({x["prompt"] for x in r["rewrites"]}) == 4
    meta = r["rewrites"][0]["image"]
    assert meta["source_size"] == [1920, 1080] and meta["observation_size"] == [1024, 576], meta
    assert meta["image_tokens"] == 576 and meta["first_frame_line_added"] is True, meta

    # 12. task identity includes the image: same image reuses, other image / no image are separate tasks
    def png(size, color, fmt="PNG"):
        buf = io.BytesIO()
        Image.new("RGB", size, color).save(buf, format=fmt)
        return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()

    other = png((720, 1280), (200, 40, 40), "JPEG")
    body = lambda u: {
        "content": [{"type": "text", "text": "same text"}]
        + (  # noqa: E731
            [{"type": "image_url", "image_url": {"url": u}, "role": "first_frame"}] if u else []
        ),
        "seed": 5,
    }
    a1 = call("POST", "/v2/h3_context_ir", body(url))[1]["task_id"]
    a2 = call("POST", "/v2/h3_context_ir", body(url))[1]["task_id"]
    b1 = call("POST", "/v2/h3_context_ir", body(other))[1]["task_id"]
    c1 = call("POST", "/v2/h3_context_ir", body(None))[1]["task_id"]
    assert a1 == a2 and len({a1, b1, c1}) == 3, (a1, a2, b1, c1)
    for tid in (a1, b1, c1):
        for _ in range(100):
            q = call("GET", f"/v2/query/video_generation/{tid}")[1]
            if q["task"]["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.2)
        assert q["task"]["status"] == "succeeded", q
    q_b = call("GET", f"/v2/query/video_generation/{b1}")[1]["task"]
    assert q_b["rewriter"]["image"]["observation_size"] == [576, 1024], q_b  # portrait 720x1280 -> 576x1024
    assert not call("GET", f"/v2/query/video_generation/{c1}")[1]["task"]["content"]["prompt"].startswith("For the")
    # 13. invalid I2V output is retried with seed + 1000; duplicate texts are resampled per (prompt, image)
    status, r = call("POST", "/v1/rewrite", {"prompt": "i2v retry", "seed": 7, "image": url})
    assert r["rewrites"][0]["used_seed"] == 1007 and r["rewrites"][0]["attempts"] == 2, r
    status, r = call("POST", "/v1/rewrite", {"prompt": "i2v dup", "n": 2, "seed": 100, "image": url})
    assert len({x["prompt"] for x in r["rewrites"]}) == 2 and any(x["duplicate_resamples"] == 1 for x in r["rewrites"])
    # 14. rejected inputs (400, nothing queued): remote URL, bad base64, GIF, last frame, two images, video, no text
    bad = [
        [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
        [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,@@@"}}],
        [
            {"type": "text", "text": "x"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/gif;base64," + base64.b64encode(b"GIF89a" + b"0" * 64).decode()},
            },
        ],
        [{"type": "text", "text": "x"}, {"type": "image_url", "image_url": {"url": url}, "role": "last_frame"}],
        [
            {"type": "text", "text": "x"},
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "image_url", "image_url": {"url": url}},
        ],
        [{"type": "text", "text": "x"}, {"type": "video_url", "video_url": {"url": url}, "role": "reference_video"}],
        [{"type": "image_url", "image_url": {"url": url}}],
    ]
    for content in bad:
        status, resp = call("POST", "/v2/h3_context_ir", {"content": content})
        assert status == 400, (status, resp)
    status, resp = call("POST", "/v1/rewrite", {"prompt": "x", "image": "data:image/png;base64,AAAA"})
    assert status == 400, resp
    # 9. finished tasks survive a gateway restart
    gw.terminate()
    gw.wait()
    gw = start_gateway()
    procs.append(gw)
    status, q = call("GET", f"/v2/query/video_generation/{a['task_id']}")
    assert status == 200 and q["task"]["status"] == "succeeded", q
    status, c = call("POST", "/v2/h3_context_ir", {"content": [{"type": "text", "text": "reuse"}], "seed": 5})
    assert c["task_id"] == a["task_id"], (c, a)
    # 15. I2V tasks survive the restart too: same (prompt, image, seed) is reused, the text keeps its first-frame line
    q = call("GET", f"/v2/query/video_generation/{a1}")[1]
    assert q["task"]["status"] == "succeeded" and q["task"]["content"]["prompt"].startswith(image_input.FL2VA_START)
    assert q["task"]["rewriter"]["task"] == "i2v", q
    assert call("POST", "/v2/h3_context_ir", body(url))[1]["task_id"] == a1
    assert call("POST", "/v2/h3_context_ir", body(None))[1]["task_id"] == c1
    print("ALL GATEWAY TESTS PASSED")
finally:
    for p in procs:
        p.terminate()
