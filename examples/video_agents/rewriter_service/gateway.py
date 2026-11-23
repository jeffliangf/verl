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
"""Authenticated HTTP gateway for the RL step-300 prompt rewriter, call-compatible with MiniMax's H3 Context-IR.

Context-IR protocol (what bestofn's adapters.py speaks; only the host and API key change):
  POST /v2/h3_context_ir  {"content": [{"type": "text", "text": PROMPT}], "seed": 42}   -> {"task_id": ...}
       Image to video: add one first-frame item, as bestofn's I2V lanes send it:
       {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}, "role": "first_frame"}
  GET  /v2/query/video_generation/{task_id}
       -> {"task": {"status": "queued|running|succeeded|failed", "content": {"prompt": REWRITE}, "error": ...}}
  "model", "duration" and "ratio" are accepted and ignored. Optional extras: "seed" (sampling seed; pass the lane's
  render seed so the four lanes differ and a rerun reproduces), "max_tokens", "attempts".
Blocking convenience call:
  POST /v1/rewrite  {"prompt": PROMPT, "n": 4, "seed": 42}  -> {"rewrites": [one per lane, seeds 42..45]}
       with an optional "image": DATA_URL for image to video.
Also GET /health (no auth) and GET /v1/stats.

Each task samples the policy's exact training input (policy.py) with the training sampling settings. An output
without a valid sol_h3 call is resampled with seed + 1000 * attempt (the eval's rule); after `attempts` invalid
outputs the task fails, and a Context-IR caller falls back to the raw prompt as it does for Context-IR failures. A
valid rewrite whose text (whitespace and case folded) was already returned for the same prompt under a different
base seed is resampled too (at most twice), so lanes of one job never receive the same text. Submitting the same
(prompt, seed, max_tokens, attempts) again returns the existing task instead of sampling again.

Image to video (image_input.py): s300 was trained on text to video only, so it gets exactly its text-to-video input
with the first frame placed at the start of the user turn, answers with the same sol_h3 call, and the gateway returns
that call's prompt behind the Sol-H3 first-frame line (image_input.FL2VA_START). Text-to-video requests are unchanged.
The image is part of the task's identity: reuse and the cross-seed duplicate check are per (prompt, image).

Generations go to the least-loaded healthy vLLM backend, at most SLOTS per backend at a time; the rest wait in
arrival order. Backend transport errors and 5xx are retried on another backend. Text-to-video work goes to
REWRITER_BACKENDS and image-to-video work to REWRITER_I2V_BACKENDS (default: the same servers). serve.sh runs a
separate image-enabled server for I2V so that text to video keeps the exact text-only engine it always had.
"""

import asyncio
import collections
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import image_input
import policy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Logs and task records are for the whole utopai group (the job may have been submitted under umask 077).
os.umask(0o022)
ENV = os.environ
BACKENDS = [u.rstrip("/") for u in ENV["REWRITER_BACKENDS"].split(",") if u.strip()]
I2V_BACKENDS = [
    u.rstrip("/") for u in ENV.get("REWRITER_I2V_BACKENDS", ENV["REWRITER_BACKENDS"]).split(",") if u.strip()
]
TOKEN = Path(ENV["REWRITER_TOKEN_FILE"]).read_text().strip()
LOG_DIR = Path(ENV.get("REWRITER_LOG_DIR", str(policy.HERE / "logs")))
MODEL_NAME = ENV.get("REWRITER_MODEL_NAME", "rewriter-s300")
MAX_MODEL_LEN = int(ENV.get("REWRITER_MAX_MODEL_LEN", "40960"))
# settings.json (optional, read at gateway start) overrides the environment, so a default can change with a
# gateway restart alone, without restarting the serving job.
SETTINGS = json.loads((policy.HERE / "settings.json").read_text()) if (policy.HERE / "settings.json").exists() else {}
DEFAULT_MAX_TOKENS = int(SETTINGS.get("max_tokens", ENV.get("REWRITER_MAX_TOKENS", "24576")))
SLOTS = int(ENV.get("REWRITER_SLOTS_PER_BACKEND", "64"))
MAX_QUEUE = int(ENV.get("REWRITER_MAX_QUEUE", "5000"))
DEFAULT_ATTEMPTS = int(ENV.get("REWRITER_ATTEMPTS", "3"))
DEDUPE_RESAMPLES = 2
SEED_STRIDE = 1000
MAX_SEED = 2**31 - 1 - SEED_STRIDE * 16
TASK_TTL_S = 7 * 86400
DEDUPE_TTL_S = 3 * 86400
MAX_PROMPT_CHARS = 20000
CHECKPOINT = "grpo-group8-16h100-20261031-resume-20261102 global_step_300"


def sha256(text):
    return hashlib.sha256(text.encode()).hexdigest()


def text_key(text):
    return hashlib.sha1(re.sub(r"\s+", " ", text).strip().casefold().encode()).hexdigest()


def err(http_status, message, code=None):
    return JSONResponse(
        {"base_resp": {"status_code": code or http_status, "status_msg": message}}, status_code=http_status
    )


class BackendRejected(RuntimeError):
    pass


class Pool:
    """Generation slots per backend; acquire() waits until a healthy backend has a free slot."""

    def __init__(self, urls, slots):
        self.urls, self.slots = urls, slots
        self.busy = dict.fromkeys(urls, 0)
        self.healthy = dict.fromkeys(urls, False)
        self.cond = asyncio.Condition()
        self.waiting = 0

    async def acquire(self):
        async with self.cond:
            self.waiting += 1
            try:
                while True:
                    free = [u for u in self.urls if self.healthy[u] and self.busy[u] < self.slots]
                    if free:
                        url = min(free, key=self.busy.__getitem__)
                        self.busy[url] += 1
                        return url
                    await self.cond.wait()
            finally:
                self.waiting -= 1

    async def release(self, url):
        async with self.cond:
            self.busy[url] -= 1
            self.cond.notify(1)

    async def set_health(self, url, ok):
        async with self.cond:
            if ok and not self.healthy[url]:
                self.cond.notify_all()
            self.healthy[url] = ok


class Task:
    def __init__(self, prompt, seed, seed_given, max_tokens, attempts, client, image=None):
        self.id = secrets.token_hex(12)
        self.prompt, self.prompt_sha = prompt, sha256(prompt)
        # image: None for text to video, else {"sha256", "observation" (PIL), "url" (PNG data URL of the
        # observation), "source_size"}; the pixels are dropped once the task finishes.
        self.kind = "t2v" if image is None else "i2v"
        self.image = image
        self.image_sha = image["sha256"] if image else None
        self.image_meta = None
        self.seed, self.seed_given = seed, seed_given
        self.max_tokens, self.attempts_allowed = max_tokens, attempts
        self.client = client
        self.status = "queued"
        self.submitted_at, self.started_at, self.finished_at = time.time(), None, None
        self.rewrite = self.error = None
        self.attempts = []  # per attempt: seed, finish_reason, tokens, elapsed, backend, error (+ text in the log)
        self.used_seed = None
        self.duplicate_of_seed = None
        self.done = asyncio.Event()

    def summary(self):
        return {
            "seed": self.seed,
            "seed_given": self.seed_given,
            "used_seed": self.used_seed,
            "attempts": len(self.attempts),
            "attempts_allowed": self.attempts_allowed,
            "max_tokens": self.max_tokens,
            "duplicate_of_seed": self.duplicate_of_seed,
            "completion_tokens": self.attempts[-1]["completion_tokens"] if self.attempts else None,
            "total_completion_tokens": sum(a["completion_tokens"] or 0 for a in self.attempts),
            "finish_reason": self.attempts[-1]["finish_reason"] if self.attempts else None,
            "queue_s": round(self.started_at - self.submitted_at, 2) if self.started_at else None,
            "elapsed_s": round((self.finished_at or time.time()) - self.submitted_at, 2),
            "duplicate_resamples": sum(1 for a in self.attempts if a.get("duplicate_of_seed") is not None),
            "model": MODEL_NAME,
            "checkpoint": CHECKPOINT,
            "task": self.kind,
            **({"image_sha256": self.image_sha, "image": self.image_meta} if self.kind == "i2v" else {}),
        }

    def view(self):
        task = {
            "task_id": self.id,
            "status": self.status,
            "created_at": int(self.submitted_at),
            "updated_at": int(self.finished_at or self.started_at or self.submitted_at),
            "rewriter": self.summary(),
        }
        if self.status == "succeeded":
            task["content"] = {"prompt": self.rewrite}
        if self.status == "failed":
            task["error"] = self.error
        return {"task": task, "base_resp": {"status_code": 0, "status_msg": "success"}}


POOL = Pool(BACKENDS, SLOTS)  # text to video
I2V_POOL = POOL if I2V_BACKENDS == BACKENDS else Pool(I2V_BACKENDS, SLOTS)  # image to video
POOLS = [POOL] if I2V_POOL is POOL else [POOL, I2V_POOL]
TASKS = {}  # task_id -> Task
BY_KEY = {}  # task_key(...) -> task_id, for tasks with a caller-given seed
SEEN = {}  # dedupe_key(...) -> {text_key: (base_seed, time)}
RECENT = collections.deque(maxlen=2000)
STATS = collections.Counter()
STARTED = time.time()
CLIENT = None


def task_key(prompt_sha, image_sha, seed, max_tokens, attempts):
    """Text-to-video keys keep their original form, so restored tasks keep matching."""
    if image_sha is None:
        return (prompt_sha, seed, max_tokens, attempts)
    return ("i2v", prompt_sha, image_sha, seed, max_tokens, attempts)


def dedupe_key(prompt_sha, image_sha):
    return prompt_sha if image_sha is None else f"i2v:{prompt_sha}:{image_sha}"


def log_task(task, attempt_texts):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task.id,
        "status": task.status,
        "submitted_at": task.submitted_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "client": task.client,
        "task": task.kind,
        "image_sha256": task.image_sha,
        "prompt": task.prompt,
        "prompt_sha256": task.prompt_sha,
        "rewrite": task.rewrite,
        "error": task.error,
        "summary": task.summary(),
        "attempts": [dict(a, text=t) for a, t in zip(task.attempts, attempt_texts, strict=False)],
    }
    path = LOG_DIR / f"tasks-{time.strftime('%Y%m%d', time.gmtime())}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


async def generate(ids, seed, max_tokens, task):
    if task.kind == "i2v":
        # vLLM's tokens-in endpoint takes our ids (image placeholder already expanded) plus the image itself.
        path = "/inference/v1/generate"
        body = {
            "model": MODEL_NAME,
            "token_ids": ids,
            "content_parts": [{"type": "image_url", "url": task.image["url"]}],
            "sampling_params": {
                "max_tokens": max_tokens,
                "seed": seed,
                "skip_special_tokens": False,
                "spaces_between_special_tokens": False,
                **policy.SAMPLING,
            },
        }
    else:
        path = "/v1/completions"
        body = {
            "model": MODEL_NAME,
            "prompt": ids,
            "max_tokens": max_tokens,
            "seed": seed,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
            **policy.SAMPLING,
        }
    pool = I2V_POOL if task.kind == "i2v" else POOL
    last = None
    for net_try in range(6):
        url = await pool.acquire()
        if task.started_at is None:
            task.started_at, task.status = time.time(), "running"
        t0 = time.time()
        try:
            r = await CLIENT.post(url + path, json=body)
            if r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            if r.status_code != 200:
                raise BackendRejected(f"HTTP {r.status_code}: {r.text[:300]}")
            resp = r.json()
            choice = resp["choices"][0]
            if task.kind == "i2v":
                out_ids = choice.get("token_ids") or []
                text = await asyncio.to_thread(policy.tokenizer().decode, out_ids, skip_special_tokens=False)
                tokens = (resp.get("usage") or {}).get("completion_tokens") or len(out_ids)
            else:
                text, tokens = choice["text"], (resp.get("usage") or {}).get("completion_tokens")
            return text, {
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": tokens,
                "elapsed_s": round(time.time() - t0, 2),
                "backend": url,
            }
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            STATS["backend_errors"] += 1
            await pool.set_health(url, False)  # the health monitor re-enables it once /health answers
        finally:
            await pool.release(url)
        await asyncio.sleep(min(60, 5 * 2**net_try))
    raise RuntimeError(f"no backend finished the generation: {last!r}")


async def run_task(task):
    texts = []
    try:
        if task.kind == "i2v":
            ids, image_tokens = await asyncio.to_thread(image_input.prompt_ids, task.prompt, task.image["observation"])
            task.image_meta = {
                "source_size": list(task.image["source_size"]),
                "observation_size": list(task.image["observation"].size),
                "image_tokens": image_tokens,
                "prompt_tokens": len(ids),
            }
        else:
            ids = await asyncio.to_thread(policy.prompt_ids, task.prompt)
        max_tokens = min(task.max_tokens, MAX_MODEL_LEN - len(ids))
        if max_tokens < 1024:
            raise BackendRejected(f"prompt too long ({len(ids)} tokens)")
        attempt, invalid_left, dedupe_left = 0, task.attempts_allowed, DEDUPE_RESAMPLES
        duplicate = None  # (rewrite, seed, prior seed): used if every resample after it is invalid
        while True:
            seed = task.seed + SEED_STRIDE * attempt
            attempt += 1
            text, meta = await generate(ids, seed, max_tokens, task)
            request, error = policy.parse(text)
            texts.append(text)
            task.attempts.append(
                {"attempt": attempt, "seed": seed, **meta, "valid": request is not None, "error": error}
            )
            if request is None:
                invalid_left -= 1
                if invalid_left > 0:
                    continue
                if duplicate:
                    task.rewrite, task.used_seed, task.duplicate_of_seed = duplicate
                    break
                task.error = {
                    "code": "rewriter_no_valid_call",
                    "message": f"no valid sol_h3 call in {task.attempts_allowed} samples: {error}",
                }
                break
            key, now = text_key(request["prompt"]), time.time()
            seen = SEEN.setdefault(dedupe_key(task.prompt_sha, task.image_sha), {})
            prior = seen.get(key)
            if prior is not None and prior[0] != task.seed:
                task.attempts[-1]["duplicate_of_seed"] = prior[0]
                if dedupe_left > 0:
                    dedupe_left -= 1
                    STATS["duplicate_resamples"] += 1
                    duplicate = duplicate or (request["prompt"], seed, prior[0])
                    continue
                task.duplicate_of_seed = prior[0]
            seen.setdefault(key, (task.seed, now))
            task.rewrite, task.used_seed = request["prompt"], seed
            break
    except Exception as exc:  # noqa: BLE001 - every failure becomes a failed task the caller can see
        code = "rewriter_rejected" if isinstance(exc, BackendRejected) else "rewriter_unavailable"
        task.error = {"code": code, "message": str(exc)[:500]}
    if task.kind == "i2v":
        task.image = None  # free the pixels; the sha256 stays
        if task.rewrite is not None:
            task.rewrite, added = image_input.render_ready(task.rewrite)
            task.image_meta["first_frame_line_added"] = added
    task.status = "succeeded" if task.rewrite is not None else "failed"
    task.finished_at = time.time()
    STATS[task.status] += 1
    STATS[f"{task.kind}_{task.status}"] += 1
    RECENT.append(task.summary() | {"status": task.status})
    try:
        log_task(task, texts)
    except Exception as exc:  # noqa: BLE001
        print(f"log write failed for {task.id}: {exc!r}", flush=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] {task.id} {task.kind} {task.status} seed={task.seed} "
        f"attempts={len(task.attempts)} tokens={task.summary()['total_completion_tokens']} "
        f"elapsed={task.summary()['elapsed_s']}s",
        flush=True,
    )
    task.done.set()


def submit(prompt, seed, max_tokens, attempts, client, image=None):
    """Create (or reuse) a task; returns (task, None) or (None, error response). `image` is image_input.prepare's
    result as a dict, or None for text to video."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
        return None, err(400, f"prompt must be 1-{MAX_PROMPT_CHARS} characters", 2013)
    seed_given = seed is not None
    if seed_given and (not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= MAX_SEED):
        return None, err(400, f"seed must be an integer in [0, {MAX_SEED}]", 2013)
    max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens
    if not isinstance(max_tokens, int) or not 1024 <= max_tokens <= MAX_MODEL_LEN - 1024:
        return None, err(400, f"max_tokens must be in [1024, {MAX_MODEL_LEN - 1024}]", 2013)
    attempts = DEFAULT_ATTEMPTS if attempts is None else attempts
    if not isinstance(attempts, int) or not 1 <= attempts <= 5:
        return None, err(400, "attempts must be in [1, 5]", 2013)
    image_sha = image["sha256"] if image else None
    if seed_given:
        existing = TASKS.get(BY_KEY.get(task_key(sha256(prompt), image_sha, seed, max_tokens, attempts)))
        if existing and not (existing.status == "failed" and existing.error["code"] == "rewriter_unavailable"):
            STATS["reused"] += 1
            return existing, None
    queued = sum(1 for t in TASKS.values() if t.status == "queued")
    if queued >= MAX_QUEUE:
        STATS["rejected_queue_full"] += 1
        return None, err(429, "rewriter queue is full, retry later", 1002)
    task = Task(
        prompt, seed if seed_given else secrets.randbelow(MAX_SEED + 1), seed_given, max_tokens, attempts, client, image
    )
    TASKS[task.id] = task
    if seed_given:
        BY_KEY[task_key(task.prompt_sha, image_sha, seed, max_tokens, attempts)] = task.id
    STATS["submitted"] += 1
    STATS[f"{task.kind}_submitted"] += 1
    asyncio.get_running_loop().create_task(run_task(task))
    return task, None


async def prepare_image(url):
    """(image dict, None) or (None, 400 response)."""
    try:
        sha, obs, png_url, source_size = await asyncio.to_thread(image_input.prepare, url)
    except image_input.ImageRejected as exc:
        STATS["rejected_image"] += 1
        return None, err(400, str(exc), 2013)
    return {"sha256": sha, "observation": obs, "url": png_url, "source_size": source_size}, None


def authorized(request):
    header = request.headers.get("authorization", "")
    return header.startswith("Bearer ") and hmac.compare_digest(header[7:].strip().encode(), TOKEN.encode())


async def health_monitor():
    while True:
        for pool in POOLS:
            for url in pool.urls:
                try:
                    ok = (await CLIENT.get(url + "/health", timeout=5)).status_code == 200
                except httpx.HTTPError:
                    ok = False
                await pool.set_health(url, ok)
        await asyncio.sleep(5)


async def janitor():
    while True:
        await asyncio.sleep(600)
        now = time.time()
        for tid in [t.id for t in TASKS.values() if t.finished_at and now - t.finished_at > TASK_TTL_S]:
            TASKS.pop(tid, None)
        for key in [k for k, tid in BY_KEY.items() if tid not in TASKS]:
            BY_KEY.pop(key, None)
        for sha in list(SEEN):
            SEEN[sha] = {k: v for k, v in SEEN[sha].items() if now - v[1] < DEDUPE_TTL_S}
            if not SEEN[sha]:
                SEEN.pop(sha)


def restore_tasks():
    """Reload finished tasks of the last TASK_TTL_S from the logs, so polls survive a gateway restart."""
    cutoff = time.time() - TASK_TTL_S
    for path in sorted(LOG_DIR.glob("tasks-*.jsonl")):
        if path.stat().st_mtime < cutoff:
            continue
        for line in path.open():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (rec.get("finished_at") or 0) < cutoff:
                continue
            s = rec["summary"]
            task = Task(rec["prompt"], s["seed"], s["seed_given"], s["max_tokens"], 0, rec.get("client"))
            if rec.get("task", "t2v") == "i2v":
                task.kind, task.image_sha, task.image_meta = "i2v", rec["image_sha256"], s.get("image")
            task.id, task.status, task.rewrite, task.error = rec["task_id"], rec["status"], rec["rewrite"], rec["error"]
            task.submitted_at, task.started_at, task.finished_at = (
                rec["submitted_at"],
                rec["started_at"],
                rec["finished_at"],
            )
            task.attempts_allowed = s.get("attempts_allowed", len(rec["attempts"]))
            task.attempts = [{k: v for k, v in a.items() if k != "text"} for a in rec["attempts"]]
            task.used_seed, task.duplicate_of_seed = s.get("used_seed"), s.get("duplicate_of_seed")
            task.done.set()
            TASKS[task.id] = task
            if task.seed_given and task.status == "succeeded":
                BY_KEY[task_key(task.prompt_sha, task.image_sha, task.seed, task.max_tokens, task.attempts_allowed)] = (
                    task.id
                )
            if task.rewrite is not None:
                # The duplicate check compares the policy's own prompt text (without the first-frame line).
                text = task.rewrite
                if task.kind == "i2v" and text.startswith(image_input.FL2VA_START):
                    text = text[len(image_input.FL2VA_START) :]
                SEEN.setdefault(dedupe_key(task.prompt_sha, task.image_sha), {}).setdefault(
                    text_key(text), (task.seed, task.finished_at)
                )
            STATS["restored"] += 1


@asynccontextmanager
async def lifespan(app):
    global CLIENT
    n_urls = sum(len(pool.urls) for pool in POOLS)
    CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(3600, connect=10),
        limits=httpx.Limits(max_connections=n_urls * SLOTS + 32, max_keepalive_connections=n_urls * SLOTS + 32),
    )
    await asyncio.to_thread(policy.tokenizer)
    await asyncio.to_thread(image_input.processor)
    restore_tasks()
    loop = asyncio.get_running_loop()
    background = [loop.create_task(health_monitor()), loop.create_task(janitor())]
    print(
        f"gateway up (t2v + i2v): backends={BACKENDS} i2v_backends={I2V_BACKENDS} slots/backend={SLOTS} "
        f"max_tokens={DEFAULT_MAX_TOKENS} "
        f"max_model_len={MAX_MODEL_LEN} restored={STATS['restored']}",
        flush=True,
    )
    yield
    for b in background:
        b.cancel()
    await CLIENT.aclose()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
async def health():
    healthy = sum(POOL.healthy.values())
    return JSONResponse(
        {
            "ok": healthy > 0,
            "backends_healthy": healthy,
            "backends": len(BACKENDS),
            "tasks": ["t2v", "i2v"],
            "i2v_backends_healthy": sum(I2V_POOL.healthy.values()),
            "i2v_backends": len(I2V_BACKENDS),
        },
        status_code=200 if healthy else 503,
    )


@app.post("/v2/h3_context_ir")
async def context_ir_submit(request: Request):
    if not authorized(request):
        return err(401, "invalid api key", 1004)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return err(400, "body must be JSON", 2013)
    content = body.get("content") if isinstance(body, dict) else None
    items = [c for c in content or [] if isinstance(c, dict)]
    texts = [c.get("text") for c in items if c.get("type") == "text"]
    images = [c for c in items if c.get("type") == "image_url"]
    if (
        not isinstance(content, list)
        or len(items) != len(content)
        or len(texts) != 1
        or len(texts) + len(images) != len(content)
        or len(images) > 1
    ):
        return err(
            400,
            "content must be one text item, optionally with one first-frame image_url item (text or image to video)",
            2013,
        )
    image = None
    if images:
        item = images[0]
        if item.get("role", "first_frame") != "first_frame":
            return err(400, "only a first_frame image is supported", 2013)
        url = item.get("image_url")
        url = url.get("url") if isinstance(url, dict) else url
        image, error = await prepare_image(url)
        if error:
            return error
    task, error = submit(
        texts[0],
        body.get("seed"),
        body.get("max_tokens"),
        body.get("attempts"),
        request.client.host if request.client else None,
        image,
    )
    if error:
        return error
    return {"task_id": task.id, "base_resp": {"status_code": 0, "status_msg": "success"}}


@app.get("/v2/query/video_generation/{task_id}")
async def context_ir_query(task_id: str, request: Request):
    if not authorized(request):
        return err(401, "invalid api key", 1004)
    task = TASKS.get(task_id)
    if task is None:
        return err(404, f"unknown task {task_id}", 2013)
    return task.view()


@app.post("/v1/rewrite")
async def rewrite(request: Request):
    """Blocking: {"prompt", "n" (1-8, default 1), "seed" (lane k samples with seed + k; default random),
    "max_tokens", "attempts", "image" (optional data URL: image to video)} -> one entry per lane once all lanes
    finish."""
    if not authorized(request):
        return err(401, "invalid api key", 1004)
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return err(400, "body must be JSON", 2013)
    if not isinstance(body, dict):
        return err(400, "body must be a JSON object", 2013)
    n, seed = body.get("n", 1), body.get("seed")
    if not isinstance(n, int) or not 1 <= n <= 8:
        return err(400, "n must be in [1, 8]", 2013)
    if seed is not None and (not isinstance(seed, int) or not 0 <= seed <= MAX_SEED - 8):
        return err(400, f"seed must be an integer in [0, {MAX_SEED - 8}]", 2013)
    image = None
    if body.get("image") is not None:
        image, error = await prepare_image(body["image"])
        if error:
            return error
    tasks = []
    for lane in range(n):
        task, error = submit(
            body.get("prompt"),
            None if seed is None else seed + lane,
            body.get("max_tokens"),
            body.get("attempts"),
            request.client.host if request.client else None,
            image,
        )
        if error:
            return error
        tasks.append(task)
    t0 = time.time()
    await asyncio.gather(*(t.done.wait() for t in tasks))
    return {
        "rewrites": [
            {"lane": lane, "task_id": t.id, "status": t.status, "prompt": t.rewrite, "error": t.error, **t.summary()}
            for lane, t in enumerate(tasks)
        ],
        "elapsed_s": round(time.time() - t0, 2),
    }


@app.get("/v1/stats")
async def stats(request: Request):
    if not authorized(request):
        return err(401, "invalid api key", 1004)

    def pct(values, q):
        values = sorted(v for v in values if v is not None)
        return values[min(len(values) - 1, int(q * len(values)))] if values else None

    recent = list(RECENT)
    by_status = collections.Counter(t.status for t in TASKS.values())
    return {
        "uptime_s": round(time.time() - STARTED),
        "counters": dict(STATS),
        "tasks_in_memory": dict(by_status),
        "waiting_for_slot": POOL.waiting,
        "busy": POOL.busy,
        "healthy": POOL.healthy,
        "slots_per_backend": SLOTS,
        **(
            {"i2v_waiting_for_slot": I2V_POOL.waiting, "i2v_busy": I2V_POOL.busy, "i2v_healthy": I2V_POOL.healthy}
            if I2V_POOL is not POOL
            else {}
        ),
        "config": {
            "model": MODEL_NAME,
            "checkpoint": CHECKPOINT,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "tasks": ["t2v", "i2v"],
            "i2v_observation_max_pixels": image_input.OBSERVATION_MAX_PIXELS,
            "max_model_len": MAX_MODEL_LEN,
            "attempts": DEFAULT_ATTEMPTS,
            "sampling": policy.SAMPLING,
            "max_queue": MAX_QUEUE,
        },
        "recent": {
            "n": len(recent),
            **{f"elapsed_s_p{int(q * 100)}": pct([t["elapsed_s"] for t in recent], q) for q in (0.5, 0.9, 0.99)},
            **{f"queue_s_p{int(q * 100)}": pct([t["queue_s"] for t in recent], q) for q in (0.5, 0.9)},
            "tokens_p50": pct([t["total_completion_tokens"] for t in recent], 0.5),
            "failed": sum(t["status"] == "failed" for t in recent),
        },
    }
