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
  GET  /v2/query/video_generation/{task_id}
       -> {"task": {"status": "queued|running|succeeded|failed", "content": {"prompt": REWRITE}, "error": ...}}
  "model", "duration" and "ratio" are accepted and ignored. Optional extras: "seed" (sampling seed; pass the lane's
  render seed so the four lanes differ and a rerun reproduces), "max_tokens", "attempts".
Blocking convenience call:
  POST /v1/rewrite  {"prompt": PROMPT, "n": 4, "seed": 42}  -> {"rewrites": [one per lane, seeds 42..45]}
Also GET /health (no auth) and GET /v1/stats.

Each task samples the policy's exact training input (policy.py) with the training sampling settings. An output
without a valid sol_h3 call is resampled with seed + 1000 * attempt (the eval's rule); after `attempts` invalid
outputs the task fails, and a Context-IR caller falls back to the raw prompt as it does for Context-IR failures. A
valid rewrite whose text (whitespace and case folded) was already returned for the same prompt under a different
base seed is resampled too (at most twice), so lanes of one job never receive the same text. Submitting the same
(prompt, seed, max_tokens, attempts) again returns the existing task instead of sampling again.

Generations go to the least-loaded healthy vLLM backend, at most SLOTS per backend at a time; the rest wait in
arrival order. Backend transport errors and 5xx are retried on another backend.
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
import policy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# Logs and task records are for the whole utopai group (the job may have been submitted under umask 077).
os.umask(0o022)
ENV = os.environ
BACKENDS = [u.rstrip("/") for u in ENV["REWRITER_BACKENDS"].split(",") if u.strip()]
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
    def __init__(self, prompt, seed, seed_given, max_tokens, attempts, client):
        self.id = secrets.token_hex(12)
        self.prompt, self.prompt_sha = prompt, sha256(prompt)
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


POOL = Pool(BACKENDS, SLOTS)
TASKS = {}  # task_id -> Task
BY_KEY = {}  # (prompt_sha, seed, max_tokens, attempts) -> task_id, for tasks with a caller-given seed
SEEN = {}  # prompt_sha -> {text_key: (base_seed, time)}
RECENT = collections.deque(maxlen=2000)
STATS = collections.Counter()
STARTED = time.time()
CLIENT = None


def log_task(task, attempt_texts):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "task_id": task.id,
        "status": task.status,
        "submitted_at": task.submitted_at,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
        "client": task.client,
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
    body = {
        "model": MODEL_NAME,
        "prompt": ids,
        "max_tokens": max_tokens,
        "seed": seed,
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        **policy.SAMPLING,
    }
    last = None
    for net_try in range(6):
        url = await POOL.acquire()
        if task.started_at is None:
            task.started_at, task.status = time.time(), "running"
        t0 = time.time()
        try:
            r = await CLIENT.post(url + "/v1/completions", json=body)
            if r.status_code >= 500:
                raise httpx.HTTPStatusError(f"HTTP {r.status_code}", request=r.request, response=r)
            if r.status_code != 200:
                raise BackendRejected(f"HTTP {r.status_code}: {r.text[:300]}")
            resp = r.json()
            choice = resp["choices"][0]
            return choice["text"], {
                "finish_reason": choice.get("finish_reason"),
                "completion_tokens": (resp.get("usage") or {}).get("completion_tokens"),
                "elapsed_s": round(time.time() - t0, 2),
                "backend": url,
            }
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            STATS["backend_errors"] += 1
            await POOL.set_health(url, False)  # the health monitor re-enables it once /health answers
        finally:
            await POOL.release(url)
        await asyncio.sleep(min(60, 5 * 2**net_try))
    raise RuntimeError(f"no backend finished the generation: {last!r}")


async def run_task(task):
    texts = []
    try:
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
            seen = SEEN.setdefault(task.prompt_sha, {})
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
    task.status = "succeeded" if task.rewrite is not None else "failed"
    task.finished_at = time.time()
    STATS[task.status] += 1
    RECENT.append(task.summary() | {"status": task.status})
    try:
        log_task(task, texts)
    except Exception as exc:  # noqa: BLE001
        print(f"log write failed for {task.id}: {exc!r}", flush=True)
    print(
        f"[{time.strftime('%H:%M:%S')}] {task.id} {task.status} seed={task.seed} "
        f"attempts={len(task.attempts)} tokens={task.summary()['total_completion_tokens']} "
        f"elapsed={task.summary()['elapsed_s']}s",
        flush=True,
    )
    task.done.set()


def submit(prompt, seed, max_tokens, attempts, client):
    """Create (or reuse) a task; returns (task, None) or (None, error response)."""
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
    if seed_given:
        existing = TASKS.get(BY_KEY.get((sha256(prompt), seed, max_tokens, attempts)))
        if existing and not (existing.status == "failed" and existing.error["code"] == "rewriter_unavailable"):
            STATS["reused"] += 1
            return existing, None
    queued = sum(1 for t in TASKS.values() if t.status == "queued")
    if queued >= MAX_QUEUE:
        STATS["rejected_queue_full"] += 1
        return None, err(429, "rewriter queue is full, retry later", 1002)
    task = Task(
        prompt, seed if seed_given else secrets.randbelow(MAX_SEED + 1), seed_given, max_tokens, attempts, client
    )
    TASKS[task.id] = task
    if seed_given:
        BY_KEY[(task.prompt_sha, seed, max_tokens, attempts)] = task.id
    STATS["submitted"] += 1
    asyncio.get_running_loop().create_task(run_task(task))
    return task, None


def authorized(request):
    header = request.headers.get("authorization", "")
    return header.startswith("Bearer ") and hmac.compare_digest(header[7:].strip().encode(), TOKEN.encode())


async def health_monitor():
    while True:
        for url in BACKENDS:
            try:
                ok = (await CLIENT.get(url + "/health", timeout=5)).status_code == 200
            except httpx.HTTPError:
                ok = False
            await POOL.set_health(url, ok)
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
                BY_KEY[(task.prompt_sha, task.seed, task.max_tokens, task.attempts_allowed)] = task.id
            if task.rewrite is not None:
                SEEN.setdefault(task.prompt_sha, {}).setdefault(text_key(task.rewrite), (task.seed, task.finished_at))
            STATS["restored"] += 1


@asynccontextmanager
async def lifespan(app):
    global CLIENT
    CLIENT = httpx.AsyncClient(
        timeout=httpx.Timeout(3600, connect=10),
        limits=httpx.Limits(
            max_connections=len(BACKENDS) * SLOTS + 32, max_keepalive_connections=len(BACKENDS) * SLOTS + 32
        ),
    )
    await asyncio.to_thread(policy.tokenizer)
    restore_tasks()
    loop = asyncio.get_running_loop()
    background = [loop.create_task(health_monitor()), loop.create_task(janitor())]
    print(
        f"gateway up: backends={BACKENDS} slots/backend={SLOTS} max_tokens={DEFAULT_MAX_TOKENS} "
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
        {"ok": healthy > 0, "backends_healthy": healthy, "backends": len(BACKENDS)}, status_code=200 if healthy else 503
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
    texts = [c.get("text") for c in content or [] if isinstance(c, dict) and c.get("type") == "text"]
    if not isinstance(content, list) or len(texts) != 1 or len(content) != 1:
        return err(400, "content must be exactly one text item (text-to-video only)", 2013)
    task, error = submit(
        texts[0],
        body.get("seed"),
        body.get("max_tokens"),
        body.get("attempts"),
        request.client.host if request.client else None,
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
    "max_tokens", "attempts"} -> one entry per lane once all lanes finish."""
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
    tasks = []
    for lane in range(n):
        task, error = submit(
            body.get("prompt"),
            None if seed is None else seed + lane,
            body.get("max_tokens"),
            body.get("attempts"),
            request.client.host if request.client else None,
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
        "config": {
            "model": MODEL_NAME,
            "checkpoint": CHECKPOINT,
            "max_tokens": DEFAULT_MAX_TOKENS,
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
