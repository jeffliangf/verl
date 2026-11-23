# RL step-300 rewriter service (for candidate-10 / candidate-11), 2026-11-13

> **Repository copy.** The service runs from `/home/jeff/rl-benchmarks/rl-rewriter-s300-service` on the cluster; relative paths below refer to that directory. The public endpoint address is redacted as `<login-node-ip>`, and `secrets/` (API token) and `logs/` (task records) are not in the repository.

**Building candidate-10/11? Start with `CANDIDATE-10-11.md` in this directory.** It is the build guide for bestofn,
with directory pointers and a prompt for Claude Code. This README covers the service itself.

The GRPO step-300 Qwen rewriter (`grpo-group8-16h100-20261031-resume-20261102`, global_step_300; Qwen3.8-27B) served
behind an API that speaks MiniMax's H3 Context-IR protocol, so bestofn can swap it in for Context-IR by changing only
the host and the API key.

- candidate-10 = this rewriter → one SMD-E draft (seed 42): candidate-8's lane 1 with the rewriter swapped.
- candidate-11 = this rewriter → four SMD-E drafts (seeds 42–45) → candidate-8's Gemini best-of-4 picker.

In the offline comparison (`../rl-rewriter-vs-context-ir-candidate8-20261112/RESULTS.md`), this rewriter beat
Context-IR 59.4% in both setups (Gemini 4 Argon, both orders, 1,000 AA prompts).

**Since 2026-11-23 it also takes image-to-video requests** (one first frame), on the same endpoint and protocol. See
"Image to video" below. Text-to-video requests are served exactly as before. Their engine is the same text-only vLLM
server, and its outputs were checked to be bit-identical to the old server's.

## Endpoint

- `http://<login-node-ip>:18400` (the login node utopai-m1). `relay.sh` forwards it to the gateway on the serving node.
  Compute-node ports are filtered from outside the cluster, while the login node's are reachable; both were checked
  from the arena VM.
- Auth: `Authorization: Bearer <token>`. The token is in `secrets/api_token` (mode 600). Share it privately.
  Every call except `GET /health` needs it.
- Plain HTTP. The token and prompts cross the internet unencrypted. If that matters, use an SSH tunnel to
  utopai-m1 instead (`ssh -N -L 18400:127.0.0.1:18400 <user>@<login-node-ip>`). It goes through the relay, so it keeps
  working when the service moves to another node.

## Calling it

**As Context-IR (recommended for bestofn).** The protocol is the same as `adapters.BestOfNMinimaxH3.context_ir`:

```python
cloud = adapters.BestOfNMinimaxH3()
cloud.host, cloud.api_key = "http://<login-node-ip>:18400", QWEN_REWRITER_TOKEN
text = cloud.context_ir(prompt, duration=10, ratio="16:9", timeout_s=450)
```

- `POST /v2/h3_context_ir` with `{"content": [{"type": "text", "text": PROMPT}], "seed": LANE_SEED}` returns
  `{"task_id"}`. `model`, `duration` and `ratio` are accepted and ignored. The content is one text item, plus
  optionally one first-frame image for image to video, exactly as bestofn's I2V lanes already send it
  (`h3_selfhost._context_ir_task`):
  `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}, "role": "first_frame"}`.
  Only base64 `data:` URLs are accepted (PNG, JPEG or WebP, at most 20 MiB); remote URLs, other roles, a second image
  and `video_url` items get HTTP 400.
- `GET /v2/query/video_generation/{task_id}` returns `task.status`: `queued`, `running`, `succeeded` (with
  `task.content.prompt`) or `failed` (with `task.error.code`). `task.rewriter` reports the seed, attempts, tokens and
  timings.
- **Seed.** Pass each lane's render seed, i.e. AA's seed plus `seed_offset`, which gives 42–45. The four lanes then
  sample differently and match the offline eval exactly, and a rerun reproduces the same text.
  - Without a seed, each call draws a random one. The lanes still differ, but a rerun does not reproduce.
  - Never send the same seed for two lanes. A repeated `(prompt, seed)` returns the existing task, by design, so
    retries are idempotent; two lanes with the same seed would therefore get the same text.
- **Failures.** `rewriter_no_valid_call` means the policy produced no valid `sol_h3` call in 3 samples.
  `rewriter_unavailable` means the backends are down, and `rewriter_rejected` means the prompt is too long. In every
  case, render AA's raw prompt, which is candidate-8's rule for Context-IR failures; `h3_selfhost` already does this
  for any exception from `context_ir`.

**Blocking call.** `POST /v1/rewrite` with `{"prompt": PROMPT, "n": 4, "seed": 42}` returns four rewrites (sampling
seeds 42–45) once all are done. Add `"image": DATA_URL` for image to video.

See `client_example.py` for a standard-library client. It runs four lanes in parallel through submit + poll.

## What the service does per task

1. **Input.** It builds the policy's exact training input, frozen in `policy_spec.json`: system prompt,
   "Generate a video of about 10 seconds of: …", the `sol_h3` tool schema, thinking on, and `reasoning_effort` xhigh.
   This is token-identical to the eval for all 1,002 AA prompts, and the parser recovers all 4,008 eval rewrites
   (`tests/check_policy_equivalence.py`).
2. **Sampling.** It uses training sampling (T 0.7, top-p 0.9) with the task's seed. An output without a valid call
   is resampled with seed + 1000 per attempt, at most 3 attempts (the eval's rule).
3. **Duplicates.** A valid rewrite whose text, with whitespace and case folded, was already returned for the same
   prompt under another seed is resampled, at most twice. So lanes of one job never get the same text, even if a
   caller's seeds collide in the sampler.
4. **Scheduling.** Each generation goes to the least-loaded healthy vLLM backend, with at most 64 in flight per
   backend; the rest wait in arrival order. A backend error or 5xx marks the backend unhealthy, and the generation
   is retried (up to 6 times) once a backend is healthy again, on another backend when there are several.
5. **Logging.** Every finished task is appended to `logs/tasks-YYYYMMDD.jsonl`, with all attempts including the
   thinking text. Finished tasks are reloaded after a gateway restart, so polls keep working.

## Image to video (2026-11-23)

- **Input.** s300 was trained on text to video only. An I2V request therefore gets exactly the text-to-video input
  above: same system prompt, same `sol_h3` tool, same "Generate a video of about 10 seconds of: …" wording. The only
  addition is the first frame at the start of the user turn.
  - `image_input.prompt_ids` checks this on every request: the token ids minus the image block must equal the
    text-to-video ids. Otherwise the task fails.
  - The model sees the frame as the joint T2V/I2V RL runs show it to their policy: EXIF-upright, at most 768 × 768
    pixels in 32-pixel tiles. A 1920 × 1080 frame becomes 1024 × 576, which is 576 image tokens.
- **Output.** The model answers with the same text-to-video `sol_h3` call. The gateway returns its `prompt` behind the
  Sol-H3 first-frame line, `For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1])
  is fully referenced.` and a blank line. This is the line Context-IR's I2V expansions open with, and the one
  bestofn's raw I2V fallback (`I2V_RAW_PREFIX`) adds.
  - Task reuse and the duplicate check key on (prompt, image).
  - `task.rewriter` adds `task: "i2v"`, `image_sha256` and the image sizes.
- **Engines.** Two vLLM servers share the job's 4 GPUs (44% of memory each).
  - **Text-only server (port 18201, image limit 0).** Serves all text to video. It is the old server's exact
    configuration except for the memory share. An image-enabled vLLM compiles a different graph (embeddings in
    instead of token ids), and that changes text-to-video log-probabilities at bf16 rounding level. That is why
    text never goes to the image server.
  - **Image server (port 18301, one image per prompt).** Serves image to video only. It starts once the text server
    is healthy, so their memory profiling never overlaps. It has its own compile cache: vLLM 0.29's AOT cache key
    ignores the image limit, and with a shared cache the image server loaded the text server's compiled forward and
    crashed.
  - KV capacity is about 1.2 M tokens per server, or ~29 concurrent requests at the full 40,960 tokens.
- **Checks (2026-11-23).**
  - **Text to video is unchanged**, measured with `tests/t2v_backend_equivalence.py` against the old server (job
    1151, scored before it stopped). The test covers 19 logged rewrites teacher-forced (35,090 tokens) plus 8 greedy
    continuations of 384 tokens. Every log-probability and every greedy token is identical.
  - For comparison, a single image-enabled server (job 1340) differed from the old server by mean |Δ log-prob| 0.0086,
    with the top-1 token differing at 0.57% of positions. Batching alone on one server gives 0.0018 and 0.13%
    (`tests/t2v_batch_noise.py`).
  - **Image to video works** (`tests/i2v_live_check.py`, results in `tests/i2v-live-check.json`).
    - bestofn's own `context_ir` client with a first-frame item returned a rewrite in 40 s.
    - 25 AA batch-2 I2V items × 4 lanes, all submitted at once: 100/100 valid on the first sample, 4 distinct
      rewrites per item, and all of them carry the first-frame line.
    - Median 63 s per task with 100 in flight (max 147 s); median 3,682 tokens; median rewrite 3,781 characters.
    - 99/100 reasoning traces work from the image (e.g. "looking at the image, the egg is behind the glass, and the
      water line is at about the middle of the egg").
  - **Not measured:** I2V quality. s300 never saw an image in training, so it is a zero-shot first-frame rewriter;
    no renders or judgments have been run.

## Measurements (2026-11-13, job 965, node 004 = 2 × TP4 H100)

Since 2026-11-19 the service runs a single TP4 server (see Operations). The numbers below were measured with two.

Load test (`tests/loadtest.py`, results in `tests/loadtest-results.json` and `tests/summary.json`):
- Real AA batch-2 prompts, each level on disjoint prompts, seeds 42–45.
- Everything goes through the Context-IR protocol, and timings are the gateway's (submit → finish).
- 428 tasks in all: 0 failures and 0 retries. 7 needed more than 8,192 tokens and now finish instead of being cut
  off.

| rewrites in flight | p50 | p90 | max | tokens/s per request | node tokens/s | rewrites/hour at this load |
|---|---|---|---|---|---|---|
| 1 (idle server) | 36 s | 69 s | 69 s | 124 | 121 | – |
| 8 (candidate-8's peak: 2 jobs × 4 lanes) | 30 s | 52 s | 52 s | 111 | 496 | 550 |
| 32 (8 best-of-4 jobs) | 40 s | 60 s | 67 s | 100 | 1,969 | 1,690 |
| 128 (all slots full) | 48 s | 72 s | 103 s | 78 | 5,008 | 4,420 |
| 256 (half wait for a slot, up to 74 s) | 75 s | 113 s | 159 s | 73 | 6,156 | 5,740 |

**Production Context-IR in candidate-8.** Measured from the 3,999 calls in the 2026-11-09..11 ledgers: median 26.6 s,
p90 37.6 s, p99 74.2 s, max 148 s. At most 8 calls ran at once, because the profile has `max_active_jobs: 2`, and the
peak was about 50 jobs per hour.

**What this means.**
- At candidate-8's real load, a Qwen rewrite takes about as long as a Context-IR call.
- With all 128 slots busy, each backend sustains about 4,900 tokens/s (vLLM's own counters), about 9,800 per node.
  That is roughly 9,000 rewrites per hour in steady state; the 256-task burst measured 5,700/h only because of its
  ramp-up and tail.
- With one server (64 slots), expect about half: roughly 4,500 rewrites per hour (estimated, not re-measured).
  Rewrites beyond 64 in flight wait for a slot.
- Even the burst rate is about 20 times what candidate-10 and candidate-11 together would need at candidate-8's
  peak (50 jobs/h × 5 rewrites = 250 rewrites/h).
- Rendering stays the bottleneck: 8 shared H200 SMD-E workers at 50–90 s per draft.

**Same input, same sampling as the eval.** For the same prompt and seed (428 items), the service and the eval agree
for a median of 733 characters before the samples diverge (low-level numerics: CUDA graphs vs eager). The
distributions match:

| | service | eval |
|---|---|---|
| median tokens | 3,619 | 3,651 |
| median rewrite length | 3,622 characters | 3,655 characters |

**Distinct lanes.** In the eval's 1,002 prompts × 4 seeds there are 5,958 within-prompt pairs:
- 0 are identical;
- the closest pair shares 28% of its word 3-grams (median 11%).

So distinct seeds give clearly different rewrites, and the duplicate check is only insurance.

## The 8,192-token cap, lifted

RL training capped a response at 8,192 tokens, and the eval kept that cap.
- 135 of the eval's 4,122 sampled attempts hit it. The policy was still thinking each time.
- Those attempts were resampled up to 3 times, and 23 lanes never got a valid call, so they rendered the raw prompt.

`tests/cap_test.py` reran all 135 with the same prompt and seed, a 32,768 cap and a single attempt:
- **All 135 produced a valid `sol_h3` call, and none reached the 32k cap.**
- Tokens: median 8,662, maximum 20,654. By band: 64 under 8k, 50 at 8–12k, 17 at 12–16k, 4 at 16–21k.
- Median time was 113 s with about 130 running at once.

These were long rewrites, not runaway loops. The default cap is therefore **24,576**, which covers the longest seen
(20,654) with room to spare. Per request, `max_tokens` can go up to 39,936.

At candidate-8's load, about 110 tokens/s per request:
- 24k tokens take about 220 s, inside the 450 s budget below.
- An output that did loop to the cap would leave about 230 s for the resample.

## bestofn settings for the Qwen lanes

- **Time budget.** Raise `h3_selfhost_smd.context_ir.timeout_s` from 180 to 450.
  - h3_selfhost gives Context-IR `min(timeout_s, 1440 − 90 − 900) = min(timeout_s, 450)` seconds at lane start
    (generation_hard_timeout_s, minus the normalize reserve, minus the draft's own budget).
  - Anything slower falls back to the raw prompt, as a Context-IR failure would.
- **Seeds.** Send `"seed"` = the lane's render seed (42–45). candidate-10 is the seed-42 lane alone.
- **Submit retries.** bestofn submits Context-IR exactly once (`attempts=1`, `tries=1`): one network blip on submit
  means a raw-prompt lane. MiniMax's API needs that, because a resubmit would create and bill a new task. Here a
  resubmit with the same seed returns the same task, so retrying the submit 3–5 times on network errors and 5xx is
  safe and recommended. Polls already tolerate errors.
- **Header.** The policy often writes its first header as `context-imultimodal_description:`. The eval rendered it
  as-is; keep it, so candidate-10/11 match what was measured.

## Operations

- **Serving job.** `umask 022; sbatch -w <node> serve.sbatch` runs the gateway on 0.0.0.0:18400 and two vLLM
  servers (TP 4) on localhost: the text-only server on 18201 and the image server on 18301. It takes half a node
  (4 GPUs, 64 CPUs, 320 GB) and shares the node with other jobs. `serve.sh` gives each group of servers its own
  TP-sized group of the job's Slurm GPUs.
  - `I2V_ENGINE=0` runs a single image-enabled server for both tasks instead (memory share 0.88). Use it only for
    a checkpoint whose text-to-video numerics need not match a text-only server.
  - No time limit (`-t UNLIMITED`; partition `all` allows it). The job runs until it is cancelled or the node fails.
    vLLM servers that exit are restarted, and the gateway queues work until a backend is healthy.
  - Startup: the text-only server is ready in about 2 min, the image server about 3 min after that.
  - Current job: **1347 on gmicloud-loki-gpu-005** (started 2026-11-23 19:29Z; text to video plus image to video).
  - Before that, job 1151 on the same node served text to video only (2026-11-19 01:45Z to 2026-11-23 19:11Z).
    Its files are in `legacy-t2v-20261123/`. Jobs 1340 and 1346 (2026-11-23) were short-lived steps toward the
    current layout.
  - Before that, job 965 ran 2 servers on all of gmicloud-loki-gpu-004 with a 7-day limit (2026-11-13 21:56Z to
    2026-11-19 01:50Z). `serve.sbatch` shows how to start that setup again.
- **Relay.** On utopai-m1, run `./relay.sh start|stop|status`. If the job moves to another node, run
  `./relay.sh stop && ./relay.sh start`.
  - socat needs `backlog=1024`. Its default backlog of 5 overflowed when 135 clients connected at once
    (`nstat` TcpExtListenOverflows), and some connections were reset.
  - The relay runs outside Slurm, so it stops if utopai-m1 reboots.
- **Stop everything.** `scancel <job>; ./relay.sh stop`.
- **Health and load.** `curl http://<login-node-ip>:18400/health`; `GET /v1/stats` (authenticated) shows the queue,
  in-flight work per backend, counters and recent latency percentiles.
- **Logs.**
  - `logs/serve-<node>-<job>.out` (supervisor)
  - `logs/gateway-*.log`
  - `logs/vllm-<node>-<port>-<job>.log`
  - `logs/tasks-*.jsonl`
  - `logs/relay.log`
- **Settings** (environment variables of `serve.sh`): `INSTANCES` (1), `TP` (4), `MAX_NUM_SEQS` (64 per backend),
  `MAX_MODEL_LEN` (40,960), `MAX_TOKENS` (the default completion cap, 24,576; per request it can go up to 39,936 via
  `max_tokens`).
  - `settings.json` (`{"max_tokens": 24576}`) overrides the default at gateway start. To change it without giving
    up the node, kill only the gateway; `serve.sh` restarts it in 10 s:
    `srun --jobid=<job> --overlap -N1 -n1 --gres=none pkill -f 'uvicorn --app-dir'`.
    Finished tasks are reloaded from the logs, but tasks still running at that moment are lost, and their
    callers fall back to the raw prompt.

## Provenance

- **Model.** `../rl-rewriter-vs-context-ir-candidate8-20261112/model/global_step_300-hf` (verl.model_merger output,
  bf16, 51 GB). Its tokenizer and chat template are byte-identical to the checkpoint's `actor/huggingface`.
- **vLLM.** 0.29.0 from `~/.venvs/video-agents-h100-f169e786`, with `--generation-config vllm` so the checkpoint's
  top-k 20 / top-p 0.95 defaults do not apply.
  - CUDA graphs are on. The eval ran eager mode at about 20 tokens/s per request; this service runs about
    120 tokens/s per request.
  - The sampling distribution is the same. Low-level numerics differ, so the same seed does not give byte-identical
    text to the eval's.
- **Tests.**
  - `tests/test_gateway_mock.py`: protocol, auth, retries, duplicates, idempotency, backend loss and restart,
    against fake backends and through bestofn's own Context-IR client. Since 2026-11-23 it also covers image to
    video:
    - bestofn's first-frame body;
    - routing to the image backend;
    - per-image task identity;
    - rejected images;
    - restart.

    Run it as `TEST_TMP=$(mktemp -d) python tests/test_gateway_mock.py`; `I2V_SHARED=1` tests the single-server
    layout.
  - `tests/t2v_backend_equivalence.py`, `tests/t2v_batch_noise.py`: text-to-video numerics of a backend against a
    stored baseline (`tests/t2v-equivalence-*.json`).
  - `tests/i2v_live_check.py`: image to video through the live gateway.
  - `tests/loadtest.py`: latency under concurrency.
  - `tests/cap_test.py`: the eval's 135 capped attempts rerun with a 32k cap.
  - `tests/summarize.py`: writes `tests/summary.json`.
