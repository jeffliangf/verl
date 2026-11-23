# Candidate-10/11 build guide

> **Repository copy.** The service runs from `/home/jeff/rl-benchmarks/rl-rewriter-s300-service` on the cluster; relative paths below refer to that directory. The public endpoint address is redacted as `<login-node-ip>`, and `secrets/` (API token) and `logs/` (task records) are not in the repository.

As of 2026-11-13. Owner: `jeff` on utopai-m1. Shared copy: https://claude.ai/artifact/DCSXC44FodYaRoMrrPoKju. This file
is the canonical version; every path below is on utopai-m1 and readable by the `utopai` group.

Candidate-10 and candidate-11 are candidate-8 with one change: the RL step-300 Qwen rewriter replaces H3 Context-IR.
The rewriter is live at `http://<login-node-ip>:18400` and speaks MiniMax's Context-IR protocol, so bestofn needs:
- one config block;
- a small client class;
- a seed passed through;
- two profiles.

## What to build

| | Rewriter (one call per lane) | SMD-E drafts, 768p | Pick | Profile |
|---|---|---|---|---|
| candidate-8 (live) | cloud H3 Context-IR | 4, seeds 42–45 | Gemini best-of-4 picker | `candidate-h3-smd-best-of-4` |
| **candidate-10** | Qwen s300, sampling seed 42 | 1, seed 42 | none: a lone survivor wins | new |
| **candidate-11** | Qwen s300, sampling seeds 42–45 | 4, seeds 42–45 | candidate-8's picker, unchanged | new |

**Why.** Offline, on AA batch-2 (1,000 prompts, judged by Gemini 4 Argon in both orders), Qwen s300 beat Context-IR
59.4% both ways: one draft against one draft, and best-of-4 against best-of-4. See `RESULTS.md` in the eval
directory below.
- Calibrated against AA's human votes for candidate-8 (AA name "Bolero", Elo 1170), that projects to about 1168 for
  candidate-10.
- The same projection gives about 1192 [1183–1206] for candidate-11.
- These are projections, not human votes.

## How a job runs

```
AA request: raw prompt, seed s (42)
  └─ bestofn job ─┬─ lane k   (candidate-11: k = 0..3; candidate-10: k = 0)
                  │    1. rewrite  POST /v2/h3_context_ir {prompt, seed: s+k} → poll → enhanced prompt
                  │                failure, or no answer within 450 s → AA's raw prompt (candidate-8's rule)
                  │    2. draft    SMD-E worker, seed s+k, 10 s, 768p
                  └─ pick: Gemini picker over the 4 drafts (candidate-11) | the only draft (candidate-10) → 768p delivery
```

## The rewriter API

- **Base URL.** `http://<login-node-ip>:18400`. That is utopai-m1; a relay forwards it to the gateway on
  gmicloud-loki-gpu-005. It is reachable from the arena VM and from inside the cluster.
- **Token.** `/home/jeff/rl-benchmarks/rl-rewriter-s300-service/secrets/api_token`, readable by the `utopai` group.
  - Send it as `Authorization: Bearer <token>`. On the VM, keep it in an env var such as `QWEN_REWRITER_TOKEN`.
  - Never commit it or paste it into chat.
- **Transport.** Plain HTTP. For an encrypted path, tunnel instead:
  `ssh -N -L 18400:127.0.0.1:18400 <you>@<login-node-ip>`. It goes through the relay, so it keeps working when the
  service moves to another node.

| Call | Body | Answer |
|---|---|---|
| `POST /v2/h3_context_ir` | `{"content": [{"type": "text", "text": PROMPT}], "seed": LANE_SEED}`. `model`, `duration` and `ratio` are accepted and ignored. Optional: `max_tokens` (default 24,576, max 39,936) and `attempts` (default 3, 1–5). | `{"task_id": "…", "base_resp": {"status_code": 0}}` |
| `GET /v2/query/video_generation/{task_id}` | none | `{"task": {"status": "queued" \| "running" \| "succeeded" \| "failed", "content": {"prompt": REWRITE}, "error": {"code", "message"}, "rewriter": {seed, attempts, completion_tokens, elapsed_s, …}}}` |
| `POST /v1/rewrite` (blocking) | `{"prompt": PROMPT, "n": 4, "seed": 42}` | `{"rewrites": [one entry per lane, sampling seeds 42..45]}` |
| `GET /health` (no token) | none | `{"ok": true, "backends_healthy": 2, "backends": 2}` |
| `GET /v1/stats` | none | queue, in-flight work per backend, counters, recent latency |

- **Seeds.**
  - Send the lane's render seed (AA seed + `seed_offset`, i.e. 42–45). Lanes then differ, the setting matches the
    offline eval, and a rerun reproduces the same text.
  - The same (prompt, seed) returns the existing task. That makes retries idempotent, and it also means two lanes
    with the same seed get the same text.
  - With no seed, each call draws a random one: the lanes still differ, but a rerun does not reproduce.
  - **Duplicate guard.** A rewrite that matches, ignoring whitespace and case, one already returned for the same
    prompt under another seed is resampled.
- **Errors.** In all three cases, render AA's raw prompt; `h3_selfhost` already does this for any exception from
  `context_ir`.
  - `rewriter_no_valid_call`: 3 samples without a valid call.
  - `rewriter_unavailable`: the backends are down.
  - `rewriter_rejected`: the prompt is too long.
  - HTTP errors: 401 is a bad token, 400 is a bad body (text-to-video only), and 429 means the queue holds more than
    5,000 tasks.
- **What it does with a prompt.**
  - It wraps the prompt in the exact RL training input: the RL system prompt, the `sol_h3` tool schema,
    "Generate a video of about 10 seconds of: {prompt}", and thinking at `xhigh`.
  - It samples at temperature 0.7, top-p 0.9 and returns only the `prompt` argument of the model's single `sol_h3`
    call.
  - The model thinks for about 3–4k tokens first. Its `duration` (always 10) is ignored, and it never sets a seed.
- **Header quirk.** The first header often reads `context-imultimodal_description:`. The eval rendered it as-is, so
  keep it.

## Changes in bestofn

Where to look in a video-generation-eval checkout. The reference checkout is `/home/jeff/video-generation-eval`,
HEAD 1fb145e; candidate-8's deployed copy is in the eval directory's `candidate8-production/service-provenance/`.
- `bestofn/config.json`:
  - `external_model_aliases` (`candidate-8` → `candidate-h3-smd-best-of-4`)
  - `external_model_settings.candidate-8` (`seed: passthrough`, `resolution: 768p`)
  - `profiles.candidate-h3-smd-best-of-4` (`max_active_jobs` 2, `job_timeout_s` 1740,
    `generation_hard_timeout_s` 1440)
  - `models[minimax-h3-smd-repeat1..4]` (`seed_offset` 0–3)
  - `h3_selfhost_smd` (the workers, and `context_ir.timeout_s` 180)
- `bestofn/adapters.py`:
  - `build_adapter()`, branch `"h3-selfhost-primary"`, which runs
    `PersistentSelfHostedH3(config[block], …, cloud=BestOfNMinimaxH3())`
  - `BestOfNMinimaxH3.context_ir()`, which submits exactly once, and `_poll_task()`
- `bestofn/h3_selfhost.py`:
  - `PersistentSelfHostedH3._generate()`: the lane seed is `params["seed"]`
  - `_expanded_prompt()`: the Context-IR budget `min(timeout_s, deadline − 90 − 900)` and the raw-prompt fallback
- `bestofn/judge.py`: `run_tournament()` returns a single candidate without judging, so candidate-10 needs no picker
  change.

Suggested minimal implementation:
1. **Config block.** Add `h3_selfhost_smd_qwen`: a copy of `h3_selfhost_smd` with its own `artifact_root` and
   ```json
   "context_ir": {"enabled": true, "timeout_s": 450, "host": "http://<login-node-ip>:18400",
                  "token_env": "QWEN_REWRITER_TOKEN", "pass_seed": true}
   ```
   450 s is the most `_expanded_prompt()` can grant at lane start (1440 − 90 − 900). At 180 s, slow rewrites would
   fall back to the raw prompt.
2. **Client.** Add `QwenRewriterClient` to `adapters.py`. This code was tested against the live service: same seed
   gives the same text, a new seed gives a new text, and `ready()` is True.
   ```python
   class QwenRewriterClient(BestOfNMinimaxH3):
       """The RL s300 rewriter service: MiniMax's Context-IR protocol plus a sampling seed and submit retries."""

       def __init__(self, host, token):
           super().__init__()
           self.host, self.api_key = host.rstrip("/"), token

       def context_ir(self, prompt_text, *, seed=None, timeout_s=450, **_ignored):
           deadline = time.time() + float(timeout_s)
           body = {"content": [{"type": "text", "text": prompt_text}]}
           if seed is not None:
               body["seed"] = int(seed)
           # the same (prompt, seed) returns the same task, so retrying an ambiguous submit is safe
           task_id = gv.http_json("POST", self.host + "/v2/h3_context_ir", headers=self._headers(),
                                  body=body, timeout=30, tries=4)["task_id"]
           return self._poll_task(task_id, "prompt", deadline=deadline, interval_s=5)
   ```
   In `build_adapter()`, build the block's client as
   `QwenRewriterClient(ir["host"], os.environ[ir["token_env"]])` when `config[block]["context_ir"]` has a `host`.
   This is safe only because the block has `regenerate_2k: false`: with that setting, the cloud client is used for
   nothing but Context-IR.
3. **Seed.** Pass `seed` from `_generate()` into `_expanded_prompt()`, and call
   `self.cloud.context_ir(..., seed=seed)` when `context_ir.pass_seed` is set. The MiniMax client does not accept
   `seed`.
4. **Lanes.** Add `minimax-h3-smd-qwen-repeat1..4`: copies of `minimax-h3-smd-repeat1..4` (`seed_offset` 0–3) with
   `selfhost_config: h3_selfhost_smd_qwen`.
5. **Profiles.**
   - `candidate-h3-smd-qwen-best-of-4`: four lanes, with candidate-8's limits and `normalize`. Alias it as
     `candidate-11`.
   - `candidate-h3-smd-qwen-one-lane`: lane repeat1 only. Alias it as `candidate-10`.
   - Copy candidate-8's `external_model_settings` to both.

Decisions for the implementer:
- **Workers.** candidate-10/11 would use the same 8 H200 SMD-E workers as candidate-8 (the VM tunnels on
  31030–31037). `build_adapter()` keeps one `InstancePool` per config block, so separate blocks on the same workers
  schedule independently, with up to one task from each pool per worker. Either share the pool or give the new
  candidates their own workers.
- **Concurrency.** `max_active_jobs` can stay at candidate-8's 2. The rewriter is not the limit; see the next section.

## Validate before AA traffic

- [ ] From the VM, `curl -s http://<login-node-ip>:18400/health` returns `"ok": true` with 2 backends.
- [ ] One candidate-11 job on an AA batch-2 prompt.
  - [ ] All four lane ledgers show `stages.context_ir.status == "complete"`, with `elapsed_s` under 450 and 2–5k
    characters.
  - [ ] The four `context-ir.txt` files differ, and the drafts use seeds 42–45.
- [ ] One candidate-10 job: one lane, no judge call, and the delivery is that lane's draft.
- [ ] Rewrite lengths match the offline eval: median about 3.6k characters. Texts will not match byte for byte.
  - Find a prompt's eval rewrites under `prompt_id = sha256(raw prompt)`, in
    `/home/jeff/rl-benchmarks/rl-rewriter-vs-context-ir-candidate8-20261112/canonical/prompts/<prompt_id>/qwen_s300/lane{1..4}/rewrite.txt`.
- [ ] A forced failure (for example, a wrong token in the env var) ends in `context_ir.status == "raw_prompt"`, and
  the lane still delivers.
- [ ] A 20-prompt dry run at production concurrency shows no `rewriter_unavailable` errors and no raw-prompt lanes.

## Capacity and latency

The load test used real AA prompts through the Context-IR protocol: 428 tasks, with 0 failures and 0 retries.

| Rewrites in flight | Median | Max | Tokens/s per request |
|---|---|---|---|
| 1 | 36 s | 69 s | 124 |
| 8 (candidate-8's peak: 2 jobs × 4 lanes) | 30 s | 52 s | 111 |
| 32 (8 best-of-4 jobs) | 40 s | 67 s | 100 |
| 128 (all slots busy) | 48 s | 103 s | 78 |
| 256 (half queued, up to 74 s) | 75 s | 159 s | 73 |

- **Compared with Context-IR.** Production Context-IR in candidate-8 (3,999 calls, 2026-11-09..11) had a median of
  26.6 s, p99 74 s and a maximum of 148 s. At candidate-8's real load, a Qwen rewrite takes about as long.
- **Capacity.** The table was measured with 2 vLLM servers (TP4 on H100), which sustained about 9,000 rewrites per
  hour. Since 2026-11-19 the service runs one server, so expect about 4,500 per hour (estimated). Candidate-10 and
  candidate-11 together need about 250 per hour at candidate-8's peak of 50 jobs per hour. SMD-E rendering stays the
  bottleneck.
- **Token cap.** RL training capped a response at 8,192 tokens, and 3.3% of the eval's samples hit that cap. Rerun
  with room to finish, all 135 of those produced a valid call: median 8,662 tokens, longest 20,654. The service's
  cap is 24,576 tokens.

## Operations

| What | Where / how |
|---|---|
| Serving job | Slurm job 1151 on gmicloud-loki-gpu-005: one vLLM server (TP 4, 4 of the node's 8 GPUs) plus the gateway. Started 2026-11-19 01:45Z, **no time limit**. It replaced job 965 (gmicloud-loki-gpu-004, 2 servers, 7-day limit). |
| Public relay | `relay.sh` on utopai-m1 (socat), outside Slurm; it stops if utopai-m1 reboots. |
| Health, load | `GET /health`; `GET /v1/stats` with the token. |
| Logs | `/home/jeff/rl-benchmarks/rl-rewriter-s300-service/logs/`. `tasks-YYYYMMDD.jsonl` holds every task with the model's full output; also `gateway-*.log`, `vllm-*.log`, `serve-*.out`. |
| Restart, stop | jeff: `scancel <job>` then `sbatch -w <node> serve.sbatch` and `./relay.sh stop && ./relay.sh start`. vLLM servers that exit restart by themselves. |
| Your own copy | Copy the service directory, point `SVC` and `LOGS` in `serve.sh` at your copy, and submit `serve.sbatch` on a free node. The model and venv below are group-readable. |

## Directory pointers

`/home/jeff/rl-benchmarks` can be entered but not listed, so use these exact paths.

- `/home/jeff/rl-benchmarks/rl-rewriter-s300-service/`: the service.
  - `README.md`: operations and measurements.
  - `gateway.py`: the API.
  - `policy.py` and `policy_spec.json`: the exact model input and the tool-call parser.
  - `serve.sh`, `serve.sbatch`, `relay.sh`, `settings.json`: how it is launched and configured.
  - `client_example.py`: a standard-library client for four lanes.
  - `tests/`: `loadtest-results.json`, `cap-test-results.json`, `summary.json`, `test_gateway_mock.py`.
  - `logs/` and `secrets/api_token`.
- `/home/jeff/rl-benchmarks/rl-rewriter-vs-context-ir-candidate8-20261112/`: the offline eval behind candidate-10/11.
  - `RESULTS.md` and `README.md`: the results and the method.
  - `canonical/prompts/<prompt_id>/`: one folder per AA prompt.
    - `aa_call.json`: AA's original request.
    - `production/lane{1..4}/`: Context-IR text, ledger and draft.
    - `qwen_s300/lane{1..4}/`: `rewrite.json` (the full model output), `rewrite.txt` and the draft.
    - `judge.json` and `winner.json` for each arm, and `comparisons/` with the Argon verdicts.
  - `canonical/index.jsonl`: one row per prompt.
  - `candidate8-production/service-provenance/`: candidate-8's bestofn code and config as deployed.
  - `aa-bolero/`: the AA human-vote analysis and Elo projection.
  - `figures/`: the plots.
  - `model/global_step_300-hf/`: the merged model, bf16, 51 GB.
- `/home/jeff/video-generation-eval/bestofn/`: the reference bestofn checkout (HEAD 1fb145e).
- `/home/jeff/.venvs/video-agents-h100-f169e786/`: the serving environment (vLLM 0.29.0).

## For your Claude Code

Paste this into a Claude Code session started in your own video-generation-eval checkout:

> Read /home/jeff/rl-benchmarks/rl-rewriter-s300-service/CANDIDATE-10-11.md. Implement candidate-10 and
> candidate-11 in bestofn as its "Changes in bestofn" section describes, check against
> /home/jeff/rl-benchmarks/rl-rewriter-vs-context-ir-candidate8-20261112/candidate8-production/service-provenance/
> for how candidate-8 is deployed, then work through the "Validate before AA traffic" checklist. Read the API token
> from /home/jeff/rl-benchmarks/rl-rewriter-s300-service/secrets/api_token into the QWEN_REWRITER_TOKEN environment
> variable; never print, log or commit it.
