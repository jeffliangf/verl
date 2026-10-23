# Video agents

This example contains two training tasks:

- **SFT baseline:** learn to rewrite a user prompt into a Context-IR description.
- **Agent RL:** learn to call a video-generation tool, using the resulting video as
  the basis for the reward. RL can start from the base model; SFT is optional.

```text
SFT: user prompt → assistant rewrite → supervised token loss

RL:  user request + tools → model reasoning + sol_h3(...) → video
                                                           ↓
                                                    Gemini comparisons
                                                           ↓
                                                      GRPO reward
```

The RL policy chooses the tool arguments itself. There is no external rewrite
stage. It currently gets **one tool call per episode**: the environment executes
`sol_h3` and ends the rollout. The model does not see or revise the generated
video. This is a native tool-calling agent with a deliberately small action space.

## Where to read the code

| File | Responsibility |
|---|---|
| `rl.py` | Prepare agent datasets; launch native GRPO or checkpoint validation |
| `agent.py` | Execute the tool, end the episode, and publish group rewards |
| `tournament.py` | Generate candidates and score training/validation comparisons |
| `self_hosted_h3.py` | Sol-H3 HTTP client, job recovery, downloads, and health checks |
| `h3.py` | Generation subprocess entry point and media checks |
| `judge.py`, `judge_worker.py`, `scoring.py` | Gemini manifests, timed requests, verified ledgers, and pair scores |
| `observability.py` | Stage timings, provider failure classification, and latency summaries |
| `dataset.py` | Validate held-out data and reference-video provenance |
| `prepare_sft.py` | Convert rewrite examples to SFT conversations |
| `sft.py`, `sft_dataset.py` | Launch native SFT and mask loss to assistant tokens |
| `config.py`, `configs/` | Dataset defaults, model/trainer settings, and tool schema |
| `storage.py` | Atomic writes, hashes, immutable inputs, and run locks |
| `cluster.sbatch` | SFT submission script for the existing eight-GPU cluster |

Start with `rl.py.dataset_records`, then `H3TerminalLoop` in `agent.py`, then
`tournament.py.score_group`. These describe the model input, action, and reward.
The optimizer, GRPO advantages, token masks, checkpointing, and TensorBoard
logging come from verl's native trainer.

## Model inputs

`system_prompt.txt` is the shared video-assistant role.

For SFT, `rewrite_instruction.txt` asks for a ten-second Context-IR description.
The assistant target is the saved `h3_context_ir` text, with thinking disabled.

For RL, `generation_instruction.txt` produces this user message:

```text
Generate a video of about 10 seconds of: {original prompt}
```

Each prepared row saves `prompt` (system/user messages), `tools` (the complete
schema from `configs/sol_h3_tool.json`), and `chat_template_kwargs` (thinking enabled,
reasoning effort `xhigh`). Training and evaluation consume those saved fields.
Teacher rewrites and reference-video paths never enter the policy input.

The only tool is `sol_h3(prompt, duration=10, seed=42)`. Its prompt must contain
1–8,000 characters; duration is 5 or 10 seconds; seed is an integer from 0 through
9223372036854775807. The model's chat template and verl's native `qwen3_coder`
parser handle tool-call syntax. Model/adapter selection and resolution are fixed
service settings, not policy arguments.

## Rewards and validation

For each training prompt, sample four trajectories. Compare every valid pair of
videos in both presentation orders, giving at most six pairs and twelve Gemini
judgments. A pair earns 1 for two wins, 0.5 for an order disagreement, or 0 for two
losses. Each candidate's reward is its total points divided by three.

An invalid tool call earns -1 and forfeits its pairings against valid calls.
Equal-reward groups have no GRPO learning signal. During training, an H3/Gemini
service failure, timeout, or missing judgment drops the whole four-rollout prompt
group. No trajectories or rewards from that group enter the optimizer. Native
`trainer.v1.sampler.sync_refill_failed_groups=true` replaces it with a fresh group
under the same policy weights, preserving the training batch size. Sibling
rollouts settle before the failed group is released. Artifact corruption,
configuration errors, and LLM failures still abort. Evaluation service failures
also abort, preserving the held-out denominator.

`video_rl.max_dropped_groups_per_step` defaults to `data.train_batch_size`.
Exceeding that many dropped groups stops the step before updating weights, so a
persistent outage cannot cause unlimited replacement. TensorBoard reports
`training/rollout_failure/evicted_samples`. Dropped groups retain `failure.json`
with their sample ID, policy step, provider stage, and reason.

Newly prepared runs use the measured latency baseline in
`configs/latency_baseline.json` from the two-step, four-H200 run at H3 concurrency
20. Timeouts are twice P99, rounded up to whole seconds:

| Measured stage | Observations | P50 | P99 | Timeout |
| --- | ---: | ---: | ---: | ---: |
| H3 accepted job through queue and generation | 64 | 49.34 s | 90.41 s | 181 s |
| One Gemini judgment, excluding upload | 192 | 11.53 s | 16.84 s | 34 s |

The H3 worker deadline includes admission, polling, download, and media checks;
its calibration measures the dominant queue/generation portion. Waiting for a
client concurrency slot is logged separately. Limits for previously unmeasured
stages remain 30 s per H3 HTTP request, 300 s for Gemini file operations, 1800 s
per Gemini presentation-order pass, and 900 s for LLM generation. Each limit can
be overridden during `prepare` using `--h3-timeout`, `--h3-request-timeout`,
`--judge-request-timeout`, `--judge-file-timeout`, `--judge-pass-timeout`, or
`--llm-timeout`. Existing frozen protocols retain their saved limits; prepare a
new run to adopt new settings. These small-sample percentiles describe successful
requests; failures are counted separately.

Stage logs are append-only JSONL, with UTC start/end times, elapsed seconds,
outcome, error type, timeout, and correlation IDs. They cover LLM inference,
concurrency waits, H3 submission/status polling/download/media checks, provider
queue/generation timings, Gemini upload/file processing/judgments/cleanup,
complete judge passes, group outcomes, and replacement dispatch. Files live at:

- `<run>/timings.jsonl`: batch dispatch and replacement counts.
- `<run>/train/step_<N>/<group>/timings.jsonl`: group, LLM, and tool stages.
- `<run>/videos/<instance>/timings.jsonl`: H3 requests and provider timings.
- `<run>/train/step_<N>/<group>/judge/timings*.jsonl`: Gemini stages.

Validation uses the same layout under `val/`. An unmatched start identifies
interrupted work, including a killed subprocess. Timing logs omit credentials,
prompts, and provider response bodies. Summarize the next run offline with:

```bash
python -m examples.video_agents.rl timings --run-dir outputs/video_agents/rl
```

This reports P50/P99 and proposed 2×P99 limits without changing the frozen run,
alongside errors, dropped groups, incomplete stages, and malformed log lines.
An accepted remote H3 job may continue after the local timeout; saved IDs remain
available for inspection and recovery, and ambiguous POSTs are never repeated.

Validation uses one greedy rollout per held-out prompt. It compares the video
against the saved original-prompt and Context-IR reference videos, each in both
orders. The validation reward averages those two scores; an invalid call scores
zero. `rl.py evaluate` uses this same loop without training.

Native metrics include `vs_prompt`, `vs_h3_context_ir`, and `valid_call` under
`val-aux/<source>/<metric>/mean@1`. Training diagnostics are under
`training/reward_extra/<metric>/mean`, including `zero_variance_group`,
`order_disagreement`, and `judge_comparisons`. The last metric counts comparisons
per prompt group, not total API requests.

## Local inputs and services

These runtime assets are supplied separately:

- Training JSONL with nonempty `id` and `prompt`; optional `source`. SFT also
  requires `h3_context_ir`. Train/evaluation IDs and prompt text must not overlap.
- Evaluation JSONL with `id`, `prompt`, `source`, `h3_context_ir`,
  `prompt_h3_self_hosted_video`, and `h3_context_ir_h3_self_hosted_video`.
- The reference directory, including `manifest.json`, both videos per prompt,
  and their generation state files. Keep its relative paths intact.
- `~/video-generation-eval`, containing `scripts/run_gemini_pairwise_eval.py`,
  `scripts/build_pairwise_manifest.py`, and `bestofn/normalize.py`.
  `--eval-repo` selects another checkout. Its source hashes and v4 rubric are
  frozen during preparation. Gemini credentials use that checkout's `arena/.env`.
- A reachable Sol-H3 gateway and token file. Set `--h3-endpoints` and
  `--h3-token-file`, or `H3_SELF_HOSTED_ENDPOINTS` and `H3_SELF_HOSTED_TOKEN_FILE`.
  Workers must be able to access the same run, data, references, evaluator, and
  token paths. MP4 remuxing needs `ffmpeg` or `imageio-ffmpeg`.

All new generation uses Sol-H3. Existing Max reference clips can still be
validated from their recorded metadata; there is no Max generation client.
The client saves job IDs before polling and does not repeat an uncertain POST.

## Run agent RL

Prepare a new run offline, without generation or judge requests:

```bash
uv run --no-project --python 3.12 python -m examples.video_agents.rl prepare \
  --run-dir outputs/video_agents/rl \
  --h3-endpoints http://10.2.201.187:18080 \
  --h3-token-file /home/jeff/sol-h3-h200/deployment/token
```

The defaults use the existing `data/prompt_rewrite_rl.jsonl`, the 200-prompt
`data/prompt_rewrite_aa_moviegen_eval_200.jsonl`, and references in
`data/prompt_rewrite_h3_eval/outputs_self_hosted_gateway_768p`.
`--train-file`, `--eval-dataset`, and `--references` select different inputs.
Prepared JSONL and readable previews go to `data/prompt_rewrite_rl`.
If you use `--data-dir`, pass the same value to preparation, training, and evaluation.

Preview the resolved command in the GPU environment:

```bash
uv run --frozen --extra fsdp --extra vllm python -m examples.video_agents.rl train \
  --run-dir outputs/video_agents/rl --check-tokens
```

`--check-tokens` renders every saved request with the model tokenizer and saves
full text previews. Execution always checks lengths before launching the trainer.
Add `--execute` to train; Hydra overrides follow `--`:

```bash
uv run --frozen --extra fsdp --extra vllm python -m examples.video_agents.rl train \
  --run-dir outputs/video_agents/rl --execute -- trainer.total_training_steps=10
```

The preset uses Qwen3.8-27B, FSDP2, TP=4 vLLM rollout, eight GPUs, sixteen prompts
per batch, and four samples per prompt. Four agent workers each allow five H3
requests, giving a client concurrency cap of 20. The ten-step pilot saves every
five steps and validates at step ten. An initial batch costs at most 64 videos
and 192 judgments, with additional work for failed-group replacements;
the 200-prompt validation costs at most 200 videos and 800 judgments. Hardware
memory fit and live provider behavior still need a GPU smoke run.

Evaluate a base model or exported checkpoint in its own output directory:

```bash
uv run --frozen --extra fsdp --extra vllm python -m examples.video_agents.rl evaluate \
  --run-dir outputs/video_agents/rl \
  --model /path/to/checkpoint/huggingface \
  --output-dir outputs/video_agents/eval-checkpoint --execute
```

Check service readiness without submitting videos:

```bash
uv run --no-project --python 3.12 python -m examples.video_agents.rl status \
  --run-dir outputs/video_agents/rl
```

A ready service is not proof of its model identity; status reports any missing
or mismatched model provenance separately.

## Run the rewrite SFT baseline

```bash
uv run --no-project --python 3.12 --with pyarrow python -m examples.video_agents.prepare_sft

uv run --frozen --extra fsdp python -m examples.video_agents.sft \
  --run-dir outputs/video_agents/sft --nproc-per-node 8 --check-tokens
```

Add `--execute` to train. Preparation preserves the original JSONL and writes
Parquet, JSONL, and complete conversation previews to `data/prompt_rewrite_sft`.
The known `h3_context_ir, ` target-key typo is repaired during conversion.
SFT uses assistant-only loss, validates loss after each epoch, and exports both
resumable state and Hugging Face weights. It does not generate videos during SFT.
The launcher reports rows dropped by the trainer's complete-batch requirement.
Agent checkpoint evaluation uses the full held-out cohort independently.

An SFT checkpoint can be passed to `rl evaluate`, but this tests tool use under
the agent request. Rewrite SFT itself does not supervise tool calls.

## Cleanup and future changes

The package was renamed from `examples/prompt_rewrite` to `examples/video_agents`.
Imports, Hydra class paths, subprocess commands, tests, and launch scripts use the
new name. Existing data/video files keep their names. Prepare a new run directory
for the new package paths; frozen old tool/training configurations still refer
to the former package.

The retired artifact CLI, `evaluation.py`, `rewrite.py`, `reporting.py`, and
`train.py` compatibility wrapper were removed, along with neutral-failure
reporting. Saved artifacts remain on disk; the removed commands belong to the
previous revision. Agent training and checkpoint evaluation share one path.

To support inspection and revision, add a concrete observation tool and a bounded
multi-turn loop. Define which video the agent submits for reward and count all
generation calls toward its budget. Keep training and evaluation on that same
loop. Avoid adding a second external rewrite pipeline.

## Verification

In the repository's CPU environment:

```bash
uv run --frozen --extra cpu python -m pytest tests/special_standalone/video_agents -q
uv run --no-project --python 3.12 --with ruff==0.12.2 ruff check \
  examples/video_agents tests/special_standalone/video_agents
```

The tests use fake model/provider responses and local HTTP servers to exercise
native tool parsing, assistant masks, scoring, deadlines, provenance, recovery,
and checkpoint isolation without submitting real videos or Gemini requests.
