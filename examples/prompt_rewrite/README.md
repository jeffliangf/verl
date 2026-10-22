# Prompt rewriting: agentic RL and evaluation

Use the same tool-calling policy interface for zero-shot, SFT, and RL checkpoints.
`rl.py train` performs GRPO and periodic native validation; `rl.py evaluate` runs
that exact validation loop without optimizer updates. Both consume the frozen
messages, tools, and thinking options in the prepared datasets. See
[Agentic GRPO](#agentic-grpo) and [Checkpoint evaluation](#checkpoint-evaluation).

**The paid H3 Max API is disabled.** New video generation uses the Sol-H3 gateway.
Saved Max runs remain readable for reporting, but cannot generate more videos.
Legacy rewrite artifacts can still be imported, generated, judged, and reported;
the text-only vLLM adapter and resident SFT video decoder have been retired.

```text
original prompt + tools → Qwen reasoning and sol_h3 call → candidate video
                                                            │
                       original prompt + two videos → Gemini, both orders
                                                            │
                                      scores vs prompt and vs h3_context_ir
```

Teacher rewrites and reference video paths never enter the policy input.
Gemini receives the original prompt and videos, with the existing v4 rubric.

## Components

| Module | Responsibility |
|---|---|
| `config.py`, `system_prompt.txt`, `rewrite_instruction.txt` | Shared system role, task instruction, and experiment settings |
| `dataset.py` | Eval schema and reference-video provenance checks |
| `rewrite.py` | Import and validate legacy rewrite records; no inference |
| `h3.py`, `self_hosted_h3.py` | Self-hosted generation, persistent job IDs, resumable polling/downloads |
| `judge.py` | Original/swapped manifests and the existing Gemini runner |
| `scoring.py` | Pure per-prompt scores and confidence intervals |
| `reporting.py` | Three-score summaries and TensorBoard scalar logging |
| `evaluation.py` | Legacy artifact orchestration and reference benchmarks |
| `storage.py` | Atomic files, immutable inputs, and run locking |
| `__main__.py` | Legacy artifact CLI |
| `sft.py`, `train.py` | Native SFT loss validation and HF checkpoint exports |
| `rl.py`, `configs/rl.yaml`, `generation_instruction.txt`, `configs/sol_h3_tool.json` | Direct RL data preparation, request template, and native trainer configuration |
| `agent.py`, `configs/h3_agent.yaml` | Native H3 tool, terminal tool loop, and group reward publication |
| `tournament.py` | Four-candidate tournaments and strict provider deadlines using the existing clients |

Video generation, judging, and scoring use the Python standard library.
TensorBoard logging uses the lightweight `tensorboard` package, already included
in the verl runtime; reporting does not load PyTorch or a GPU.
The MP4 reader, v4 rubric, and Gemini runner come from `~/video-generation-eval`;
their source hashes are frozen in each run. The self-hosted HTTP client lives here;
new runs do not load or depend on the old Max HTTP helpers.

## Shared system role and task instructions

`system_prompt.txt` defines a general video assistant: preserve user intent, use
available tools, ground observations in actual results, and improve outputs from
feedback. It does not prescribe a model/provider, duration, output schema, or
fixed number of steps.

For the current rewrite task, `rewrite_instruction.txt` specifies the 10-second
target, three Context-IR sections, and returning only the rewritten prompt.
`build_messages` places that instruction and the original request together in
the user message. SFT preparation and legacy artifact identities use this builder.
The agentic RL preparation uses `generation_instruction.txt` and `configs/sol_h3_tool.json`
with the same system role to populate each dataset row. During training, the saved
row supplies all messages, tool definitions, and chat-template options. It learns
a single generation action directly from video rewards. Inspection and revision
would require additional tools and turns.

Legacy artifact preparation freezes both texts in its protocol and run directory. Set
`prepare --system-prompt-file` or `--rewrite-instruction-file` to override them for
a new experiment. Existing saved runs retain their original system and user
messages, including older runs without a separate task instruction. Regenerate
the SFT Parquet files after editing either shared prompt file, and prepare a new
RL dataset when changing the tool-calling evaluation request.

## Required local inputs

A fresh checkout needs these assets supplied separately:

- The evaluation JSONL, including `id`, `prompt`, `h3_context_ir`, `source`,
  `prompt_h3_self_hosted_video`, and `h3_context_ir_h3_self_hosted_video`. Use accessible absolute
  video paths; the two source names are `movie_gen_video_bench` and `aa_eval`.
- The reference directory with `manifest.json`, both videos per prompt, and their
  generation state files. Copy the complete directory so the manifest's relative
  paths continue to resolve.
- A checkout of `video-generation-eval` containing
  `scripts/run_gemini_pairwise_eval.py`, `scripts/build_pairwise_manifest.py`, and
  `bestofn/normalize.py`. Set `prepare --eval-repo` if it lives elsewhere.
- For live execution, the model weights and GPU runtime for inference, plus the
  H3 and Gemini credentials described below. Importing saved rewrites skips the
  GPU requirement.

Full datasets, videos, checkpoints, credentials, generated reports, and TensorBoard
event files are runtime inputs or outputs; they are not included with this code.
Small SFT conversation previews are included in `data/prompt_rewrite_sft`.

## Agentic GRPO

This path starts from `Qwen/Qwen3.8-27B` with thinking enabled and no SFT stage.
Each training/evaluation row is the source of truth for the model request:
`prompt` contains the system/user messages, `tools` contains the complete
`sol_h3(prompt, duration=10, seed=42)` JSON Schema, and `chat_template_kwargs` contains
`enable_thinking: true` and `reasoning_effort: xhigh`. The user message is exactly:

```text
Generate a video of about 10 seconds of: {original prompt}
```

The `sol_h3` tool executes self-hosted H3-Base/FastH3. Its definition in every
row exposes all request arguments accepted by Sol-H3's `serve.py:GenerateRequest`,
which the gateway imports for `POST /v1/videos`:

| Argument | Required | Values | Default |
|---|---|---|---|
| `prompt` | Yes | Nonblank string, 1–8,000 characters | None |
| `duration` | No | Integer seconds: `5` or `10` | `10` |
| `seed` | No | Integer from `0` through `9223372036854775807` | `42` |

The general tool description is "Generate a video with synchronized audio from
a text description using Sol-H3." The user request supplies the desired duration;
for this dataset a suitable call is `sol_h3(prompt="...", duration=10, seed=42)`.
Omitted arguments use the defaults saved in the row. Validation uses that row's
JSON Schema, and the executor forwards all three arguments to the existing client.
The tool explicitly sends its ten-second default; the gateway's default for bare
HTTP requests remains five seconds.
The protocol's candidate settings contain fixed service configuration; duration
and seed are recorded with each actual generation request.
Generated media is checked against the requested duration. A supported five-second
request is a valid tool call; infrastructure checks do not force it to ten seconds.
Evaluation uses the
completed self-hosted original-prompt and Context-IR clips, with their actual
model and endpoint provenance frozen in the protocol.

The standalone Sol-H3 engine also supports 15-second clips, I2V images, and Ref2VA
references. Those options are not accepted by the deployed HTTP gateway. Resolution,
frame rate, inference steps, and model/adapter selection are service settings, not
request arguments. The tool schema only advertises the gateway's supported fields.

There are no additional task instructions in that user message. The environment
accepts one tool call and ends the episode after it; the video goes to the judge.
Teacher rewrites and reference video paths are excluded from policy inputs.

This follows [Qwen's messages/tools format](https://qwen.readthedocs.io/en/latest/framework/function_call.html)
and the separate tools column in [MS-SWIFT agent datasets](https://swift.readthedocs.io/en/v4.0/Customization/Custom-dataset.html#agent).
Here `tools` is a JSON array, which verl's native JSON dataset loader retains.
Qwen3.8's own chat template renders the tools into the system turn and specifies
its `<tool_call><function=...><parameter=...>` syntax. The local DeepEyes RL and
LongVT SFT examples instead embed tool descriptions in system text for their older
templates; copying their call syntax would be wrong for this model.

`H3TerminalLoop` passes each row's fields to the existing continuous token builder
and native tool parser. `tools.json` only registers the executor and dataset path;
it contains no model-facing tool schema. `SolH3Tool` reads data to initialize native
tool registration, then each rollout uses its own row's tool definition. Saved
dataset hashes protect resume consistency; training never reconstructs model
inputs from preparation defaults.

Reasoning/tool boundaries follow the native Qwen XML parser. verl receives raw
token IDs from its rollout server; it does not consume OpenAI chat response fields
such as `reasoning_content`. Its parser scans that output for tool markup. Do not
assume that everything before `</think>` is inert: vLLM 0.29's unified Qwen parser
also treats `<tool_call>` as an implicit end of reasoning
([source](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/parser/qwen3.py)).
A tool call can therefore be valid without a closing thinking tag. Literal
reserved tool markup is interpreted as an action; two parsed calls are invalid
in this single-action environment. Training and checkpoint evaluation share
these semantics. This example does not impose an extra reasoning-strip rule.

The integration reuses verl's synchronous V1 trainer, native RL dataset,
`ToolAgentLoop`, `qwen3_coder` parser, continuous token masks, TransferQueue
publication, GRPO advantage computation, and checkpointing. `TournamentWorker`
holds the four sibling outputs until their tournament is complete, then passes
their scalar rewards to the existing worker publisher. H3 requests/task states,
Gemini requests/ledgers, reference checks, and scoring use the existing modules
above and `video-generation-eval`. There is no separate optimizer or HTTP client.

For each training prompt:

1. Sample four tool-calling trajectories and generate one H3 clip for each valid call.
2. Compare all six unordered pairs in both presentation orders: **12 Gemini judgments**.
   Every judgment uses the original user prompt and the two videos, with the existing v4 rubric.
3. Award 1 point for winning in both orders, 0.5 for an order disagreement, and 0 for two losses.
   Each valid candidate's reward is its total points divided by three.
4. Give an invalid tool call reward **-1**. Valid candidates win their matchups against invalid calls
   without a judge request. Native GRPO computes the four within-prompt advantages.

All-invalid or otherwise equal-reward groups have zero relative advantage;
`zero_variance_group` records this in the saved rewards. Repeated prompt text with
distinct IDs retains its original dataset weighting. IDs must be unique, and
preparation rejects train/evaluation overlap by ID or prompt text.

Validation uses one greedy agent rollout per held-out prompt and compares its
video with the frozen **original-prompt and `h3_context_ir` reference videos** in
both orders. The default 200-prompt evaluation costs at most 200 new videos and
800 judgments. Invalid calls receive zero against both references. Native
validation metrics include `vs_prompt`, `vs_h3_context_ir`, and `valid_call`,
grouped by `aa_eval` and `movie_gen_video_bench`; they are logged under
`val-aux/<source>/<metric>/mean@1`. The overall validation reward averages the
two reference scores for each prompt.

Training diagnostics reach the normal console/TensorBoard logger under
`training/reward_extra/<metric>/mean`: `score`, `valid_call`,
`zero_variance_group`, `order_disagreement`, and `judge_comparisons`. Padding and
nonfinite/nonnumeric metadata are excluded; missing metrics are not zero-filled.
Group-level values are repeated for the four siblings, so their mean is a group
average. In particular, `judge_comparisons` is comparisons per group, not a total
API-request counter. Existing validation tags remain `val-aux/<source>/<metric>/mean@1`.

Prepare the actual RL data and verify the existing reference artifacts offline:

```bash
uv run --no-project --python 3.12 python -m examples.prompt_rewrite.rl prepare \
  --run-dir outputs/prompt_rewrite/qwen38-h3-rl \
  --data-dir data/prompt_rewrite_rl \
  --h3-endpoints http://10.2.201.187:18080 \
  --h3-token-file /home/jeff/sol-h3-h200/deployment/token
```

These connection settings target the H200 cluster; the self-hosted service
section below describes a tunnel and local token file for other clients.
Preparation defaults to `data/prompt_rewrite_rl.jsonl` and the 200-prompt evaluation
assets in `data/prompt_rewrite_h3_eval/outputs_self_hosted_gateway_768p`.
`--train-file`, `--eval-dataset`, `--references`, and
`--eval-repo` select other local inputs. Prepared conversations are written to
`data/prompt_rewrite_rl/train.jsonl` and `data/prompt_rewrite_rl/val.jsonl`, beside
the SFT data directory. `--data-dir` overrides this default on both `prepare`
and `train`; pass the same value to both commands when using a custom location.
The run directory holds the frozen protocol, dataset hashes, evaluator source
hashes, reference inventory, and executable tool configuration. The preparation
templates are only used to create data; editing them cannot change an existing
run's prompts. Use a new run directory for changed settings and a new data
directory for changed requests.

In the repository's GPU environment, preview the resolved command and request
counts, optionally checking complete tool prompts with the model tokenizer:

```bash
uv run --frozen --extra fsdp --extra vllm python -m examples.prompt_rewrite.rl train \
  --run-dir outputs/prompt_rewrite/qwen38-h3-rl --check-tokens
```

`train.preview.json` and `val.preview.json` contain complete, indented first rows
beside the datasets. `--check-tokens` also writes `train.preview.txt` and
`val.preview.txt`: the exact rendered initial inputs, including Qwen's tool
instructions, special tokens, and assistant generation prefix. It checks every
saved row with its own tools and chat-template options. No generation is needed.

The default preset is a 10-step pilot on eight GPUs: FSDP2 actor training with
CPU offload, TP=4 vLLM rollout, 16 prompts per batch, four samples per prompt,
and checkpoints every five steps. It allows 4,096 prompt tokens and 8,192 response
tokens. Each batch costs at most **64 videos and 192 judgments**. Validation runs
at step 10; it does not run before training. Hardware memory fit still needs a
GPU smoke test on the target machine.

Start training by adding `--execute`. Hydra overrides follow `--`:

```bash
uv run --frozen --extra fsdp --extra vllm python -m examples.prompt_rewrite.rl train \
  --run-dir outputs/prompt_rewrite/qwen38-h3-rl --execute -- \
  trainer.total_training_steps=10
```

The launcher checks tool-bearing prompt lengths before execution. It freezes the
resolved training configuration and invokes `verl.trainer.main_ppo`. Use the
shared self-hosted H3 token file and the existing Gemini credentials, as for
standalone evaluation. The evaluator checkout, run directory, reference videos,
gateway, and token file must be accessible from rollout workers. Four agent
workers each allow one concurrent group and two H3 requests by default, for
eight concurrent generations; preparation options
`--groups-per-worker` and `--h3-concurrency-per-worker` adjust these bounds.
Gemini handles one prompt group per worker subprocess using its existing runner.

Provider errors stop the current batch before an optimizer update. They never
become losses, ties, or neutral fallback scores in this RL path. This includes
aborted/empty LLM responses, H3 failures, missing judgments, and invalid artifacts.
Malformed or missing policy tool calls are the separate, penalized case.

Default deadlines are 300 seconds per LLM request, 1,800 seconds per H3 job,
300 seconds per Gemini request/file operation, and 1,800 seconds per judging
orientation. Set `prepare --llm-timeout`, `--h3-timeout`,
`--judge-request-timeout`, or `--judge-pass-timeout` to change them. H3 and Gemini
run in subprocesses that are killed and waited on at their deadlines. The H3
worker preserves task IDs and refuses to resubmit uncertain POSTs; the judge
preserves completed comparison ledgers. No automatic provider retry loop is added.

After fixing a provider problem, rerun the same training command. Native
`resume_mode=auto` restores the latest complete checkpoint, including optimizer
and dataloader state. Work since that checkpoint is regenerated; this is **not
exact resumption of an incomplete batch**, and can incur additional API cost.
The original job states and comparisons remain available for inspection.

Artifacts live under the prepared run directory: `videos/` holds requests, task
states and clips; `train/step_*/` and `val/step_*/` hold trajectory tokens,
rollout log probabilities, manifests, judgments, scalar rewards and failures.
`checkpoints/`, `rollouts/`, `validation/`, and `tensorboard/` use the native
trainer's persistence and logging. Preparation and training preview make no
H3/Gemini calls; `--check-tokens` may download tokenizer files if not cached.

## Self-hosted H3 on H200

Self-hosted H3 is the only enabled generation backend. `--h3-backend self-hosted`
is the default; `--h3-backend max` is rejected. The policy calls
`sol_h3(prompt, duration=10, seed=42)`, and the generator is **H3-Base FL2VA with the four-step FastH3
adapter**. Set `--h3-endpoints` or `H3_SELF_HOSTED_ENDPOINTS` to a reachable gateway.
`H3_HOST` and `H3_TOKEN` cannot enable the retired Max API.

Use the single gateway **`http://10.2.201.187:18080`**. It dispatches internally
to healthy workers; this URL and the bearer token stay the same when GPU counts,
worker addresses, or Slurm jobs change. As of 2026-10-22, the pool has **20 H200
GPUs / ten two-GPU workers**. Check `/healthz` for current capacity.
The service preserves its correctness limits and uses dense attention for
rejected shapes; completed job records report `used_gate_fallback` and attention
counters. Every generation/Ray worker must be
able to reach the gateway. This is a private cluster address and is not directly
reachable from the `/home/jeff/Sana` workstation. A tunnel on
one machine serves clients on that machine only. Distributed training needs
network routing or equivalent forwarding on every client host.

On the H200 cluster, the shared bearer-token file is
`/home/jeff/sol-h3-h200/deployment/token`. Pass its filename at preparation time;
the protocol stores the path, never the token. Ray workers read the file
directly, so this does not depend on forwarding the driver's environment.
For clients elsewhere, use a protected token file accessible to those clients.
Standalone generation also accepts `H3_SELF_HOSTED_TOKEN_FILE` or
`H3_SELF_HOSTED_TOKEN`; the paid provider's `H3_TOKEN` is not used.

The public cluster login is **`jeff@157.66.255.187`, SSH port `44222`**;
Jeff's key on this workstation is `/home/jeff/.ssh/gmi`. The full standalone
service guide is `/home/jeff/Sana/models/minimax_h3/Sol-H3/README.md`, under
**Access the deployed H200 service**. For direct cluster access:

```bash
ssh -i ~/.ssh/gmi -p 44222 jeff@157.66.255.187
```

For a single-host client outside that network, this SSH tunnel exposes the gateway
on that client's loopback interface. Keep it running in another terminal on the
**training/client host**:

```bash
ssh -i ~/.ssh/gmi -p 44222 -N \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:28080:10.2.201.187:18080 jeff@157.66.255.187
```

Use `--h3-endpoints http://127.0.0.1:28080` in this case. Copy the token securely
to a local private file and pass that path with `--h3-token-file`. This one
tunnel reaches all GPU workers through the gateway. A tunnel
running only on a login machine does not provide connectivity to Ray actors on
another compute machine.

From the verl checkout in a network-connected environment:

```bash
# Stable across GPU allocation and worker updates.
H3_ENDPOINTS=http://10.2.201.187:18080

curl --noproxy '*' --fail "$H3_ENDPOINTS/healthz"

python -m examples.prompt_rewrite.rl prepare \
  --run-dir outputs/prompt_rewrite/qwen38-h3-rl \
  --data-dir data/prompt_rewrite_rl \
  --h3-backend self-hosted \
  --h3-endpoints "$H3_ENDPOINTS" \
  --h3-token-file /home/jeff/sol-h3-h200/deployment/token \
  --groups-per-worker 1 --h3-concurrency-per-worker 2 --poll-interval 1

# Inspect the configuration first; --execute starts the training run.
python -m examples.prompt_rewrite.rl train \
  --run-dir outputs/prompt_rewrite/qwen38-h3-rl \
  --data-dir data/prompt_rewrite_rl
```

The existing `--train-file`, `--eval-dataset`, `--references`, and `--eval-repo`
options still select your datasets and evaluator checkout. With the default
four agent workers, `--h3-concurrency-per-worker 2` allows eight active tool
requests. Each service worker has one active generation and eight waiting slots.
The gateway prefers shorter queues. The client retries explicitly rejected
429/503 submissions within the H3 timeout and saves the gateway URL and job ID
for recovery. All polling and downloads use that same URL; the gateway retains
the accepting worker internally and archives completed videos independently of
GPU allocations. Neither the client nor gateway automatically repeats an
ambiguous POST. Provider failures remain infrastructure errors in RL.
Request concurrency is independent of the number of configured URLs: a single
gateway supports concurrent jobs across the entire GPU pool.

For the legacy artifact CLI, use the same three connection
options during `prepare`, then generate with eight concurrent requests:

```bash
python -m examples.prompt_rewrite prepare \
  --run-dir outputs/prompt_rewrite/qwen38-self-hosted-eval \
  --h3-backend self-hosted --h3-endpoints "$H3_ENDPOINTS" \
  --h3-token-file /home/jeff/sol-h3-h200/deployment/token

# Import existing legacy rewrites with import-rewrites before generation.
python -m examples.prompt_rewrite generate \
  --run-dir outputs/prompt_rewrite/qwen38-self-hosted-eval \
  --execute --concurrency 8 --poll-interval 1
```

Standalone rewrite evaluation requests 10-second clips with seed 42. Agentic RL
uses the duration and seed from `sol_h3` calls, including defaults from the dataset.
The service produces 1344x768/24-fps video with 32-kHz stereo audio. Warm generation took
about 31 seconds per clip when sparse attention passed the original H200
benchmark; repaired workers using dense fallback measured about 45 seconds on
four recovered prompts. New prompt shapes can take longer. Each tool prompt is
limited to 8,000 characters. Existing run directories remain frozen to their
original provider and endpoint configuration. Prepare a new run to switch an
old direct-worker run to the gateway; do not rewrite accepted jobs' recovery
state. Once a run uses the gateway, its internal worker pool can change without
altering that run's configuration.
Install `ffmpeg` or `imageio-ffmpeg` in the client environment. Sol-H3's
fragmented MP4s are remuxed into regular MP4s for the evaluator's existing parser;
the encoded video and audio are copied without re-encoding.

The RL and standalone default reference directory is
`data/prompt_rewrite_h3_eval/outputs_self_hosted_gateway_768p`: 200 original-prompt
and 200 Context-IR videos generated with this same Base/FastH3 configuration.
Preparation verifies their request bodies, endpoint pool, saved job state, file
sizes, checksums, native media profile, and the dataset's self-hosted video paths.
`reference_h3` records the actual reference model and endpoint pool separately
from the candidate connection. Explicitly selecting historical Max references
remains possible for offline comparisons; it does not enable Max generation.
Gemini still uses its existing API and credentials.

## Checkpoint evaluation

Prepare an experiment with `rl prepare` as above, then evaluate any compatible
base model or exported Hugging Face checkpoint with the same native validation
used during GRPO:

```bash
uv run --frozen --extra fsdp --extra vllm python -m examples.prompt_rewrite.rl evaluate \
  --run-dir outputs/prompt_rewrite/qwen38-h3-rl \
  --data-dir data/prompt_rewrite_rl \
  --model Qwen/Qwen3.8-27B \
  --output-dir outputs/prompt_rewrite/eval-base --check-tokens
```

This previews the command. Add `--execute` to generate and judge videos. For an
SFT or RL checkpoint, pass its exported HF directory to `--model` and use a new
`--output-dir`. All checkpoints receive the saved tools and thinking-enabled
request; the previous plain-text rewrite evaluation is no longer available.
The SFT training objective remains teacher-forced plain-text rewriting, so its
checkpoints must demonstrate tool use under this evaluation like any other policy.

The launcher sets `trainer.val_only=true`, `trainer.val_before_train=true`, and
`trainer.resume_mode=disable`; overrides that would train or silently resume other
weights are rejected. It launches `verl.trainer.main_ppo` with the same
`H3TerminalLoop`, native parser, and tournament scorer as training. The default
hardware configuration is the training preset's eight GPUs. Native initialization
still loads the training stack even though no optimizer step runs.

Each evaluation freezes `evaluation-config.json` in its output directory. Videos,
trajectory tokens, comparisons, native validation JSONL, and TensorBoard events
also go there, separate from training artifacts. Repeating the command regenerates
the cohort; it is not a resume of partially completed evaluation. The default
200-prompt cohort costs at most 200 videos and 800 judgments.

## Live H3 status

From a machine that can reach the frozen endpoints:

```bash
uv run --no-project --python 3.12 python -m examples.prompt_rewrite.rl status \
  --run-dir outputs/prompt_rewrite/qwen38-h3-rl --timeout 5
```

This only reads public `/healthz` endpoints, without credentials or generation.
It reports readiness, capacity, and whatever model settings the server supplies,
including mismatches and missing fields. HTTP 503 health JSON is retained for
warming/unavailable workers. The command exits nonzero if any endpoint is not ready.

The currently deployed gateway reports worker/GPU counts and queues. Individual
workers also report dimensions, frame rate, four transformer forwards, and
attention settings. Neither exposes the loaded model/adapter IDs or revisions,
and the gateway does not propagate worker identity. Status therefore reports
`settings_check.status=unverified`; healthy does not mean the frozen model was
verified. Full runtime verification requires the server to publish model/adapter
revisions from the loaded artifacts and the gateway to attest that all routed
workers agree. `reported_match` means reported fields match, not a weight checksum.
The client still does not enforce model identity before generation.

## Legacy scores and artifacts

This section describes preserved schema-v1 rewrite reports. New policy evaluations
use the native validation metrics and artifacts described above.

Against **each** reference, each prompt contributes 1 point for two candidate
wins, 0.5 for disagreement between orders, or 0 for two candidate losses. The
dataset score is total points / completed prompts in that dataset. Split orders
do not receive random tie breaking.

SFT evaluation enables an explicit neutral fallback policy: if either presentation
order fails or times out, that prompt/reference comparison scores **0.5**. Each
request has a 300-second timeout, uploads/file processing use the same limit,
and each presentation-order pass has a hard 1,800-second process deadline.
There is one request attempt. A hung judging process is terminated before missing
results are recorded. Standalone judging opts in with `judge --neutral-on-failure`;
`--request-timeout-seconds` and `--pass-timeout-seconds` override these limits.

Fallbacks are stored in separate `judge/fallbacks-<order>.json` files, with the
policy in `judge/failure-policy.json`. Original Gemini ledgers remain unchanged.
Completed judgments and recorded fallbacks are reused without repeating API calls.
The report distinguishes `fallback_prompts`, `fallback_judgments`, and
`judged_coverage` from score coverage. `complete` means every comparison is
accounted for; `judging_complete` additionally requires no fallback scores.
TensorBoard logs these counts and actual judging coverage alongside the scores.

SFT also enables neutral handling of H3 rejections, provider errors, uncertain
submissions, and timeouts. The entire H3 generation pass has a **1,800-second hard
deadline**, including submission, polling, and downloading. A timed-out worker
is terminated before its missing results are recorded. Standalone generation
opts in with `generate --neutral-on-failure --pass-timeout-seconds 1800`.

An unavailable candidate video contributes **0.5 against each reference** and
stays in the 200-prompt denominator. Only existing candidate videos are sent to
Gemini. The original H3 state/task ID is preserved; separate
`videos/fallbacks.json` and `videos/failure-policy.json` files record the decision
and deadline. Resuming reuses both successful clips and explicit fallback scores
without resubmitting failed or uncertain jobs. Invalid media, mismatched inputs,
and worker setup failures still stop evaluation.

Reports and TensorBoard distinguish `h3_fallback_prompts` from
`gemini_fallback_prompts`. `generation_complete` indicates whether every candidate
video exists, while `complete` indicates whether every comparison has a score.
`fallback_judgments` counts only Gemini fallback decisions; H3 fallbacks do not
pretend that a judge call was made. To recover an interrupted evaluation whose
rewrites are already saved, no training process or GPU is required:

```bash
python -m examples.prompt_rewrite generate --run-dir <eval-directory> --execute --neutral-on-failure
python -m examples.prompt_rewrite judge --run-dir <eval-directory> --execute --neutral-on-failure
python -m examples.prompt_rewrite report --run-dir <eval-directory> --step <checkpoint-step> \
  --tensorboard-dir <training-tensorboard-directory>
```

By default every evaluation reports **three scores against each reference**:
MovieGen, AA eval, and **average = (MovieGen + AA eval) / 2**. The average gives
each dataset equal weight, even when completed counts differ. If a dataset has
no completed comparisons, its score and the average are unavailable.

`report.json` stores the three default scores in `comparisons.<reference>.scores`
as `moviegen`, `aa_eval`, and `average`. It also retains the pooled per-prompt
`overall` score and detailed AA/MovieGen results: wins, splits,
losses, points, `win_rate_half_split`, 95% prompt-bootstrap intervals (2,000
resamples), order-disagreement rate, first/second-position win rates, and coverage.
`per_prompt_scores.jsonl` includes individual scores and both Gemini explanations.
The CLI displays the three scores and each dataset's coverage. For example,
110 wins, 50 splits, and 40 losses
would yield 135/200 points, or 67.5%; this is an illustration, not a measured result.

Reporting also writes TensorBoard scalars by default. The default directory is
`tensorboard_log/prompt-rewrite/<run-directory-name>` under the repository root.
An explicit `--tensorboard-dir` overrides `TENSORBOARD_DIR`, which in turn
overrides that default. Use the trainer's log directory and `--step` to place
checkpoint evaluations on the same training curve; zero-shot defaults to step 0.

```bash
uv run --no-project --python 3.12 --with tensorboard==2.20.0 python -m examples.prompt_rewrite report \
  --run-dir outputs/prompt_rewrite/sft-step-500 \
  --tensorboard-dir tensorboard_log/prompt-rewrite/sft --step 500

uv run --no-project --python 3.12 --with tensorboard==2.20.0 tensorboard \
  --logdir tensorboard_log/prompt-rewrite
```

The six score tags are `eval/prompt_rewrite/vs_<reference>/<score>`, where
`<reference>` is `prompt` or `h3_context_ir`, and `<score>` is `moviegen`,
`aa_eval`, or `average`. Values are fractions in [0, 1]. Dataset coverage and
overall completion are logged separately. Missing scores are omitted instead
of appearing as zero. `--no-tensorboard` produces JSON-only reports without
requiring the TensorBoard package. `report --allow-partial` retains the explicit
incomplete status in both JSON and TensorBoard.

The run directory also contains the protocol and system prompt, per-prompt
rewrite records and `rewrites.jsonl`, H3 requests and task states, candidate clips,
the two comparison manifests, and the original Gemini result ledgers. Completed
artifacts are reused on restart. Use `status` to inspect rewrite/video progress.

An incomplete comparison is not scored as a loss or a split. A full report
requires both votes for every pair; `report --allow-partial` explicitly produces
an incomplete report with coverage and the missing comparisons listed. Such a
score is conditional on the completed subset and should not select a checkpoint.
Both presentation orders are attempted even when one pass contains provider
errors. Content-filter blocks remain visible in the ledgers and unscored; do not
change their prompts or videos to force a judgment.
Interrupted JSONL lines are preserved and counted in
`report.json` under `ledger_counts.<order>.malformed_lines`. They contribute no
vote; a valid judgment appended on resume can complete that comparison.

An uncertain H3 submission remains blocked in its state file instead of being
automatically submitted again. Polling/download failures with a saved task ID
can resume. Terminal provider failures also remain visible; investigate them
before creating replacement work. Changed inputs or evaluator code require a
new run directory. Reference and candidate checksums are checked before judging.
Existing runs keep their frozen generation settings even when defaults change.

To reuse the completed zero-shot rewrites with self-hosted H3, prepare a separate run
and import the saved outputs. This skips GPU inference and preserves the old run:

```bash
uv run --no-project --python 3.12 python -m examples.prompt_rewrite prepare \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot-self-hosted \
  --name qwen38-zero-shot-self-hosted --h3-endpoints "$H3_SELF_HOSTED_ENDPOINTS"

uv run --no-project --python 3.12 python -m examples.prompt_rewrite import-rewrites \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot-self-hosted \
  --input outputs/prompt_rewrite/qwen38-zero-shot/rewrites.jsonl
```

Then run `generate`, `judge`, and `report` with the new `--run-dir`. Both reference
videos and newly generated candidate videos use self-hosted H3-Base/FastH3.

## Direct reference benchmark

Compare the existing Context-IR video with the existing original-prompt video
for each sample, using the exact cohort and settings of a saved evaluation:

```bash
uv run --no-project --python 3.12 python -m examples.prompt_rewrite benchmark-references \
  --source-run outputs/prompt_rewrite/qwen38-zero-shot-self-hosted \
  --run-dir outputs/prompt_rewrite/h3-context-ir-vs-prompt-self-hosted \
  --name h3-context-ir-vs-prompt-self-hosted --execute --concurrency 500

uv run --no-project --python 3.12 --with tensorboard==2.20.0 python -m examples.prompt_rewrite report \
  --run-dir outputs/prompt_rewrite/h3-context-ir-vs-prompt-self-hosted
```

The candidate is `h3_context_ir`, scored against `prompt`. Thus, a score above
50% favors Context-IR. This reuses all 400 reference clips, makes 400 judgments
(200 prompts × two orders), and runs neither Qwen inference nor H3 generation.
The same rubric, three dataset scores, partial-coverage handling, and TensorBoard
logging apply. The source evaluation remains unchanged. Omit `--execute` to
preview; `judge`, `report`, and `status` also work on the benchmark directory.

## SFT and saved rewrite artifacts

Evaluate exported SFT/RL checkpoints with `rl evaluate` above. There is one live
policy-evaluation path. The older `EvaluationRun` API only supports saved rewrite
artifacts, reference comparisons, and reports; `EvaluationRun.rewrite()` gives a
migration error. `rewrite.py` retains record validation/import/export only.

For existing schema-v1 runs, `import-rewrites --run-dir ... --input ...` accepts
saved `id, rewritten_prompt` rows. Optional `prompt`, `finish_reason`, and
`output_tokens` fields preserve provenance and truncation checks. `generate`,
`judge`, and `report` still process those artifacts. Imported text does not measure
tool-calling success and should not be compared as the same policy evaluation.

### Prepare SFT data

Export the existing training rows and the established 200-example evaluation
cohort to the standard `MultiTurnSFTDataset` format:

```bash
uv run --no-project --python 3.12 --with pyarrow python -m examples.prompt_rewrite.prepare_sft
```

This writes `data/prompt_rewrite_sft/train.parquet` from
`data/prompt_rewrite_sft.jsonl` and `data/prompt_rewrite_sft/val.parquet` from
`data/prompt_rewrite_aa_moviegen_eval_200.jsonl`. Source JSONL files are preserved.
The converter repairs the known `"h3_context_ir, "` field-name typo in the export,
checks nonempty inputs/targets and unique IDs, and rejects train/validation
overlap by ID or stripped prompt text. It preserves training row order and
duplicates; `manifest.json` records input hashes, counts, and repaired keys.
Use `--train-file`, `--val-file`, or `--output-dir` to override these paths.

The same directory also contains `train.jsonl` and `val.jsonl` with every
converted row, plus `train.preview.txt` and `val.preview.txt` showing the first
three examples with full system/user/assistant messages and readable line breaks.
These inspection files are refreshed whenever the converter runs.

Every Parquet row has `id`, `source`, `split`, `enable_thinking=false`, and
`messages`: the shared system prompt, the rewrite task instruction and original
request in one user message, then the assistant's `h3_context_ir` target. Teacher
text appears only in the assistant turn. The example's `PromptRewriteSFTDataset`
renders the complete conversation with the training tokenizer, verifies that it
starts with the exact inference prompt, and applies loss only after that prefix.
This avoids rendering isolated system turns, which the Qwen template rejects.
Reference video paths remain in the evaluation JSONL and are not SFT inputs.

The `configs/sft.yaml` preset inherits verl's standard SFT trainer and sets both
thinking controls to false, `data.max_length=4096`, `data.truncation=error`, and
`data.pad_mode=no_padding`. From the repository root, in the training environment,
select it with:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=<GPU_COUNT> \
  -m verl.trainer.sft_trainer \
  --config-path=../../examples/prompt_rewrite/configs --config-name=sft \
  trainer.n_gpus_per_node=<GPU_COUNT>
```

Choose the engine, batch size, optimizer, and any LoRA settings for your hardware;
the preset supplies the data/template settings and Qwen3.8-27B model path.
SFT validation computes teacher-forced loss on 192 of the 200 examples after each
epoch; the final incomplete batch is dropped. Video evaluation uses all 200
examples through `rl evaluate`, independently of SFT loss validation.
These 200 prompts stay out of both SFT and RL training.

`data.max_length` limits the **complete tokenized conversation**, including the
system/user messages, assistant target, and template tokens. It is not the output
generation budget. `no_padding` keeps shorter sequences at their actual lengths.
The preset uses fixed microbatches of one example per GPU, with dynamic batching
disabled. On eight GPUs, eight microbatches accumulate into each global batch of 64.
If enabled for a later run, `data.max_token_len_per_gpu` controls the dynamic
microbatch token budget and defaults to `${data.max_length}`.
The canonical tool evaluation uses the RL preset's prompt and response budgets. Tokenize with the exact training tokenizer before launching a GPU job,
and inspect assistant loss masks and the chat-template consistency check.

### Train and export checkpoints

Preview SFT, adding `--execute` to start it:

```bash
uv run --frozen --extra fsdp python -m examples.prompt_rewrite.sft \
  --run-dir outputs/prompt_rewrite/qwen38-sft --nproc-per-node 8 \
  -- trainer.total_epochs=3
```

The launcher invokes native `verl.trainer.sft_trainer`, validates dataset isolation
and assistant masks, and freezes `sft-inputs.json`. Each epoch validates
teacher-forced loss and saves resumable state plus HF weights. It makes no H3 or
Gemini requests. The old `--eval-before-training` video-baseline flag is removed;
evaluate the base model with `rl evaluate` for that baseline.
Use `rl evaluate --model <checkpoint>/huggingface` after releasing the training
GPUs to evaluate any exported checkpoint. SFT training logs and standalone video
evaluation logs reside in their respective output directories.

## Local verification

In a verl CPU-capable environment with PyTorch, Transformers, Ray, TransferQueue,
pytest, and TensorBoard installed:

```bash
uv run --frozen --extra cpu python -m pytest \
  tests/special_standalone/prompt_rewrite \
  tests/trainer/ppo/v1/test_trainer_base_on_cpu.py \
  tests/trainer/ppo/v1/test_metrics_aggregator_on_cpu.py -q

uv run --no-project --python 3.12 --with ruff==0.12.2 ruff check \
  examples/prompt_rewrite tests/special_standalone/prompt_rewrite
```

Tests exercise the native tool loop with a fake model/provider, reward scoring,
policy/evaluation config parity, isolated output directories, TensorBoard event
contents, public HTTP health checks, artifact compatibility, and job recovery.
GPU memory fit, actual policy quality, and end-to-end provider execution still
need a GPU smoke run.
