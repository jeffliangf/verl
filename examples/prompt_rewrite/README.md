# Prompt rewrite evaluation

Evaluate a zero-shot, SFT, or RL rewriter using the same 200 prompts, H3 Max generation
settings, reference videos, and Gemini rubric. Each run represents one immutable
checkpoint and decoding configuration. Training is a separate step.

```text
original prompt → rewriter → rewritten_prompt → H3 → candidate video
                                                     │
                    original prompt + two videos → Gemini, both orders
                                                     │
                                  scores vs prompt and vs h3_context_ir
```

The rewriter sees a shared system role, the rewrite task instruction, and the
original prompt. The teacher Context-IR is never included in its input. Gemini
also receives the original prompt; model identities, teacher text, and rewritten
text are excluded from its request by the existing evaluator.

## Components

| Module | Responsibility |
|---|---|
| `config.py`, `system_prompt.txt`, `rewrite_instruction.txt` | Shared system role, task instruction, and experiment settings |
| `dataset.py` | Eval schema and reference-video provenance checks |
| `rewrite.py` | Lazy vLLM adapter, injectable policy callable, imported rollout outputs |
| `h3.py` | Direct generation, persistent task IDs, resumable polling/downloads |
| `judge.py` | Original/swapped manifests and the existing Gemini runner |
| `scoring.py` | Pure per-prompt scores and confidence intervals |
| `reporting.py` | Three-score summaries and TensorBoard scalar logging |
| `evaluation.py` | `EvaluationRun`, the reusable orchestration API |
| `storage.py` | Atomic files, immutable inputs, and run locking |
| `__main__.py` | Thin CLI for each stage |

Video generation, judging, and scoring use the Python standard library.
TensorBoard logging uses the lightweight `tensorboard` package, already included
in the verl runtime; reporting does not load PyTorch or a GPU.
The H3 HTTP helpers, MP4 reader, v4 rubric, and Gemini runner come from
`~/video-generation-eval`; their source hashes are frozen in each run.

## Shared system role and task instructions

`system_prompt.txt` defines a general video assistant: preserve user intent, use
available tools, ground observations in actual results, and improve outputs from
feedback. It does not prescribe a model/provider, duration, output schema, or
fixed number of steps.

For the current rewrite task, `rewrite_instruction.txt` specifies the 10-second
target, three Context-IR sections, and returning only the rewritten prompt.
`build_messages` places that instruction and the original request together in
the user message. Both SFT preparation and new evaluations use this same builder.
Future generation/inspection/revision tasks can supply their own task instructions
and tool schemas while keeping the system role. Tool-call and revision behavior
will also require suitable training trajectories and a tool-enabled rollout loop;
the current examples supervise only the first rewrite.

Each new evaluation freezes both texts in its protocol and run directory. Set
`prepare --system-prompt-file` or `--rewrite-instruction-file` to override them for
a new experiment. Existing saved runs retain their original system and user
messages, including older runs without a separate task instruction. Regenerate
the SFT Parquet files after editing either shared prompt file, and prepare a new
zero-shot evaluation when comparing checkpoints under the new instructions.

## Required local inputs

A fresh checkout needs these assets supplied separately:

- The evaluation JSONL, including `id`, `prompt`, `h3_context_ir`, `source`,
  `prompt_h3_max_video`, and `h3_context_ir_h3_max_video`. Use accessible absolute
  video paths; the two source names are `movie_gen_video_bench` and `aa_eval`.
- The reference directory with `manifest.json`, both videos per prompt, and their
  generation state files. Copy the complete directory so the manifest's relative
  paths continue to resolve.
- A checkout of `video-generation-eval` containing `scripts/generate_h3_repeats.py`,
  `scripts/run_gemini_pairwise_eval.py`, `scripts/build_pairwise_manifest.py`, and
  `bestofn/normalize.py`. Set `prepare --eval-repo` if it lives elsewhere.
- For live execution, the model weights and GPU runtime for inference, plus the
  H3 and Gemini credentials described below. Importing saved rewrites skips the
  GPU requirement.

Full datasets, videos, checkpoints, credentials, generated reports, and TensorBoard
event files are runtime inputs or outputs; they are not included with this code.
Small SFT conversation previews are included in `data/prompt_rewrite_sft`.

## Zero-shot run

Run from the verl repository root. First prepare the full evaluation and inspect
the rewrite input. These commands make no network requests and load no model:

```bash
uv run --no-project --python 3.12 python -m examples.prompt_rewrite prepare \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot \
  --name qwen38-zero-shot --model Qwen/Qwen3.8-27B

uv run --no-project --python 3.12 python -m examples.prompt_rewrite rewrite \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot
```

Preparation defaults to `data/prompt_rewrite_aa_moviegen_eval_200.jsonl` and the
existing `data/prompt_rewrite_h3_eval/outputs_h3max_disabled_768p/manifest.json`. It verifies both
reference files for every prompt, including their request text, settings,
checksum, and native video/audio format. The JSONL's `prompt_h3_max_video` and
`h3_context_ir_h3_max_video` paths are required and must match the reference manifest.
These become the `prompt` and `h3_context_ir` reference scores, respectively.

For a small smoke run, prepare a **different directory** with `--limit 2`.
`--dataset`, `--references`, and `--eval-repo` can select other local inputs.

On a GPU worker with the repository's vLLM environment, generate the rewrites:

```bash
uv run --frozen --extra fsdp --extra vllm python -m examples.prompt_rewrite rewrite \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot --execute
```

The default is BF16, one GPU, thinking disabled, greedy decoding, and 2,048 output
tokens within an 8,192-token context. Set `--tensor-parallel-size`,
`--max-model-len`, `--max-new-tokens`, and other decoding options at **prepare**
time. Use `--revision` for a pinned HF revision, or `--model` for an immutable
local checkpoint directory. The adapter uses vLLM's
[offline inference interface](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/).
It schedules at most 32 sequences at once to fit hybrid-attention cache on an
80 GB H100; larger input batches are queued internally.

Generate and judge videos after reviewing the rewrites:

```bash
uv run --no-project --python 3.12 python -m examples.prompt_rewrite generate \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot --execute --concurrency 500

uv run --no-project --python 3.12 python -m examples.prompt_rewrite judge \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot --execute --concurrency 500

uv run --no-project --python 3.12 --with tensorboard==2.20.0 python -m examples.prompt_rewrite report \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot
```

`rewrite`, `generate`, and `judge` default to dry runs; `--execute` starts work.
`generate` requires all rewrites, and `judge` requires all candidate videos.
Credentials use the same environment variables or `arena/.env` as the existing
eval repository (`H3_HOST`, `H3_TOKEN`, and the Gemini runner's API key).

For 200 prompts this creates **200 new videos and 800 Gemini judgments**. Both
reference cohorts are reused. H3 concurrency defaults to 500, so all 200 candidate
tasks can be active at once. Gemini concurrency also defaults to 500 and is
configured separately; its existing runner limits active workers to available
prompt batches.
No Context-IR endpoint is called. Generation uses `MiniMax-H3-Max` with
`extra.prompt_expansion_mode: "disabled"`, 768P,
10 seconds, requested ratio 16:9, seed 42, and the same host as the references.
Its native reference output is 1344×768 at 24 fps with audio; candidate media must
match. A fixed seed does not establish deterministic behavior of the remote API.

The judge settings are `gemini-3.7-flash`, high thinking, standard video
processing, seed 0, unchanged v4 instructions, and no explicit token ceiling.
Both presentation orders use the same candidate/reference files. The rubric
emphasizes overall viewing quality and core prompt adherence, including audio.

## Scores and artifacts

Against **each** reference, each prompt contributes 1 point for two candidate
wins, 0.5 for disagreement between orders, or 0 for two candidate losses. The
dataset score is total points / completed prompts in that dataset. Split orders
do not receive random tie breaking.

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

To reuse the completed zero-shot rewrites with H3 Max, prepare a separate run
and import the saved outputs. This skips GPU inference and preserves the old run:

```bash
uv run --no-project --python 3.12 python -m examples.prompt_rewrite prepare \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot-h3-max \
  --name qwen38-zero-shot-h3-max

uv run --no-project --python 3.12 python -m examples.prompt_rewrite import-rewrites \
  --run-dir outputs/prompt_rewrite/qwen38-zero-shot-h3-max \
  --input outputs/prompt_rewrite/qwen38-zero-shot/rewrites.jsonl
```

Then run `generate`, `judge`, and `report` with the new `--run-dir`. Both reference
videos and newly generated candidate videos use H3 Max with expansion disabled.

## Direct reference benchmark

Compare the existing Context-IR video with the existing original-prompt video
for each sample, using the exact cohort and settings of a saved evaluation:

```bash
uv run --no-project --python 3.12 python -m examples.prompt_rewrite benchmark-references \
  --source-run outputs/prompt_rewrite/qwen38-zero-shot-h3-max \
  --run-dir outputs/prompt_rewrite/h3-context-ir-vs-prompt-h3-max \
  --name h3-context-ir-vs-prompt-h3-max --execute --concurrency 500

uv run --no-project --python 3.12 --with tensorboard==2.20.0 python -m examples.prompt_rewrite report \
  --run-dir outputs/prompt_rewrite/h3-context-ir-vs-prompt-h3-max
```

The candidate is `h3_context_ir`, scored against `prompt`. Thus, a score above
50% favors Context-IR. This reuses all 400 reference clips, makes 400 judgments
(200 prompts × two orders), and runs neither Qwen inference nor H3 generation.
The same rubric, three dataset scores, partial-coverage handling, and TensorBoard
logging apply. The source evaluation remains unchanged. Omit `--execute` to
preview; `judge`, `report`, and `status` also work on the benchmark directory.

## Reuse for SFT and RL

For an exported HF checkpoint (including a merged LoRA checkpoint), prepare a new
run with `--name sft-epoch-1 --model /path/to/checkpoint/huggingface` and repeat the
same stages. Keep the system prompt, dataset, and decoding settings fixed.
For RL checkpoints use the same interface; no evaluation logic depends on how
the model was trained.

A trainer that already generated the eval rewrites can avoid loading vLLM:

```bash
uv run --no-project --python 3.12 python -m examples.prompt_rewrite import-rewrites \
  --run-dir outputs/prompt_rewrite/sft-epoch-1 --input /path/to/eval-rollouts.jsonl
```

Each row must have `id` and `rewritten_prompt`; optionally include `prompt`,
`finish_reason`, and `output_tokens`. Every eval ID must occur exactly once.
Preserve the trainer's truncation indicator: `finish_reason: "length"` is rejected.
Export only the assistant's final rewrite, using the run's saved system prompt
and decoding settings. The importer checks alignment but cannot verify how an
external trainer actually generated its outputs.

The equivalent Python API is:

```python
from pathlib import Path
from examples.prompt_rewrite.config import RewriteConfig, build_messages
from examples.prompt_rewrite.evaluation import EvaluationRun

run = EvaluationRun.prepare(
    Path("outputs/prompt_rewrite/sft-epoch-1"),
    name="sft-epoch-1",
    config=RewriteConfig(model="/path/to/checkpoint/huggingface"),
)
run.import_rewrites(eval_rollout_rows)
# Alternatively: run.rewrite(execute=True, rewriter=policy_callable)
# The callable receives batches of chat messages and returns RewriteResult objects.
run.generate(execute=True)
run.judge(execute=True)
metrics = run.report()["comparisons"]
# During training, use run.report(tensorboard_dir=trainer_log_dir, step=global_step).
```

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
text appears only in the assistant turn. The standard dataset applies the chat
template and assistant-only loss.
Reference video paths remain in the evaluation JSONL and are not SFT inputs.

The `configs/sft.yaml` preset inherits verl's standard SFT trainer and sets both
thinking controls to false, `data.max_length=8192`, `data.truncation=error`, and
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
SFT validation computes teacher-forced loss on the same 200 examples after each
epoch. Continue using the existing rewrite → H3 → judge → report workflow to
measure video quality; the SFT trainer does not invoke that pipeline itself.
These 200 prompts stay out of both SFT and RL training.

`data.max_length` limits the **complete tokenized conversation**, including the
system/user messages, assistant target, and template tokens. It is not the output
generation budget. `no_padding` keeps shorter sequences at their actual lengths.
With dynamic batching, `data.max_token_len_per_gpu` controls the microbatch token
budget; the preset defaults it to `${data.max_length}` so one full-length example
fits that budget. Actual GPU memory capacity still determines viable batch sizes.
Override `data.max_length=8192` on a training command to set this limit normally;
in this preset the microbatch token budget follows it unless separately overridden.
The existing evaluation keeps its 8,192-token context and 2,048-token generation
budget. Tokenize with the exact training tokenizer before launching a GPU job,
and inspect assistant loss masks and the chat-template consistency check.

## Local verification

```bash
uv run --no-project --python 3.12 --with pytest --with tensorboard==2.20.0 \
  --with pandas --with pyarrow python -m pytest tests/special_standalone/prompt_rewrite -q

uv run --no-project --python 3.12 --with ruff==0.12.2 ruff check \
  examples/prompt_rewrite tests/special_standalone/prompt_rewrite
```

The tests use fake inference and HTTP transports to check the complete workflow,
teacher-input isolation, A/B reversal and scoring, missing/duplicate votes,
checkpoint isolation, frozen-input validation, and H3 crash recovery. Actual
Qwen inference and live provider calls require a separate smoke run.
