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
"""Staged CLI; GPU inference and network stages are dry runs unless --execute is set."""

import argparse
import json
import subprocess
from pathlib import Path

from . import self_hosted_h3
from .config import (
    DEFAULT_DATASET,
    DEFAULT_EVAL_REPO,
    DEFAULT_REWRITE_INSTRUCTION,
    DEFAULT_SELF_HOSTED_REFERENCES,
    DEFAULT_SYSTEM_PROMPT,
    JUDGE_CONCURRENCY,
    RewriteConfig,
)
from .evaluation import EvaluationRun
from .reporting import print_report
from .storage import read_jsonl


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run-dir", type=Path, required=True)
    sub = parser.add_subparsers(dest="stage", required=True)
    prepare = sub.add_parser("prepare", parents=[common], help="Validate references and freeze a checkpoint's inputs")
    prepare.add_argument("--name", default="qwen38-zero-shot")
    prepare.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    prepare.add_argument("--references", type=Path, default=DEFAULT_SELF_HOSTED_REFERENCES)
    prepare.add_argument("--eval-repo", type=Path, default=DEFAULT_EVAL_REPO)
    self_hosted_h3.add_arguments(prepare)
    prepare.add_argument("--system-prompt-file", type=Path)
    prepare.add_argument("--rewrite-instruction-file", type=Path)
    prepare.add_argument("--limit", type=int, help="Freeze a small cohort for a smoke run in its own directory")
    for field in (
        "model",
        "revision",
        "max_new_tokens",
        "temperature",
        "seed",
        "tensor_parallel_size",
        "max_model_len",
        "gpu_memory_utilization",
        "backend",
    ):
        default = getattr(RewriteConfig(), field)
        prepare.add_argument(
            "--" + field.replace("_", "-"), default=default, type=type(default) if default is not None else str
        )
    for stage in ("rewrite", "generate", "judge", "benchmark-references"):
        command = sub.add_parser(stage, parents=[common])
        mode = command.add_mutually_exclusive_group()
        mode.add_argument("--execute", action="store_true")
        mode.add_argument("--dry-run", action="store_true", help="Preview only (the default)")
        if stage == "rewrite":
            command.add_argument("--batch-size", type=int, default=8)
        else:
            command.add_argument(
                "--concurrency",
                type=int,
                default=self_hosted_h3.DEFAULT_CONCURRENCY if stage == "generate" else JUDGE_CONCURRENCY,
            )
        if stage == "generate":
            command.add_argument("--poll-interval", type=float, default=1)
        if stage in ("generate", "judge"):
            command.add_argument("--neutral-on-failure", action="store_true")
            command.add_argument("--pass-timeout-seconds", type=float, default=1800)
        if stage == "judge":
            command.add_argument("--request-timeout-seconds", type=float, default=300)
        if stage == "benchmark-references":
            command.add_argument("--source-run", type=Path, required=True, help="Run whose reference videos to compare")
            command.add_argument("--name", default="h3-context-ir-vs-prompt")
    imported = sub.add_parser("import-rewrites", parents=[common], help="Use saved SFT/RL rollout outputs")
    imported.add_argument("--input", type=Path, required=True)
    report = sub.add_parser("report", parents=[common], help="Score both orientations and write per-prompt results")
    report.add_argument("--allow-partial", action="store_true")
    report.add_argument(
        "--tensorboard-dir", type=Path, help="Override TENSORBOARD_DIR or the default run log directory"
    )
    report.add_argument("--step", type=int, default=0, help="Training step for TensorBoard (zero-shot: 0)")
    report.add_argument("--no-tensorboard", action="store_true", help="Write JSON reports without TensorBoard events")
    sub.add_parser("status", parents=[common])
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.stage == "prepare":
            config = RewriteConfig(**{field: getattr(args, field) for field in RewriteConfig.__dataclass_fields__})
            system = (
                args.system_prompt_file.read_text(encoding="utf-8").strip()
                if args.system_prompt_file
                else DEFAULT_SYSTEM_PROMPT
            )
            instruction = (
                args.rewrite_instruction_file.read_text(encoding="utf-8").strip()
                if args.rewrite_instruction_file
                else DEFAULT_REWRITE_INSTRUCTION
            )
            run = EvaluationRun.prepare(
                args.run_dir,
                name=args.name,
                config=config,
                dataset=args.dataset,
                references=args.references,
                eval_repo=args.eval_repo,
                system_prompt=system,
                rewrite_instruction=instruction,
                limit=args.limit,
                h3_backend=args.h3_backend,
                h3_endpoints=args.h3_endpoints,
                h3_token_file=args.h3_token_file,
            )
            result = run.status()
        elif args.stage == "benchmark-references":
            run = EvaluationRun.prepare_reference_benchmark(args.run_dir, source_run=args.source_run, name=args.name)
            print(json.dumps(run.status(), indent=2))
            run.judge(execute=args.execute, concurrency=args.concurrency)
            return 0
        else:
            run = EvaluationRun(args.run_dir)
            if args.stage == "rewrite":
                result = run.rewrite(execute=args.execute, batch_size=args.batch_size)
            elif args.stage == "import-rewrites":
                result = run.import_rewrites(read_jsonl(args.input.expanduser()))
            elif args.stage == "generate":
                result = run.generate(
                    execute=args.execute,
                    concurrency=args.concurrency,
                    poll_interval=args.poll_interval,
                    neutral_on_failure=args.neutral_on_failure,
                    pass_timeout_seconds=args.pass_timeout_seconds,
                )
                print(json.dumps(result, indent=2))
                return int(args.execute and result["videos_accounted_for"] != result["prompts"])
            elif args.stage == "judge":
                run.judge(
                    execute=args.execute,
                    concurrency=args.concurrency,
                    neutral_on_failure=args.neutral_on_failure,
                    request_timeout_seconds=args.request_timeout_seconds,
                    pass_timeout_seconds=args.pass_timeout_seconds,
                )
                return 0
            elif args.stage == "report":
                result = run.report(
                    allow_partial=args.allow_partial,
                    tensorboard_dir=args.tensorboard_dir,
                    step=args.step,
                    log_tensorboard=not args.no_tensorboard,
                )
                print_report(result)
                print(f"Saved {run.path / 'report.json'} and per_prompt_scores.jsonl")
                if "tensorboard" in result:
                    print(f"TensorBoard: {result['tensorboard']['log_dir']} (step {result['tensorboard']['step']})")
                return 0
            else:
                result = run.status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Evaluation stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
