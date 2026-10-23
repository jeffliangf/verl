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
"""Prepare, train, or evaluate the same tool-calling policy; execution is opt-in."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from math import isfinite
from pathlib import Path

from . import h3
from .config import DEFAULT_DATASET, DEFAULT_EVAL_REPO, DEFAULT_REFERENCES, DEFAULT_SYSTEM_PROMPT, JUDGE_SETTINGS, ROOT
from .dataset import load_samples, reference_inventory
from .judge import EXTERNAL_FILES
from .observability import summarize_timings
from .storage import (
    atomic_text,
    file_sha256,
    fingerprint,
    freeze_json,
    load_external,
    read_json,
    read_jsonl,
    run_lock,
    write_json,
)
from .tournament import GROUP_SIZE, load_protocol

CONFIG_DIR = Path(__file__).with_name("configs")
DEFAULT_DATA_DIR = ROOT / "data/prompt_rewrite_rl"
GENERATION_INSTRUCTION = Path(__file__).with_name("generation_instruction.txt").read_text().strip()


def local_path(value) -> Path:
    """Resolve all example paths relative to the checkout."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def prompt_rows(path: Path) -> list[dict]:
    """Keep only prompt metadata; teacher rewrites are not training inputs."""
    result, ids = [], set()
    for row in read_jsonl(path):
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("id", "prompt")):
            raise ValueError("Every RL row needs a nonempty id and prompt")
        if row["id"] in ids:
            raise ValueError(f"Duplicate RL ID: {row['id']}")
        source = row.get("source", "unknown")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Invalid data source: {row['id']}")
        ids.add(row["id"])
        result.append({"id": row["id"], "prompt": row["prompt"], "source": source})
    if not result:
        raise ValueError("The RL training dataset is empty")
    return result


def dataset_records(samples: list[dict], split: str, tools: list[dict]) -> list[dict]:
    """Materialize complete model inputs; rollout consumes these saved fields verbatim."""
    return [
        {
            "data_source": sample["source"],
            "agent_name": "h3_terminal",
            "prompt": [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": GENERATION_INSTRUCTION.format(prompt=sample["prompt"])},
            ],
            "tools": deepcopy(tools),
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "xhigh"},
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {"id": sample["id"], "index": index, "split": split},
        }
        for index, sample in enumerate(samples)
    ]


def tool_config(directory: Path, train_file: Path) -> dict:
    """Register the executable; its model-facing schema comes from the dataset."""
    return {
        "tools": [
            {
                "class_name": "examples.video_agents.agent.SolH3Tool",
                "config": {
                    "type": "native",
                    "protocol": str(directory / "protocol.json"),
                    "dataset": str(train_file),
                },
            }
        ]
    }


def prepare(args) -> dict:
    """Freeze data, tool schema, evaluation references, and provider settings offline."""
    directory = local_path(args.run_dir)
    data_dir = local_path(args.data_dir)
    train_file, val_file = local_path(args.train_file), local_path(args.eval_dataset)
    references, repo = local_path(args.references), local_path(args.eval_repo)
    train, validation = prompt_rows(train_file), load_samples(val_file)
    if {r["id"] for r in train} & {r["id"] for r in validation} or {r["prompt"].strip() for r in train} & {
        r["prompt"].strip() for r in validation
    }:
        raise ValueError("Train/evaluation overlap by ID or prompt")
    if not isfinite(args.poll_interval) or args.poll_interval <= 0:
        raise ValueError("Polling interval must be positive and finite")
    if min(args.groups_per_worker, args.h3_concurrency_per_worker) < 1:
        raise ValueError("Worker concurrency must be positive")
    dependencies = {relative: file_sha256(repo / relative) for relative in EXTERNAL_FILES}
    rubric = load_external(repo / "scripts/build_pairwise_manifest.py")
    if rubric.INSTRUCTIONS_VERSION != JUDGE_SETTINGS["instructions_version"]:
        raise ValueError("Expected the existing v4 Gemini rubric")
    val, profile = reference_inventory(validation, references, load_external(repo / "bestofn/normalize.py").probe)
    protocol = {
        "schema_version": 2,
        "task": "h3_tournament",
        "group_size": GROUP_SIZE,
        "samples": {"train": train, "val": val},
        "source_files": {str(path): file_sha256(path) for path in (train_file, val_file, references / "manifest.json")},
        "reference_manifest": str(references / "manifest.json"),
        "media_profile": profile,
        "judge": JUDGE_SETTINGS,
        "rubric": rubric.INSTRUCTIONS,
        "eval_repo": str(repo),
        "dependencies": dependencies,
        "poll_interval": args.poll_interval,
        "groups_per_worker": args.groups_per_worker,
        "h3_concurrency_per_worker": args.h3_concurrency_per_worker,
    }
    protocol.update(
        h3.configuration(args.h3_endpoints, profile, read_json(references / "manifest.json"), args.h3_token_file)
    )
    # Policy-controlled arguments and their defaults live in the dataset. Keep
    # only fixed service settings here; reference_h3 retains the reference seed/duration.
    for key in ("duration", "seed"):
        protocol["h3"].pop(key)
    schema = read_json(CONFIG_DIR / "sol_h3_tool.json")
    records = {split: dataset_records(protocol["samples"][split], split, [schema]) for split in ("train", "val")}
    contents = {
        split: "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows) for split, rows in records.items()
    }
    protocol["datasets"] = {
        split: {"path": str(data_dir / f"{split}.jsonl"), "sha256": sha256(content.encode()).hexdigest()}
        for split, content in contents.items()
    }
    with run_lock(directory):
        for split, content in contents.items():
            path = data_dir / f"{split}.jsonl"
            if path.exists() and path.read_text() != content:
                raise ValueError(f"Prepared dataset changed: {path}")
        freeze_json(directory / "protocol.json", protocol)
        freeze_json(directory / "protocol-sha256.json", fingerprint(protocol))
        for split, content in contents.items():
            path = data_dir / f"{split}.jsonl"
            atomic_text(path, content)
            write_json(data_dir / f"{split}.preview.json", records[split][0])
        freeze_json(directory / "tools.json", tool_config(directory, data_dir / "train.jsonl"))
    return {
        "run_dir": str(directory),
        "data_dir": str(data_dir),
        "train_prompts": len(train),
        "eval_prompts": len(val),
        "network_requests": 0,
    }


def training_config(args):
    """Compose the exact config passed to the native V1 entry point."""
    from hydra import compose, initialize_config_dir

    directory = local_path(args.run_dir)
    data_dir = local_path(args.data_dir)
    evaluation = args.command == "evaluate"
    output = local_path(args.output_dir) if evaluation else directory
    if evaluation and output == directory:
        raise ValueError("Evaluation needs its own --output-dir, separate from the training run")
    overrides = [
        f"hydra.searchpath=[file://{ROOT / 'verl/trainer/config'}]",
        f"video_rl.protocol={json.dumps(str(directory / 'protocol.json'))}",
        f"video_rl.output_dir={json.dumps(str(output))}",
        f"data.train_files={json.dumps(str(data_dir / 'train.jsonl'))}",
        f"data.val_files={json.dumps(str(data_dir / 'val.jsonl'))}",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={json.dumps(str(directory / 'tools.json'))}",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={json.dumps(str(CONFIG_DIR / 'h3_agent.yaml'))}",
        f"trainer.default_local_dir={json.dumps(str(output / 'checkpoints'))}",
        f"trainer.rollout_data_dir={json.dumps(str(output / 'rollouts'))}",
        f"trainer.validation_data_dir={json.dumps(str(output / 'validation'))}",
        *(
            [
                f"actor_rollout_ref.model.path={json.dumps(args.model)}",
                "trainer.val_before_train=true",
                "trainer.val_only=true",
                "trainer.resume_mode=disable",
            ]
            if evaluation
            else []
        ),
        *args.overrides,
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name="rl", overrides=overrides)
    if evaluation and (
        not config.trainer.val_before_train
        or not config.trainer.val_only
        or config.trainer.resume_mode != "disable"
        or config.trainer.resume_from_path is not None
        or config.actor_rollout_ref.model.path != args.model
    ):
        raise ValueError("Evaluation loads --model and requires val_only, val_before_train, and resume_mode=disable")
    if (
        not config.trainer.use_v1
        or config.trainer.v1.trainer_mode != "sync"
        or config.algorithm.adv_estimator != "grpo"
        or config.actor_rollout_ref.rollout.n != GROUP_SIZE
        or config.actor_rollout_ref.rollout.val_kwargs.n != 1
    ):
        raise ValueError("Keep synchronous V1 GRPO, four training rollouts, and one validation rollout")
    multi_turn = config.actor_rollout_ref.rollout.multi_turn
    if multi_turn.max_assistant_turns is not None or multi_turn.max_user_turns is not None:
        raise ValueError("The terminal H3 loop enforces its own turn limit after executing the tool")
    if config.data.apply_chat_template_kwargs:
        raise ValueError("Set chat-template options in dataset rows, not in the trainer config")
    if config.data.filter_overlong_prompts:
        raise ValueError("Use the launcher's token check, which includes each row's tools and chat-template options")
    if config.distillation.enabled:
        raise ValueError("This preset uses video rewards without policy distillation")
    if (
        config.actor_rollout_ref.rollout.agent.agent_loop_manager_class
        != "examples.video_agents.agent.TournamentManager"
        or not multi_turn.enable
        or multi_turn.format != "qwen3_coder"
        or multi_turn.max_parallel_calls != 1
        or config.reward.reward_model.enable
    ):
        raise ValueError("Keep the terminal H3 agent, native Qwen tool parser, and tournament reward manager")
    for actual, expected in (
        (config.video_rl.protocol, directory / "protocol.json"),
        (config.video_rl.output_dir, output),
        (config.trainer.default_local_dir, output / "checkpoints"),
        (config.trainer.rollout_data_dir, output / "rollouts"),
        (config.trainer.validation_data_dir, output / "validation"),
        (multi_turn.tool_config_path, directory / "tools.json"),
        (config.actor_rollout_ref.rollout.agent.agent_loop_config_path, CONFIG_DIR / "h3_agent.yaml"),
    ):
        if local_path(actual) != expected:
            raise ValueError("Use the prepared experiment's protocol and tool/agent configuration")
    return config, overrides


def check_tokens(config):
    """Render saved rows with the actual tokenizer, check lengths, and save previews."""
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(
        config.actor_rollout_ref.model.path, trust_remote_code=config.actor_rollout_ref.model.trust_remote_code
    )
    lengths = {}
    for split in ("train", "val"):
        path = local_path(config.data[f"{split}_files"])
        rows = read_jsonl(path)
        sizes = [
            len(
                tokenizer.apply_chat_template(
                    row["prompt"],
                    tools=row["tools"],
                    **row["chat_template_kwargs"],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                )
            )
            for row in rows
        ]
        if max(sizes) > config.data.max_prompt_length:
            raise ValueError(f"{split} tool prompt exceeds data.max_prompt_length: {max(sizes)}")
        lengths[split] = max(sizes)
        write_json(path.with_suffix(".preview.json"), rows[0])
        atomic_text(
            path.with_suffix(".preview.txt"),
            tokenizer.apply_chat_template(
                rows[0]["prompt"],
                tools=rows[0]["tools"],
                **rows[0]["chat_template_kwargs"],
                add_generation_prompt=True,
                tokenize=False,
            ),
        )
    return lengths


def train(args) -> dict:
    """Launch native training or validation; preview cost/configuration by default."""
    from omegaconf import OmegaConf

    directory = local_path(args.run_dir)
    evaluation = args.command == "evaluate"
    output = local_path(args.output_dir) if evaluation else directory
    protocol = load_protocol(directory / "protocol.json")
    config, overrides = training_config(args)
    if read_json(directory / "tools.json") != tool_config(directory, local_path(config.data.train_files)):
        raise ValueError("Prepared H3 tool configuration changed")
    for split in ("train", "val"):
        path = local_path(config.data[f"{split}_files"])
        if {"path": str(path), "sha256": file_sha256(path)} != protocol["datasets"][split]:
            raise ValueError(f"{split} data differs from the frozen experiment")
    batch = config.data.train_batch_size
    if batch < 1 or batch > len(protocol["samples"]["train"]):
        raise ValueError("Training batch must fit in the prepared dataset")
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={CONFIG_DIR}",
        "--config-name=rl",
        *overrides,
    ]
    summary = {
        "mode": "evaluate" if evaluation else "train",
        "output_dir": str(output),
        "model": config.actor_rollout_ref.model.path,
        "h3_model": protocol["h3"]["model"],
        "h3_endpoints": protocol["h3_endpoints"],
        "reference_h3": protocol["reference_h3"],
        "train_prompts": len(protocol["samples"]["train"]),
        "eval_prompts": len(protocol["samples"]["val"]),
        "videos_per_training_batch": 0 if evaluation else GROUP_SIZE * batch,
        "judgments_per_training_batch": 0 if evaluation else GROUP_SIZE * (GROUP_SIZE - 1) * batch,
        "videos_per_evaluation": len(protocol["samples"]["val"]),
        "judgments_per_evaluation": 4 * len(protocol["samples"]["val"]),
        "execute": args.execute,
        "command": shlex.join(command),
    }
    if args.check_tokens or args.execute:
        summary["max_prompt_tokens"] = check_tokens(config)
    print(json.dumps(summary, indent=2), flush=True)
    if args.execute:
        with run_lock(output):
            name = "evaluation-config.json" if evaluation else "training-config.json"
            freeze_json(output / name, OmegaConf.to_container(config, resolve=True))
            env = {**os.environ, "TENSORBOARD_DIR": str(output / "tensorboard")}
            subprocess.run(command, cwd=ROOT, env=env, check=True)
    return summary


def parse_args(argv=None):
    """Keep preparation offline and make paid training an explicit command flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--run-dir", type=Path, required=True)
    prep.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    prep.add_argument("--train-file", type=Path, default=ROOT / "data/prompt_rewrite_rl.jsonl")
    prep.add_argument("--eval-dataset", type=Path, default=DEFAULT_DATASET)
    prep.add_argument("--references", type=Path, default=DEFAULT_REFERENCES)
    prep.add_argument("--eval-repo", type=Path, default=DEFAULT_EVAL_REPO)
    h3.add_arguments(prep)
    prep.add_argument("--poll-interval", type=float, default=1)
    prep.add_argument("--groups-per-worker", type=int, default=5)
    prep.add_argument("--h3-concurrency-per-worker", type=int, default=5)
    status = commands.add_parser("status", help="Read H3 readiness and any reported model provenance; no generation")
    status.add_argument("--run-dir", type=Path, required=True)
    status.add_argument("--timeout", type=float, default=5)
    timings = commands.add_parser("timings", help="Summarize stage latency, failures, and interrupted work offline")
    timings.add_argument("--run-dir", type=Path, required=True)
    for mode in ("train", "evaluate"):
        launch = commands.add_parser(mode)
        launch.add_argument("--run-dir", type=Path, required=True)
        launch.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
        if mode == "evaluate":
            launch.add_argument("--output-dir", type=Path, required=True)
            launch.add_argument("--model", required=True, help="Base model or exported Hugging Face checkpoint")
        launch.add_argument("--execute", action="store_true")
        launch.add_argument("--check-tokens", action="store_true")
        launch.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if getattr(args, "overrides", None) and args.overrides[0] == "--":
        args.overrides = args.overrides[1:]
    return args


def main(argv=None):
    """Prepare, train, evaluate, or inspect the configured H3 service."""
    args = parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args), indent=2))
    elif args.command == "status":
        protocol = load_protocol(local_path(args.run_dir) / "protocol.json")
        result = h3.server_status(protocol, timeout=args.timeout)
        print(json.dumps(result, indent=2))
        return 0 if result["all_ready"] else 1
    elif args.command == "timings":
        print(json.dumps(summarize_timings(local_path(args.run_dir)), indent=2))
    else:
        train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
