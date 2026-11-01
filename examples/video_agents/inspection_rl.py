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
"""Prepare and run V1-1 inspection RL from a frozen video-only experiment."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from . import rl
from .config import ROOT
from .h3_deadlines import phase_budgets, worker_timeout
from .inspection_agent import feedback
from .storage import atomic_text, file_sha256, fingerprint, freeze_json, read_json, read_jsonl, run_lock
from .tournament import load_protocol, runtime_protocol

SYSTEM_EXTENSION = (
    "Balance expected improvements in final video quality against the resource cost of additional tool calls. "
    "Decide which available actions are worthwhile. Use `finish_video` when you choose to stop."
)
CONFIG_DIR = Path(__file__).with_name("configs")
AGENT_CONFIG = CONFIG_DIR / "inspection_agent.yaml"
MANAGER = "examples.video_agents.inspection_loop.InspectionManager"


def tool_config(directory, train_file):
    return {
        "tools": [
            {
                "class_name": "examples.video_agents.inspection_loop.InspectionActionTool",
                "config": {"type": "native", "name": name, "dataset": str(train_file)},
            }
            for name in ("sol_h3", "inspect_video", "finish_video")
        ]
    }


def prepare(args):
    directory, data_dir, source = map(rl.local_path, (args.run_dir, args.data_dir, args.reference_run))
    if directory == source:
        raise ValueError("Inspection RL needs a new run directory")
    original = load_protocol(source / "protocol.json")
    protocol = deepcopy(original)
    added_tools = read_json(CONFIG_DIR / "inspection_tools.json")
    contents = {}
    for split in ("train", "val"):
        path = Path(original["datasets"][split]["path"])
        if data_dir / f"{split}.jsonl" == path:
            raise ValueError("Inspection RL needs new dataset files")
        if file_sha256(path) != original["datasets"][split]["sha256"]:
            raise ValueError("The reference dataset changed")
        rows = read_jsonl(path)
        if [r["extra_info"]["id"] for r in rows] != [s["id"] for s in protocol["samples"][split]]:
            raise ValueError("Reference row order differs from its frozen protocol")
        for row in rows:
            if len(row["tools"]) != 1 or row["tools"][0]["function"]["name"] != "sol_h3":
                raise ValueError("Expected the original single-generator dataset")
            row["agent_name"] = "h3_inspection"
            row["prompt"][0]["content"] += "\n\n" + SYSTEM_EXTENSION
            row["tools"].extend(deepcopy(added_tools))
        contents[split] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    protocol["datasets"] = {
        split: {"path": str(data_dir / f"{split}.jsonl"), "sha256": sha256(content.encode()).hexdigest()}
        for split, content in contents.items()
    }
    protocol["groups_per_worker"] = args.groups_per_worker
    protocol["inspection"] = {
        "variant": "V1-1",
        "reference_protocol": str(source / "protocol.json"),
        "reference_protocol_sha256": fingerprint(original),
        "source_datasets": original["datasets"],
        "settings": feedback.SETTINGS,
        "prompt_sha256": file_sha256(feedback.PROMPT_PATH),
        "schema_sha256": fingerprint(feedback.SCHEMA),
        "implementation_sha256": file_sha256(Path(feedback.__file__)),
        "system_extension": SYSTEM_EXTENSION,
        "max_generations": 2,
        "max_inspections": 1,
        "second_generation_requires_inspection": True,
        "reward": "original final-video quality only",
    }
    with run_lock(directory):
        for split, content in contents.items():
            path = data_dir / f"{split}.jsonl"
            if path.exists() and path.read_text() != content:
                raise ValueError(f"Prepared data changed: {path}")
        freeze_json(directory / "protocol.json", protocol)
        freeze_json(directory / "protocol-sha256.json", fingerprint(protocol))
        for split, content in contents.items():
            atomic_text(data_dir / f"{split}.jsonl", content)
        freeze_json(directory / "tools.json", tool_config(directory, data_dir / "train.jsonl"))
    return {
        "run_dir": str(directory),
        "train_prompts": len(protocol["samples"]["train"]),
        "eval_prompts": len(protocol["samples"]["val"]),
        "group_size": protocol["group_size"],
    }


def training_config(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    directory, data_dir = map(rl.local_path, (args.run_dir, args.data_dir))
    overrides = [
        *list(OmegaConf.load(CONFIG_DIR / "inspection_rl.yaml")["overrides"]),
        f"hydra.searchpath=[file://{ROOT / 'verl/trainer/config'}]",
        f"video_rl.protocol={json.dumps(str(directory / 'protocol.json'))}",
        f"video_rl.output_dir={json.dumps(str(directory))}",
        f"data.train_files={json.dumps(str(data_dir / 'train.jsonl'))}",
        f"data.val_files={json.dumps(str(data_dir / 'val.jsonl'))}",
        f"actor_rollout_ref.rollout.multi_turn.tool_config_path={json.dumps(str(directory / 'tools.json'))}",
        f"actor_rollout_ref.rollout.agent.agent_loop_config_path={json.dumps(str(AGENT_CONFIG))}",
        f"trainer.default_local_dir={json.dumps(str(directory / 'checkpoints'))}",
        f"trainer.rollout_data_dir={json.dumps(str(directory / 'rollouts'))}",
        f"trainer.validation_data_dir={json.dumps(str(directory / 'validation'))}",
        *args.overrides,
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name="rl", overrides=overrides)
    protocol = load_protocol(directory / "protocol.json")
    runtime_protocol(protocol, config.video_rl)
    rollout = config.actor_rollout_ref.rollout
    if (
        not config.trainer.use_v1
        or config.trainer.v1.trainer_mode != "sync"
        or config.algorithm.adv_estimator != "grpo"
        or rollout.n != protocol["group_size"]
        or rollout.val_kwargs.n != 1
        or rollout.agent.agent_loop_manager_class != MANAGER
        or rollout.agent.default_agent_loop != "h3_inspection"
        or config.reward.reward_model.enable
        or config.distillation.enabled
    ):
        raise ValueError("Keep native synchronous GRPO, the inspection loop, and original quality rewards")
    multi = rollout.multi_turn
    if (
        not multi.enable
        or multi.format != "qwen3_coder"
        or multi.max_parallel_calls != 1
        or multi.max_assistant_turns is not None
        or multi.max_user_turns is not None
        or config.data.apply_chat_template_kwargs
        or config.data.filter_overlong_prompts
    ):
        raise ValueError("Keep native parsing, episode limits, and per-row chat-template settings")
    if config.inspection_rl.turn_max_tokens < 1 or rollout.max_model_len < (
        config.data.max_prompt_length + config.data.max_response_length
    ):
        raise ValueError("Model context must fit the complete prompt and continuation budgets")
    for actual, expected in (
        (config.video_rl.protocol, directory / "protocol.json"),
        (config.video_rl.output_dir, directory),
        (multi.tool_config_path, directory / "tools.json"),
        (rollout.agent.agent_loop_config_path, AGENT_CONFIG),
    ):
        if rl.local_path(actual) != expected:
            raise ValueError("Use this run's frozen protocol, tools and native loop")
    return config, overrides


def train(args):
    from omegaconf import OmegaConf

    directory = rl.local_path(args.run_dir)
    protocol = load_protocol(directory / "protocol.json")
    config, overrides = training_config(args)
    frozen = protocol["inspection"]
    if (
        frozen["settings"] != feedback.SETTINGS
        or frozen["prompt_sha256"] != file_sha256(feedback.PROMPT_PATH)
        or frozen["schema_sha256"] != fingerprint(feedback.SCHEMA)
        or frozen["implementation_sha256"] != file_sha256(Path(feedback.__file__))
    ):
        raise ValueError("The frozen V1 inspector changed")
    if read_json(directory / "tools.json") != tool_config(directory, rl.local_path(config.data.train_files)):
        raise ValueError("The prepared native tools changed")
    for split in ("train", "val"):
        path = rl.local_path(config.data[f"{split}_files"])
        if {"path": str(path), "sha256": file_sha256(path)} != protocol["datasets"][split]:
            raise ValueError(f"Frozen {split} dataset changed")
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={CONFIG_DIR}",
        "--config-name=rl",
        *overrides,
    ]
    episodes = config.data.train_batch_size * protocol["group_size"]
    summary = {
        "run_dir": str(directory),
        "model": config.actor_rollout_ref.model.path,
        "train_prompts": len(protocol["samples"]["train"]),
        "eval_prompts": len(protocol["samples"]["val"]),
        "training_episodes_per_update": episodes,
        "generation_calls_per_update": [episodes, 2 * episodes],
        "max_inspections_per_update": episodes,
        "h3_phase_timeouts": phase_budgets(config.video_rl),
        "h3_worker_timeout_seconds": worker_timeout(config.video_rl, config.video_rl.timeouts.h3),
        "reward": frozen["reward"],
        "context": {
            "prompt": config.data.max_prompt_length,
            "continuation": config.data.max_response_length,
            "model": config.actor_rollout_ref.rollout.max_model_len,
            "per_policy_turn": config.inspection_rl.turn_max_tokens,
        },
        "execute": args.execute,
        "command": shlex.join(command),
    }
    if args.check_tokens or args.execute:
        summary["max_prompt_tokens"] = rl.check_tokens(config)
    print(json.dumps(summary, indent=2), flush=True)
    if args.execute:
        with run_lock(directory):
            freeze_json(directory / "training-config.json", OmegaConf.to_container(config, resolve=True))
            subprocess.run(
                command, cwd=ROOT, env={**os.environ, "TENSORBOARD_DIR": str(directory / "tensorboard")}, check=True
            )
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "train"):
        child = commands.add_parser(name)
        child.add_argument("--run-dir", type=Path, required=True)
        child.add_argument("--data-dir", type=Path, required=True)
        if name == "prepare":
            child.add_argument("--reference-run", type=Path, required=True)
            child.add_argument("--groups-per-worker", type=int, default=4)
        else:
            child.add_argument("--execute", action="store_true")
            child.add_argument("--check-tokens", action="store_true")
            child.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if getattr(args, "overrides", None) and args.overrides[0] == "--":
        args.overrides = args.overrides[1:]
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.command == "prepare":
        print(json.dumps(prepare(args), indent=2))
    else:
        train(args)


if __name__ == "__main__":
    main()
