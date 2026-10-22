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
"""Launch SFT with loss validation and HF exports; use rl evaluate for video evaluation."""

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .config import (
    DEFAULT_DATASET,
    DEFAULT_REWRITE_INSTRUCTION,
    DEFAULT_SYSTEM_PROMPT,
    ROOT,
    build_messages,
)
from .dataset import load_samples
from .storage import file_sha256, freeze_json, run_lock

CONFIG_DIR = Path(__file__).with_name("configs")


def local_path(value) -> Path:
    """Resolve example inputs relative to the repository root."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def training_config(args):
    """Compose the Hydra configuration passed to the native SFT trainer."""
    from hydra import compose, initialize_config_dir

    overrides = [
        f"hydra.searchpath=[file://{ROOT / 'verl/trainer/config'}]",
        f"trainer.default_local_dir={json.dumps(str(args.run_dir / 'checkpoints'))}",
        f"trainer.n_gpus_per_node={args.nproc_per_node}",
    ]
    overrides.extend(args.overrides)
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        config = compose(config_name="sft", overrides=overrides)
    if config.trainer.nnodes != 1 or config.trainer.n_gpus_per_node != args.nproc_per_node:
        raise ValueError("This launcher uses one node; set its GPU count with --nproc-per-node")
    if config.engine.strategy != "fsdp2" or config.engine.ulysses_sequence_parallel_size != 1:
        raise ValueError("This SFT preset requires FSDP2 with sequence parallel size 1")
    if config.trainer.resume_mode != "auto" or config.trainer.resume_from_path is not None:
        raise ValueError("This SFT launcher uses trainer.resume_mode=auto and resume_from_path=null")
    if config.trainer.total_training_steps is not None or config.trainer.total_epochs < 1:
        raise ValueError("Set a positive trainer.total_epochs; leave total_training_steps=null for complete epochs")
    if config.trainer.save_freq != "after_each_epoch" or config.trainer.test_freq != "after_each_epoch":
        raise ValueError("Keep save_freq and test_freq set to after_each_epoch")
    if not {"model", "optimizer", "extra", "hf_model"} <= set(config.checkpoint.save_contents):
        raise ValueError("Save model, optimizer, extra, and hf_model for epoch resume and inference")
    if not {"model", "optimizer", "extra"} <= set(config.checkpoint.load_contents):
        raise ValueError("Load model, optimizer, and extra to preserve optimizer/scheduler/RNG state")
    if config.model.custom_chat_template is not None:
        raise ValueError("Use the model's shared chat template for this SFT/evaluation preset")
    if (
        config.data.custom_cls.path != "pkg://examples.prompt_rewrite.sft_dataset"
        or config.data.custom_cls.name != "PromptRewriteSFTDataset"
    ):
        raise ValueError("Keep PromptRewriteSFTDataset so preflight and training use the same assistant masks")
    return config, overrides


def validate_data(config, dataset: Path) -> dict:
    """Record dropped SFT rows and check isolation/alignment with the held-out cohort."""
    import pyarrow.parquet as pq

    if config.data.train_max_samples != -1 or config.data.val_max_samples != -1:
        raise ValueError("Use all prepared train/validation rows (max_samples=-1)")
    if (
        config.data.messages_key != "messages"
        or config.data.enable_thinking_default is not False
        or config.data.apply_chat_template_kwargs.get("enable_thinking") is not False
        or config.data.ignore_input_ids_mismatch
        or config.data.truncation != "error"
    ):
        raise ValueError("Keep the SFT preset's message, thinking, truncation, and template-consistency settings")
    batch, world = config.data.train_batch_size, config.trainer.n_gpus_per_node
    sp = config.engine.ulysses_sequence_parallel_size
    if sp < 1 or world < 1 or world % sp or batch < 1 or batch % (world // sp):
        raise ValueError("Sequence-parallel size must divide GPU count; data-parallel size must divide global batch")
    micro = config.data.micro_batch_size_per_gpu
    if not config.data.use_dynamic_bsz and (micro < 1 or (batch // (world // sp)) % micro):
        raise ValueError("Static microbatch size must divide the per-GPU batch")
    splits, files = {}, {}
    for split in ("train", "val"):
        path = local_path(config.data[f"{split}_files"])
        rows = pq.read_table(path).to_pylist()
        if len(rows) < batch:
            raise ValueError(f"{split}: need at least one complete global batch of {batch}; got {len(rows)} rows")
        ids = [row["id"] for row in rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate {split} IDs")
        prefix = DEFAULT_REWRITE_INSTRUCTION + "\n\nUser request:\n"
        for row in rows:
            messages = row["messages"]
            if (
                row.get("enable_thinking") is not False
                or [m["role"] for m in messages] != ["system", "user", "assistant"]
                or messages[0]["content"] != DEFAULT_SYSTEM_PROMPT
                or not messages[1]["content"].startswith(prefix)
                or not messages[1]["content"][len(prefix) :].strip()
                or not messages[2]["content"].strip()
            ):
                raise ValueError(f"SFT messages differ from the shared rewrite task: {split}/{row['id']}")
        splits[split] = rows
        files[split] = {
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": len(rows),
            "used_rows_per_epoch": len(rows) // batch * batch,
            "dropped_rows_per_epoch": len(rows) % batch,
        }
    train_ids, val_ids = ({row["id"] for row in splits[split]} for split in ("train", "val"))
    train_prompts, val_prompts = (
        {row["messages"][1]["content"][len(prefix) :].strip() for row in splits[split]} for split in ("train", "val")
    )
    if train_ids & val_ids or train_prompts & val_prompts:
        raise ValueError("Train/validation overlap by ID or prompt")
    evaluation = {row["id"]: row for row in load_samples(dataset)}
    if set(evaluation) != {row["id"] for row in splits["val"]}:
        raise ValueError("SFT validation IDs differ from the video evaluation cohort")
    for row in splits["val"]:
        sample = evaluation[row["id"]]
        expected = build_messages(sample["prompt"]) + [{"role": "assistant", "content": sample["h3_context_ir"]}]
        if row["messages"] != expected or row["source"] != sample["source"]:
            raise ValueError(f"SFT validation text/source differs from video evaluation: {row['id']}")
    return {
        "files": files,
        "eval_dataset_sha256": file_sha256(dataset),
        "eval_sources": dict(Counter(row["source"] for row in evaluation.values())),
        "steps_per_epoch": len(splits["train"]) // batch,
        "new_videos_per_epoch": 0,
        "judgments_per_epoch": 0,
    }


def check_tokens(config):
    """Exercise the actual SFT dataset and assistant masks before GPU training."""
    from verl.utils import hf_tokenizer

    from .sft_dataset import PromptRewriteSFTDataset

    path = config.model.tokenizer_path or config.model.path
    tokenizer = hf_tokenizer(path, trust_remote_code=config.model.trust_remote_code)
    for split in ("train", "val"):
        data = PromptRewriteSFTDataset(str(local_path(config.data[f"{split}_files"])), tokenizer, config.data)
        lengths, targets = [], []
        for index in range(len(data)):
            item = data[index]
            length, target = len(item["input_ids"]), int(item["loss_mask"].sum())
            if not 0 < target < length:
                raise ValueError(f"Invalid assistant loss mask: {split}/{index}: {target}/{length}")
            lengths.append(length)
            targets.append(target)
        print(json.dumps({"split": split, "max_tokens": max(lengths), "min_target_tokens": min(targets)}), flush=True)


def launch(command: list[str], execute: bool, env: dict):
    """Display a reviewable command and optionally execute it without a shell."""
    print(shlex.join(command), flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def run(args) -> int:
    """Start native SFT; checkpoints are evaluated separately through the tool agent."""
    from omegaconf import OmegaConf

    args.run_dir = local_path(args.run_dir)
    args.dataset = local_path(args.dataset)
    config, overrides = training_config(args)
    summary = validate_data(config, args.dataset)
    tensorboard_dir = local_path(
        args.tensorboard_dir
        or os.environ.get("TENSORBOARD_DIR")
        or (ROOT / "tensorboard_log/prompt-rewrite" / args.run_dir.name)
    )
    env = {**os.environ, "TENSORBOARD_DIR": str(tensorboard_dir)}
    print(
        json.dumps(
            {
                **summary,
                "epochs": config.trainer.total_epochs,
                "tensorboard_dir": str(tensorboard_dir),
            },
            indent=2,
        )
    )
    if args.check_tokens:
        check_tokens(config)
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc-per-node={args.nproc_per_node}",
        "-m",
        "verl.trainer.sft_trainer",
        f"--config-path={CONFIG_DIR}",
        "--config-name=sft",
        *overrides,
    ]
    if not args.execute:
        launch(command, False, env)
        return 0
    with run_lock(args.run_dir):
        freeze_json(
            args.run_dir / "sft-inputs.json",
            {
                "config": OmegaConf.to_container(config, resolve=True),
                "data": summary,
                "tensorboard_dir": str(tensorboard_dir),
            },
        )
        if not args.check_tokens:
            check_tokens(config)
        launch(command, True, env)
    return 0


def parse_args(argv=None):
    """Keep hardware/evaluation flags separate from trailing Hydra overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tensorboard-dir", type=Path)
    parser.add_argument(
        "--execute", action="store_true", help="Train, validate loss, and export checkpoints after every epoch"
    )
    parser.add_argument("--check-tokens", action="store_true", help="Check tokenization/masks even in preview mode")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides after --, e.g. trainer.total_epochs=3")
    args = parser.parse_args(argv)
    if args.nproc_per_node < 1:
        parser.error("GPU count must be positive")
    return args


def main(argv=None) -> int:
    """Report failures without continuing to another training epoch."""
    try:
        return run(parse_args(argv))
    except (ValueError, RuntimeError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"SFT stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
