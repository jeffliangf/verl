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
"""Validate complete native datasets in the training environment before submission."""

import argparse
import importlib.metadata
import json
import socket
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

from .config import ROOT
from .storage import file_sha256, fingerprint, read_json, read_jsonl, write_json
from .tournament import load_protocol


def validate_native_data(config, protocol, tokenizer=None, processor=None):
    """Exercise the actual trainer loader, every raw observation, and mixed collation."""
    from verl.trainer.ppo.utils import create_rl_dataset
    from verl.utils.dataset.rl_dataset import collate_fn

    results = {}
    for split in ("train", "val"):
        path = Path(config.data[f"{split}_files"])
        spec = protocol["datasets"][split]
        if str(path) != spec["path"] or file_sha256(path) != spec["sha256"]:
            raise ValueError(f"Frozen {split} data changed")
        rows = read_jsonl(path)
        dataset = create_rl_dataset(str(path), config.data, tokenizer=tokenizer, processor=processor, max_samples=-1)
        if len(dataset) != len(rows):
            raise ValueError(f"The native loader dropped {split} records")
        counts, representatives = Counter(), {}
        for index, row in enumerate(rows):
            if dataset.dataframe[index] != row:
                raise ValueError(f"Native {split} record {index} differs from the frozen JSON")
            item = dataset[index]
            task = row["extra_info"]["task"]
            images = [
                part
                for message in item["raw_prompt"]
                if isinstance(message["content"], list)
                for part in message["content"]
                if part.get("type") == "image"
            ]
            if len(images) != (1 if task == "i2v" else 0):
                raise ValueError(f"Incorrect source-image binding at {split} record {index}")
            if images and images[0]["image"] != row["images"][0]["image"]:
                raise ValueError(f"Source image changed at {split} record {index}")
            if item["extra_info"]["id"] != protocol["samples"][split][index]["id"]:
                raise ValueError("Native record and frozen reward sample disagree")
            counts[task] += 1
            representatives.setdefault(task, item)
        batch = collate_fn(list(representatives.values()))
        if any(len(value) != len(representatives) for value in batch.values()):
            raise ValueError("Mixed native collation lost batch alignment")
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"Frozen {split} data changed during validation")
        results[split] = {"rows": len(rows), "native_items_checked": len(rows), "by_task": dict(counts)}
    return results


def preflight(run):
    from verl.utils import hf_processor, hf_tokenizer

    from .rl import check_tokens

    manifest = read_json(run / "launch.json")
    config = read_json(run / "preview-config.json")
    protocol = load_protocol(run / "protocol.json")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != manifest["code_commit"] or subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT):
        raise ValueError("Preflight requires the clean, frozen runtime checkout")
    if sys.prefix != manifest["python_environment"]:
        raise ValueError("Run data preflight with the production training Python environment")
    composed = OmegaConf.create(config)
    model = composed.actor_rollout_ref.model
    tokenizer = hf_tokenizer(model.path, trust_remote_code=model.trust_remote_code)
    processor = hf_processor(model.path, trust_remote_code=model.trust_remote_code)
    if processor is None or getattr(processor, "image_processor", None) is None:
        raise ValueError("Mixed video preflight requires the policy's image processor")
    native = validate_native_data(composed, protocol, tokenizer=tokenizer, processor=processor)
    tokens = check_tokens(composed)
    assets = {}
    for samples in protocol["samples"].values():
        for sample in samples:
            for asset in [sample.get("image"), *sample.get("references", {}).values()]:
                if asset:
                    if assets.setdefault(asset["path"], asset["sha256"]) != asset["sha256"]:
                        raise ValueError("Conflicting frozen media hashes")

    def verify_asset(item):
        path, digest = item
        if file_sha256(Path(path)) != digest:
            raise ValueError(f"Frozen media changed: {path}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(verify_asset, assets.items()))
    result = {
        "passed": True,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "runtime_commit": commit,
        "python_environment": sys.prefix,
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "datasets", "pyarrow", "transformers", "vllm")
        },
        "configuration_sha256": fingerprint(config),
        "protocol_sha256": file_sha256(run / "protocol.json"),
        "datasets": protocol["datasets"],
        "native": native,
        "max_prompt_tokens": tokens,
        "verified_media_assets": len(assets),
    }
    write_json(run / "data-preflight.json", result)
    return result


def require_preflight(run):
    """Reject missing, partial, or stale results before the scheduler allocates GPUs."""
    manifest = read_json(run / "launch.json")
    report = read_json(run / "data-preflight.json")
    protocol = load_protocol(run / "protocol.json")
    config = read_json(run / "preview-config.json")
    if (
        report.get("passed") is not True
        or report.get("runtime_commit") != manifest["code_commit"]
        or report.get("python_environment") != manifest["python_environment"]
        or report.get("configuration_sha256") != fingerprint(config)
        or report.get("protocol_sha256") != file_sha256(run / "protocol.json")
        or report.get("datasets") != protocol["datasets"]
    ):
        raise ValueError("Missing or stale full-data preflight; do not submit this run")
    for split, spec in protocol["datasets"].items():
        expected = len(protocol["samples"][split])
        native = report.get("native", {}).get(split, {})
        if (
            native.get("native_items_checked") != expected
            or native.get("rows") != expected
            or native.get("by_task") != dict(Counter(sample["task"] for sample in protocol["samples"][split]))
            or not 0 < report.get("max_prompt_tokens", {}).get(split, 0) <= config["data"]["max_prompt_length"]
            or file_sha256(Path(spec["path"])) != spec["sha256"]
        ):
            raise ValueError(f"Incomplete or stale {split} data preflight")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.run_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
