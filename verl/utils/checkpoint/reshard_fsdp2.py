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
"""Offline contraction of native, evenly sharded FSDP2 training checkpoints.

This deliberately supports only a one-dimensional full-world Shard(0) mesh,
an integer reduction in world size, and dimensions divisible by the old size.
Model and optimizer precision are preserved. No GPUs or process groups are
created. PyTorch 2.13's checkpoint metadata APIs are required; validate a native
restore on the destination before using the converted checkpoint for training.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta


def signature(value):
    """Compare replicated values and global tensor layout without reading shards."""
    if isinstance(value, DTensor):
        return ("dtensor", tuple(value.shape), value.stride(), value.dtype, value.placements)
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Only scalar replicated tensor entries are supported")
        return ("tensor", tuple(value.shape), value.dtype, value.item())
    if isinstance(value, dict):
        return (
            type(value).__name__,
            tuple((key, signature(item)) for key, item in value.items()),
            signature(getattr(value, "_metadata", None)),
        )
    if isinstance(value, list | tuple):
        return (type(value).__name__, tuple(signature(item) for item in value))
    if value is None or type(value) in (str, int, float, bool):
        return (type(value).__name__, value)
    raise ValueError(f"Unsupported checkpoint entry: {type(value).__name__}")


def merge_shards(values, ranks, old_world_size, mesh):
    """Join neighboring shards while retaining global shape and original precision."""
    first = values[0]
    if any(signature(value) != signature(first) for value in values[1:]):
        raise ValueError("Shard layouts or replicated optimizer values disagree")
    if isinstance(first, DTensor):
        for rank, value in zip(ranks, values, strict=True):
            source_mesh = value.device_mesh
            if (
                source_mesh.ndim != 1
                or source_mesh.mesh.tolist() != list(range(old_world_size))
                or tuple(source_mesh.get_coordinate() or ()) != (rank,)
                or source_mesh.device_type != mesh.device_type
                or source_mesh.mesh_dim_names != mesh.mesh_dim_names
                or getattr(source_mesh, "_dim_group_names", None) != ["0"]
                or len(value.placements) != 1
                or type(value.placements[0]) is not Shard
                or value.placements[0].dim != 0
                or not value.ndim
                or value.shape[0] % old_world_size
                or tuple(value.to_local().shape) != (value.shape[0] // old_world_size, *value.shape[1:])
                or value.requires_grad
            ):
                raise ValueError("Expected even Shard(0) tensors on the full default FSDP2 mesh")
        local = torch.cat([value.to_local() for value in values], dim=0)
        # These are detached serialized state tensors. The low-level constructor
        # preserves the CUDA mesh metadata while all storage remains on CPU;
        # from_local would otherwise move the offline conversion onto CUDA.
        return DTensor(
            local,
            DTensorSpec(mesh, (Shard(0),), TensorMeta(first.shape, first.stride(), first.dtype)),
            requires_grad=False,
        )
    if isinstance(first, torch.Tensor):
        return first.clone()
    if isinstance(first, dict):
        result = type(first)(
            (key, merge_shards([value[key] for value in values], ranks, old_world_size, mesh)) for key in first
        )
        if hasattr(first, "_metadata"):
            result._metadata = copy.deepcopy(first._metadata)
        return result
    if isinstance(first, list | tuple):
        return type(first)(
            merge_shards([value[index] for value in values], ranks, old_world_size, mesh) for index in range(len(first))
        )
    return first


def verify_roundtrip(expected, actual):
    """Check every serialized value, tensor byte value, mesh and rank after saving."""
    if signature(expected) != signature(actual):
        raise ValueError("Serialized checkpoint metadata changed")
    if isinstance(expected, DTensor):
        if (
            expected.device_mesh != actual.device_mesh
            or expected.device_mesh.get_coordinate() != actual.device_mesh.get_coordinate()
            or not torch.equal(expected.to_local(), actual.to_local())
        ):
            raise ValueError("Serialized distributed tensor changed")
    elif isinstance(expected, torch.Tensor):
        if not torch.equal(expected, actual):
            raise ValueError("Serialized scalar changed")
    elif isinstance(expected, dict):
        for key in expected:
            verify_roundtrip(expected[key], actual[key])
    elif isinstance(expected, list | tuple):
        for left, right in zip(expected, actual, strict=True):
            verify_roundtrip(left, right)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def convert(source, destination, world_size):
    """Write and verify a new checkpoint; publish its directory only when complete."""
    if not torch.__version__.startswith("2.13."):
        raise ValueError("This offline checkpoint conversion requires the validated PyTorch 2.13 format")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    cfg = json.loads((source / "actor/fsdp_config.json").read_text())
    old_size = cfg["world_size"]
    if cfg.get("FSDP_version") != 2 or not 1 <= world_size < old_size or old_size % world_size:
        raise ValueError("Require FSDP2 and an integer contraction of the source world size")
    if (source / "transfer_queue").exists() or (source / "video-cohorts.json").exists():
        raise ValueError("Pending async work requires a separate migration; use a synchronous checkpoint")
    marker = json.loads((source / "checkpoint_complete.json").read_text())
    if source.name != f"global_step_{marker['step']}" or destination.name != source.name:
        raise ValueError("Source and destination must retain the saved global step")
    temporary = destination.with_name(f".{destination.name}.resharding")
    if destination.exists() or temporary.exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint or interrupted conversion")
    paths = [source / "data.pt", source / "checkpoint_complete.json", source / "actor/fsdp_config.json"]
    paths += [
        source / "actor" / f"{kind}_world_size_{old_size}_rank_{rank}.pt"
        for rank in range(old_size)
        for kind in ("model", "optim", "extra_state")
    ]
    before = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}
    extra = [
        torch.load(source / "actor" / f"extra_state_world_size_{old_size}_rank_{rank}.pt", weights_only=False)
        for rank in range(old_size)
    ]
    if any(signature(value["lr_scheduler"]) != signature(extra[0]["lr_scheduler"]) for value in extra[1:]):
        raise ValueError("Source ranks disagree on scheduler state")
    temporary.mkdir(parents=True)
    (temporary / "actor").mkdir()
    report = {
        "source": str(source),
        "source_world_size": old_size,
        "world_size": world_size,
        "step": marker["step"],
        "torch": torch.__version__,
        "rng_mapping": "new rank r retains the RNG state of source rank r * contraction_factor",
        "files": {},
    }
    metadata = {}
    ratio = old_size // world_size
    for rank in range(world_size):
        ranks = list(range(rank * ratio, (rank + 1) * ratio))
        for kind in ("model", "optim"):
            shards = [
                torch.load(
                    source / "actor" / f"{kind}_world_size_{old_size}_rank_{r}.pt",
                    mmap=True,
                    map_location="cpu",
                    weights_only=False,
                )
                for r in ranks
            ]
            current = signature(shards[0])
            if kind in metadata and current != metadata[kind]:
                raise ValueError("Source ranks disagree on global state or optimizer metadata")
            metadata[kind] = current
            # Source validation below checks the actual names and full-world mesh.
            mesh = DeviceMesh(
                "cuda", list(range(world_size)), mesh_dim_names=("fsdp",), _init_backend=False, _rank=rank
            )
            mesh._dim_group_names = ["0"]
            merged = merge_shards(shards, ranks, old_size, mesh)
            filename = f"actor/{kind}_world_size_{world_size}_rank_{rank}.pt"
            output = temporary / filename
            torch.save(merged, output)
            loaded = torch.load(output, mmap=True, map_location="cpu", weights_only=False)
            verify_roundtrip(merged, loaded)
            report["files"][filename] = {"bytes": output.stat().st_size, "sha256": digest(output)}
            del shards, merged, loaded
            gc.collect()
            print(f"Verified {filename}", flush=True)
        torch.save(extra[ranks[0]], temporary / "actor" / f"extra_state_world_size_{world_size}_rank_{rank}.pt")
    for name in ("data.pt", "checkpoint_complete.json"):
        shutil.copy2(source / name, temporary / name)
    shutil.copytree(source / "actor/huggingface", temporary / "actor/huggingface")
    (temporary / "actor/fsdp_config.json").write_text(json.dumps({**cfg, "world_size": world_size}, indent=2) + "\n")
    for path in sorted(temporary.rglob("*")):
        if path.is_file() and str(path.relative_to(temporary)) not in report["files"]:
            report["files"][str(path.relative_to(temporary))] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    if before != {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in paths}:
        raise RuntimeError("Source checkpoint changed during conversion")
    (temporary / "reshard-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    if destination.exists():
        raise FileExistsError("Destination appeared during conversion; leaving verified temporary checkpoint intact")
    os.rename(temporary, destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = convert(args.source, args.destination, args.world_size)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
