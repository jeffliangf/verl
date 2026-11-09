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
"""Verify exact training-state contraction and the next AdamW update."""

import json
from collections import OrderedDict

import pytest
import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

from verl.utils.checkpoint.reshard_fsdp2 import convert


def distributed(value, rank, size=4):
    mesh = DeviceMesh("cuda", list(range(size)), mesh_dim_names=("fsdp",), _init_backend=False, _rank=rank)
    mesh._dim_group_names = ["0"]
    return DTensor(
        value.chunk(size)[rank].clone(),
        DTensorSpec(mesh, (Shard(0),), TensorMeta(value.shape, value.stride(), value.dtype)),
        requires_grad=False,
    )


@pytest.fixture
def checkpoint(tmp_path):
    parameter = torch.nn.Parameter(torch.linspace(-0.8, 0.9, 24).reshape(8, 3))
    optimizer = torch.optim.AdamW([parameter], lr=0.007, weight_decay=0.013, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.9)
    for step in range(3):
        parameter.grad = torch.sin(parameter.detach() * (step + 1))
        optimizer.step()
        scheduler.step()
    source = tmp_path / "source/global_step_3"
    actor = source / "actor"
    (actor / "huggingface").mkdir(parents=True)
    (actor / "huggingface/config.json").write_text("{}\n")
    (actor / "fsdp_config.json").write_text(json.dumps({"FSDP_version": 2, "world_size": 4}))
    (source / "checkpoint_complete.json").write_text('{"step": 3}\n')
    torch.save({"cursor": 19, "generator": torch.Generator().manual_seed(42).get_state()}, source / "data.pt")
    state = optimizer.state_dict()
    for rank in range(4):
        model_state = OrderedDict(weight=distributed(parameter.detach(), rank))
        model_state._metadata = {"": {"version": 1}}
        torch.save(model_state, actor / f"model_world_size_4_rank_{rank}.pt")
        torch.save(
            {
                "state": {
                    0: {
                        key: distributed(value, rank) if value.ndim else value.clone()
                        for key, value in state["state"][0].items()
                    }
                },
                "param_groups": state["param_groups"],
            },
            actor / f"optim_world_size_4_rank_{rank}.pt",
        )
        torch.save(
            {"lr_scheduler": scheduler.state_dict(), "rng": {"torch": torch.tensor([rank])}},
            actor / f"extra_state_world_size_4_rank_{rank}.pt",
        )
    return source, tmp_path / "destination/global_step_3", parameter, optimizer


def load(path):
    return torch.load(path, weights_only=False)


def test_conversion_preserves_model_optimizer_cursor_and_next_adam_update(checkpoint):
    source, target, parameter, optimizer = checkpoint
    original_files = {str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    report = convert(source, target, 2)
    assert report["world_size"] == 2 and report["source_world_size"] == 4
    assert (target / "data.pt").read_bytes() == (source / "data.pt").read_bytes()
    assert json.loads((target / "actor/fsdp_config.json").read_text())["world_size"] == 2
    gradient = torch.cos(parameter.detach() * 1.7)
    expected_before = parameter.detach().clone()
    parameter.grad = gradient
    optimizer.step()
    next_shards = []
    for rank in range(2):
        model_state = load(target / "actor" / f"model_world_size_2_rank_{rank}.pt")
        assert model_state._metadata == {"": {"version": 1}}
        weights = model_state["weight"]
        saved = load(target / "actor" / f"optim_world_size_2_rank_{rank}.pt")
        assert weights.dtype == torch.float32
        assert weights.device_mesh.mesh.tolist() == [0, 1]
        assert tuple(weights.device_mesh.get_coordinate()) == (rank,)
        assert torch.equal(weights.to_local(), expected_before.chunk(2)[rank])
        local_parameter = torch.nn.Parameter(weights.to_local().clone())
        local_optimizer = torch.optim.AdamW([local_parameter])
        saved["state"][0] = {
            key: value.to_local().clone() if isinstance(value, DTensor) else value
            for key, value in saved["state"][0].items()
        }
        local_optimizer.load_state_dict(saved)
        local_parameter.grad = gradient.chunk(2)[rank].clone()
        local_optimizer.step()
        next_shards.append(local_parameter.detach())
        extra = load(target / "actor" / f"extra_state_world_size_2_rank_{rank}.pt")
        old_extra = load(source / "actor" / f"extra_state_world_size_4_rank_{rank * 2}.pt")
        assert extra["lr_scheduler"] == old_extra["lr_scheduler"]
        assert torch.equal(extra["rng"]["torch"], old_extra["rng"]["torch"])
    assert torch.equal(torch.cat(next_shards), parameter.detach())
    assert original_files == {
        str(path.relative_to(source)): path.read_bytes() for path in source.rglob("*") if path.is_file()
    }
    assert (target / "reshard-verification.json").is_file()


@pytest.mark.parametrize("problem", ["uneven", "optimizer_step", "rank", "global_metadata", "async", "existing"])
def test_invalid_checkpoint_never_publishes_converted_state(checkpoint, problem):
    source, target, _, _ = checkpoint
    actor = source / "actor"
    if problem == "uneven":
        for rank in range(4):
            torch.save({"weight": distributed(torch.ones(10, 3), rank)}, actor / f"model_world_size_4_rank_{rank}.pt")
    elif problem in {"optimizer_step", "global_metadata"}:
        path = actor / f"optim_world_size_4_rank_{1 if problem == 'optimizer_step' else 2}.pt"
        value = load(path)
        if problem == "optimizer_step":
            value["state"][0]["step"] += 1
        else:
            value["param_groups"][0]["lr"] *= 2
        torch.save(value, path)
    elif problem == "rank":
        (actor / "model_world_size_4_rank_1.pt").write_bytes((actor / "model_world_size_4_rank_0.pt").read_bytes())
    elif problem == "async":
        (source / "transfer_queue").mkdir()
    else:
        target.mkdir(parents=True)
    with pytest.raises((ValueError, FileExistsError)):
        convert(source, target, 2)
    assert not (target / "checkpoint_complete.json").exists()


def test_rejects_non_integral_contraction(checkpoint):
    source, target, _, _ = checkpoint
    with pytest.raises(ValueError, match="integer contraction"):
        convert(source, target, 3)
    assert not target.exists()
