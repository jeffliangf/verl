import datetime
import json
import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
rank, world = dist.get_rank(), dist.get_world_size()
value = torch.full((1024, 1024), float(rank), device="cuda")
dist.all_reduce(value)
torch.cuda.synchronize()
assert world == 16 and torch.all(value == world * (world - 1) / 2).item()
info = {
    "rank": rank,
    "host": socket.gethostname(),
    "gpu": local_rank,
    "name": torch.cuda.get_device_name(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "memory_bytes": torch.cuda.get_device_properties(local_rank).total_memory,
}
nodes = [None] * world
dist.all_gather_object(nodes, info)
if rank == 0:
    assert len({row["host"] for row in nodes}) == 2
    path = Path(os.environ["RL_RUN_DIR"]) / f"nccl-preflight-{os.environ['SLURM_JOB_ID']}.json"
    path.write_text(json.dumps({"passed": True, "world_size": world, "ranks": nodes}, indent=2) + "\n")
    print("NCCL_PREFLIGHT_PASSED: 16 H100 GPUs on two nodes", flush=True)
dist.barrier()
dist.destroy_process_group()
