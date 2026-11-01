import json
import os
import shutil
import socket
from pathlib import Path

source = Path(os.environ["RL_MODEL_SOURCE"])
target = Path(os.environ["RL_LOCAL_MODEL"])
index = json.loads((source / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
assert shards and all((source / name).is_file() for name in shards)
target.mkdir(mode=0o700, parents=True, exist_ok=True)
for original in sorted(source.iterdir()):
    if not original.is_file():
        continue
    destination = target / original.name
    if destination.exists() and destination.stat().st_size == original.stat().st_size:
        continue
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(original, temporary)
    assert temporary.stat().st_size == original.stat().st_size
    temporary.replace(destination)
assert all((target / name).stat().st_size == (source / name).stat().st_size for name in shards)
print(
    json.dumps({"host": socket.gethostname(), "model": str(target), "shards": len(shards), "ready": True}), flush=True
)
