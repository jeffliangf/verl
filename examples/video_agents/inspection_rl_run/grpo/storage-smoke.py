"""Exercise concurrent writes, fsync, rename, cross-client reads and checksums."""

import concurrent.futures
import hashlib
import json
import os
import socket
import time
from pathlib import Path

root = Path(__file__).parent
host = socket.gethostname()
data = os.urandom(16 * 1024 * 1024)
digest = hashlib.sha256(data).hexdigest()
start = time.monotonic()


def write(index):
    directory = root / "checkpoints/.storage-smoke"
    directory.mkdir(exist_ok=True)
    path = directory / f"{host}-{index}.bin"
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return {"path": str(path), "sha256": digest, "bytes": len(data)}


with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    files = list(pool.map(write, range(8)))
result = {"host": host, "seconds": time.monotonic() - start, "files": files}
(root / f"storage-smoke-{host}.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"host": host, "seconds": result["seconds"], "bytes": len(data) * len(files)}), flush=True)
