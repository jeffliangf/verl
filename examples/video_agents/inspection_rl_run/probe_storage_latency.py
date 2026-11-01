"""Measure small artifact writes under load, removing only this probe's files."""
import json
import os
import socket
import tempfile
import time
from pathlib import Path

root = Path(__file__).parent / "grpo"
results = []
payload = b"storage-latency-probe\n" * 3000
for label, parent in [("episode_mount", root / "episodes"), ("shared_home", root), ("local_tmp", Path("/tmp"))]:
    rows = []
    with tempfile.TemporaryDirectory(prefix=".inspection-io-probe-", dir=parent) as directory:
        path = Path(directory) / "probe"
        for _ in range(3):
            started = time.monotonic()
            with path.open("wb") as stream:
                stream.write(payload)
                stream.flush()
                written = time.monotonic()
                os.fsync(stream.fileno())
                synced = time.monotonic()
            target = path.with_suffix(".done")
            path.replace(target)
            renamed = time.monotonic()
            assert target.read_bytes() == payload
            rows.append({"write": written-started, "fsync": synced-written,
                         "rename": renamed-synced, "round_trip": time.monotonic()-started})
            target.unlink()
    results.append({"storage": label, "samples": rows})
output = {"host": socket.gethostname(), "at_epoch": time.time(), "results": results}
path = root / f"storage-latency-{socket.gethostname()}.json"
path.write_text(json.dumps(output, indent=2)+"\n")
print(json.dumps(output), flush=True)
