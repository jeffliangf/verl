"""Keep raw GPU samples and live TensorBoard system charts for this allocation."""

import argparse
import csv
import io
import json
import math
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path


def parse_samples(text, expected=8):
    samples = []
    for row in csv.DictReader(io.StringIO(text), skipinitialspace=True):
        sample = {
            "gpu": int(row["index"]),
            "utilization_percent": float(row["utilization.gpu [%]"].split()[0]),
            "memory_used_gib": float(row["memory.used [MiB]"].split()[0]) / 1024,
            "memory_total_gib": float(row["memory.total [MiB]"].split()[0]) / 1024,
        }
        if not all(math.isfinite(v) for v in sample.values()):
            raise ValueError("GPU sample is not finite")
        samples.append(sample)
    if len(samples) != expected or {r["gpu"] for r in samples} != set(range(expected)):
        raise ValueError("GPU telemetry must account for all allocated GPUs")
    return samples


def main():
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    root = args.run_dir
    started = json.loads((root / "run-budget.json").read_text())["started_at_epoch"]
    writer = EventFileWriter(str(root / "tensorboard/system" / socket.gethostname()), flush_secs=15)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    path = root / f"gpus-{args.job_id}-{socket.gethostname()}.csv"
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv",
    ]
    try:
        with path.open("a", buffering=1) as raw:
            while not stop.is_set():
                try:
                    text = subprocess.check_output(command, text=True, timeout=10)
                    samples = parse_samples(text)
                    now = time.time()
                    raw.write(text if raw.tell() == 0 else text.split("\n", 1)[1])
                    raw.flush()
                    values = [
                        Summary.Value(tag=f"gpu_{s['gpu']}/{key}", simple_value=value)
                        for s in samples
                        for key, value in s.items()
                        if key != "gpu"
                    ]
                    values.append(
                        Summary.Value(
                            tag="all_gpus/mean_utilization_percent",
                            simple_value=sum(s["utilization_percent"] for s in samples) / len(samples),
                        )
                    )
                    writer.add_event(
                        Event(wall_time=now, step=max(0, int(now - started)), summary=Summary(value=values))
                    )
                    writer.flush()
                except Exception as exc:
                    print(f"GPU_TELEMETRY_ERROR {type(exc).__name__}: {exc}", flush=True)
                stop.wait(15)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
