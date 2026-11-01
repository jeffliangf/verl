"""Mount only this experiment's bulk artifacts on available node-local disks."""

import argparse
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BINARY = ROOT.parent / "storage-tools/package/usr/bin/sshfs"
KEY = "/home/jeff/.ssh/gmi"
STORES = {
    "episodes": ("176.56.202.237", "/tmp/jeff-inspection-v1-1-20261101/episodes"),
    "checkpoints": ("176.56.202.238", "/tmp/jeff-inspection-v1-1-20261101/checkpoints"),
}


def ensure():
    result = {"host": socket.gethostname(), "stores": {}}
    for name, (host, remote_path) in STORES.items():
        target = ROOT / name
        source = f"jeff@{host}:{remote_path}"
        target.mkdir(exist_ok=True)
        if not os.path.ismount(target):
            if any(target.iterdir()):
                raise RuntimeError(f"Refusing to hide existing files under a new mount: {target}")
            subprocess.run([
                "ssh", "-i", KEY, "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=yes", "-o", "ConnectTimeout=10", f"jeff@{host}",
                f"mkdir -p {remote_path}",
            ], check=True, timeout=30)
            subprocess.run([
                str(BINARY), source, str(target), "-o",
                "reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,ConnectTimeout=10,"
                f"IdentityFile={KEY},IdentitiesOnly=yes,BatchMode=yes,StrictHostKeyChecking=yes,"
                "dir_cache=no,attr_timeout=0,entry_timeout=0,negative_timeout=0,max_conns=8",
            ], check=True, timeout=30)
        mounted = subprocess.check_output(["findmnt", "-n", "-o", "SOURCE", "--target", str(target)], text=True).strip()
        if mounted != source:
            raise RuntimeError(f"Wrong backing storage: {target}: {mounted}")
        free = shutil.disk_usage(target).free
        result["stores"][name] = {"source": source, "free_bytes": free, "mount": str(target)}
        probe = target / f".storage-probe-{socket.gethostname()}"
        with probe.open("w") as handle:
            handle.write(socket.gethostname())
            handle.flush()
            os.fsync(handle.fileno())
        if probe.read_text() != socket.gethostname():
            raise RuntimeError("Storage probe did not round trip")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(ensure()), flush=True)
