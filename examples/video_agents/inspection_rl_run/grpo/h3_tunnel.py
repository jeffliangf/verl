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
"""Keep one launcher-owned SSH tunnel alive across short disconnections."""

import argparse
import http.client
import json
import math
import signal
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} H3 tunnel: {message}", flush=True)


def ready(url, timeout):
    """Probe H3 through the forwarded port, without credentials or generation."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            health = json.load(response)
        return isinstance(health, dict) and health.get("status") == "ready"
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        return False


def supervise(command, health_url, ready_file, recovery_timeout=120):
    """Keep this PID stable; fail only after one continuous recovery deadline."""
    if not math.isfinite(recovery_timeout) or recovery_timeout <= 0:
        raise ValueError("Recovery timeout must be positive and finite")
    child = None
    healthy = False
    deadline = time.monotonic() + recovery_timeout
    delay = 2
    ready_file.unlink(missing_ok=True)
    try:
        while True:
            if child is None or child.poll() is not None:
                if healthy:
                    log(f"SSH exited with code {child.returncode}; reconnecting for up to {recovery_timeout:g}s")
                    healthy = False
                    deadline = time.monotonic() + recovery_timeout
                    delay = 2
                    ready_file.unlink(missing_ok=True)
                if time.monotonic() < deadline:
                    child = subprocess.Popen(command, stdin=subprocess.DEVNULL)

            if healthy:
                try:
                    child.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log(f"ERROR: recovery deadline of {recovery_timeout:g}s expired")
                return 1
            if ready(health_url, min(5, remaining)) and child.poll() is None and time.monotonic() < deadline:
                healthy = True
                ready_file.touch()
                log("H3 health check passed; tunnel ready")
                continue

            # A failed SSH attempt or an open port without healthy H3 must not
            # reset the deadline. Only a successful health check resets it.
            time.sleep(min(delay, max(0, deadline - time.monotonic())))
            delay = min(delay * 2, 10)
    finally:
        ready_file.unlink(missing_ok=True)
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-url", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--recovery-timeout", type=float, default=120)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Supply the SSH command after --")

    def stop(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return supervise(command, args.health_url, args.ready_file, args.recovery_timeout)


if __name__ == "__main__":
    raise SystemExit(main())
