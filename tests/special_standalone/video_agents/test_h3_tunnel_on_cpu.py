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
"""Inject tunnel outages using real child processes and a local HTTP endpoint."""

import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "examples/video_agents/inspection_rl_run/grpo"


def eventually(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("Condition did not become true before the deadline")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def tunnel(root, recovery_timeout=4, use_uv=False):
    """Use an HTTP server in place of SSH, leaving supervision unmodified."""
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    child_script = root / "ssh.py"
    child_script.write_text(
        textwrap.dedent("""\
        import os, signal, sys, time
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from pathlib import Path

        root, port = Path(sys.argv[1]), int(sys.argv[2])
        with (root / 'pids').open('a') as stream:
            stream.write(str(os.getpid()) + '\\n')
        if (root / 'refuse').exists():
            sys.exit(255)
        if (root / 'ignore-term').exists():
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if (root / 'stall').exists():
                    time.sleep(30)
                payload = b'{"status":"warming"}' if (root / 'unhealthy').exists() else b'{"status":"ready"}'
                self.send_response(200)
                if (root / 'truncated').exists():
                    self.send_header('Content-Length', '1000')
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args):
                pass

        class Server(HTTPServer):
            allow_reuse_address = True

        Server(('127.0.0.1', port), Handler).serve_forever()
        """)
    )
    ready_file = root / "ready"
    log_file = root / "tunnel.log"
    runner = [sys.executable]
    if use_uv:
        uv = shutil.which("uv")
        if uv is None:
            pytest.skip("uv is not installed")
        runner = [uv, "run", "--no-project", "--python", sys.executable, "python"]
    with log_file.open("w") as output:
        process = subprocess.Popen(
            [
                *runner,
                str(SCRIPTS / "h3_tunnel.py"),
                "--health-url",
                f"http://127.0.0.1:{port}/healthz",
                "--ready-file",
                str(ready_file),
                "--recovery-timeout",
                str(recovery_timeout),
                "--",
                sys.executable,
                str(child_script),
                str(root),
                str(port),
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        state = SimpleNamespace(
            process=process,
            ready=ready_file,
            log=log_file,
            pids=lambda: [int(pid) for pid in (root / "pids").read_text().splitlines()],
        )
        try:
            yield state
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=8)
            finally:
                # Also clean up injected failures if a test assertion fails.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=3)


@pytest.mark.parametrize("use_uv", [False, True])
def test_disconnect_recovers_without_stopping_worker_or_ray(tmp_path, use_uv):
    with tunnel(tmp_path, use_uv=use_uv) as state, subprocess.Popen(["sleep", "60"]) as ray:
        waiter = None
        try:
            eventually(state.ready.exists)
            # A healthy connection may outlive the recovery timeout. The budget
            # starts again at the next disconnect, not at launcher startup.
            time.sleep(4.1)
            waiter = subprocess.Popen(
                [
                    "bash",
                    "-c",
                    'set -euo pipefail; source "$1"; rl_control=$2; rl_ssh_pid=$3; rl_ray_pid=$4; '
                    "RL_RUN_DIR=$2; SLURM_JOB_ID=test; rl_node=worker; rl_wait_for_driver_and_acknowledge",
                    "bash",
                    str(SCRIPTS / "shutdown-sync.sh"),
                    str(tmp_path),
                    str(state.process.pid),
                    str(ray.pid),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            first_child = state.pids()[0]
            os.kill(first_child, signal.SIGTERM)
            eventually(lambda: len(state.pids()) >= 2 and state.ready.exists())
            assert state.process.poll() is None
            assert ray.poll() is None
            assert waiter.poll() is None
            assert not alive(first_child)
            assert alive(state.pids()[-1])

            (tmp_path / "driver.done").touch()
            assert waiter.wait(timeout=5) == 0
            assert (tmp_path / "worker-worker.done").exists()
            state.process.terminate()
            assert state.process.wait(timeout=7) == 143, state.log.read_text()
            assert not any(alive(pid) for pid in state.pids())
        finally:
            if waiter is not None:
                if waiter.poll() is None:
                    waiter.terminate()
                waiter.communicate(timeout=5)
            ray.terminate()


@pytest.mark.parametrize("failure", ["refuse", "unhealthy", "stall", "truncated"])
def test_persistent_outage_expires_and_cleans_up(tmp_path, failure):
    with tunnel(tmp_path) as state:
        eventually(state.ready.exists)
        (tmp_path / failure).touch()
        start = time.monotonic()
        os.kill(state.pids()[0], signal.SIGTERM)
        assert state.process.wait(timeout=7) == 1, state.log.read_text()
        assert 3 <= time.monotonic() - start < 7
        assert "recovery deadline of 4s expired" in state.log.read_text()
        assert not state.ready.exists()
        assert len(state.pids()) >= (3 if failure == "refuse" else 2)
        assert not any(alive(pid) for pid in state.pids())


def test_startup_requires_h3_readiness_and_removes_stale_marker(tmp_path):
    (tmp_path / "ready").touch()
    (tmp_path / "unhealthy").touch()
    with tunnel(tmp_path, recovery_timeout=2) as state:
        assert state.process.wait(timeout=5) == 1, state.log.read_text()
        assert not state.ready.exists()
        assert "health check passed" not in state.log.read_text()
        assert not any(alive(pid) for pid in state.pids())


@pytest.mark.parametrize("ignore_term", [False, True])
def test_shutdown_reaps_ssh_child(tmp_path, ignore_term):
    if ignore_term:
        (tmp_path / "ignore-term").touch()
    with tunnel(tmp_path) as state:
        eventually(state.ready.exists)
        state.process.terminate()
        assert state.process.wait(timeout=7) == 143, state.log.read_text()
        assert not state.ready.exists()
        assert not any(alive(pid) for pid in state.pids())
