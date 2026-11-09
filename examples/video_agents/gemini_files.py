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
"""Retain upload ownership across retries and judge-process termination."""

import argparse
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .storage import load_external, read_json


def valid_name(name):
    return isinstance(name, str) and re.fullmatch(r"files/[A-Za-z0-9_-]{1,128}", name) is not None


class UploadJournal:
    """Append before polling an upload, so failed processing cannot lose its name."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def record(self, event, name, **fields):
        if not valid_name(name):
            raise ValueError("Invalid Gemini file resource name")
        row = json.dumps({"event": event, "name": name, **fields}) + "\n"
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as handle:
                handle.write(row)
                handle.flush()
                os.fsync(handle.fileno())

    def created(self, name, path):
        path = Path(path)
        display = "eval-" + hashlib.sha256(str(path).encode()).hexdigest()[:20] + path.suffix.lower()
        self.record("created", name, display_name=display)

    def deleted(self, name):
        self.record("deleted", name)

    def pending(self):
        pending = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                row = json.loads(line)
                if not valid_name(row.get("name")):
                    raise ValueError("Invalid Gemini upload journal")
                if row["event"] == "created":
                    pending[row["name"]] = row["display_name"]
                elif row["event"] == "deleted":
                    pending.pop(row["name"], None)
                else:
                    raise ValueError("Unknown Gemini upload journal event")
        return pending


def reap(journal, runner, client):
    """Call only after the owning subprocess has exited and all threads are gone."""

    def remove(item):
        name, display = item
        try:
            remote, _ = client.request_json(
                "GET", runner.API_BASE + "/" + name, retry_network=True, context="Verify owned upload cleanup"
            )
        except runner.HttpFailure as exc:
            if exc.status != 404:
                raise
        else:
            if remote.get("name") != name or remote.get("displayName") != display:
                raise ValueError("Gemini cleanup ownership mismatch")
            client.delete_file(name)
        journal.deleted(name)

    pending = journal.pending()
    failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(remove, item) for item in pending.items()]
        for future in futures:
            try:
                future.result()
            except (runner.EvalError, OSError):
                failed += 1  # Durable pending entries remain available for recovery.
    return {"pending_before": len(pending), "failed": failed, "pending_after": len(journal.pending())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    journal = UploadJournal(args.directory / "judge/gemini-files.jsonl")
    if not journal.pending():
        return
    protocol = read_json(args.directory / "judge/stream/context.json")["protocol"]
    runner = load_external(Path(protocol["eval_repo"]) / "scripts/run_gemini_pairwise_eval.py")
    env = runner.read_env_file(Path(protocol["eval_repo"]) / "arena/.env")
    key = os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY", "")
    client = runner.GeminiRestClient(key, max_attempts=1, request_timeout=5, file_timeout=5)
    from .judge_worker import gemini_address_family

    with gemini_address_family("ipv4"):
        result = reap(journal, runner, client)
    print(json.dumps(result), flush=True)
    if result["pending_after"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
