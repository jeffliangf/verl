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
"""Upload and judge arriving clips in one isolated evaluator process per group.

Each pair has an immutable manifest and ledger, including its presentation order.
Final scoring validates those manifests against the complete group. Uploads are
shared across both orders; rewards still wait for the whole synchronous group.
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from . import judge
from .dataset import verify_references
from .judge_worker import gemini_address_family, instrument_client, parallel_pairs, parse_options
from .observability import ProviderError, record_event, timed_stage
from .storage import file_sha256, freeze_json, load_external, read_json, write_json
from .tournament import check_dependencies, tournament_manifests


def pair_manifest(manifest, key):
    return {**manifest, "metadata": {**manifest["metadata"], "pair_count": 1}, "pairs": {key: manifest["pairs"][key]}}


def ready_manifests(protocol, sample, candidates, validate):
    if validate:
        valid = candidates[0]["status"] == "succeeded"
        return judge.build_manifests(
            {**protocol, "name": "h3-agent", "samples": [sample]},
            {sample["id"]: candidates[0]} if valid else {},
        )
    return tournament_manifests(sample, candidates, protocol)


def pair_directory(directory, key):
    # Pair IDs are constructed locally, never from the model or provider.
    from .storage import item_key

    return directory / "judge/stream/pairs" / item_key(key)


@contextmanager
def record_fatal(root):
    try:
        yield
    except BaseException as exc:
        # Write before thread-pool cleanup can block. A timeout must never
        # neutralize an unexpected crash or a fatal authentication failure.
        write_json(
            root / "fatal.json",
            {
                "type": type(exc).__name__,
                "message": "See pair ledgers and runner.log",
                "provider_stage": exc.stage if isinstance(exc, ProviderError) else None,
            },
        )
        raise


def read_pair(directory, protocol, manifest, order, *, allow_fallback, timed_out=False):
    manifest_path = directory / "judge" / f"manifest-{order}.json"
    freeze_json(manifest_path, manifest)
    path = directory / "judge" / f"judgments-{order}.jsonl"
    completed, _ = judge.read_judgments(path, manifest, manifest_path, protocol["judge"])
    fallback_path = directory / "judge" / f"fallbacks-{order}.json"
    fallbacks = judge.read_fallbacks(fallback_path, manifest, manifest_path, protocol["judge"])
    if set(completed) & set(fallbacks) or (fallbacks and not allow_fallback):
        raise ValueError("Conflicting or disallowed streamed fallbacks")
    errors = judge.read_failure_rows(path, manifest, manifest_path, protocol["judge"])
    for key in set(manifest["pairs"]) - set(completed) - set(fallbacks):
        if key not in errors and not timed_out:
            continue
        # Classify even during a timeout: auth/configuration errors stay fatal.
        reason = judge.fallback_reason(errors[key]) if key in errors else "pass_timeout"
        if not allow_fallback:
            raise ProviderError("gemini", f"Streamed {order} comparison {key} failed: {reason}; see {path}")
        fallbacks[key] = judge.neutral_fallback(
            key, manifest, manifest_path, protocol["judge"], reason, errors.get(key)
        )
    if fallbacks:
        write_json(fallback_path, [fallbacks[key] for key in sorted(fallbacks)])
    return {**completed, **fallbacks}


def read_streamed_judgments(directory, protocol, manifests, *, allow_fallback):
    """Keep original pair-ledger identities; never rewrite their manifest hashes."""
    outcome = read_json(directory / "judge/stream/outcome.json")
    judge.freeze_manifests(directory, manifests)
    results, index = {}, {}
    for order, manifest in manifests.items():
        results[order], index[order] = {}, {}
        for key in manifest["pairs"]:
            pair_dir = pair_directory(directory, key)
            result = read_pair(
                pair_dir,
                protocol,
                pair_manifest(manifest, key),
                order,
                allow_fallback=allow_fallback,
                timed_out=outcome["timed_out"],
            )
            if key not in result:
                raise RuntimeError(f"Streaming runner left {order}/{key} without a result or recorded failure")
            results[order].update(result)
            index[order][key] = str(pair_dir / "judge" / f"manifest-{order}.json")
    freeze_json(directory / "judge/stream-index.json", index)
    return results


class StreamingJudge:
    """Async parent: publish arrivals, bound the final pass, and reap on cancellation."""

    def __init__(self, directory, protocol, sample, *, validate):
        self.directory, self.protocol = directory, protocol
        self.root = directory / "judge/stream"
        self.context = {
            "protocol": {k: v for k, v in protocol.items() if k != "samples"},
            "sample": sample,
            "validate": validate,
        }
        self.process = None
        self.log = None

    async def start(self):
        check_dependencies(self.protocol)
        freeze_json(self.root / "context.json", self.context)
        for name in ("outcome.json", "fatal.json", "finished.json"):
            (self.root / name).unlink(missing_ok=True)
        self.log = (self.root / "runner.log").open("ab")
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "examples.video_agents.streaming_judge",
                str(self.directory),
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            self.log.close()
            raise

    def offer(self, index, candidate):
        freeze_json(self.root / f"candidate-{index}.json", candidate)

    async def finish(self):
        write_json(self.root / "finished.json", True)
        timeout = self.protocol["timeouts"]["judge_pass"]
        timed_out = False
        with timed_stage(self.directory / "judge/timings.jsonl", "gemini_stream_tail", timeout_seconds=timeout):
            try:
                await asyncio.wait_for(self.process.wait(), timeout=timeout)
            except TimeoutError:
                timed_out = True
                self.process.kill()
                await self.process.wait()
            if (self.root / "fatal.json").exists():
                failure = read_json(self.root / "fatal.json")
                message = f"Streaming evaluator failed: {failure}; see {self.root / 'runner.log'}"
                if failure.get("provider_stage"):
                    raise ProviderError(failure["provider_stage"], message)
                raise RuntimeError(message)
            if self.process.returncode != 0 and not timed_out:
                raise RuntimeError(
                    f"Streaming evaluator exited {self.process.returncode}; see {self.root / 'runner.log'}"
                )
        write_json(self.root / "outcome.json", {"timed_out": timed_out, "returncode": self.process.returncode})
        if timed_out:
            record_event(
                self.directory / "judge/timings.jsonl", "gemini_stream", "timeout", cleanup_status="unconfirmed"
            )

    async def close(self):
        if self.process is not None and self.process.returncode is None:
            self.process.kill()
            await self.process.wait()
            record_event(
                self.directory / "judge/timings.jsonl", "gemini_stream", "aborted", cleanup_status="unconfirmed"
            )
        if self.log is not None:
            self.log.close()


def run_stream(directory, runner):
    """Reuse the frozen runner, its retry budget, and a shared upload cache."""
    root = directory / "judge/stream"
    context = read_json(root / "context.json")
    protocol, sample, validate = context["protocol"], context["sample"], context["validate"]
    check_dependencies(protocol)
    if validate:
        verify_references([sample])
    count = 1 if validate else protocol["group_size"]
    candidates = [{"status": "pending"} for _ in range(count)]
    allow_fallback = not validate and protocol.get("judge_failure_policy") == "neutral"
    env = runner.read_env_file(Path(protocol["eval_repo"]) / "arena/.env")
    api_key = os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY", "")
    client = runner.GeminiRestClient(
        api_key,
        max_attempts=protocol["judge_max_attempts"],
        request_timeout=protocol["timeouts"]["judge_request"],
        file_timeout=protocol["timeouts"]["judge_file"],
    )
    uploads, submitted = {}, {}

    def upload(path):
        try:
            client.upload_video(Path(path))
        except (runner.EvalError, OSError):
            # The shared cache retains the exception. The runner records it in
            # each affected pair's ledger with its usual error classification.
            pass

    def compare(pair_dir, manifest, order):
        key = next(iter(manifest["pairs"]))
        _, forwarded = parse_options(judge.command(pair_dir, protocol, order, [key], allow_fallback=allow_fallback)[3:])
        with timed_stage(directory / "judge/timings.jsonl", "gemini_stream_pair", pair_id=key, order=order):
            code = runner.main(forwarded)
            result = read_pair(pair_dir, protocol, manifest, order, allow_fallback=allow_fallback)
            if key not in result:
                raise RuntimeError(f"Streaming runner exited {code} without recording {order}/{key}")

    # Uploads have their own pool, so comparisons cannot starve an upload they need.
    with (
        ThreadPoolExecutor(max_workers=count + 3) as upload_pool,
        ThreadPoolExecutor(max_workers=protocol.get("judge_concurrency", 1)) as pair_pool,
        record_fatal(root),
    ):
        while True:
            for i in range(count):
                path = root / f"candidate-{i}.json"
                if candidates[i]["status"] != "pending" or not path.exists():
                    continue
                candidate = read_json(path)
                if candidate["status"] not in {"succeeded", "invalid_call"}:
                    raise ValueError("Only completed videos or invalid calls may enter a streamed group")
                if candidate["status"] == "succeeded":
                    if file_sha256(Path(candidate["path"])) != candidate["sha256"]:
                        raise ValueError("Candidate video changed before streaming upload")
                    if candidate["path"] not in uploads:
                        uploads[candidate["path"]] = upload_pool.submit(upload, candidate["path"])
                candidates[i] = candidate
                record_event(directory / "judge/timings.jsonl", "gemini_stream_candidate", "ready", candidate=i)
            manifests = ready_manifests(protocol, sample, candidates, validate)
            # Interleave A/B and B/A tasks so both orders start concurrently.
            for key in manifests["original"]["pairs"]:
                for order, manifest in manifests.items():
                    if (order, key) in submitted:
                        continue
                    pair = manifest["pairs"][key]
                    paths = [pair["video_path_A"], pair["video_path_B"]]
                    for path in paths:
                        if path not in uploads:
                            uploads[path] = upload_pool.submit(upload, path)
                    if not all(uploads[path].done() for path in paths):
                        continue
                    for path in paths:
                        uploads[path].result()  # Do not mask unexpected upload bugs.
                    single, pair_dir = pair_manifest(manifest, key), pair_directory(directory, key)
                    completed = read_pair(pair_dir, protocol, single, order, allow_fallback=allow_fallback)
                    submitted[order, key] = (
                        None if key in completed else pair_pool.submit(compare, pair_dir, single, order)
                    )
            for future in submitted.values():
                if future is not None and future.done():
                    future.result()
            finished = (root / "finished.json").exists() and all(c["status"] != "pending" for c in candidates)
            if finished and len(submitted) == 2 * len(manifests["original"]["pairs"]):
                break
            time.sleep(0.1)
        for future in submitted.values():
            if future is not None:
                future.result()
        for future in uploads.values():
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory / "judge/stream"
    protocol = read_json(root / "context.json")["protocol"]
    runner = load_external(Path(protocol["eval_repo"]) / "scripts/run_gemini_pairwise_eval.py")
    instrument_client(runner, args.directory / "judge/timings-stream.jsonl", protocol["timeouts"]["judge_control"])
    with record_fatal(root), gemini_address_family("ipv4"), parallel_pairs(runner):
        run_stream(args.directory, runner)


if __name__ == "__main__":
    main()
