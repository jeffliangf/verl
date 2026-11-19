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
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path

from . import judge
from .config import judging_protocol
from .dataset import verify_references
from .gemini_files import UploadJournal
from .gemini_transport import read_api_key, vertex_runner
from .image_assets import image_bytes
from .judge_worker import gemini_address_family, instrument_client, parallel_pairs, parse_options, reference_images
from .observability import ProviderError, record_event, timed_stage
from .storage import file_sha256, freeze_json, guard_parent, load_external, read_json, write_json
from .tournament import check_dependencies, tournament_manifests

# A training pair whose primary comparison failed is re-judged in both orders by
# the secondary judge; this marker in its pair directory records that routing.
SECONDARY_LEDGER = "secondary"
SECONDARY_MARKER = "secondary.json"


def pair_manifest(manifest, key):
    return {**manifest, "metadata": {**manifest["metadata"], "pair_count": 1}, "pairs": {key: manifest["pairs"][key]}}


def ready_manifests(protocol, sample, candidates, validate):
    protocol = judging_protocol(protocol, validate=validate)
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


def secondary_routed(directory):
    return (directory / "judge" / SECONDARY_MARKER).exists()


def secondary_reason(row):
    """Any secondary failure imputes the neutral vote it replaced; it never stops training."""
    try:
        return judge.fallback_reason(row)
    except RuntimeError:
        return "nonrecoverable"


def read_pair(directory, protocol, manifest, order, *, allow_fallback, timed_out=False, final=False):
    """Read one order of one pair.

    With a secondary judge, a recoverable primary failure is left open until the pair is
    routed; a routed pair takes the secondary judge's verdicts in both orders. A final
    read (scoring) imputes the neutral vote for anything still unresolved.
    """
    manifest_path = directory / "judge" / f"manifest-{order}.json"
    freeze_json(manifest_path, manifest)
    path = directory / "judge" / f"judgments-{order}.jsonl"
    completed, _ = judge.read_judgments(path, manifest, manifest_path, protocol["judge"])
    fallback_path = directory / "judge" / f"fallbacks-{order}.json"
    fallbacks = judge.read_fallbacks(fallback_path, manifest, manifest_path, protocol["judge"])
    errors = judge.read_failure_rows(path, manifest, manifest_path, protocol["judge"])
    secondary = judge.secondary_settings(protocol) if allow_fallback else None
    routed = secondary is not None and secondary_routed(directory)
    if routed:
        # The primary verdicts of a routed pair are kept in their ledger but not used.
        path = directory / "judge" / f"judgments-{order}-{SECONDARY_LEDGER}.jsonl"
        completed, _ = judge.read_judgments(path, manifest, manifest_path, secondary)
        completed = {key: {**row, "judge_role": "secondary"} for key, row in completed.items()}
        errors = judge.read_failure_rows(path, manifest, manifest_path, secondary)
    if set(completed) & set(fallbacks) or (fallbacks and not allow_fallback):
        raise ValueError("Conflicting or disallowed streamed fallbacks")
    for key in set(manifest["pairs"]) - set(completed) - set(fallbacks):
        if key not in errors and not timed_out and not (final and routed):
            continue
        if routed:
            missing = "pass_timeout" if timed_out else "missing"
            reason = "secondary_" + (secondary_reason(errors[key]) if key in errors else missing)
        else:
            # Classify even during a timeout: auth/configuration errors stay fatal.
            reason = judge.fallback_reason(errors[key]) if key in errors else "pass_timeout"
            if secondary is not None and key in errors and not (timed_out or final):
                continue
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
    repair_path = directory / "judge/stream/repair.json"
    repair_exhausted = repair_path.exists() and read_json(repair_path).get("exhausted", False)
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
                timed_out=outcome["timed_out"] or repair_exhausted,
                final=True,
            )
            if key not in result:
                raise RuntimeError(f"Streaming runner left {order}/{key} without a result or recorded failure")
            results[order].update(result)
            index[order][key] = str(pair_dir / "judge" / f"manifest-{order}.json")
    freeze_json(directory / "judge/stream-index.json", index)
    return results


def kill_if_running(process):
    """Kill a child that is still running; return False if it had already exited.

    A blocked event loop can notice a child's exit only after a wait deadline has
    fired. uvloop then raises ProcessLookupError from kill(), so check first and
    treat that error as the same already-exited outcome.
    """
    if process.returncode is not None:
        return False
    try:
        process.kill()
    except ProcessLookupError:
        return False
    return True


class StreamingJudge:
    """Async parent: publish arrivals, bound the final pass, and reap on cancellation."""

    def __init__(self, directory, protocol, sample, *, validate):
        protocol = judging_protocol(protocol, validate=validate)
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
        if self.protocol.get("judge_background_cleanup"):
            from .cleanup import ensure_running

            ensure_running(self.protocol)
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
                "--parent-pid",
                str(os.getpid()),
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
                # A child that exited on its own is judged by its exit status, not as a timeout.
                timed_out = kill_if_running(self.process)
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
        # The killed child cannot execute its finally blocks. Reap only its
        # recorded uploads after it exits; never list or delete another group's files.
        journal = UploadJournal(self.directory / "judge/gemini-files.jsonl")
        if self.protocol.get("judge_background_cleanup"):
            from .cleanup import enqueue, ensure_running

            if journal.pending():
                enqueue(self.directory, self.protocol)
                ensure_running(self.protocol)
            if self.log is not None:
                self.log.close()
            return
        if journal.pending():
            cleanup = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "examples.video_agents.gemini_files",
                str(self.directory),
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            try:
                await asyncio.wait_for(cleanup.wait(), timeout=60)
            except TimeoutError:
                kill_if_running(cleanup)
                await cleanup.wait()
            except BaseException:
                if cleanup.returncode is None:
                    cleanup.kill()
                    await cleanup.wait()
                raise
            record_event(
                self.directory / "judge/timings.jsonl",
                "gemini_upload_cleanup",
                "completed" if cleanup.returncode == 0 else "unconfirmed",
                pending=len(journal.pending()),
            )
        if self.log is not None:
            self.log.close()


def run_stream(directory, runner, secondary_runner=None):
    """Reuse the frozen runner, its retry budget, and a shared upload cache.

    In training with a secondary judge, a pair whose primary comparison failed in either
    order is re-judged by the secondary judge in both orders, once both primary attempts
    have settled; a secondary failure leaves that order's neutral vote.
    """
    root = directory / "judge/stream"
    context = read_json(root / "context.json")
    protocol, sample, validate = context["protocol"], context["sample"], context["validate"]
    protocol = judging_protocol(protocol, validate=validate)
    check_dependencies(protocol)
    if isinstance(sample.get("image"), dict):
        image_bytes(sample["image"])
    if validate:
        verify_references([sample])
        if protocol.get("eval_repair", {}).get("enabled", False):
            from .eval_recovery import run_evaluation

            with record_fatal(root):
                return run_evaluation(directory, runner, protocol, sample)
    count = 1 if validate else protocol["group_size"]
    candidates = [{"status": "pending"} for _ in range(count)]
    allow_fallback = not validate and protocol.get("judge_failure_policy") == "neutral"
    secondary = judge.secondary_settings(protocol) if allow_fallback else None
    if secondary is not None and secondary_runner is None:
        raise RuntimeError("The secondary judge requires its own evaluator instance")
    api_key = read_api_key(runner, protocol["judge"]["judge_model"], Path(protocol["eval_repo"]) / "arena/.env")
    client = runner.GeminiRestClient(
        api_key,
        max_attempts=protocol["judge_max_attempts"],
        request_timeout=protocol["timeouts"]["judge_request"],
        file_timeout=protocol["timeouts"]["judge_file"],
    )
    uploads, submitted, settling = {}, {}, []
    settled, settled_lock = defaultdict(set), threading.Lock()
    secondary_slots = threading.BoundedSemaphore(protocol["judge_secondary"]["concurrency"]) if secondary else None

    def upload(path):
        try:
            client.upload_video(Path(path))
        except (runner.EvalError, OSError):
            # The shared cache retains the exception. The runner records it in
            # each affected pair's ledger with its usual error classification.
            pass

    def ledger_state(pair_dir, manifest, order, settings, ledger=""):
        """(completed row, error row) of one order's ledger for its only pair."""
        key = next(iter(manifest["pairs"]))
        manifest_path = pair_dir / "judge" / f"manifest-{order}.json"
        path = pair_dir / "judge" / f"judgments-{order}{'-' + ledger if ledger else ''}.jsonl"
        completed, _ = judge.read_judgments(path, manifest, manifest_path, settings)
        errors = judge.read_failure_rows(path, manifest, manifest_path, settings)
        return completed.get(key), errors.get(key)

    def compare(pair_dir, manifest, order):
        key = next(iter(manifest["pairs"]))
        _, forwarded = parse_options(judge.command(pair_dir, protocol, order, [key], allow_fallback=allow_fallback)[3:])
        with timed_stage(directory / "judge/timings.jsonl", "gemini_stream_pair", pair_id=key, order=order):
            code = runner.main(forwarded)
            result = read_pair(pair_dir, protocol, manifest, order, allow_fallback=allow_fallback)
            # A failure kept open for the secondary judge is still a recorded result.
            if key not in result and not (secondary and ledger_state(pair_dir, manifest, order, protocol["judge"])[1]):
                raise RuntimeError(f"Streaming runner exited {code} without recording {order}/{key}")

    def judge_secondary(pair_dir, manifest, order):
        key = next(iter(manifest["pairs"]))
        if any(ledger_state(pair_dir, manifest, order, secondary, SECONDARY_LEDGER)):
            return
        command = judge.command(
            pair_dir,
            protocol,
            order,
            [key],
            allow_fallback=True,
            settings=secondary,
            ledger=SECONDARY_LEDGER,
            max_attempts=protocol["judge_secondary"]["attempts"],
        )
        _, forwarded = parse_options(command[3:])
        timings = directory / "judge/timings.jsonl"
        with (
            secondary_slots,
            timed_stage(timings, "gemini_secondary_pair", pair_id=key, order=order, model=secondary["judge_model"]),
        ):
            try:
                secondary_runner.main(forwarded)
            except Exception as exc:
                # The secondary judge may only replace a neutral vote, never stop training.
                manifest_path = pair_dir / "judge" / f"manifest-{order}.json"
                fallback_path = pair_dir / "judge" / f"fallbacks-{order}.json"
                rows = judge.read_fallbacks(fallback_path, manifest, manifest_path, protocol["judge"])
                error = {"error_type": type(exc).__name__, "error": str(exc)[:500]}
                rows[key] = judge.neutral_fallback(
                    key, manifest, manifest_path, protocol["judge"], "secondary_exception", error
                )
                write_json(fallback_path, [rows[name] for name in sorted(rows)])
                record_event(timings, "gemini_secondary", "exception", pair_id=key, order=order, **error)
        if key not in read_pair(pair_dir, protocol, manifest, order, allow_fallback=True):
            raise RuntimeError(f"Secondary judge did not record {order}/{key}")

    def settle(pair_dir, manifests, key):
        """Route the pair once both primary orders have a result; nonrecoverable errors stay fatal."""
        if not secondary_routed(pair_dir):
            failures = {}
            for order, manifest in manifests.items():
                manifest_path = pair_dir / "judge" / f"manifest-{order}.json"
                fallback_path = pair_dir / "judge" / f"fallbacks-{order}.json"
                if key in judge.read_fallbacks(fallback_path, manifest, manifest_path, protocol["judge"]):
                    return  # A neutral vote recorded before this judge configuration stays.
                completed, error = ledger_state(pair_dir, manifest, order, protocol["judge"])
                if completed is None and error is not None:
                    failures[order] = judge.fallback_reason(error)
            if not failures:
                return
            write_json(pair_dir / "judge" / SECONDARY_MARKER, {"model": secondary["judge_model"], "failures": failures})
            record_event(
                directory / "judge/timings.jsonl",
                "gemini_secondary",
                "routed",
                pair_id=key,
                model=secondary["judge_model"],
                failures=failures,
            )
        with ThreadPoolExecutor(max_workers=len(manifests)) as orders:
            futures = [
                orders.submit(judge_secondary, pair_dir, manifest, order) for order, manifest in manifests.items()
            ]
            for future in futures:
                future.result()

    def task(pair_dir, manifests, order):
        key = next(iter(manifests[order]["pairs"]))
        if not (secondary and secondary_routed(pair_dir)):
            compare(pair_dir, manifests[order], order)
        if secondary is None:
            return
        with settled_lock:
            settled[key].add(order)
            last = settled[key] == set(manifests)
        if last:
            settle(pair_dir, manifests, key)

    # Uploads have their own pool, so comparisons cannot starve an upload they need.
    with (
        ThreadPoolExecutor(max_workers=count + 3) as upload_pool,
        # Shared admission bounds requests; also bound threads waiting for it.
        ThreadPoolExecutor(
            max_workers=min(protocol.get("judge_concurrency", 1), 32)
            if protocol.get("judge_limits")
            else protocol.get("judge_concurrency", 1)
        ) as pair_pool,
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
                    singles = {name: pair_manifest(value, key) for name, value in manifests.items()}
                    single, pair_dir = singles[order], pair_directory(directory, key)
                    completed = read_pair(pair_dir, protocol, single, order, allow_fallback=allow_fallback)
                    if key in completed:
                        submitted[order, key] = None
                        with settled_lock:
                            settled[key].add(order)
                            last = settled[key] == set(singles)
                        if last and secondary is not None:
                            # The other order's task finished first; settle the pair here.
                            settling.append(pair_pool.submit(settle, pair_dir, singles, key))
                    else:
                        submitted[order, key] = pair_pool.submit(task, pair_dir, singles, order)
            for future in [*submitted.values(), *settling]:
                if future is not None and future.done():
                    future.result()
            finished = (root / "finished.json").exists() and all(c["status"] != "pending" for c in candidates)
            if finished and len(submitted) == 2 * len(manifests["original"]["pairs"]):
                break
            time.sleep(0.1)
        for future in [*submitted.values(), *settling]:
            if future is not None:
                future.result()
        for future in uploads.values():
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args()
    if args.parent_pid is not None:
        guard_parent(args.parent_pid)
    root = args.directory / "judge/stream"
    context = read_json(root / "context.json")
    protocol = judging_protocol(context["protocol"], validate=context["validate"])
    runner_path = Path(protocol["eval_repo"]) / "scripts/run_gemini_pairwise_eval.py"
    runner = load_external(runner_path)
    # The primary instance is routed to Vertex; the secondary judge uses the Gemini API
    # through its own instance, without the run's Vertex comparison slots.
    use_secondary = (
        not context["validate"]
        and protocol.get("judge_failure_policy") == "neutral"
        and judge.secondary_settings(protocol) is not None
    )
    secondary_runner = load_external(runner_path, namespace="_video_agents_secondary_") if use_secondary else None
    from .cleanup import readers
    from .provider_limits import provider_limits
    from .upload_admission import UploadAdmission

    runner._verl_provider_limits = provider_limits(protocol)
    if protocol.get("provider_recovery_root"):
        runner._verl_upload_admission = UploadAdmission(
            Path(protocol["provider_recovery_root"]) / "gemini-upload-quota",
            # Training streams can start before a long policy response. This
            # gate never waits while blocked; the parent owns the pass deadline.
            float("inf"),
        )
    with vertex_runner(runner, protocol["judge"]["judge_model"]):
        instrument_client(runner, args.directory / "judge/timings-stream.jsonl", protocol["timeouts"]["judge_control"])
        if secondary_runner is not None:
            # Same upload journal as the group's, so a killed pass leaves its files for cleanup.
            instrument_client(
                secondary_runner,
                args.directory / "judge/timings-secondary.jsonl",
                protocol["timeouts"]["judge_control"],
            )
        deferred = protocol.get("judge_background_cleanup", False)
        with (
            readers(args.directory, protocol) if deferred else nullcontext(),
            record_fatal(root),
            gemini_address_family("ipv4"),
            reference_images(runner),
            parallel_pairs(runner, keep_uploaded_files=deferred),
            reference_images(secondary_runner) if secondary_runner is not None else nullcontext(),
            # Delete the secondary judge's uploads as soon as its pass ends.
            parallel_pairs(secondary_runner, keep_uploaded_files=False)
            if secondary_runner is not None
            else nullcontext(),
        ):
            run_stream(args.directory, runner, secondary_runner)


if __name__ == "__main__":
    main()
