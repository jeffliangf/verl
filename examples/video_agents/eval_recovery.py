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
"""Repair individual held-out judgments without resampling policy outputs or votes."""

import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import judge
from .dataset import verify_references
from .observability import record_event
from .storage import file_sha256, freeze_json, read_json, run_lock, write_json
from .upload_admission import UploadAdmission, storage_quota


def run_evaluation(directory, runner, protocol, sample):
    """All retries for a pair share one ledger, attempt budget and hard deadline.

    The enclosing streaming subprocess owns shared uploads until every reader
    settles. Holding this lock also prevents a second repair process from judging
    the same pair concurrently. The native validation callback runs after return.
    """
    from .judge_worker import parse_options
    from .streaming_judge import pair_directory, pair_manifest, ready_manifests

    root = directory / "judge/stream"
    candidate_path = root / "candidate-0.json"
    while not candidate_path.exists():
        time.sleep(0.1)  # Parent bounds H3 and kills this child on cancellation.
    candidate = read_json(candidate_path)
    if candidate["status"] not in {"succeeded", "invalid_call"}:
        raise ValueError("Evaluation repair requires a completed video or invalid policy call")
    if candidate["status"] == "succeeded" and file_sha256(Path(candidate["path"])) != candidate["sha256"]:
        raise ValueError("Candidate video changed before evaluation repair")
    manifests = ready_manifests(protocol, sample, [candidate], True)
    settings = protocol["judge"]
    state_path = root / "repair.json"
    with run_lock(root / "repair-lock"):
        freeze_json(root / "repair-identity.json", {"manifests": manifests, "judge": settings, "candidate": candidate})
        state = (
            read_json(state_path)
            if state_path.exists()
            else {
                "deadline_at": time.time() + protocol["timeouts"]["judge_pass"],
                "pairs": {},
            }
        )
        deadline = state["deadline_at"]
        maximum = protocol["judge_max_attempts"]
        admission = UploadAdmission(
            Path(protocol.get("provider_recovery_root", root)) / "gemini-upload-quota", deadline
        )
        runner._verl_upload_admission = admission
        # The outer scheduler owns retries. A runner invocation gets ONE attempt,
        # avoiding three HTTP attempts multiplied by three repair passes.
        once = {**protocol, "judge_max_attempts": 1}
        env = runner.read_env_file(Path(protocol["eval_repo"]) / "arena/.env")
        client = runner.GeminiRestClient(
            os.environ.get("GEMINI_API_KEY") or env.get("GEMINI_API_KEY", ""),
            max_attempts=1,
            request_timeout=protocol["timeouts"]["judge_request"],
            file_timeout=protocol["timeouts"]["judge_file"],
        )
        jobs = {}
        for order, manifest in manifests.items():
            for key in manifest["pairs"]:
                single, pair_dir = pair_manifest(manifest, key), pair_directory(directory, key)
                manifest_path = pair_dir / "judge" / f"manifest-{order}.json"
                freeze_json(manifest_path, single)
                jobs[f"{order}/{key}"] = (order, key, single, pair_dir, manifest_path)

        def inspect(identifier):
            order, key, single, pair_dir, manifest_path = jobs[identifier]
            path = pair_dir / "judge" / f"judgments-{order}.jsonl"
            completed, _ = judge.read_judgments(path, single, manifest_path, settings)
            # Validate error identities as well; never reuse a ledger from other weights/inputs.
            errors = judge.read_failure_rows(path, single, manifest_path, settings)
            if key in completed:
                return "completed", False
            if key not in errors:
                return "pending", False
            reason = judge.fallback_reason(errors[key])  # Auth/configuration/code errors remain fatal.
            if "Gemini uploads paused for storage recovery" in errors[key]["error"]:
                return "deferred", False
            return ("retryable" if reason == "provider_failure" else reason), storage_quota(errors[key]["error"])

        def compare(identifier):
            order, key, _, pair_dir, _ = jobs[identifier]
            _, forwarded = parse_options(judge.command(pair_dir, once, order, [key])[3:])
            runner.main(forwarded)
            status, quota = inspect(identifier)
            if status == "pending":
                raise RuntimeError("Evaluator returned without a completed vote or failure row")
            return status, quota

        try:
            while True:
                pending = []
                for identifier in jobs:
                    status, _ = inspect(identifier)
                    entry = state["pairs"].setdefault(identifier, {"attempts": 0})
                    entry["status"] = status
                    if status in {"pending", "retryable", "deferred"} and entry["attempts"] < maximum:
                        pending.append(identifier)
                write_json(state_path, state)
                if not pending or time.time() >= deadline:
                    break
                verify_references([sample])
                if candidate["status"] == "succeeded" and file_sha256(Path(candidate["path"])) != candidate["sha256"]:
                    raise ValueError("Candidate video changed during evaluation repair")
                if admission.blocked():
                    # No comparison readers remain here. Release only this
                    # process's files and perform one real capacity probe BEFORE
                    # spending another comparison attempt or starting a wave.
                    reset = runner._verl_reset_uploads
                    reset(failed_only=False)
                    _, key, single, _, _ = jobs[pending[0]]
                    admission.wait_for_probe = True
                    try:
                        client.upload_video(Path(single["pairs"][key]["video_path_A"]))
                    except (runner.EvalError, OSError) as exc:
                        judge.fallback_reason({"error_type": type(exc).__name__, "error": str(exc), "stage": "upload"})
                        continue
                    finally:
                        admission.wait_for_probe = False
                quota = False
                # Failures settle independently: a timeout cannot cancel the other
                # reference or presentation-order comparisons in this prompt.
                with ThreadPoolExecutor(max_workers=min(protocol.get("judge_concurrency", 1), len(pending))) as pool:
                    futures = {}
                    for identifier in pending:
                        if time.time() >= deadline:
                            break
                        state["pairs"][identifier]["attempts"] += 1
                        write_json(state_path, state)  # Charge before sending; crashes cannot reset attempts.
                        futures[pool.submit(compare, identifier)] = identifier
                    for future in as_completed(futures):
                        identifier = futures[future]
                        status, hit_quota = future.result()
                        quota |= hit_quota
                        state["pairs"][identifier]["status"] = status
                        if status == "deferred":
                            # Admission refused BEFORE any interaction was sent.
                            state["pairs"][identifier]["attempts"] -= 1
                        write_json(state_path, state)
                        record_event(
                            directory / "judge/timings.jsonl",
                            "gemini_eval_repair",
                            status,
                            comparison=identifier,
                            attempts=state["pairs"][identifier]["attempts"],
                        )
                retry = any(
                    row["status"] in {"retryable", "deferred"} and row["attempts"] < maximum
                    for row in state["pairs"].values()
                )
                if not retry or time.time() >= deadline:
                    break
                # Every reader in this process has settled. Keep successful
                # uploads for timeouts; on quota, release this prompt's own files
                # before probing capacity. Never delete another process's files.
                reset = getattr(runner, "_verl_reset_uploads", None)
                if reset is not None:
                    reset(failed_only=not (quota or admission.blocked()))
                delay = 5 + random.uniform(0, 2)
                time.sleep(max(0, min(delay, deadline - time.time())))
            state["complete"] = all(row["status"] == "completed" for row in state["pairs"].values())
            state["exhausted"] = any(
                row["status"] in {"pending", "retryable", "deferred"} for row in state["pairs"].values()
            )
            write_json(state_path, state)
        finally:
            del runner._verl_upload_admission
