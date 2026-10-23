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
"""Video tournaments using the existing H3 worker, Gemini runner, and pair scorer."""

import itertools
import subprocess
from math import isfinite
from pathlib import Path

from . import h3, judge, scoring, self_hosted_h3
from .config import CANDIDATE, REFERENCE_FIELDS
from .dataset import verify_references
from .observability import ProviderError, record_event, timed_stage
from .storage import file_sha256, fingerprint, freeze_json, read_json, run_lock, write_json

GROUP_SIZE = 4
INVALID_REWARD = -1.0
TIMEOUT_STAGES = ("llm", "h3", "h3_request", "judge_request", "judge_file", "judge_control", "judge_pass")


class GeminiPolicyRefusal(RuntimeError):
    """An identity-verified content-policy rejection for this prompt group."""

    def __init__(self, order: str, pair_ids: list[str]):
        super().__init__(f"Gemini prohibited_content rejection in {order}: {pair_ids}")
        self.order = order
        self.pair_ids = pair_ids


def check_dependencies(protocol: dict):
    """Refuse to mix evaluator versions inside a training run."""
    for relative, digest in protocol["dependencies"].items():
        if file_sha256(Path(protocol["eval_repo"]) / relative) != digest:
            raise ValueError(f"Evaluator changed since preparation: {relative}")


def load_protocol(path: Path) -> dict:
    """Load a frozen experiment without importing models or reading credentials."""
    protocol = read_json(path)
    if read_json(path.with_name("protocol-sha256.json")) != fingerprint(protocol):
        raise ValueError("RL protocol changed; prepare a new run directory")
    if protocol.get("task") != "h3_tournament" or protocol.get("group_size") != GROUP_SIZE:
        raise ValueError("Expected a four-candidate H3 tournament protocol")
    check_dependencies(protocol)
    return protocol


def runtime_protocol(protocol: dict, settings) -> dict:
    """Overlay launch-time provider limits from video_rl; the frozen identity is kept."""
    timeouts = dict(settings.get("timeouts") or {})
    if unknown := set(timeouts) - set(TIMEOUT_STAGES):
        raise ValueError(f"Unknown video_rl.timeouts stages: {sorted(unknown)}")
    seconds = timeouts.values()
    if any(isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v) or v <= 0 for v in seconds):
        raise ValueError("video_rl.timeouts must be positive, finite seconds")
    attempts = settings.get("judge_max_attempts", 1)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("video_rl.judge_max_attempts must be a positive integer")
    return {
        **protocol,
        "timeouts": {**protocol["timeouts"], **timeouts},
        "judge_max_attempts": attempts,
        "protocol_sha256": protocol.get("protocol_sha256", fingerprint(protocol)),
    }


def generate_video(directory: Path, protocol: dict, request: dict) -> dict:
    """Execute the policy's complete Sol-H3 request using the existing durable worker."""
    task = {
        "id": "candidate",
        "request": request,
        "state": str(directory / "state.json"),
        "output": str(directory / "video.mp4"),
    }
    worker_protocol = {key: protocol[key] for key in ("h3", "h3_host", "media_profile", "eval_repo")}
    for key in ("h3_backend", "h3_endpoints", "h3_token_file", "reference_h3"):
        if key in protocol:
            worker_protocol[key] = protocol[key]
    worker_protocol["rl_protocol_sha256"] = protocol.get("protocol_sha256") or fingerprint(protocol)
    if "h3_request" in protocol["timeouts"]:
        worker_protocol["timeouts"] = {"h3_request": protocol["timeouts"]["h3_request"]}
    with run_lock(directory):
        freeze_json(directory / "protocol.json", worker_protocol)
        freeze_json(directory / "generation.json", [task])
        timeout = protocol["timeouts"]["h3"]
        # This worker already persists the task ID before polling and refuses to
        # resubmit an uncertain POST. subprocess.run kills and waits on timeout.
        try:
            with timed_stage(directory / "timings.jsonl", "h3_worker", timeout_seconds=timeout):
                subprocess.run(
                    h3.worker_command(directory, 1, protocol["poll_interval"], timeout),
                    check=True,
                    timeout=timeout,
                )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError("h3", f"H3 exceeded {timeout}s; inspect the saved job in {directory}") from exc
        except subprocess.CalledProcessError as exc:
            state = read_json(Path(task["state"])) if Path(task["state"]).exists() else {}
            if state.get("status") in h3.TERMINAL_FAILURES or (
                state.get("status") == "error" and state.get("error_stage") == "provider"
            ):
                raise ProviderError("h3", f"H3 service failed: {state.get('error')}; inspect {directory}") from exc
            raise  # Artifact, configuration, and worker-code errors must remain fatal.
        state, path = read_json(Path(task["state"])), Path(task["output"])
        if (
            state.get("status") != "succeeded"
            or any(state.get(key) != value for key, value in self_hosted_h3.identity(task, protocol).items())
            or state.get("sha256") != file_sha256(path)
            or state.get("bytes") != path.stat().st_size
            or h3.media_profile(state.get("media", {}), duration=request["duration"]) != protocol["media_profile"]
        ):
            raise RuntimeError(f"Incomplete or mismatched H3 result: {directory}")
        return {
            "status": "succeeded",
            "prompt": request["prompt"],
            "request": request,
            "path": str(path),
            "sha256": state["sha256"],
            "task_id": state["task_id"],
            "state": task["state"],
        }


def tournament_manifests(sample: dict, candidates: list[dict], protocol: dict) -> dict:
    """Build six unordered matchups and their reverse orientations."""
    if len(candidates) != GROUP_SIZE:
        raise ValueError("Training requires exactly four candidate trajectories")
    pairs = {}
    for i, j in itertools.combinations(range(GROUP_SIZE), 2):
        if candidates[i]["status"] != "succeeded" or candidates[j]["status"] != "succeeded":
            continue
        pairs[f"{i}_vs_{j}"] = {
            "prompt_id": sample["id"],
            "prompt": sample["prompt"],
            "source": sample["source"],
            "reference": f"candidate_{j}",
            "model_A": f"candidate_{i}",
            "model_B": f"candidate_{j}",
            "video_path_A": candidates[i]["path"],
            "video_path_B": candidates[j]["path"],
        }
    base = {
        "schema_version": 1,
        "instructions_version": protocol["judge"]["instructions_version"],
        "instructions_how_to_compare": protocol["rubric"],
        "metadata": {"pair_count": len(pairs), "prompt_count": 1, "excluded_models": []},
    }
    return {
        "original": {**base, "pairs": pairs},
        "swapped": {**base, "pairs": {key: judge.swap_pair(pair) for key, pair in pairs.items()}},
    }


def require_judgments(manifest: dict, rows: dict):
    """Missing comparisons are infrastructure failures, never ties or losses."""
    if set(rows) != set(manifest["pairs"]) or any(row.get("status") != "completed" for row in rows.values()):
        raise ProviderError("gemini", "Tournament requires every judgment; missing results cannot train the policy")


def run_judgments(directory: Path, protocol: dict, manifests: dict) -> dict:
    """Reuse the judge CLI and ledgers with strict, bounded failure handling."""
    judge.freeze_manifests(directory, manifests)

    def read_pass(order):
        manifest = manifests[order]
        path = directory / "judge" / f"judgments-{order}.jsonl"
        manifest_path = directory / "judge" / f"manifest-{order}.json"
        completed, counts = judge.read_judgments(path, manifest, manifest_path, protocol["judge"])
        refusals = judge.read_policy_refusals(path, manifest, manifest_path, protocol["judge"], completed)
        if refusals:
            raise GeminiPolicyRefusal(order, sorted({row["pair_id"] for row in refusals}))
        return completed, counts

    results = {}
    for order in judge.ORDERS:
        manifest = manifests[order]
        path = directory / "judge" / f"judgments-{order}.jsonl"
        completed, _ = read_pass(order)
        pending = set(manifest["pairs"]) - set(completed)
        if pending:
            command = judge.command(directory, protocol, order)
            request_timeout = str(protocol["timeouts"]["judge_request"])
            command.extend(
                [
                    "--request-timeout",
                    request_timeout,
                    "--file-timeout",
                    str(protocol["timeouts"].get("judge_file", protocol["timeouts"]["judge_request"])),
                    "--max-attempts",
                    str(protocol.get("judge_max_attempts", 1)),
                    "--max-errors",
                    "1",
                ]
            )
            if "judge_control" in protocol["timeouts"]:
                command.extend(["--file-control-timeout", str(protocol["timeouts"]["judge_control"])])
            for pair_id in sorted(pending):
                command.extend(["--pair-id", pair_id])
            timeout = protocol["timeouts"]["judge_pass"]
            try:
                with timed_stage(
                    directory / "judge/timings.jsonl", "gemini_pass", order=order, timeout_seconds=timeout
                ):
                    subprocess.run(command, check=True, timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                # The runner writes judgments before deleting its remote files.
                # subprocess.run has killed and reaped the timed-out process;
                # complete, identity-validated results remain usable even if
                # cleanup did not finish. Missing or invalid results still fail.
                completed, _ = read_pass(order)
                if set(completed) != set(manifest["pairs"]):
                    raise ProviderError("gemini", f"Gemini {order} pass exceeded {timeout}s") from exc
                record_event(
                    directory / "judge/timings.jsonl",
                    "gemini_pass",
                    "timeout_with_complete_ledger",
                    order=order,
                    completed_pairs=len(completed),
                    cleanup_status="unconfirmed",
                )
            except subprocess.CalledProcessError as exc:
                _, counts = read_pass(order)
                if counts.get("error", 0):
                    raise ProviderError("gemini", f"Gemini failed comparisons; inspect {path}") from exc
                raise
            completed, _ = read_pass(order)
        require_judgments(manifest, completed)
        results[order] = completed
    return results


def tournament_rewards(candidates: list[dict], manifests: dict, judgments: dict) -> list[dict]:
    """Average pair wins, with explicit forfeits for invalid policy calls."""
    if len(candidates) != GROUP_SIZE:
        raise ValueError("Training requires exactly four candidate trajectories")
    if any(c["status"] not in {"succeeded", "invalid_call"} for c in candidates):
        raise RuntimeError("Provider failures must stop training, not receive a policy penalty")
    valid = [c["status"] == "succeeded" for c in candidates]
    invalid_count = GROUP_SIZE - sum(valid)
    points = [float(invalid_count) if ok else 0.0 for ok in valid]
    split_count = 0
    for order in judge.ORDERS:
        require_judgments(manifests[order], judgments[order])
    expected_pairs = {f"{i}_vs_{j}" for i, j in itertools.combinations(range(GROUP_SIZE), 2) if valid[i] and valid[j]}
    if set(manifests["original"]["pairs"]) != expected_pairs:
        raise ValueError("Incomplete tournament manifest")
    for key, pair in manifests["original"]["pairs"].items():
        i, j = (int(value) for value in key.split("_vs_"))
        result = scoring.score_pair(
            pair, judgments["original"][key], judgments["swapped"][key], candidate=f"candidate_{i}"
        )
        points[i] += result["score"]
        points[j] += 1 - result["score"]
        split_count += result["outcome"] == "split"
    rewards = [point / (GROUP_SIZE - 1) if ok else INVALID_REWARD for point, ok in zip(points, valid, strict=True)]
    return [
        {
            "score": reward,
            "valid_call": float(ok),
            "zero_variance_group": float(len(set(rewards)) == 1),
            "order_disagreement": split_count / len(expected_pairs) if expected_pairs else 0.0,
            "judge_comparisons": 2 * len(expected_pairs),
        }
        for reward, ok in zip(rewards, valid, strict=True)
    ]


def score_group(directory: Path, protocol: dict, sample: dict, candidates: list[dict], *, validate: bool) -> list[dict]:
    """Score a training tournament or a held-out candidate against both references."""
    check_dependencies(protocol)
    if any(c["status"] not in {"succeeded", "invalid_call"} for c in candidates):
        raise RuntimeError(f"An H3 job failed; inspect {directory} and restart from a training checkpoint")
    for candidate in candidates:
        if candidate["status"] == "succeeded" and file_sha256(Path(candidate["path"])) != candidate["sha256"]:
            raise ValueError("Candidate video changed before judging")
    with run_lock(directory):
        freeze_json(directory / "inputs.json", {"sample": sample, "candidates": candidates, "validate": validate})
        if validate:
            if len(candidates) != 1:
                raise ValueError("Held-out evaluation requires one candidate per prompt")
            verify_references([sample])
            valid = candidates[0]["status"] == "succeeded"
            eval_protocol = {**protocol, "name": "h3-agent", "samples": [sample]}
            manifests = judge.build_manifests(eval_protocol, {sample["id"]: candidates[0]} if valid else {})
        else:
            manifests = tournament_manifests(sample, candidates, protocol)
        try:
            judgments = run_judgments(directory, protocol, manifests)
        except GeminiPolicyRefusal as exc:
            # Skip the entire example, including any partial comparisons. Equal
            # rewards give a training group zero GRPO advantage. A held-out
            # example is dropped from the evaluation denominator instead. Keep
            # the real error ledger untouched.
            policy = "dropped_example" if validate else "neutral_example"
            refusal = {
                "sample_id": sample["id"],
                "reason": "prohibited_content",
                "policy": policy,
                **({} if validate else {"score": 0.5}),
                "order": exc.order,
                "pair_ids": exc.pair_ids,
                "ledger": f"judge/judgments-{exc.order}.jsonl",
                "validate": validate,
            }
            write_json(directory / "gemini-refusal.json", refusal)
            record_event(directory / "judge/timings.jsonl", "gemini_refusal", policy, **refusal)
            if validate:
                raise ProviderError("gemini_refusal", f"Gemini refused held-out example {sample['id']}") from exc
            rewards = [
                {
                    "score": 0.5,
                    "valid_call": float(candidate["status"] == "succeeded"),
                    "judge_comparisons": 0,
                    "gemini_refusal": 1.0,
                    "zero_variance_group": 1.0,
                    "order_disagreement": 0.0,
                }
                for candidate in candidates
            ]
        else:
            if validate:
                result = {"valid_call": float(valid), "judge_comparisons": 4 if valid else 0}
                for reference in REFERENCE_FIELDS:
                    matches = [
                        key for key, pair in manifests["original"]["pairs"].items() if pair["reference"] == reference
                    ]
                    result[f"vs_{reference}"] = (
                        scoring.score_pair(
                            manifests["original"]["pairs"][matches[0]],
                            judgments["original"][matches[0]],
                            judgments["swapped"][matches[0]],
                            candidate=CANDIDATE,
                        )["score"]
                        if valid
                        else 0.0
                    )
                result["score"] = sum(result[f"vs_{ref}"] for ref in REFERENCE_FIELDS) / len(REFERENCE_FIELDS)
                rewards = [result]
            else:
                rewards = tournament_rewards(candidates, manifests, judgments)
        for reward in rewards:
            reward.setdefault("gemini_refusal", 0.0)
        write_json(directory / "rewards.json", rewards)
        return rewards
