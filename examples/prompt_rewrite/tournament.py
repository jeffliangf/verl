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
from pathlib import Path

from . import h3, judge, scoring
from .config import CANDIDATE, REFERENCE_FIELDS
from .dataset import verify_references
from .storage import file_sha256, fingerprint, freeze_json, read_json, run_lock, write_json

GROUP_SIZE = 4
INVALID_REWARD = -1.0


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
    worker_protocol["rl_protocol_sha256"] = fingerprint(protocol)
    with run_lock(directory):
        freeze_json(directory / "protocol.json", worker_protocol)
        freeze_json(directory / "generation.json", [task])
        timeout = protocol["timeouts"]["h3"]
        # This worker already persists the task ID before polling and refuses to
        # resubmit an uncertain POST. subprocess.run kills and waits on timeout.
        subprocess.run(
            h3.worker_command(directory, 1, protocol["poll_interval"], timeout),
            check=True,
            timeout=timeout,
        )
        state, path = read_json(Path(task["state"])), Path(task["output"])
        if (
            state.get("status") != "succeeded"
            or any(state.get(key) != value for key, value in h3.task_identity(task, protocol).items())
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
        raise RuntimeError("Tournament requires every judgment; missing/fallback results cannot train the policy")


def run_judgments(directory: Path, protocol: dict, manifests: dict) -> dict:
    """Reuse the judge CLI and ledgers with strict, bounded failure handling."""
    judge.freeze_manifests(directory, manifests)
    results = {}
    for order in judge.ORDERS:
        manifest = manifests[order]
        path = directory / "judge" / f"judgments-{order}.jsonl"
        manifest_path = directory / "judge" / f"manifest-{order}.json"
        completed, _ = judge.read_judgments(path, manifest, manifest_path, protocol["judge"])
        if any(row.get("status") != "completed" for row in completed.values()):
            raise RuntimeError("Evaluation fallback ledgers cannot be used as RL rewards")
        pending = set(manifest["pairs"]) - set(completed)
        if pending:
            command = judge.command(directory, protocol, order, True, 1)
            request_timeout = str(protocol["timeouts"]["judge_request"])
            command.extend(
                [
                    "--request-timeout",
                    request_timeout,
                    "--file-timeout",
                    request_timeout,
                    "--max-attempts",
                    "1",
                    "--max-errors",
                    str(len(pending) + 1),
                ]
            )
            for pair_id in sorted(pending):
                command.extend(["--pair-id", pair_id])
            subprocess.run(command, check=True, timeout=protocol["timeouts"]["judge_pass"])
            completed, _ = judge.read_judgments(path, manifest, manifest_path, protocol["judge"])
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
            judgments = run_judgments(directory, protocol, manifests)
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
            manifests = tournament_manifests(sample, candidates, protocol)
            rewards = tournament_rewards(candidates, manifests, run_judgments(directory, protocol, manifests))
        write_json(directory / "rewards.json", rewards)
        return rewards
