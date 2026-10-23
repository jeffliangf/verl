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

from . import h3, judge, scoring
from .config import CANDIDATE, REFERENCE_FIELDS
from .dataset import verify_references
from .observability import ProviderError, record_event, timed_stage
from .storage import file_sha256, fingerprint, freeze_json, read_json, run_lock, write_json

GROUP_SIZE = 4
INVALID_REWARD = -1.0
TIMEOUT_STAGES = ("llm", "h3", "h3_request", "judge_request", "judge_file", "judge_control", "judge_pass")


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
    if protocol.get("h3_backend") != h3.BACKEND:
        raise ValueError("Expected a self-hosted Sol-H3 protocol; the H3 Max API is disabled")
    check_dependencies(protocol)
    return protocol


def runtime_protocol(protocol: dict, settings) -> dict:
    """Attach the launch-time provider limits from video_rl; the frozen identity is kept."""
    timeouts = dict(settings.get("timeouts") or {})
    if set(timeouts) != set(TIMEOUT_STAGES):
        raise ValueError(f"video_rl.timeouts must set exactly: {', '.join(TIMEOUT_STAGES)}")
    seconds = timeouts.values()
    if any(isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v) or v <= 0 for v in seconds):
        raise ValueError("video_rl.timeouts must be positive, finite seconds")
    attempts = settings.get("judge_max_attempts")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("video_rl.judge_max_attempts must be a positive integer")
    return {
        **protocol,
        "timeouts": timeouts,
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
    worker_protocol = {
        **{key: protocol[key] for key in ("h3", "h3_endpoints", "media_profile", "eval_repo")},
        **({"h3_token_file": protocol["h3_token_file"]} if "h3_token_file" in protocol else {}),
        "timeouts": {"h3_request": protocol["timeouts"]["h3_request"]},
        "rl_protocol_sha256": protocol["protocol_sha256"],
    }
    log_path = directory / "h3-worker.log"
    with run_lock(directory):
        freeze_json(directory / "protocol.json", worker_protocol)
        freeze_json(directory / "generation.json", [task])
        timeout = protocol["timeouts"]["h3"]
        # This worker already persists the task ID before polling and refuses to
        # resubmit an uncertain POST. subprocess.run kills and waits on timeout.
        try:
            with (
                timed_stage(directory / "timings.jsonl", "h3_worker", timeout_seconds=timeout),
                log_path.open("ab") as log,
            ):
                subprocess.run(
                    h3.worker_command(directory, protocol["poll_interval"], timeout),
                    check=True,
                    timeout=timeout,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.TimeoutExpired as exc:
            raise ProviderError("h3", f"H3 exceeded {timeout}s; see {log_path} and the saved job state") from exc
        except subprocess.CalledProcessError as exc:
            state = read_json(Path(task["state"])) if Path(task["state"]).exists() else {}
            detail = f"{state.get('status')}: {state.get('error')}; see {log_path}"
            if state.get("status") in h3.TERMINAL_FAILURES or (
                state.get("status") == "error" and state.get("error_stage") == "provider"
            ):
                raise ProviderError("h3", f"H3 service failed ({detail})") from exc
            # Artifact, configuration, and worker-code errors must remain fatal.
            raise RuntimeError(f"H3 worker failed ({detail})") from exc
        state, path = read_json(Path(task["state"])), Path(task["output"])
        if (
            state.get("status") != "succeeded"
            or any(state.get(key) != value for key, value in h3.identity(task, protocol).items())
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


def run_judgments(directory: Path, protocol: dict, manifests: dict) -> dict:
    """Reuse the judge CLI and ledgers; any missing comparison is a provider failure."""
    judge.freeze_manifests(directory, manifests)
    results = {}
    for order in judge.ORDERS:
        manifest = manifests[order]
        path = directory / "judge" / f"judgments-{order}.jsonl"
        manifest_path = directory / "judge" / f"manifest-{order}.json"
        log_path = directory / "judge" / f"runner-{order}.log"
        completed, _ = judge.read_judgments(path, manifest, manifest_path, protocol["judge"])
        pending = set(manifest["pairs"]) - set(completed)
        if pending:
            timeout = protocol["timeouts"]["judge_pass"]
            try:
                with (
                    timed_stage(directory / "judge/timings.jsonl", "gemini_pass", order=order, timeout_seconds=timeout),
                    log_path.open("ab") as log,
                ):
                    subprocess.run(
                        judge.command(directory, protocol, order, pending),
                        check=True,
                        timeout=timeout,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
            except subprocess.TimeoutExpired as exc:
                # The runner writes judgments before deleting its remote files.
                # subprocess.run has killed and reaped the timed-out process;
                # complete, identity-validated results remain usable even if
                # cleanup did not finish. Missing or invalid results still fail.
                completed, _ = judge.read_judgments(path, manifest, manifest_path, protocol["judge"])
                if set(completed) != set(manifest["pairs"]):
                    detail = judgment_failure(path, set(manifest["pairs"]) - set(completed), log_path)
                    raise ProviderError("gemini", f"Gemini {order} pass exceeded {timeout}s; {detail}") from exc
                record_event(
                    directory / "judge/timings.jsonl",
                    "gemini_pass",
                    "timeout_with_complete_ledger",
                    order=order,
                    completed_pairs=len(completed),
                    cleanup_status="unconfirmed",
                )
            except subprocess.CalledProcessError as exc:
                if not judge.judgment_errors(path, pending):
                    raise RuntimeError(f"Gemini runner crashed without recording an error; see {log_path}") from exc
                detail = judgment_failure(path, pending, log_path)
                raise ProviderError("gemini", f"Gemini {order} comparisons failed; {detail}") from exc
            completed, _ = judge.read_judgments(path, manifest, manifest_path, protocol["judge"])
        if missing := set(manifest["pairs"]) - set(completed):
            detail = judgment_failure(path, missing, log_path)
            raise ProviderError("gemini", f"Gemini {order} pass left comparisons unjudged; {detail}")
        results[order] = completed
    return results


def judgment_failure(path: Path, pair_ids: set, log_path: Path) -> str:
    """Name the failed comparisons, their recorded errors, and where to look."""
    errors = judge.judgment_errors(path, pair_ids)
    listed = "; ".join(f"{pair_id}: {errors.get(pair_id, 'no result')[:300]}" for pair_id in sorted(pair_ids))
    return f"{listed}; see {path} and {log_path}"


def tournament_rewards(candidates: list[dict], manifests: dict, judgments: dict) -> list[dict]:
    """Average pair wins, with explicit forfeits for invalid policy calls."""
    valid = [c["status"] == "succeeded" for c in candidates]
    invalid_count = GROUP_SIZE - sum(valid)
    points = [float(invalid_count) if ok else 0.0 for ok in valid]
    pairs = manifests["original"]["pairs"]
    split_count = 0
    for key, pair in pairs.items():
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
            "order_disagreement": split_count / len(pairs) if pairs else 0.0,
            "judge_comparisons": 2 * len(pairs),
        }
        for reward, ok in zip(rewards, valid, strict=True)
    ]


def score_group(directory: Path, protocol: dict, sample: dict, candidates: list[dict], *, validate: bool) -> list[dict]:
    """Score a training tournament or a held-out candidate against both references."""
    check_dependencies(protocol)
    if any(c["status"] not in {"succeeded", "invalid_call"} for c in candidates):
        raise RuntimeError(f"Only generated videos and invalid calls can be scored; inspect {directory}")
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
            pairs = manifests["original"]["pairs"]
            by_reference = {pair["reference"]: key for key, pair in pairs.items()}
            result = {"valid_call": float(valid), "judge_comparisons": 2 * len(pairs)}
            for reference in REFERENCE_FIELDS:
                key = by_reference.get(reference)
                result[f"vs_{reference}"] = (
                    scoring.score_pair(
                        pairs[key], judgments["original"][key], judgments["swapped"][key], candidate=CANDIDATE
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
