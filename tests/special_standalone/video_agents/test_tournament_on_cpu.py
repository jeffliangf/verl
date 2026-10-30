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
"""Exercise tournaments, provider deadlines, and native RL data without paid calls."""

import asyncio
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from examples.video_agents import h3, judge, rl, tournament
from examples.video_agents.config import CANDIDATE, DEFAULT_REFERENCES, JUDGE_SETTINGS, REFERENCE_VIDEO_FIELDS
from examples.video_agents.judge import EXTERNAL_FILES
from examples.video_agents.observability import ProviderError
from examples.video_agents.storage import file_sha256, fingerprint, read_json, read_jsonl, write_json, write_jsonl

MEDIA = {
    "duration_s": 10.125,
    "video": {"width": 1344, "height": 768, "fps": 24.0, "codec": "avc1"},
    "audio": {"sample_rate": 32000, "channels": 2, "codec": "mp4a"},
}
HOST = "https://h3.invalid"
REFERENCE_GATEWAY = "http://reference-gateway:18080"
# The launch-time settings every runtime protocol takes from configs/rl.yaml.
VIDEO_RL = OmegaConf.load(rl.CONFIG_DIR / "rl.yaml").video_rl


@pytest.fixture
def prepared_rl(tmp_path):
    repo, references, directory = tmp_path / "eval-repo", tmp_path / "references", tmp_path / "run"
    for relative in EXTERNAL_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake evaluator dependency\n")
    (repo / "scripts/build_pairwise_manifest.py").write_text(
        'INSTRUCTIONS_VERSION = 4\nINSTRUCTIONS = "Compare videos."\n'
    )
    (repo / "bestofn/normalize.py").write_text(f"def probe(path):\n    return {MEDIA!r}\n")
    manifest = {
        "h3_backend": h3.BACKEND,
        "h3_endpoints": [REFERENCE_GATEWAY],
        "h3": h3.SETTINGS,
        "settings": h3.SETTINGS,
        "endpoint": h3.ENDPOINT,
        "media_profile": h3.media_profile(MEDIA),
        "context_ir_rewriter_enabled": False,
        "tasks": [],
    }
    rows = []
    for i, source in enumerate(["aa_eval", "movie_gen_video_bench"]):
        row = {"id": f"eval-{i}", "prompt": f"Original {i}", "source": source, "h3_context_ir": f"SECRET {i}"}
        for field in ("prompt", "h3_context_ir"):
            video = references / f"{i}-{field}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"reference {i} {field}".encode())
            state = video.with_suffix(".json")
            task = {
                "item_id": row["id"],
                "field": field,
                "request": h3.request_body(row[field], h3.SETTINGS),
                "output": video.name,
                "state": state.name,
            }
            write_json(
                state,
                {
                    **h3.identity(task, manifest),
                    "status": "succeeded",
                    "worker_url": REFERENCE_GATEWAY,
                    "task_id": f"reference-{i}-{field}",
                    "sha256": file_sha256(video),
                    "bytes": video.stat().st_size,
                    "media": MEDIA,
                },
            )
            manifest["tasks"].append(task)
            row[f"{field}_h3_self_hosted_video"] = str(video)
        best_of_4 = tmp_path / "best-of-4"
        video = best_of_4 / "videos" / row["id"] / "best_of_4.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"best-of-4 {i}".encode())
        write_json(
            best_of_4 / "items" / f"{row['id']}.json",
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "status": "succeeded",
                "winner_model_id": "minimax-h3-sol-repeat4",
                "selection": {"selected_model": "minimax-h3-sol-repeat4"},
                "lanes": [
                    {
                        "model_id": f"minimax-h3-sol-repeat{j}",
                        "status": "completed",
                        "processing_path": "context-ir-sol-h3-native",
                    }
                    for j in range(1, 5)
                ],
                "video": {
                    "path": str(video),
                    "sha256": file_sha256(video),
                    "bytes": video.stat().st_size,
                    "media": MEDIA,
                },
            },
        )
        row[REFERENCE_VIDEO_FIELDS["best_of_4"]] = str(video)
        rows.append(row)
    write_json(references / "manifest.json", manifest)
    train_file, val_file = tmp_path / "train.jsonl", tmp_path / "eval.jsonl"
    write_jsonl(
        train_file,
        [
            {"id": f"train-{i}", "prompt": f"Train {i}", "source": "vbench", "h3_context_ir": "SECRET TRAIN"}
            for i in range(16)
        ],
    )
    write_jsonl(val_file, rows)
    args = rl.parse_args(
        [
            "prepare",
            "--h3-endpoints",
            HOST,
            "--run-dir",
            str(directory),
            "--data-dir",
            str(tmp_path / "prepared-data"),
            "--train-file",
            str(train_file),
            "--eval-dataset",
            str(val_file),
            "--references",
            str(references),
            "--eval-repo",
            str(repo),
        ]
    )
    rl.prepare(args)
    return directory, tournament.runtime_protocol(tournament.load_protocol(directory / "protocol.json"), VIDEO_RL), args


def candidates(tmp_path, invalid=(), group_size=4):
    result = []
    for i in range(group_size):
        if i in invalid:
            result.append({"status": "invalid_call"})
        else:
            path = tmp_path / f"candidate-{i}.mp4"
            path.write_bytes(f"video {i}".encode())
            result.append({"status": "succeeded", "path": str(path), "sha256": file_sha256(path)})
    return result


def votes(manifests, *, position_bias=False):
    result = {}
    for order, manifest in manifests.items():
        result[order] = {}
        for key, pair in manifest["pairs"].items():
            winner = pair["model_A"] if position_bias else min(pair["model_A"], pair["model_B"])
            result[order][key] = {
                "status": "completed",
                "preferred_model": winner,
                "reason": "Visible evidence",
            }
    return result


def install_judge(monkeypatch, *, winner=CANDIDATE):
    calls = []

    def execute(command, *, check, timeout, stdout, stderr):
        calls.append(command)
        manifest_path = Path(command[command.index("--manifest") + 1])
        output_path = Path(command[command.index("--output") + 1])
        manifest = read_json(manifest_path)
        selected = [command[i + 1] for i, arg in enumerate(command) if arg == "--pair-id"]
        rows = read_jsonl(output_path) if output_path.exists() else []
        for key in selected:
            pair = manifest["pairs"][key]
            selected_winner = (
                winner if winner in (pair["model_A"], pair["model_B"]) else min(pair["model_A"], pair["model_B"])
            )
            rows.append(
                {
                    "status": "completed",
                    "pair_id": key,
                    "prompt_id": pair["prompt_id"],
                    "model_A": pair["model_A"],
                    "model_B": pair["model_B"],
                    "preferred_video": "A" if selected_winner == pair["model_A"] else "B",
                    "preferred_model": selected_winner,
                    "reason": "Visible evidence",
                    "manifest_sha256": file_sha256(manifest_path),
                    "evaluation_signature": judge.evaluation_signature(pair, manifest["instructions_how_to_compare"]),
                    **JUDGE_SETTINGS,
                }
            )
        write_jsonl(output_path, rows)
        assert check is True and timeout == VIDEO_RL.timeouts.judge_pass
        assert stderr == subprocess.STDOUT and Path(stdout.name).name.startswith("runner-")
        assert int(command[command.index("--max-attempts") + 1]) == VIDEO_RL.judge_max_attempts
        assert float(command[command.index("--request-timeout") + 1]) == VIDEO_RL.timeouts.judge_request
        assert float(command[command.index("--file-timeout") + 1]) == VIDEO_RL.timeouts.judge_file
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tournament.subprocess, "run", execute)
    return calls


def test_six_pairs_twelve_votes_and_native_grpo(prepared_rl, tmp_path):
    import numpy as np
    import torch

    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

    _, protocol, _ = prepared_rl
    clips = candidates(tmp_path)
    manifests = tournament.tournament_manifests(protocol["samples"]["train"][0], clips, protocol)
    assert len(manifests["original"]["pairs"]) == len(manifests["swapped"]["pairs"]) == 6
    for key, pair in manifests["original"]["pairs"].items():
        assert manifests["swapped"]["pairs"][key] == judge.swap_pair(pair)
        assert pair["prompt"] == "Train 0"
    rewards = tournament.tournament_rewards(clips, manifests, votes(manifests))
    scores = [r["score"] for r in rewards]
    assert scores == pytest.approx([1, 2 / 3, 1 / 3, 0])
    advantages, _ = compute_grpo_outcome_advantage(
        torch.tensor(scores).unsqueeze(-1), torch.ones(4, 1), np.array(["same"] * 4)
    )
    assert advantages[:, 0].tolist() == pytest.approx([1.161895, 0.387298, -0.387298, -1.161895], abs=1e-5)


@pytest.mark.parametrize("invalid", [(), (5,), (1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)])
def test_six_candidates_have_thirty_votes_and_normalized_forfeits(prepared_rl, tmp_path, invalid):
    _, original, _ = prepared_rl
    protocol = {**original, "group_size": 6}
    clips = candidates(tmp_path, invalid, group_size=6)
    manifests = tournament.tournament_manifests(protocol["samples"]["train"][0], clips, protocol)
    count = (6 - len(invalid)) * (5 - len(invalid))
    assert sum(len(m["pairs"]) for m in manifests.values()) == count
    rewards = tournament.tournament_rewards(clips, manifests, votes(manifests))
    assert [r["score"] for r in rewards] == pytest.approx([1 - i / 5 if i not in invalid else -1 for i in range(6)])
    assert all(r["judge_comparisons"] == r["judge_completed_comparisons"] == count for r in rewards)
    with pytest.raises(ValueError, match="exactly 6"):
        tournament.tournament_manifests(protocol["samples"]["train"][0], clips[:4], protocol)


def test_six_candidate_preparation_preview_and_parallel_judging(fresh_args, monkeypatch):
    args = fresh_args
    args.group_size = 6
    rl.prepare(args)
    monkeypatch.setattr(rl.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected paid work"))
    launch = rl.parse_args(
        [
            "train",
            "--run-dir",
            str(args.run_dir),
            "--data-dir",
            str(args.data_dir),
            "--",
            "actor_rollout_ref.rollout.n=6",
            "video_rl.judge_concurrency=15",
        ]
    )
    summary = rl.train(launch)
    assert (summary["videos_per_training_batch"], summary["judgments_per_training_batch"]) == (96, 480)
    config, _ = rl.training_config(launch)
    runtime = tournament.runtime_protocol(tournament.load_protocol(args.run_dir / "protocol.json"), config.video_rl)
    for pairs, expected in [(set(range(15)), "15"), ({"remaining"}, "1")]:
        command = judge.command(args.run_dir, runtime, "original", pairs, allow_fallback=True)
        assert command[command.index("--concurrency") + 1] == expected
        assert "--parallel-pairs" in command
    launch.overrides = ["actor_rollout_ref.rollout.n=4"]
    with pytest.raises(ValueError, match="6 training rollouts"):
        rl.training_config(launch)


@pytest.mark.parametrize("value", [0, 1, True, 6.0, "6", None])
def test_invalid_group_sizes_are_rejected(fresh_args, value):
    fresh_args.group_size = value
    with pytest.raises(ValueError, match="group_size"):
        rl.prepare(fresh_args)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "15"])
def test_invalid_judge_concurrency_is_rejected(prepared_rl, value):
    settings = OmegaConf.to_container(VIDEO_RL, resolve=False)
    settings["judge_concurrency"] = value
    with pytest.raises(ValueError, match="judge_concurrency"):
        tournament.runtime_protocol(prepared_rl[1], settings)


def test_order_disagreement_is_split_and_equal_groups_have_no_signal(prepared_rl, tmp_path):
    _, protocol, _ = prepared_rl
    clips = candidates(tmp_path)
    manifests = tournament.tournament_manifests(protocol["samples"]["train"][0], clips, protocol)
    rewards = tournament.tournament_rewards(clips, manifests, votes(manifests, position_bias=True))
    assert [r["score"] for r in rewards] == [0.5] * 4
    assert all(r["zero_variance_group"] == r["order_disagreement"] == 1 for r in rewards)


@pytest.mark.parametrize(
    "invalid,expected,count",
    [
        ((3,), [1, 2 / 3, 1 / 3, -1], 6),
        ((1, 2, 3), [1, -1, -1, -1], 0),
        ((0, 1, 2, 3), [-1, -1, -1, -1], 0),
    ],
)
def test_invalid_calls_forfeit_without_video_judgments(prepared_rl, tmp_path, invalid, expected, count):
    _, protocol, _ = prepared_rl
    clips = candidates(tmp_path, invalid)
    manifests = tournament.tournament_manifests(protocol["samples"]["train"][0], clips, protocol)
    rewards = tournament.tournament_rewards(clips, manifests, votes(manifests))
    assert [r["score"] for r in rewards] == pytest.approx(expected)
    assert all(r["judge_comparisons"] == count for r in rewards)


@pytest.mark.parametrize("damage", ["error", "fallback", "missing"])
@pytest.mark.parametrize("policy", ["drop", "neutral"])
def test_no_incomplete_tournament_can_receive_rewards(prepared_rl, tmp_path, monkeypatch, damage, policy):
    directory, protocol, _ = prepared_rl
    protocol = {**protocol, "judge_failure_policy": policy}
    install_judge(monkeypatch)
    execute = tournament.subprocess.run

    def incomplete(command, **kwargs):
        execute(command, **kwargs)
        path = Path(command[command.index("--output") + 1])
        rows = read_jsonl(path)
        if damage == "missing":
            rows.pop()
        else:
            rows[0]["status"] = damage
        write_jsonl(path, rows)

    monkeypatch.setattr(tournament.subprocess, "run", incomplete)
    group = directory / "group"
    with pytest.raises(RuntimeError, match="unjudged" if policy == "drop" else "Nonrecoverable|without a result"):
        tournament.score_group(group, protocol, protocol["samples"]["train"][0], candidates(tmp_path), validate=False)
    assert not (group / "rewards.json").exists()


def test_failed_candidates_are_never_scored(prepared_rl, tmp_path, monkeypatch):
    directory, protocol, _ = prepared_rl
    monkeypatch.setattr(tournament.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected judge request"))
    clips = candidates(tmp_path)
    clips[0]["status"] = "infrastructure_error"
    with pytest.raises(RuntimeError, match="Only generated videos"):
        tournament.score_group(directory / "group", protocol, protocol["samples"]["train"][0], clips, validate=False)


def test_judge_runner_reuses_all_completed_comparisons(prepared_rl, tmp_path, monkeypatch):
    directory, protocol, _ = prepared_rl
    clips = candidates(tmp_path)
    calls = install_judge(monkeypatch)
    group = directory / "group"
    sample = protocol["samples"]["train"][0]
    first = tournament.score_group(group, protocol, sample, clips, validate=False)
    second = tournament.score_group(group, protocol, sample, clips, validate=False)
    assert first == second and len(calls) == 2
    assert sum(c.count("--pair-id") for c in calls) == 12
    assert [r["score"] for r in first] == pytest.approx([1, 2 / 3, 1 / 3, 0])


def test_held_out_scoring_uses_both_fixed_references(prepared_rl, tmp_path, monkeypatch):
    directory, protocol, _ = prepared_rl
    calls = install_judge(monkeypatch)
    rewards = tournament.score_group(
        directory / "eval-group", protocol, protocol["samples"]["val"][0], candidates(tmp_path)[:1], validate=True
    )
    assert rewards[0]["vs_best_of_4"] == rewards[0]["vs_h3_context_ir"] == rewards[0]["score"] == 1
    assert "vs_prompt" not in rewards[0]
    assert sum(c.count("--pair-id") for c in calls) == 4
    manifest = read_json(directory / "eval-group/judge/manifest-original.json")
    assert {p["reference"] for p in manifest["pairs"].values()} == {"h3_context_ir", "best_of_4"}
    assert all(p["prompt"] == "Original 0" for p in manifest["pairs"].values())


@pytest.mark.parametrize("validate", [False, True])
@pytest.mark.parametrize(
    "error,exit_mode",
    [
        ('HTTP 400: {"error":{"code":"prohibited_content","message":"Input blocked"}}', "error"),
        ('HTTP 400: {"error":{"code":"prohibited_content","message":"Input blocked"}}', "timeout"),
        ('HTTP 400: {"error":{"code":"prohibited_content","message":"Input blocked"}}', "success"),
        ('HTTP 403: {"error":{"code":"PERMISSION_DENIED"}}', "error"),
        ('HTTP 429: {"error":{"code":"RESOURCE_EXHAUSTED"}}', "error"),
        ("Network request failed: The read operation timed out", "error"),
    ],
)
def test_gemini_refusals_and_errors_fail_the_example_with_their_reason(
    prepared_rl, tmp_path, monkeypatch, validate, error, exit_mode
):
    directory, protocol, _ = prepared_rl
    protocol = {**protocol, "judge_failure_policy": "drop"}
    calls = install_judge(monkeypatch)
    execute = tournament.subprocess.run

    def fail(command, **kwargs):
        execute(command, **kwargs)
        path = Path(command[command.index("--output") + 1])
        row = read_jsonl(path)[0]
        for key in ("preferred_video", "preferred_model", "reason"):
            row.pop(key)
        row.update(status="error", stage="interaction", error_type="HttpFailure", error=error)
        write_jsonl(path, [row])
        if exit_mode == "error":
            raise subprocess.CalledProcessError(1, command)
        if exit_mode == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(tournament.subprocess, "run", fail)
    group = directory / "failed-example"
    sample = protocol["samples"]["val" if validate else "train"][0]
    clips = candidates(tmp_path)[: 1 if validate else 4]
    with pytest.raises(ProviderError) as failure:
        tournament.score_group(group, protocol, sample, clips, validate=validate)
    # A refusal is an ordinary provider failure: the example is dropped, never
    # scored, and the message names the comparison, its error, and both logs.
    message = str(failure.value)
    assert failure.value.stage == "gemini" and error[:40] in message
    assert "judgments-original.jsonl" in message and "runner-original.log" in message
    assert len(calls) == 1 and (group / "judge/runner-original.log").exists()
    assert read_jsonl(group / "judge/judgments-original.jsonl")[0]["status"] == "error"
    assert not (group / "rewards.json").exists()


@pytest.mark.parametrize("stage", ["h3", "judge"])
def test_hard_timeouts_raise_without_neutral_fallbacks(prepared_rl, tmp_path, monkeypatch, stage):
    directory, protocol, _ = prepared_rl
    protocol = {**protocol, "judge_failure_policy": "drop"}
    calls = []

    def timeout(command, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(tournament.subprocess, "run", timeout)
    with pytest.raises(ProviderError):
        if stage == "h3":
            tournament.generate_video(
                directory / "candidate", protocol, {"prompt": "A valid video prompt", "duration": 10, "seed": 42}
            )
        else:
            tournament.score_group(
                directory / "group", protocol, protocol["samples"]["train"][0], candidates(tmp_path), validate=False
            )
    assert [call["timeout"] for call in calls] == [protocol["timeouts"]["h3" if stage == "h3" else "judge_pass"]]
    assert calls[0]["check"] is True and calls[0]["stderr"] == subprocess.STDOUT
    log = "candidate/h3-worker.log" if stage == "h3" else "group/judge/runner-original.log"
    assert (directory / log).exists()
    assert not list(directory.rglob("rewards.json"))


def test_cleanup_timeout_keeps_complete_validated_judgments(prepared_rl, tmp_path, monkeypatch):
    directory, protocol, _ = prepared_rl
    calls = install_judge(monkeypatch)
    execute = tournament.subprocess.run

    def timeout_after_judgments(command, **kwargs):
        execute(command, **kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(tournament.subprocess, "run", timeout_after_judgments)
    group = directory / "group"
    rewards = tournament.score_group(
        group, protocol, protocol["samples"]["train"][0], candidates(tmp_path), validate=False
    )
    assert len(calls) == 2
    assert [r["score"] for r in rewards] == pytest.approx([1, 2 / 3, 1 / 3, 0])
    records = read_jsonl(group / "judge/timings.jsonl")
    recovered = [r for r in records if r["event"] == "timeout_with_complete_ledger"]
    assert len(recovered) == 2 and all(r["completed_pairs"] == 6 for r in recovered)
    assert all(r["cleanup_status"] == "unconfirmed" for r in recovered)


@pytest.mark.parametrize("broken", ["missing", "identity"])
@pytest.mark.parametrize("policy", ["drop", "neutral"])
def test_cleanup_timeout_preserves_real_votes_and_rejects_mismatched_judgments(
    prepared_rl, tmp_path, monkeypatch, broken, policy
):
    directory, protocol, _ = prepared_rl
    protocol = {**protocol, "judge_failure_policy": policy}
    install_judge(monkeypatch)
    execute = tournament.subprocess.run

    def timeout_after_bad_ledger(command, **kwargs):
        execute(command, **kwargs)
        path = Path(command[command.index("--output") + 1])
        rows = read_jsonl(path)
        if broken == "missing":
            rows.pop()
        else:
            rows[0]["manifest_sha256"] = "wrong"
        write_jsonl(path, rows)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(tournament.subprocess, "run", timeout_after_bad_ledger)
    group = directory / "group"
    if broken == "missing" and policy == "neutral":
        rewards = tournament.score_group(
            group, protocol, protocol["samples"]["train"][0], candidates(tmp_path), validate=False
        )
        assert [r["score"] for r in rewards] == pytest.approx([1, 2 / 3, 1 / 6, 1 / 6])
        assert all(r["judge_completed_comparisons"] == 10 and r["judge_fallback_comparisons"] == 2 for r in rewards)
        return
    with pytest.raises(ProviderError if broken == "missing" else ValueError):
        tournament.score_group(group, protocol, protocol["samples"]["train"][0], candidates(tmp_path), validate=False)
    assert not (group / "rewards.json").exists()


@pytest.mark.parametrize(
    "state,provider_failure",
    [
        ({"status": "submit_unknown", "error": "POST HTTP 502"}, True),
        ({"status": "failed", "error": "Worker stopped"}, True),
        ({"status": "error", "error_stage": "provider", "error": "Poll timed out"}, True),
        ({"status": "error", "error_stage": "artifact", "error": "Corrupted media"}, False),
        (None, False),
    ],
)
def test_h3_worker_exit_only_classifies_service_failures_as_droppable(
    prepared_rl, monkeypatch, state, provider_failure
):
    directory, protocol, _ = prepared_rl

    def fail(command, **kwargs):
        if state is not None:
            write_json(directory / "candidate/state.json", state)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(tournament.subprocess, "run", fail)
    with pytest.raises(RuntimeError) as error:
        tournament.generate_video(directory / "candidate", protocol, {"prompt": "A video", "duration": 10, "seed": 42})
    assert isinstance(error.value, ProviderError) == provider_failure
    assert "h3-worker.log" in str(error.value)
    if state is not None:
        assert state["error"] in str(error.value)


def test_gemini_failed_judgment_is_droppable_but_runner_crash_is_fatal(prepared_rl, tmp_path, monkeypatch):
    directory, protocol, _ = prepared_rl
    manifest = tournament.tournament_manifests(protocol["samples"]["train"][0], candidates(tmp_path), protocol)
    output = directory / "group"
    write_error = False

    def fail(command, **kwargs):
        if write_error:
            write_jsonl(
                output / "judge/judgments-original.jsonl",
                [{"status": "error", "pair_id": "0_vs_1", "stage": "upload", "error": "HTTP 503"}],
            )
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(tournament.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="crashed without recording an error") as crash:
        tournament.run_judgments(output, protocol, manifest)
    assert not isinstance(crash.value, ProviderError)
    write_error = True
    with pytest.raises(ProviderError, match="0_vs_1: upload: HTTP 503"):
        tournament.run_judgments(output, protocol, manifest)


@pytest.mark.parametrize("duration,seed", [(5, 0), (10, 9223372036854775807)])
def test_h3_adapter_reuses_existing_worker_and_validates_artifacts(prepared_rl, monkeypatch, duration, seed):
    directory, protocol, _ = prepared_rl
    posts = []
    job_id = "338a55c9-c8f3-4dc4-982b-df72221cd893"
    client = SimpleNamespace(
        submit=lambda origin, request, **kw: posts.append(request) or {"id": job_id},
        get_json=lambda url, **kw: (
            {"status": "ready", "queued": 0, "queue_capacity": 8}
            if url.endswith("/healthz")
            else {"id": job_id, "status": "succeeded"}
        ),
        download=lambda origin, job_id, path, **kw: path.write_bytes(b"video"),
    )

    def run(command, **kwargs):
        work = Path(command[command.index("--run-dir") + 1])
        task = read_json(work / "generation.json")[0]
        assert h3.generate_one(task, protocol, client, lambda path: {**MEDIA, "duration_s": duration + 0.125})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tournament.subprocess, "run", run)
    request = {"prompt": "Exact prompt", "duration": duration, "seed": seed}
    first = tournament.generate_video(directory / "candidate", protocol, request)
    second = tournament.generate_video(directory / "candidate", protocol, request)
    assert first == second and len(posts) == 1
    assert posts[0] == first["request"] == request
    assert first["task_id"] == job_id
    Path(first["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        tournament.generate_video(directory / "candidate", protocol, request)


def test_prepare_has_no_teacher_leakage_and_is_immutable(prepared_rl):
    directory, protocol, args = prepared_rl
    assert rl.prepare(args)["network_requests"] == 0
    for split in ("train", "val"):
        rows = read_jsonl(args.data_dir / f"{split}.jsonl")
        assert all([m["role"] for m in row["prompt"]] == ["system", "user"] for row in rows)
        assert "SECRET" not in (args.data_dir / f"{split}.jsonl").read_text()
        assert all(row["agent_name"] == "h3_terminal" for row in rows)
        for row, sample in zip(rows, protocol["samples"][split], strict=True):
            assert row["prompt"][1]["content"] == f"Generate a video of about 10 seconds of: {sample['prompt']}"
            assert row["tools"][0]["function"]["name"] == "sol_h3"
            assert row["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "xhigh"}
        assert protocol["datasets"][split]["sha256"] == file_sha256(args.data_dir / f"{split}.jsonl")
        assert read_json(args.data_dir / f"{split}.preview.json") == rows[0]
    assert "tool_schema" not in read_json(directory / "tools.json")["tools"][0]
    changed = copy.copy(args)
    changed.poll_interval = 2
    with pytest.raises(ValueError, match="Frozen input changed"):
        rl.prepare(changed)
    protocol["group_size"] = 8
    write_json(directory / "protocol.json", protocol)
    with pytest.raises(ValueError, match="protocol changed"):
        tournament.load_protocol(directory / "protocol.json")


def test_prepare_rejects_train_eval_overlap(prepared_rl):
    _, _, args = prepared_rl
    rows = read_jsonl(args.train_file)
    rows[0]["prompt"] = "Original 0"
    write_jsonl(args.train_file, rows)
    with pytest.raises(ValueError, match="overlap"):
        rl.prepare(args)


def test_prepare_records_tool_schema_and_reference_provenance(prepared_rl, fresh_args):
    _, original, _ = prepared_rl
    args = fresh_args
    args.h3_endpoints = "http://worker-a:18080,http://worker-b:18080"
    args.h3_concurrency_per_worker = 2
    assert rl.prepare(args)["network_requests"] == 0
    protocol = tournament.load_protocol(args.run_dir / "protocol.json")
    assert protocol["h3"]["model"] == "MiniMax-H3-Base-FL2VA"
    assert protocol["h3_endpoints"] == ["http://worker-a:18080", "http://worker-b:18080"]
    assert protocol["reference_h3"] == {
        "backend": "self-hosted",
        "settings": h3.SETTINGS,
        "endpoint": "/v1/videos",
        "endpoints": [REFERENCE_GATEWAY],
    }
    assert protocol["h3"] == {k: v for k, v in h3.SETTINGS.items() if k not in {"duration", "seed"}}
    assert protocol["samples"]["val"] == original["samples"]["val"]
    # Provider limits come from the launch config; none are frozen at preparation.
    assert not {"timeouts", "latency_baseline", "h3_host", "reference_video_fields"} & set(protocol)
    for split in ("train", "val"):
        row = read_jsonl(args.data_dir / f"{split}.jsonl")[0]
        schema = row["tools"][0]["function"]
        assert "Sol-H3" in schema["description"] and "10 seconds" not in schema["description"]
        assert set(schema["parameters"]["properties"]) == {"prompt", "duration", "seed"}
        assert schema["parameters"]["properties"]["prompt"]["maxLength"] == 8000
        assert "SECRET" not in json.dumps(row)


def test_rl_defaults_use_self_hosted_references(tmp_path):
    args = rl.parse_args(["prepare", "--run-dir", str(tmp_path)])
    assert args.references == DEFAULT_REFERENCES
    assert (args.groups_per_worker, args.h3_concurrency_per_worker, args.poll_interval) == (5, 5, 1)


@pytest.fixture
def fresh_args(prepared_rl, tmp_path):
    """Preparation arguments for a second run over the same inputs."""
    args = copy.copy(prepared_rl[2])
    args.run_dir, args.data_dir = tmp_path / "second-run", tmp_path / "second-data"
    return args


@pytest.mark.parametrize("invalid", ["backend", "settings", "pool", "request", "state_media", "video_path", "bytes"])
def test_prepare_rejects_mismatched_reference_provenance(fresh_args, invalid):
    args = fresh_args
    manifest = read_json(args.references / "manifest.json")
    task = next(task for task in manifest["tasks"] if task["field"] == "h3_context_ir")
    state_path = args.references / task["state"]
    state = read_json(state_path)
    rows = read_jsonl(args.eval_dataset)
    if invalid == "backend":
        manifest["h3_backend"] = "unknown"
    elif invalid == "settings":
        manifest["settings"]["model"] = "another-model"
    elif invalid == "pool":
        state["endpoints"] = ["http://another-gateway:18080"]
    elif invalid == "request":
        task["request"]["prompt"] = "Another prompt"
    elif invalid == "state_media":
        state["media"]["video"]["width"] = 1280
    elif invalid == "video_path":
        rows[0][REFERENCE_VIDEO_FIELDS[task["field"]]] = "/wrong/video.mp4"
    else:
        (args.references / task["output"]).write_bytes(b"changed reference clip")
    write_json(args.references / "manifest.json", manifest)
    write_json(state_path, state)
    write_jsonl(args.eval_dataset, rows)
    with pytest.raises(ValueError):
        rl.prepare(args)
    assert not (args.run_dir / "protocol.json").exists()


def test_prepare_excludes_original_prompt_videos(fresh_args):
    args = fresh_args
    rows = read_jsonl(args.eval_dataset)
    for row in rows:
        Path(row.pop("prompt_h3_self_hosted_video")).unlink()
    write_jsonl(args.eval_dataset, rows)
    rl.prepare(args)
    samples = tournament.load_protocol(args.run_dir / "protocol.json")["samples"]["val"]
    assert all(set(sample["references"]) == {"h3_context_ir", "best_of_4"} for sample in samples)


@pytest.mark.parametrize(
    "invalid",
    ["missing", "id", "prompt", "status", "winner", "lanes", "backend", "path", "sha256", "bytes", "media"],
)
def test_prepare_rejects_mismatched_best_of_4_provenance(fresh_args, invalid):
    args = fresh_args
    rows = read_jsonl(args.eval_dataset)
    video = Path(rows[0][REFERENCE_VIDEO_FIELDS["best_of_4"]])
    record_path = video.parent.parent.parent / "items" / f"{video.parent.name}.json"
    record = read_json(record_path)
    if invalid == "missing":
        del rows[0][REFERENCE_VIDEO_FIELDS["best_of_4"]]
    elif invalid in {"id", "prompt", "status"}:
        record[invalid] = "different"
    elif invalid == "winner":
        record["selection"]["selected_model"] = "minimax-h3-sol-repeat1"
    elif invalid == "lanes":
        record["lanes"].pop()
    elif invalid == "backend":
        record["lanes"][0]["processing_path"] = "other-provider"
    elif invalid == "media":
        record["video"]["media"]["video"]["width"] = 1280
    else:
        record["video"][invalid] = 0 if invalid == "bytes" else "different"
    write_json(record_path, record)
    write_jsonl(args.eval_dataset, rows)
    with pytest.raises(ValueError):
        rl.prepare(args)
    assert not (args.run_dir / "protocol.json").exists()


def test_launcher_rejects_old_reference_pair_before_launch(prepared_rl, monkeypatch):
    directory, _, prepared_args = prepared_rl
    path = directory / "protocol.json"
    protocol = read_json(path)
    for sample in protocol["samples"]["val"]:
        sample["references"]["prompt"] = sample["references"].pop("best_of_4")
    write_json(path, protocol)
    write_json(path.with_name("protocol-sha256.json"), fingerprint(protocol))
    monkeypatch.setattr(rl.subprocess, "run", lambda *a, **k: pytest.fail("Must not launch with old references"))
    args = rl.parse_args(["train", "--run-dir", str(directory), "--data-dir", str(prepared_args.data_dir)])
    with pytest.raises(ValueError, match="Evaluation references changed; prepare a new run directory"):
        rl.train(args)


@pytest.mark.parametrize("changed", ["candidate", "reference", "evaluator"])
def test_changed_artifacts_stop_before_judging(prepared_rl, tmp_path, monkeypatch, changed):
    directory, protocol, _ = prepared_rl
    sample = protocol["samples"]["val"][0]
    clips = candidates(tmp_path)[:1]
    if changed == "candidate":
        path = Path(clips[0]["path"])
    elif changed == "reference":
        path = Path(sample["references"]["best_of_4"]["path"])
    else:
        path = Path(protocol["eval_repo"]) / EXTERNAL_FILES[0]
    path.write_bytes(b"changed artifact")
    monkeypatch.setattr(tournament.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected judge request"))
    with pytest.raises(ValueError, match="changed"):
        tournament.score_group(directory / "changed", protocol, sample, clips, validate=True)


@pytest.mark.parametrize("backend", [None, "max"])
def test_launcher_rejects_legacy_max_run_before_launch(prepared_rl, monkeypatch, backend):
    directory, protocol, prepared_args = prepared_rl
    protocol["h3_backend"] = backend
    write_json(directory / "protocol.json", protocol)
    write_json(directory / "protocol-sha256.json", fingerprint(protocol))
    monkeypatch.setattr(rl.subprocess, "run", lambda *a, **k: pytest.fail("Must not launch a legacy Max run"))
    args = rl.parse_args(["train", "--run-dir", str(directory), "--data-dir", str(prepared_args.data_dir)])
    with pytest.raises(ValueError, match="H3 Max API is disabled"):
        rl.train(args)


def test_prompt_reader_preserves_repeat_weighting_but_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [{"id": str(i), "prompt": "Repeated prompt", "source": "vbench"} for i in range(2)]
    write_jsonl(path, rows)
    assert rl.prompt_rows(path) == rows
    write_jsonl(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate RL ID"):
        rl.prompt_rows(path)


def test_invalid_held_out_call_scores_zero_without_judge_requests(prepared_rl, monkeypatch):
    directory, protocol, _ = prepared_rl
    monkeypatch.setattr(tournament.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected judge call"))
    rewards = tournament.score_group(
        directory / "invalid-eval",
        protocol,
        protocol["samples"]["val"][0],
        [{"status": "invalid_call"}],
        validate=True,
    )
    assert rewards == [
        {
            "valid_call": 0.0,
            "judge_comparisons": 0,
            "vs_best_of_4": 0.0,
            "vs_h3_context_ir": 0.0,
            "score": 0.0,
        }
    ]


def test_launcher_preview_uses_base_model_and_never_executes(prepared_rl, monkeypatch):
    directory, _, prepared_args = prepared_rl
    monkeypatch.setattr(rl.subprocess, "run", lambda *a, **k: pytest.fail("Unexpected paid work"))
    args = rl.parse_args(["train", "--run-dir", str(directory), "--data-dir", str(prepared_args.data_dir)])
    summary = rl.train(args)
    assert summary["model"] == "Qwen/Qwen3.8-27B"
    assert summary["videos_per_training_batch"] == 64
    assert summary["judgments_per_training_batch"] == 192
    assert summary["execute"] is False
    config, _ = rl.training_config(args)
    assert config.data.train_files == str(prepared_args.data_dir / "train.jsonl")
    assert config.data.val_files == str(prepared_args.data_dir / "val.jsonl")
    assert config.trainer.use_v1 and config.trainer.v1.trainer_mode == "sync"
    assert config.actor_rollout_ref.rollout.agent.agent_loop_manager_class.endswith("TournamentManager")
    args.overrides = ["actor_rollout_ref.rollout.n=8"]
    with pytest.raises(ValueError, match="4 training rollouts"):
        rl.training_config(args)


def test_launcher_reads_saved_data_without_rebuilding_prompts(prepared_rl, monkeypatch):
    directory, _, prepared_args = prepared_rl
    args = rl.parse_args(["train", "--run-dir", str(directory), "--data-dir", str(prepared_args.data_dir)])
    monkeypatch.setattr(rl, "dataset_records", lambda *a, **k: pytest.fail("Must consume the saved data"))
    rl.train(args)
    rows = read_jsonl(prepared_args.data_dir / "train.jsonl")
    rows[0]["tools"][0]["function"]["description"] = "Changed after this experiment was frozen"
    write_jsonl(prepared_args.data_dir / "train.jsonl", rows)
    with pytest.raises(ValueError, match="train data differs"):
        rl.train(args)


def test_evaluate_uses_the_training_agent_and_isolates_checkpoint_artifacts(prepared_rl, monkeypatch, tmp_path):
    from omegaconf import OmegaConf

    directory, _, prepared_args = prepared_rl
    common = ["--run-dir", str(directory), "--data-dir", str(prepared_args.data_dir)]
    training, _ = rl.training_config(rl.parse_args(["train", *common]))
    output = tmp_path / "checkpoint-eval"
    args = rl.parse_args(
        [
            "evaluate",
            *common,
            "--model",
            "/checkpoints/huggingface",
            "--output-dir",
            str(output),
            "--execute",
        ]
    )
    evaluation, _ = rl.training_config(args)
    # The Prometheus model label resolves to the checkpoint being evaluated.
    assert OmegaConf.to_container(evaluation.actor_rollout_ref.rollout) == OmegaConf.to_container(
        training.actor_rollout_ref.rollout
    )
    assert evaluation.data == training.data
    assert evaluation.video_rl.protocol == training.video_rl.protocol
    assert evaluation.trainer.val_only and evaluation.trainer.val_before_train
    assert evaluation.trainer.resume_mode == "disable"
    assert evaluation.video_rl.output_dir == str(output)
    calls = []
    monkeypatch.setattr(rl, "check_tokens", lambda config: {"train": 100, "val": 100})
    monkeypatch.setattr(rl.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    summary = rl.train(args)
    assert summary["videos_per_training_batch"] == summary["judgments_per_training_batch"] == 0
    assert summary["videos_per_evaluation"] == 2 and summary["judgments_per_evaluation"] == 8
    assert len(calls) == 1 and "verl.trainer.main_ppo" in calls[0][0]
    assert calls[0][1]["env"]["TENSORBOARD_DIR"] == str(output / "tensorboard")
    assert read_json(output / "evaluation-config.json") == OmegaConf.to_container(evaluation, resolve=True)
    assert not (directory / "training-config.json").exists()
    args.model = "/different/checkpoint"
    with pytest.raises(ValueError, match="changed"):
        rl.train(args)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "override", ["trainer.val_only=false", "trainer.val_before_train=false", "trainer.resume_mode=auto"]
)
def test_evaluate_cannot_switch_to_training_or_resume_other_weights(prepared_rl, tmp_path, override):
    directory, _, prepared_args = prepared_rl
    args = rl.parse_args(
        [
            "evaluate",
            "--run-dir",
            str(directory),
            "--data-dir",
            str(prepared_args.data_dir),
            "--model",
            "Qwen/Qwen3.8-27B",
            "--output-dir",
            str(tmp_path / "eval"),
            "--",
            override,
        ]
    )
    with pytest.raises(ValueError, match="Evaluation loads"):
        rl.training_config(args)
    args.overrides = []
    args.output_dir = directory
    with pytest.raises(ValueError, match="separate from the training run"):
        rl.training_config(args)


class TemplateTokenizer:
    """Small local tokenizer for exercising the real tool loop without model downloads."""

    def encode(self, text, **kwargs):
        return list(text.encode())

    def decode(self, tokens, **kwargs):
        return bytes(tokens).decode()

    def apply_chat_template(self, messages, *, add_generation_prompt=False, tokenize=True, **kwargs):
        text = json.dumps({k: kwargs.get(k) for k in ("tools", "enable_thinking", "reasoning_effort")}) + "\n"
        text += "".join(f"<{m['role']}>{m.get('content', '')}\n" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text


VALID_CALL = "<think>Plan the scene.</think><tool_call><function=sol_h3>"
VALID_CALL += "<parameter=prompt>A clear scene</parameter></function></tool_call>"


@pytest.fixture
def model_request(prepared_rl):
    row = read_jsonl(prepared_rl[2].data_dir / "train.jsonl")[0]
    row["raw_prompt"] = row.pop("prompt")
    return row


@pytest.fixture
def loop_factory(prepared_rl, monkeypatch):
    from examples.video_agents import agent
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap
    from verl.tools.tool_registry import initialize_tools_from_config
    from verl.workers.rollout.replica import TokenOutput

    directory, _, prepared_args = prepared_rl
    config, _ = rl.training_config(
        rl.parse_args(["train", "--run-dir", str(directory), "--data-dir", str(prepared_args.data_dir)])
    )
    generated = []

    def video(directory, protocol, request):
        generated.append(request)
        return {
            "status": "succeeded",
            "path": str(directory / "video.mp4"),
            "prompt": request["prompt"],
            "request": request,
        }

    monkeypatch.setattr(agent, "generate_video", video)
    tool_list = initialize_tools_from_config(directory / "tools.json")

    def make(text=VALID_CALL, *, stop_reason="completed", hang=False):
        async def generate(**kwargs):
            if hang:
                await asyncio.sleep(1)
            ids = list(text.encode())
            return TokenOutput(token_ids=ids, log_probs=[-0.1] * len(ids), stop_reason=stop_reason)

        config.video_rl.timeouts.llm = 0.01 if hang else VIDEO_RL.timeouts.llm
        return agent.H3TerminalLoop(
            trainer_config=DictConfigWrap(config),
            server_manager=SimpleNamespace(generate=generate),
            tokenizer=TemplateTokenizer(),
            processor=None,
            dataset_cls=None,
            data_config=DictConfigWrap(config.data),
            tools=ToolListWrap(tool_list),
        )

    return make, generated


@pytest.mark.asyncio
async def test_native_tool_loop_executes_once_and_preserves_policy_masks(loop_factory, model_request):
    make, generated = loop_factory
    output = await make().run({}, **model_request)
    assert generated == [{"prompt": "A clear scene", "duration": 10, "seed": 42}]
    assert output.extra_fields["h3_result"]["status"] == "succeeded"
    assert output.response_ids[: len(VALID_CALL)] == list(VALID_CALL.encode())
    assert output.response_mask[: len(VALID_CALL)] == [1] * len(VALID_CALL)
    assert output.response_mask[len(VALID_CALL) :] and not any(output.response_mask[len(VALID_CALL) :])
    assert len(output.response_ids) == len(output.response_mask) == len(output.response_logprobs)
    assert output.response_logprobs[: len(VALID_CALL)] == [-0.1] * len(VALID_CALL)


@pytest.mark.asyncio
async def test_checkpoint_evaluation_keeps_videos_in_its_output_directory(loop_factory, model_request, tmp_path):
    make, _ = loop_factory
    loop = make()
    original_dir = loop.tools["sol_h3"].output_dir
    output_dir = tmp_path / "evaluation"
    loop.config.video_rl.output_dir = str(output_dir)
    output = await loop.run({}, **model_request)
    assert Path(output.extra_fields["h3_result"]["path"]).is_relative_to(output_dir)
    assert not Path(output.extra_fields["h3_result"]["path"]).is_relative_to(original_dir)


@pytest.mark.asyncio
async def test_row_tools_and_template_options_are_the_actual_model_input(loop_factory, model_request):
    make, _ = loop_factory
    loop = make()
    shared_tool = loop.tools["sol_h3"]
    description = shared_tool.tool_schema.function.description
    model_request["tools"][0]["function"]["description"] = "ROW_TOOL_DESCRIPTION"
    model_request["tools"][0]["function"]["parameters"]["properties"]["prompt"]["maxLength"] = 7777
    model_request["chat_template_kwargs"]["reasoning_effort"] = "medium"
    output = await loop.run({}, **model_request)
    expected = loop.tokenizer.apply_chat_template(
        model_request["raw_prompt"],
        tools=model_request["tools"],
        **model_request["chat_template_kwargs"],
        add_generation_prompt=True,
    )
    assert output.prompt_ids == expected
    rendered = loop.tokenizer.decode(output.prompt_ids)
    assert rendered.count("ROW_TOOL_DESCRIPTION") == 1
    assert '"maxLength": 7777' in rendered
    assert '"reasoning_effort": "medium"' in rendered
    assert shared_tool.tool_schema.function.description == description
    assert loop.tools["sol_h3"].slots is shared_tool.slots


@pytest.mark.asyncio
@pytest.mark.parametrize("duration,seed", [(5, 0), (10, 9223372036854775807)])
async def test_native_sol_h3_call_forwards_duration_and_seed(loop_factory, model_request, duration, seed):
    make, generated = loop_factory
    text = VALID_CALL.replace(
        "</function>",
        f"<parameter=duration>{duration}</parameter><parameter=seed>{seed}</parameter></function>",
    )
    output = await make(text).run({}, **model_request)
    assert generated == [{"prompt": "A clear scene", "duration": duration, "seed": seed}]
    assert output.extra_fields["h3_result"]["status"] == "succeeded"


@pytest.mark.asyncio
async def test_omitted_tool_arguments_use_defaults_from_the_row(loop_factory, model_request):
    make, generated = loop_factory
    properties = model_request["tools"][0]["function"]["parameters"]["properties"]
    properties["duration"]["default"] = 5
    properties["seed"]["default"] = 7
    await make().run({}, **model_request)
    assert generated == [{"prompt": "A clear scene", "duration": 5, "seed": 7}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,value",
    [
        ("duration", "15"),
        ("duration", "5.5"),
        ("seed", "-1"),
        ("seed", "9223372036854775808"),
        ("seed", "true"),
        ("width", "1344"),
    ],
)
async def test_invalid_sol_h3_arguments_are_policy_errors(loop_factory, model_request, name, value):
    make, generated = loop_factory
    text = VALID_CALL.replace("</function>", f"<parameter={name}>{value}</parameter></function>")
    output = await make(text).run({}, **model_request)
    result = output.extra_fields["h3_result"]
    assert result["status"] == "invalid_call" and result["reason"].startswith("invalid arguments: ")
    assert not generated


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["tools", "chat_template_kwargs"])
async def test_missing_row_input_fails_before_generation(loop_factory, model_request, missing):
    make, generated = loop_factory
    model_request.pop(missing)
    with pytest.raises(ValueError, match="Each dataset row must supply"):
        await make().run({}, **model_request)
    assert not generated


def test_token_check_and_previews_render_the_saved_row(prepared_rl, monkeypatch):
    import verl.utils

    directory, _, prepared_args = prepared_rl
    args = rl.parse_args(["train", "--run-dir", str(directory), "--data-dir", str(prepared_args.data_dir)])
    config, _ = rl.training_config(args)
    monkeypatch.setattr(verl.utils, "hf_tokenizer", lambda *a, **k: TemplateTokenizer())
    monkeypatch.setattr(rl, "dataset_records", lambda *a, **k: pytest.fail("Must render the saved data"))
    rows = read_jsonl(prepared_args.data_dir / "train.jsonl")
    rows[0]["tools"][0]["function"]["description"] = "SAVED_DATA_CANARY"
    write_jsonl(prepared_args.data_dir / "train.jsonl", rows)
    assert rl.check_tokens(config)["train"] > 0
    preview = (prepared_args.data_dir / "train.preview.txt").read_text()
    assert "SAVED_DATA_CANARY" in preview
    assert rows[0]["prompt"][1]["content"] in preview
    config.data.max_prompt_length = 1
    with pytest.raises(ValueError, match="tool prompt exceeds"):
        rl.check_tokens(config)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,reason",
    [
        ("I cannot generate a video.", "no tool call"),
        (VALID_CALL.replace("function=sol_h3", "function=unknown"), "unknown tool 'unknown'"),
        (VALID_CALL.replace("function=sol_h3", "function=h3_max"), "unknown tool 'h3_max'"),
        (VALID_CALL.replace("A clear scene", " "), "invalid arguments: Sol-H3 requires nonblank prompt"),
        (VALID_CALL.replace("<parameter=prompt>", "<parameter=unexpected>"), "invalid arguments: "),
        (VALID_CALL + VALID_CALL, "expected one tool call, got 2"),
    ],
)
async def test_invalid_native_calls_never_invoke_h3(loop_factory, model_request, text, reason):
    make, generated = loop_factory
    output = await make(text).run({}, **model_request)
    result = output.extra_fields["h3_result"]
    # The reason and the model's own output tail make invalid calls debuggable.
    assert result["status"] == "invalid_call" and result["reason"].startswith(reason)
    assert result["response_tail"] == text[-512:]
    assert not generated


@pytest.mark.asyncio
async def test_token_budget_exhaustion_is_a_policy_failure(loop_factory, model_request):
    make, generated = loop_factory
    loop = make("Still reasoning", stop_reason="length")
    loop.response_length = len("Still reasoning")
    output = await loop.run({}, **model_request)
    assert output.extra_fields["h3_result"] == {
        "status": "invalid_call",
        "reason": "no tool call before the length limit",
        "response_tail": "Still reasoning",
    }
    assert not generated


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["aborted", "empty", "timeout", "h3"])
async def test_infrastructure_errors_are_not_invalid_policy_calls(loop_factory, model_request, monkeypatch, failure):
    from examples.video_agents import agent

    make, generated = loop_factory
    if failure == "h3":

        def fail(*args):
            raise subprocess.TimeoutExpired("h3-worker", 1800)

        monkeypatch.setattr(agent, "generate_video", fail)
    loop = make(
        "" if failure == "empty" else VALID_CALL,
        stop_reason="aborted" if failure == "aborted" else "completed",
        hang=failure == "timeout",
    )
    with pytest.raises((RuntimeError, TimeoutError)) as error:
        await loop.run({}, **model_request)
    assert not generated
    if failure == "timeout":
        # A slow-tail LLM turn drops its group instead of stopping training.
        assert isinstance(error.value, ProviderError) and error.value.stage == "llm"


@pytest.fixture
def native_worker(prepared_rl, monkeypatch):
    from examples.video_agents import agent

    directory, protocol, prepared_args = prepared_rl
    worker = object.__new__(agent.TournamentWorker)
    worker.config, _ = rl.training_config(
        rl.parse_args(["train", "--run-dir", str(directory), "--data-dir", str(prepared_args.data_dir)])
    )
    worker.rollout_config = worker.config.actor_rollout_ref.rollout
    worker.output_dir = directory
    worker.protocol_path, worker.protocol = directory / "protocol.json", protocol
    worker.samples = {split: {s["id"]: s for s in protocol["samples"][split]} for split in ("train", "val")}
    worker.group_slots = asyncio.Semaphore(2)
    worker.background_tasks = set()
    worker.processor, worker.reward_loop_worker_handles = None, None
    worker.distillation_enabled = False
    events, published = [], []

    async def put(*, key, partition_id, tag):
        events.append(tag["status"])

    async def batch_put(**kwargs):
        events.append("publish")
        published.append(kwargs)

    monkeypatch.setattr(agent.tq, "async_kv_put", put)
    monkeypatch.setattr(agent.tq, "async_kv_batch_put", batch_put)
    return worker, events, published


@pytest.mark.asyncio
@pytest.mark.parametrize("group_size", [4, 6])
@pytest.mark.parametrize("failure", [None, "rollout", "judge", "h3_provider", "judge_provider", "mixed"])
async def test_native_dispatch_waits_for_group_rewards_and_propagates_failures(
    native_worker, monkeypatch, failure, group_size
):
    import numpy as np

    from examples.video_agents import agent
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
    from verl.utils.tensordict_utils import assign_non_tensor_data, get_tensordict

    worker, events, published = native_worker
    worker.protocol["group_size"] = group_size
    worker.config.actor_rollout_ref.rollout.n = group_size
    scores = [1 - i / (group_size - 1) for i in range(group_size - 1)] + [-1]
    settled = set()

    async def rollout(*args, session_id, **kwargs):
        await asyncio.sleep(session_id * 0.001)
        settled.add(session_id)
        if failure == "rollout" and session_id == 0:
            raise RuntimeError("LLM interrupted")
        if failure in {"h3_provider", "mixed"} and session_id == 0:
            raise ProviderError("h3", "H3 timed out")
        if failure == "mixed" and session_id == 1:
            raise ValueError("Corrupted artifact")
        return AgentLoopOutput(
            prompt_ids=[101, 102],
            response_ids=[11, 12, 13],
            response_mask=[1, 1, 0],
            response_logprobs=[-0.1, -0.2, 0],
            metrics=AgentLoopMetrics(),
            num_turns=3,
            extra_fields={"h3_result": {"status": "invalid_call"}},
        )

    def score(*args, **kwargs):
        assert settled == set(range(group_size)) and not published
        events.append("score")
        if failure == "judge":
            raise RuntimeError("Gemini interrupted")
        if failure == "judge_provider":
            raise ProviderError("gemini", "Gemini timed out")
        return [{"score": s, "valid_call": 1.0} for s in scores]

    monkeypatch.setattr(worker, "_run_agent_loop", rollout)
    monkeypatch.setattr(agent, "score_group", score)
    batch = get_tensordict(
        {
            "uid": np.array(["uid0"], dtype=object),
            "index": np.array([0]),
            "extra_info": np.array([{"id": "train-0"}], dtype=object),
        }
    )
    assign_non_tensor_data(batch, "global_steps", 1)
    if failure in {"h3_provider", "judge_provider"}:
        dropped = await worker.generate_sequences(batch)
        assert len(dropped) == 1 and dropped[0]["dropped"] is True
        assert settled == set(range(group_size))
        assert not published and events[-1] == "failure"
        assert not list(worker.output_dir.rglob("rewards.json"))
        failure = read_json(next(worker.output_dir.rglob("failure.json")))
        assert failure["dropped"] is True and "ProviderError" in failure["traceback"]
        end = [row for row in read_jsonl(worker.output_dir / "timings.jsonl") if row["event"] == "end"][-1]
        assert (end["stage"], end["status"], end["sample_id"]) == ("group", "dropped", "train-0")
        assert end["failure_stage"] == failure["stage"] and Path(end["directory"]).is_dir()
    elif failure:
        with pytest.raises((RuntimeError, ValueError), match="Corrupted|interrupted"):
            await worker.generate_sequences(batch)
        assert settled == set(range(group_size))
        assert not published and events[-1] == "failure"
    else:
        records = await worker.generate_sequences(batch)
        assert [(r["dropped"], r["sample_id"]) for r in records] == [(False, "train-0")]
        assert [r["score"] for r in records[0]["rewards"]] == pytest.approx(scores)
        assert events == ["running", "score", *(["publish"] * group_size), "finished"]
        end = [row for row in read_jsonl(worker.output_dir / "timings.jsonl") if row["event"] == "end"][-1]
        assert end["status"] == "succeeded" and end["scores"] == pytest.approx(scores)
        for i, row in enumerate(published):
            assert row["keys"] == [f"uid0_{i}_0"]
            fields = row["fields"]
            assert fields["rm_scores"][0].sum().item() == pytest.approx(scores[i])
            assert fields["loss_mask"][0].tolist() == [1, 1, 0]
            assert fields["rollout_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2, 0])
            assert fields["extra_fields"][0]["reward_extra_info"]["valid_call"] == 1


@pytest.mark.asyncio
async def test_provider_failure_cannot_silently_pad_disabled_refill(native_worker):
    worker, events, published = native_worker
    worker.config.trainer.v1.sampler.sync_refill_failed_groups = False

    async def rollout(*args, **kwargs):
        raise ProviderError("h3", "H3 timed out")

    worker._run_agent_loop = rollout
    prompt = {"uid": "test", "global_steps": 1, "extra_info": {"id": "train-0"}}
    with pytest.raises(ProviderError):
        await worker._run_prompt(prompt, {}, {"validate": False})
    assert not published and events[-1] == "failure"
    assert read_json(next(worker.output_dir.rglob("failure.json")))["dropped"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ProviderError("h3", "H3 timed out"),
        ProviderError("gemini", "Gemini failed comparisons"),
        ProviderError("gemini_refusal", "Gemini refused held-out example"),
        RuntimeError("Incomplete or mismatched H3 result"),
        TimeoutError(),
    ],
)
@pytest.mark.parametrize("stage", ["rollout", "judge"])
async def test_any_held_out_failure_drops_the_prompt_without_publishing(native_worker, monkeypatch, error, stage):
    from examples.video_agents import agent
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    worker, events, published = native_worker

    async def rollout(*args, **kwargs):
        if stage == "rollout":
            raise error
        return AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[2],
            response_mask=[1],
            metrics=AgentLoopMetrics(),
            extra_fields={"h3_result": {"status": "succeeded"}},
        )

    def score(*args, **kwargs):
        raise error

    worker._run_agent_loop = rollout
    monkeypatch.setattr(agent, "score_group", score)
    prompt = {"uid": "test", "global_steps": 5, "extra_info": {"id": "eval-0"}}
    record = await worker._run_prompt(prompt, {}, {"validate": True})
    assert record["dropped"] is True and record["validate"] is True and record["sample_id"] == "eval-0"
    assert record["stage"] == getattr(error, "stage", "internal")
    assert not published and events[-1] == "failure"
    assert read_json(next(worker.output_dir.rglob("failure.json")))["dropped"] is True


@pytest.mark.asyncio
async def test_held_out_success_returns_its_reward_record(native_worker, monkeypatch):
    from examples.video_agents import agent
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

    worker, events, published = native_worker
    reward = {"score": 0.75, "valid_call": 1.0, "vs_best_of_4": 1.0, "vs_h3_context_ir": 0.5}

    async def rollout(*args, **kwargs):
        return AgentLoopOutput(
            prompt_ids=[1],
            response_ids=[2],
            response_mask=[1],
            metrics=AgentLoopMetrics(),
            extra_fields={"h3_result": {"status": "succeeded"}},
        )

    async def publish(*args, **kwargs):
        published.append(kwargs)

    worker._run_agent_loop = rollout
    monkeypatch.setattr(agent, "score_group", lambda *a, **k: [dict(reward)])
    monkeypatch.setattr(agent.AgentLoopWorkerTQBase, "_agent_loop_postprocess", publish)
    prompt = {"uid": "test", "global_steps": 5, "extra_info": {"id": "eval-0"}}
    record = await worker._run_prompt(prompt, {}, {"validate": True})
    assert record == {"dropped": False, "uid": "test", "sample_id": "eval-0", "rewards": [reward]}
    assert len(published) == 1 and events[-1] == "finished"


def held_out_records(valid_scores, dropped):
    records = [
        {
            "dropped": False,
            "sample_id": f"eval-{i}",
            "rewards": [{"score": s, "valid_call": float(s > 0), "vs_best_of_4": s, "vs_h3_context_ir": s}],
        }
        for i, s in enumerate(valid_scores)
    ]
    records += [
        {
            "dropped": True,
            "validate": True,
            "sample_id": f"dropped-{i}",
            "stage": "gemini",
            "type": "ProviderError",
            "error": "x" * 900,
        }
        for i in range(dropped)
    ]
    return records


def test_evaluation_summary_reports_valid_only_means_and_dropped_reasons():
    from examples.video_agents.agent import evaluation_summary

    summary = evaluation_summary(5, held_out_records([1.0, 0.5, 0.0], dropped=1), 0.9)
    assert (summary["prompts"], summary["valid"], summary["dropped"]) == (4, 3, 1)
    assert summary["score"] == summary["vs_best_of_4"] == summary["vs_h3_context_ir"] == pytest.approx(0.5)
    # Invalid policy calls stay in the denominator with their zero score.
    assert summary["invalid_calls"] == 1
    assert summary["valid_fraction"] == 0.75 and summary["degraded"] is True
    assert summary["dropped_examples"] == [
        {"sample_id": "dropped-0", "stage": "gemini", "type": "ProviderError", "error": "x" * 500}
    ]
    assert evaluation_summary(5, held_out_records([1.0] * 9, dropped=1), 0.9)["degraded"] is False


def test_manager_summarizes_evaluation_without_counting_training_drops(native_worker, monkeypatch, tmp_path):
    import numpy as np

    from examples.video_agents import agent
    from verl.utils.tensordict_utils import assign_non_tensor_data, get_tensordict

    worker, _, _ = native_worker
    manager = object.__new__(agent.TournamentManager)
    manager.config = worker.config
    manager.config.video_rl.output_dir = str(tmp_path)
    manager.config.video_rl.max_dropped_groups_per_step = 1
    manager._drop_step, manager._dropped_groups = None, 0
    manager._eval_step, manager._eval_records = None, []
    manager.agent_loop_workers = [SimpleNamespace(generate_sequences=SimpleNamespace(remote=lambda chunk: 0))]
    batch = get_tensordict({"uid": np.array(["test"], dtype=object)})
    assign_non_tensor_data(batch, "global_steps", 5)
    assign_non_tensor_data(batch, "validate", True)
    replies = [held_out_records([1.0], dropped=2), held_out_records([0.0], dropped=0)]
    monkeypatch.setattr(agent.ray, "get", lambda refs: [replies.pop(0)])
    manager.generate_sequences(batch)
    manager.generate_sequences(batch)
    summary = read_json(tmp_path / "val/step_5/eval-summary.json")
    assert (summary["prompts"], summary["valid"], summary["dropped"]) == (4, 2, 2)
    assert summary["score"] == 0.5
    assert manager._dropped_groups == 0
    monkeypatch.setattr(agent.ray, "get", lambda refs: [held_out_records([], dropped=2)])
    with pytest.raises(RuntimeError, match="Every held-out prompt"):
        manager.generate_sequences(batch)


def test_manager_bounds_failed_group_replacements(native_worker, monkeypatch, tmp_path):
    import numpy as np

    from examples.video_agents import agent
    from verl.utils.tensordict_utils import assign_non_tensor_data, get_tensordict

    worker, _, _ = native_worker
    manager = object.__new__(agent.TournamentManager)
    manager.config = worker.config
    manager.config.video_rl.max_dropped_groups_per_step = 1
    manager._drop_step, manager._dropped_groups = None, 0
    manager.agent_loop_workers = [SimpleNamespace(generate_sequences=SimpleNamespace(remote=lambda chunk: 0))]
    manager.config.video_rl.output_dir = str(tmp_path)
    monkeypatch.setattr(agent.ray, "get", lambda refs: [[{"dropped": True, "stage": "h3"}]])
    batch = get_tensordict({"uid": np.array(["test"], dtype=object)})
    assign_non_tensor_data(batch, "global_steps", 1)
    manager.generate_sequences(batch)
    with pytest.raises(RuntimeError, match="stopping before an optimizer update"):
        manager.generate_sequences(batch)
    assign_non_tensor_data(batch, "global_steps", 2)
    manager.generate_sequences(batch)
    assert manager._dropped_groups == 1
    ends = [row for row in read_jsonl(tmp_path / "timings.jsonl") if row["event"] == "end"]
    # The dispatch itself completes; the drop limit stops the step afterwards.
    assert [(row["status"], row["dropped"], row["total_dropped_groups"]) for row in ends] == [
        ("succeeded", {"h3": 1}, 1),
        ("succeeded", {"h3": 1}, 2),
        ("succeeded", {"h3": 1}, 1),
    ]


def test_launch_config_supplies_every_limit_and_keeps_the_frozen_identity(prepared_rl):
    directory, _, _ = prepared_rl
    frozen = tournament.load_protocol(directory / "protocol.json")
    assert "timeouts" not in frozen
    runtime = tournament.runtime_protocol(frozen, VIDEO_RL)
    assert runtime["timeouts"] == dict(VIDEO_RL.timeouts) and set(runtime["timeouts"]) == set(tournament.TIMEOUT_STAGES)
    assert runtime["judge_max_attempts"] == VIDEO_RL.judge_max_attempts
    assert {key: runtime[key] for key in h3.RECOVERY_DEFAULTS} == dict(h3.RECOVERY_DEFAULTS)
    assert all(VIDEO_RL[key] == default for key, default in h3.RECOVERY_DEFAULTS.items())
    assert runtime["max_judge_fallback_fraction"] == VIDEO_RL.max_judge_fallback_fraction == 0.75
    assert runtime["protocol_sha256"] == fingerprint(frozen) == read_json(directory / "protocol-sha256.json")
    assert tournament.runtime_protocol(runtime, VIDEO_RL)["protocol_sha256"] == fingerprint(frozen)
    # Limits frozen by older runs are replaced, not merged.
    legacy = {**frozen, "timeouts": {"llm": 1, "unmeasured": 2}}
    assert tournament.runtime_protocol(legacy, VIDEO_RL)["timeouts"] == dict(VIDEO_RL.timeouts)


@pytest.mark.parametrize(
    "timeouts,attempts",
    [
        ({"unknown": 1}, 2),
        ({"llm": None}, 2),
        ({"llm": 0}, 2),
        ({"llm": float("inf")}, 2),
        ({"llm": True}, 2),
        ({"llm": "240"}, 2),
        ({}, 0),
        ({}, 1.5),
    ],
)
def test_runtime_settings_are_validated(prepared_rl, timeouts, attempts):
    complete = {**dict(VIDEO_RL.timeouts), **timeouts}
    if timeouts.get("llm", 0) is None:
        complete.pop("llm")
    with pytest.raises(ValueError, match="video_rl"):
        tournament.runtime_protocol(prepared_rl[1], {"timeouts": complete, "judge_max_attempts": attempts})


def test_rl_config_timeouts_reach_the_judge_runner(prepared_rl, tmp_path, monkeypatch):
    directory, protocol, prepared_args = prepared_rl
    config, _ = rl.training_config(
        rl.parse_args(["train", "--run-dir", str(directory), "--data-dir", str(prepared_args.data_dir)])
    )
    runtime = tournament.runtime_protocol(tournament.load_protocol(directory / "protocol.json"), config.video_rl)
    assert runtime["timeouts"] == protocol["timeouts"] and runtime["judge_max_attempts"] == 3
    assert runtime["judge_failure_policy"] == "neutral"
    runtime["judge_failure_policy"] = "drop"
    calls = []

    def execute(command, **kwargs):
        calls.append((command, kwargs["timeout"]))
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(tournament.subprocess, "run", execute)
    with pytest.raises(ProviderError, match="exceeded 300s"):
        tournament.score_group(
            directory / "group", runtime, protocol["samples"]["train"][0], candidates(tmp_path), validate=False
        )
    command, timeout = calls[0]
    assert timeout == 300
    assert command[command.index("--max-attempts") + 1] == "3"
    assert float(command[command.index("--request-timeout") + 1]) == 45
    assert float(command[command.index("--file-timeout") + 1]) == 60
    assert float(command[command.index("--file-control-timeout") + 1]) == 15


def test_h3_worker_receives_runtime_limit_and_frozen_protocol_identity(prepared_rl, monkeypatch):
    directory, protocol, _ = prepared_rl
    protocol = {**protocol, "h3_max_request_attempts": 5, "h3_max_generation_attempts": 3}
    calls = []

    def run(command, **kwargs):
        calls.append(kwargs["timeout"])
        assert command[command.index("--max-request-attempts") + 1] == "5"
        assert command[command.index("--max-generation-attempts") + 1] == "3"
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(tournament.subprocess, "run", run)
    with pytest.raises(ProviderError, match="150"):
        tournament.generate_video(
            directory / "candidate", protocol, {"prompt": "A valid video prompt", "duration": 10, "seed": 42}
        )
    assert calls == [150]
    worker_protocol = read_json(directory / "candidate/protocol.json")
    assert worker_protocol["rl_protocol_sha256"] == read_json(directory / "protocol-sha256.json")
    assert worker_protocol["timeouts"] == {"h3_request": 30}
    expected = {"h3", "h3_endpoints", "media_profile", "eval_repo", "timeouts", "rl_protocol_sha256"}
    assert set(worker_protocol) == expected


@pytest.mark.parametrize("error_stage", ["configuration", "protocol", "artifact"])
def test_h3_fatal_errors_cannot_be_hidden_by_unknown_submission_status(prepared_rl, monkeypatch, error_stage):
    directory, protocol, _ = prepared_rl

    def run(command, **kwargs):
        write_json(directory / "candidate/state.json", {"status": "submit_unknown", "error_stage": error_stage})
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(tournament.subprocess, "run", run)
    with pytest.raises(RuntimeError) as raised:
        tournament.generate_video(directory / "candidate", protocol, {"prompt": "Video", "duration": 10, "seed": 42})
    assert not isinstance(raised.value, ProviderError)


@pytest.mark.parametrize(
    "error_type,error,reason",
    [
        ("NetworkFailure", "Network request failed: The read operation timed out", "provider_failure"),
        ("HttpFailure", 'HTTP 400: {"error":{"code":"prohibited_content"}}', "content_refusal"),
        (
            "HttpFailure",
            'HTTP 400: {"error":{"message":"Request contains an invalid argument.","code":"invalid_request"}}',
            "judge_request_rejected",
        ),
    ],
)
def test_training_preserves_votes_and_videos_with_a_persisted_neutral_order(
    prepared_rl, tmp_path, monkeypatch, error_type, error, reason
):
    directory, protocol, _ = prepared_rl
    clips = candidates(tmp_path)
    before = {c["path"]: Path(c["path"]).read_bytes() for c in clips}
    calls = install_judge(monkeypatch)
    execute = tournament.subprocess.run

    def fail_one(command, **kwargs):
        assert command[command.index("--max-errors") + 1] == "6"
        execute(command, **kwargs)
        path = Path(command[command.index("--output") + 1])
        if path.name == "judgments-original.jsonl":
            rows = read_jsonl(path)
            row = next(r for r in rows if r["pair_id"] == "0_vs_1")
            for field in ("preferred_video", "preferred_model", "reason"):
                row.pop(field)
            row.update(status="error", stage="interaction", error_type=error_type, error=error)
            write_jsonl(path, rows)
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(tournament.subprocess, "run", fail_one)
    monkeypatch.setattr(tournament, "generate_video", lambda *a, **k: pytest.fail("must reuse existing H3 videos"))
    group = directory / "neutral-group"
    rewards = tournament.score_group(group, protocol, protocol["samples"]["train"][0], clips, validate=False)
    assert [r["score"] for r in rewards] == pytest.approx([11 / 12, 3 / 4, 1 / 3, 0])
    assert all(r["judge_fallback_comparisons"] == 1 and r["judge_completed_comparisons"] == 11 for r in rewards)
    assert all(r["order_disagreement"] == 0 for r in rewards)
    fallback = read_json(group / "judge/fallbacks-original.json")
    assert len(fallback) == 1 and fallback[0]["fallback_reason"] == reason
    assert fallback[0]["status"] == "fallback" and "preferred_model" not in fallback[0]
    assert fallback[0]["error"]["error"] == error
    assert len(calls) == 2
    assert tournament.score_group(group, protocol, protocol["samples"]["train"][0], clips, validate=False) == rewards
    assert len(calls) == 2
    assert before == {c["path"]: Path(c["path"]).read_bytes() for c in clips}
    # Fallback identity is verified just like an actual vote before resuming.
    fallback[0]["manifest_sha256"] = "wrong-manifest"
    write_json(group / "judge/fallbacks-original.json", fallback)
    with pytest.raises(ValueError, match="frozen manifest"):
        tournament.score_group(group, protocol, protocol["samples"]["train"][0], clips, validate=False)


@pytest.mark.parametrize("invalid", [(), (3,)])
def test_pass_deadlines_neutralize_only_judging_and_keep_invalid_call_penalties(
    prepared_rl, tmp_path, monkeypatch, invalid
):
    directory, protocol, _ = prepared_rl
    calls = []

    def timeout(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 300
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(tournament.subprocess, "run", timeout)
    clips = candidates(tmp_path, invalid)
    group = directory / "timeout-group"
    rewards = tournament.score_group(group, protocol, protocol["samples"]["train"][0], clips, validate=False)
    assert [r["score"] for r in rewards] == pytest.approx([2 / 3] * 3 + [-1] if invalid else [0.5] * 4)
    assert all(r["judge_fallback_fraction"] == 1 and r["order_disagreement"] == 0 for r in rewards)
    assert all(r["zero_variance_group"] == (not invalid) for r in rewards)
    assert len(calls) == 2
    assert tournament.score_group(group, protocol, protocol["samples"]["train"][0], clips, validate=False) == rewards
    assert len(calls) == 2


@pytest.mark.parametrize("exit_mode", ["error", "timeout"])
def test_auth_failure_is_fatal_even_when_the_pass_later_times_out(prepared_rl, tmp_path, monkeypatch, exit_mode):
    directory, protocol, _ = prepared_rl
    install_judge(monkeypatch)
    execute = tournament.subprocess.run

    def unauthorized(command, **kwargs):
        execute(command, **kwargs)
        path = Path(command[command.index("--output") + 1])
        rows = read_jsonl(path)
        row = rows[0]
        for field in ("preferred_video", "preferred_model", "reason"):
            row.pop(field)
        row.update(status="error", stage="interaction", error_type="HttpFailure", error="HTTP 401: unauthorized")
        write_jsonl(path, rows)
        if exit_mode == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(tournament.subprocess, "run", unauthorized)
    group = directory / "auth-error"
    with pytest.raises(RuntimeError, match="Nonrecoverable Gemini error"):
        tournament.score_group(group, protocol, protocol["samples"]["train"][0], candidates(tmp_path), validate=False)
    assert not list(group.rglob("fallbacks-*.json")) and not (group / "rewards.json").exists()


def test_validation_never_uses_training_fallbacks(prepared_rl, tmp_path, monkeypatch):
    directory, protocol, _ = prepared_rl

    def timeout(command, **kwargs):
        assert command[command.index("--max-errors") + 1] == "1"
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(tournament.subprocess, "run", timeout)
    group = directory / "eval-timeout"
    with pytest.raises(ProviderError):
        tournament.score_group(group, protocol, protocol["samples"]["val"][0], candidates(tmp_path)[:1], validate=True)
    assert not list(group.rglob("fallbacks-*.json")) and not (group / "rewards.json").exists()


@pytest.mark.parametrize("limit", [0.25, 0.75])
def test_fallback_guard_counts_groups_once_across_replacements(native_worker, tmp_path, limit):
    from examples.video_agents import agent

    worker, _, _ = native_worker
    manager = object.__new__(agent.TournamentManager)
    manager.config = worker.config
    manager.config.video_rl.output_dir = str(tmp_path)
    manager.config.video_rl.max_judge_fallback_fraction = limit
    manager.config.data.train_batch_size = 4

    def record(fallbacks):
        return {"dropped": False, "rewards": [{"judge_comparisons": 12, "judge_fallback_comparisons": fallbacks}] * 4}

    neutral_groups = int(limit * 4)
    # A partial dispatch can be entirely neutral; the completed batch decides.
    manager._check_judge_coverage(1, [record(12)] * neutral_groups + [{"dropped": True}])
    manager._check_judge_coverage(1, [record(0)] * (4 - neutral_groups))
    assert (manager._judge_comparisons, manager._judge_fallbacks) == (48, 12 * neutral_groups)
    manager._check_judge_coverage(2, [record(12)] * neutral_groups + [record(0)] * (3 - neutral_groups))
    with pytest.raises(RuntimeError, match="stopping before an optimizer update"):
        # One additional missing judgment crosses the boundary.
        manager._check_judge_coverage(2, [record(1)])
    manager._check_judge_coverage(3, [record(0)] * 4)
    assert manager._judge_fallbacks == 0


@pytest.mark.parametrize("value", [-0.1, 1.1, True, float("nan"), "0.25"])
def test_fallback_fraction_limit_is_validated(prepared_rl, value):
    settings = OmegaConf.to_container(VIDEO_RL, resolve=False)
    settings["max_judge_fallback_fraction"] = value
    with pytest.raises(ValueError, match="fallback_fraction"):
        tournament.runtime_protocol(prepared_rl[1], settings)
