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

from examples.video_agents import h3, judge, rl, self_hosted_h3, tournament
from examples.video_agents.config import (
    CANDIDATE,
    DEFAULT_SELF_HOSTED_REFERENCES,
    JUDGE_SETTINGS,
    MAX_REFERENCE_ENDPOINT,
    MAX_REFERENCE_SETTINGS,
    MAX_REFERENCE_VIDEO_FIELDS,
    SELF_HOSTED_REFERENCE_VIDEO_FIELDS,
)
from examples.video_agents.dataset import reference_request
from examples.video_agents.judge import EXTERNAL_FILES
from examples.video_agents.storage import file_sha256, fingerprint, read_json, read_jsonl, write_json, write_jsonl

MEDIA = {
    "duration_s": 10.125,
    "video": {"width": 1344, "height": 768, "fps": 24.0, "codec": "avc1"},
    "audio": {"sample_rate": 32000, "channels": 2, "codec": "mp4a"},
}
HOST = "https://h3.invalid"


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
    rows, tasks = [], []
    for i, source in enumerate(["aa_eval", "movie_gen_video_bench"]):
        row = {"id": f"eval-{i}", "prompt": f"Original {i}", "source": source, "h3_context_ir": f"SECRET {i}"}
        for field in ("prompt", "h3_context_ir"):
            video = references / f"{i}-{field}.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"reference {i} {field}".encode())
            state = video.with_suffix(".json")
            request = reference_request(row[field], {"settings": MAX_REFERENCE_SETTINGS})
            write_json(
                state,
                {
                    "status": "succeeded",
                    "request_sha256": fingerprint(request),
                    "endpoint": MAX_REFERENCE_ENDPOINT,
                    "host": HOST,
                    "sha256": file_sha256(video),
                    "bytes": video.stat().st_size,
                    "media": MEDIA,
                },
            )
            tasks.append(
                {
                    "item_id": row["id"],
                    "field": field,
                    "request": request,
                    "request_sha256": fingerprint(request),
                    "output": video.name,
                    "state": state.name,
                }
            )
            row[MAX_REFERENCE_VIDEO_FIELDS[field]] = str(video)
        rows.append(row)
    write_json(
        references / "manifest.json",
        {
            "settings": MAX_REFERENCE_SETTINGS,
            "endpoint": MAX_REFERENCE_ENDPOINT,
            "context_ir_rewriter_enabled": False,
            "tasks": tasks,
        },
    )
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
    return directory, tournament.load_protocol(directory / "protocol.json"), args


def candidates(tmp_path, invalid=()):
    result = []
    for i in range(4):
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

    def execute(command, *, check, timeout):
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
        assert check is True and timeout == 1800
        assert command[command.index("--max-attempts") + 1] == "1"
        assert command[command.index("--request-timeout") + 1] == "300"
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


@pytest.mark.parametrize("failure", ["missing", "fallback", "provider"])
def test_no_incomplete_tournament_can_receive_rewards(prepared_rl, tmp_path, failure):
    _, protocol, _ = prepared_rl
    clips = candidates(tmp_path)
    manifests = tournament.tournament_manifests(protocol["samples"]["train"][0], clips, protocol)
    judgments = votes(manifests)
    if failure == "missing":
        judgments["swapped"].pop("0_vs_1")
    elif failure == "fallback":
        judgments["swapped"]["0_vs_1"]["status"] = "fallback"
    else:
        clips[0]["status"] = "infrastructure_error"
    with pytest.raises(RuntimeError):
        tournament.tournament_rewards(clips, manifests, judgments)


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
    assert rewards[0]["vs_prompt"] == rewards[0]["vs_h3_context_ir"] == 1
    assert sum(c.count("--pair-id") for c in calls) == 4
    manifest = read_json(directory / "eval-group/judge/manifest-original.json")
    assert {p["reference"] for p in manifest["pairs"].values()} == {"prompt", "h3_context_ir"}
    assert all(p["prompt"] == "Original 0" for p in manifest["pairs"].values())


@pytest.mark.parametrize("stage", ["h3", "judge"])
def test_hard_timeouts_raise_without_neutral_fallbacks(prepared_rl, tmp_path, monkeypatch, stage):
    directory, protocol, _ = prepared_rl
    calls = []

    def timeout(command, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(tournament.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        if stage == "h3":
            tournament.generate_video(
                directory / "candidate", protocol, {"prompt": "A valid video prompt", "duration": 10, "seed": 42}
            )
        else:
            tournament.score_group(
                directory / "group", protocol, protocol["samples"]["train"][0], candidates(tmp_path), validate=False
            )
    assert calls == [{"check": True, "timeout": 1800}]
    assert not list(directory.rglob("*fallback*"))
    assert not list(directory.rglob("rewards.json"))


@pytest.mark.parametrize("duration,seed", [(5, 0), (10, 9223372036854775807)])
def test_h3_adapter_reuses_existing_worker_and_validates_artifacts(prepared_rl, monkeypatch, duration, seed):
    directory, protocol, _ = prepared_rl
    posts = []
    job_id = "338a55c9-c8f3-4dc4-982b-df72221cd893"
    client = SimpleNamespace(
        submit=lambda origin, request, **kw: (posts.append(request) or {"id": job_id}),
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
        assert self_hosted_h3.generate_one(
            task, protocol, client, lambda path: {**MEDIA, "duration_s": duration + 0.125}
        )
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
    changed.h3_timeout = 1
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


def test_prepare_self_hosted_rl_keeps_tool_and_max_reference_provenance(prepared_rl, tmp_path):
    _, original, args = prepared_rl
    args = copy.copy(args)
    args.run_dir = tmp_path / "self-hosted-rl"
    args.data_dir = tmp_path / "self-hosted-data"
    args.h3_backend = "self-hosted"
    args.h3_endpoints = "http://worker-a:18080,http://worker-b:18080"
    args.poll_interval = 1
    args.h3_concurrency_per_worker = 2
    assert rl.prepare(args)["network_requests"] == 0
    protocol = tournament.load_protocol(args.run_dir / "protocol.json")
    schema = read_jsonl(args.data_dir / "train.jsonl")[0]["tools"][0]
    assert schema["function"]["name"] == "sol_h3"
    assert schema["function"]["parameters"]["properties"]["prompt"]["maxLength"] == 8000
    assert protocol["h3"]["model"] == "MiniMax-H3-Base-FL2VA"
    assert protocol["reference_h3"]["settings"] == MAX_REFERENCE_SETTINGS
    assert protocol["reference_h3"]["backend"] == "max"
    assert protocol["reference_video_fields"] == MAX_REFERENCE_VIDEO_FIELDS
    assert protocol["samples"]["val"] == original["samples"]["val"]
    assert "duration" not in protocol["h3"] and "seed" not in protocol["h3"]


def test_rl_defaults_use_self_hosted_references(tmp_path):
    args = rl.parse_args(["prepare", "--run-dir", str(tmp_path)])
    assert args.references == DEFAULT_SELF_HOSTED_REFERENCES
    assert args.h3_backend == "self-hosted"
    assert (args.groups_per_worker, args.h3_concurrency_per_worker, args.poll_interval) == (1, 2, 1)


@pytest.fixture
def hosted_references(prepared_rl, tmp_path):
    _, _, args = prepared_rl
    args = copy.copy(args)
    args.run_dir, args.data_dir = tmp_path / "hosted-run", tmp_path / "hosted-data"
    manifest = read_json(args.references / "manifest.json")
    manifest.update(
        h3_backend="self-hosted",
        h3_endpoints=["http://reference-gateway:18080"],
        h3=self_hosted_h3.SETTINGS,
        settings=self_hosted_h3.SETTINGS,
        endpoint=self_hosted_h3.ENDPOINT,
        media_profile=h3.media_profile(MEDIA),
    )
    rows = read_jsonl(args.eval_dataset)
    by_id = {row["id"]: row for row in rows}
    for index, task in enumerate(manifest["tasks"]):
        row = by_id[task["item_id"]]
        task["request"] = self_hosted_h3.request_body(row[task["field"]], manifest["settings"])
        task["request_sha256"] = fingerprint(task["request"])
        state_path = args.references / task["state"]
        state = read_json(state_path)
        state.update(
            self_hosted_h3.identity(task, manifest),
            worker_url=manifest["h3_endpoints"][0],
            task_id=f"reference-job-{index}",
        )
        write_json(state_path, state)
        row[SELF_HOSTED_REFERENCE_VIDEO_FIELDS[task["field"]]] = str(args.references / task["output"])
    write_json(args.references / "manifest.json", manifest)
    write_jsonl(args.eval_dataset, rows)
    return args


def test_prepare_rl_records_self_hosted_reference_provenance(hosted_references):
    args = hosted_references
    assert rl.prepare(args)["network_requests"] == 0
    protocol = tournament.load_protocol(args.run_dir / "protocol.json")
    assert protocol["reference_video_fields"] == SELF_HOSTED_REFERENCE_VIDEO_FIELDS
    assert protocol["reference_h3"] == {
        "backend": "self-hosted",
        "settings": self_hosted_h3.SETTINGS,
        "endpoint": "/v1/videos",
        "endpoints": ["http://reference-gateway:18080"],
    }
    assert protocol["h3_endpoints"] == [HOST]
    assert protocol["h3"] == {
        key: value for key, value in protocol["reference_h3"]["settings"].items() if key not in {"duration", "seed"}
    }
    for split in ("train", "val"):
        row = read_jsonl(args.data_dir / f"{split}.jsonl")[0]
        schema = row["tools"][0]["function"]
        assert "Sol-H3" in schema["description"]
        assert "10 seconds" not in schema["description"]
        assert set(schema["parameters"]["properties"]) == {"prompt", "duration", "seed"}
        assert schema["parameters"]["properties"]["prompt"]["maxLength"] == 8000
        assert "SECRET" not in json.dumps(row)


@pytest.mark.parametrize("invalid", ["backend", "settings", "pool", "request", "state_media", "video_path", "bytes"])
def test_prepare_rejects_mismatched_reference_provenance(hosted_references, invalid):
    args = hosted_references
    manifest = read_json(args.references / "manifest.json")
    task = manifest["tasks"][0]
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
        rows[0][SELF_HOSTED_REFERENCE_VIDEO_FIELDS[task["field"]]] = "/wrong/video.mp4"
    else:
        (args.references / task["output"]).write_bytes(b"changed reference clip")
    write_json(args.references / "manifest.json", manifest)
    write_json(state_path, state)
    write_jsonl(args.eval_dataset, rows)
    with pytest.raises(ValueError):
        rl.prepare(args)
    assert not (args.run_dir / "protocol.json").exists()


@pytest.mark.parametrize("changed", ["candidate", "reference", "evaluator"])
def test_changed_artifacts_stop_before_judging(prepared_rl, tmp_path, monkeypatch, changed):
    directory, protocol, _ = prepared_rl
    sample = protocol["samples"]["val"][0]
    clips = candidates(tmp_path)[:1]
    if changed == "candidate":
        path = Path(clips[0]["path"])
    elif changed == "reference":
        path = Path(sample["references"]["prompt"]["path"])
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
        {"valid_call": 0.0, "judge_comparisons": 0, "vs_prompt": 0.0, "vs_h3_context_ir": 0.0, "score": 0.0}
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
    with pytest.raises(ValueError, match="four training rollouts"):
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

        if hang:
            tool_list[0].protocol["timeouts"]["llm"] = 0.01
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
    assert output.extra_fields["h3_result"] == {"status": "invalid_call"}
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
    "text",
    [
        "I cannot generate a video.",
        VALID_CALL.replace("function=sol_h3", "function=unknown"),
        VALID_CALL.replace("function=sol_h3", "function=h3_max"),
        VALID_CALL.replace("A clear scene", " "),
        VALID_CALL.replace("<parameter=prompt>", "<parameter=unexpected>"),
        VALID_CALL + VALID_CALL,
    ],
)
async def test_invalid_native_calls_never_invoke_h3(loop_factory, model_request, text):
    make, generated = loop_factory
    output = await make(text).run({}, **model_request)
    assert output.extra_fields["h3_result"] == {"status": "invalid_call"}
    assert not generated


@pytest.mark.asyncio
async def test_token_budget_exhaustion_is_a_policy_failure(loop_factory, model_request):
    make, generated = loop_factory
    loop = make("Still reasoning", stop_reason="length")
    loop.response_length = len("Still reasoning")
    output = await loop.run({}, **model_request)
    assert output.extra_fields["h3_result"] == {"status": "invalid_call"}
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
    with pytest.raises((RuntimeError, TimeoutError)):
        await loop.run({}, **model_request)
    assert not generated


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
@pytest.mark.parametrize("failure", [None, "rollout", "judge"])
async def test_native_dispatch_waits_for_group_rewards_and_propagates_failures(native_worker, monkeypatch, failure):
    import numpy as np

    from examples.video_agents import agent
    from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
    from verl.utils.tensordict_utils import assign_non_tensor_data, get_tensordict

    worker, events, published = native_worker
    settled = set()

    async def rollout(*args, session_id, **kwargs):
        await asyncio.sleep(session_id * 0.001)
        settled.add(session_id)
        if failure == "rollout" and session_id == 0:
            raise RuntimeError("LLM interrupted")
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
        assert settled == set(range(4)) and not published
        events.append("score")
        if failure == "judge":
            raise RuntimeError("Gemini interrupted")
        return [{"score": s, "valid_call": 1.0} for s in [1, 2 / 3, 1 / 3, -1]]

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
    if failure:
        with pytest.raises(RuntimeError, match="interrupted"):
            await worker.generate_sequences(batch)
        assert settled == set(range(4))
        assert not published and events[-1] == "failure"
    else:
        await worker.generate_sequences(batch)
        assert events == ["running", "score", *(["publish"] * 4), "finished"]
        for i, row in enumerate(published):
            assert row["keys"] == [f"uid0_{i}_0"]
            fields = row["fields"]
            assert fields["rm_scores"][0].sum().item() == pytest.approx([1, 2 / 3, 1 / 3, -1][i])
            assert fields["loss_mask"][0].tolist() == [1, 1, 0]
            assert fields["rollout_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2, 0])
            assert fields["extra_fields"][0]["reward_extra_info"]["valid_call"] == 1
