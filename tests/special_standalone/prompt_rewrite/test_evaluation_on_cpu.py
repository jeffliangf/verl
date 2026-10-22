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
"""Exercise the full workflow without GPUs, credentials, or network calls."""

import copy
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from examples.prompt_rewrite import evaluation, h3, judge, rewrite, scoring, self_hosted_h3
from examples.prompt_rewrite.__main__ import main
from examples.prompt_rewrite.config import (
    CANDIDATE,
    H3_ENDPOINT,
    H3_SETTINGS,
    JUDGE_SETTINGS,
    REFERENCE_VIDEO_FIELDS,
    SELF_HOSTED_REFERENCE_VIDEO_FIELDS,
)
from examples.prompt_rewrite.evaluation import EXTERNAL_FILES, EvaluationRun
from examples.prompt_rewrite.storage import file_sha256, fingerprint, read_json, write_json, write_jsonl

MEDIA = {
    "duration_s": 10.125,
    "video": {"width": 1344, "height": 768, "fps": 24.0, "codec": "avc1"},
    "audio": {"sample_rate": 32000, "channels": 2, "codec": "mp4a"},
}
HOST = "https://h3.invalid"


class FakeH3:
    def __init__(self):
        self.posts = []
        self.queries = []
        self.fail_post = False
        self.fail_query = False

    def submit(self, origin, request, **kwargs):
        self.posts.append((origin + self_hosted_h3.ENDPOINT, request))
        if self.fail_post:
            raise TimeoutError("ambiguous submission")
        return {"id": str(UUID(int=len(self.posts)))}

    def get_json(self, url, **kwargs):
        if url.endswith("/healthz"):
            return {"status": "ready", "queued": 0, "queue_capacity": 8}
        self.queries.append(url)
        if self.fail_query:
            raise OSError("temporary polling failure")
        return {"id": url.rsplit("/", 1)[-1], "status": "succeeded"}

    def download(self, origin, job_id, path, **kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"candidate clip")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    repo, references = tmp_path / "eval-repo", tmp_path / "references"
    for relative in EXTERNAL_FILES:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fake integration module\n")
    rows, tasks = [], []
    for i in range(3):
        row = {
            "id": f"sample:{i}",
            "prompt": f"Original prompt {i}",
            "h3_context_ir": f"SECRET teacher target {i}",
            "source": "aa_eval" if i < 2 else "movie_gen_video_bench",
        }
        for field in ("prompt", "h3_context_ir"):
            video = references / "videos" / str(i) / (field + ".mp4")
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(f"reference {i}/{field}".encode())
            request = h3.request_body(row[field])
            state = video.with_suffix(".state.json")
            write_json(
                state,
                {
                    "status": "succeeded",
                    "request_sha256": fingerprint(request),
                    "endpoint": H3_ENDPOINT,
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
                    "state": str(state.relative_to(references)),
                    "output": str(video.relative_to(references)),
                }
            )
            row[REFERENCE_VIDEO_FIELDS[field]] = str(video)
            # Legacy video fields coexist, but must never be used for this run.
            row[field + "_h3_video"] = str(tmp_path / "legacy" / (field + ".mp4"))
        rows.append(row)
    dataset = tmp_path / "eval.jsonl"
    write_jsonl(dataset, rows)
    write_json(
        references / "manifest.json",
        {"settings": H3_SETTINGS, "endpoint": H3_ENDPOINT, "context_ir_rewriter_enabled": False, "tasks": tasks},
    )
    client = FakeH3()

    def external(path):
        if path.name == "build_pairwise_manifest.py":
            return SimpleNamespace(INSTRUCTIONS_VERSION=4, INSTRUCTIONS="Frozen comparison rubric")
        if path.name == "normalize.py":
            return SimpleNamespace(probe=lambda _: MEDIA)
        raise AssertionError(f"Unexpected module: {path}")

    monkeypatch.setattr(evaluation, "load_external", external)
    monkeypatch.setattr(self_hosted_h3.Client, "from_environment", classmethod(lambda cls, protocol=None: client))
    monkeypatch.setenv("H3_SELF_HOSTED_ENDPOINTS", HOST)
    monkeypatch.setenv("H3_SELF_HOSTED_TOKEN", "test-token")
    monkeypatch.setenv("H3_HOST", HOST)
    monkeypatch.setenv("H3_TOKEN", "test-token")
    monkeypatch.setenv("TENSORBOARD_DIR", str(tmp_path / "tensorboard"))
    run = EvaluationRun.prepare(tmp_path / "run", dataset=dataset, references=references, eval_repo=repo)
    return run, client, rows


def import_outputs(run):
    run.import_rewrites(
        [
            {"id": sample["id"], "rewritten_prompt": f"Rewrite for {sample['prompt']}"}
            for sample in run.protocol["samples"]
        ]
    )


@pytest.fixture
def hosted_references(prepared, tmp_path):
    source, client, rows = prepared
    references = tmp_path / "hosted-references"
    shutil.copytree(Path(source.protocol["reference_manifest"]).parent, references)
    manifest = read_json(references / "manifest.json")
    manifest.update(
        h3_backend="self-hosted",
        h3_endpoints=["http://reference-a:18080", "http://reference-b:18080"],
        h3=copy.deepcopy(self_hosted_h3.SETTINGS),
        settings=copy.deepcopy(self_hosted_h3.SETTINGS),
        endpoint=self_hosted_h3.ENDPOINT,
        media_profile=h3.media_profile(MEDIA),
    )
    rows = copy.deepcopy(rows)
    by_id = {row["id"]: row for row in rows}
    for index, task in enumerate(manifest["tasks"]):
        row = by_id[task["item_id"]]
        task["request"] = self_hosted_h3.request_body(row[task["field"]], manifest["settings"])
        task["request_sha256"] = fingerprint(task["request"])
        state_path = references / task["state"]
        state = read_json(state_path)
        state.pop("host")
        state.update(
            self_hosted_h3.identity(task, manifest),
            worker_url=manifest["h3_endpoints"][index % 2],
            task_id=str(UUID(int=index + 100)),
        )
        write_json(state_path, state)
        row[SELF_HOSTED_REFERENCE_VIDEO_FIELDS[task["field"]]] = str(references / task["output"])
    write_json(references / "manifest.json", manifest)
    dataset = tmp_path / "hosted.jsonl"
    write_jsonl(dataset, rows)
    return dict(dataset=dataset, references=references, eval_repo=Path(source.protocol["eval_repo"]))


def test_self_hosted_references_keep_pool_and_model_provenance(hosted_references, tmp_path):
    run = EvaluationRun.prepare(tmp_path / "hosted-run", **hosted_references)
    protocol = run.protocol
    assert protocol["h3_backend"] == "self-hosted"
    assert protocol["h3_endpoints"] == [HOST]
    assert protocol["reference_video_fields"] == SELF_HOSTED_REFERENCE_VIDEO_FIELDS
    assert protocol["reference_h3"] == {
        "backend": "self-hosted",
        "settings": self_hosted_h3.SETTINGS,
        "endpoint": "/v1/videos",
        "endpoints": ["http://reference-a:18080", "http://reference-b:18080"],
    }
    assert "scripts/generate_h3_repeats.py" not in protocol["dependencies"]
    import_outputs(run)
    assert run.generate(execute=True, concurrency=1)["video_statuses"] == {"succeeded": 3}
    assert len(run.comparison_inputs()["original"]["pairs"]) == 6


@pytest.mark.parametrize("invalid", ["missing_field", "max_path", "pool", "worker", "settings", "request", "bytes"])
def test_self_hosted_references_reject_mismatches_without_using_max(hosted_references, tmp_path, invalid):
    dataset, references = hosted_references["dataset"], hosted_references["references"]
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    manifest = read_json(references / "manifest.json")
    task = manifest["tasks"][0]
    state_path = references / task["state"]
    state = read_json(state_path)
    if invalid == "missing_field":
        del rows[0]["prompt_h3_self_hosted_video"]
    elif invalid == "max_path":
        rows[0]["prompt_h3_self_hosted_video"] = rows[0]["prompt_h3_max_video"]
    elif invalid == "pool":
        state["endpoints"] = ["http://unrelated:18080"]
    elif invalid == "worker":
        state["worker_url"] = "http://unrelated:18080"
    elif invalid == "settings":
        manifest["settings"]["seed"] = 99
    elif invalid == "request":
        task["request"]["prompt"] = "Different reference prompt"
    else:
        (references / task["output"]).write_bytes(b"changed video")
    write_jsonl(dataset, rows)
    write_json(references / "manifest.json", manifest)
    write_json(state_path, state)
    with pytest.raises(ValueError):
        EvaluationRun.prepare(tmp_path / "invalid-hosted", **hosted_references)


@pytest.mark.parametrize("backend", [None, "max"])
@pytest.mark.parametrize("neutral", [False, True])
def test_legacy_runs_reject_generation_before_credentials_or_fallbacks(prepared, monkeypatch, backend, neutral):
    run, client, _ = prepared
    run.protocol.pop("h3_backend")
    if backend:
        run.protocol["h3_backend"] = backend
    run.protocol["h3"] = copy.deepcopy(H3_SETTINGS)
    write_json(run.path / "protocol.json", run.protocol)
    write_json(run.path / "protocol-sha256.json", fingerprint(run.protocol))
    monkeypatch.setattr(evaluation, "load_external", lambda *a: pytest.fail("Loaded a provider client"))
    monkeypatch.setattr(h3.subprocess, "run", lambda *a, **k: pytest.fail("Started a generation worker"))
    monkeypatch.setattr(
        self_hosted_h3.Client, "from_environment", classmethod(lambda *a: pytest.fail("Read credentials"))
    )
    assert run.status()["rewrites_complete"] == 0
    with pytest.raises(ValueError, match="H3 Max API is disabled"):
        run.generate(execute=True, neutral_on_failure=neutral)
    monkeypatch.setattr("sys.argv", ["h3", "--run-dir", str(run.path)])
    with pytest.raises(ValueError, match="H3 Max API is disabled"):
        h3.main()
    assert not (run.path / "generation.json").exists()
    assert not (run.path / "videos/fallbacks.json").exists()
    assert not client.posts


def test_completed_legacy_artifacts_remain_readable(prepared):
    run, client, _ = prepared
    run.protocol.pop("h3_backend")
    run.protocol["h3"] = copy.deepcopy(H3_SETTINGS)
    write_json(run.path / "protocol.json", run.protocol)
    write_json(run.path / "protocol-sha256.json", fingerprint(run.protocol))
    import_outputs(run)
    for task in run.generation_tasks():
        path = Path(task["output"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"historical Max clip")
        write_json(
            Path(task["state"]),
            {
                **h3.task_identity(task, run.protocol),
                "status": "succeeded",
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
                "media": MEDIA,
            },
        )
    add_judgments(run, run.comparison_inputs())
    assert run.report(log_tensorboard=False)["complete"]
    assert not client.posts


def test_self_hosted_evaluation_uses_distinct_protocol_and_generation_path(prepared, tmp_path, monkeypatch):
    from examples.prompt_rewrite import self_hosted_h3

    source, paid_client, _ = prepared
    run = EvaluationRun.prepare(
        tmp_path / "self-hosted",
        dataset=Path(source.protocol["dataset"]),
        references=Path(source.protocol["reference_manifest"]).parent,
        eval_repo=Path(source.protocol["eval_repo"]),
        h3_backend="self-hosted",
        h3_endpoints="http://worker:18080",
    )
    import_outputs(run)
    tasks = run.generation_tasks()
    assert tasks[0]["request"] == {"prompt": "Rewrite for Original prompt 0", "duration": 10, "seed": 42}
    calls = []

    def generate(actual_tasks, protocol, probe, concurrency, poll_interval, timeout):
        calls.append(actual_tasks)
        for task in actual_tasks:
            output = Path(task["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"self-hosted clip")
            write_json(
                Path(task["state"]),
                {
                    **h3.task_identity(task, protocol),
                    "status": "succeeded",
                    "sha256": file_sha256(output),
                    "bytes": output.stat().st_size,
                    "media": MEDIA,
                },
            )
        return {task["id"]: True for task in actual_tasks}

    monkeypatch.setattr(self_hosted_h3, "generate_tasks", generate)
    result = run.generate(execute=True, concurrency=8, poll_interval=1)
    assert result["videos_accounted_for"] == 3
    assert len(run._generated_candidates()) == 3
    assert len(calls) == 1 and not paid_client.posts
    assert run.protocol["reference_h3"]["settings"] == H3_SETTINGS


def test_prepare_cli_keeps_vllm_as_default_backend(prepared, tmp_path):
    run, _, _ = prepared
    destination = tmp_path / "cli-run"
    assert (
        main(
            [
                "prepare",
                "--run-dir",
                str(destination),
                "--dataset",
                run.protocol["dataset"],
                "--references",
                str(Path(run.protocol["reference_manifest"]).parent),
                "--eval-repo",
                run.protocol["eval_repo"],
            ]
        )
        == 0
    )
    assert EvaluationRun(destination).protocol["rewriter"]["backend"] == "vllm"


def add_judgments(run, manifests, *, omit=None):
    """Three prompts score 1, 0.5, 0 against each reference, independent of displayed side."""
    candidate = run.protocol.get("candidate_field", CANDIDATE)
    for order in judge.ORDERS:
        rows = []
        path = run.path / "judge" / f"manifest-{order}.json"
        for pair_id, pair in manifests[order]["pairs"].items():
            if (order, pair_id) == omit:
                continue
            i = int(pair["prompt_id"].split(":")[-1])
            candidate_wins = i == 0 or (i == 1 and order == "original")
            winner = candidate if candidate_wins else pair["reference"]
            preferred = "A" if pair["model_A"] == winner else "B"
            rows.append(
                {
                    "status": "completed",
                    "pair_id": pair_id,
                    "prompt_id": pair["prompt_id"],
                    "model_A": pair["model_A"],
                    "model_B": pair["model_B"],
                    "preferred_video": preferred,
                    "preferred_model": winner,
                    "reason": "Visible evidence",
                    "manifest_sha256": file_sha256(path),
                    "evaluation_signature": judge.evaluation_signature(pair, run.protocol["rubric"]),
                    **JUDGE_SETTINGS,
                }
            )
        write_jsonl(run.path / "judge" / f"judgments-{order}.jsonl", rows)


def test_full_pipeline_and_repeated_resume(prepared):
    run, client, _ = prepared
    import_outputs(run)
    assert run.generate()["planned_judgments"] == 12
    assert client.posts == []
    assert run.generate(execute=True, concurrency=1)["video_statuses"] == {"succeeded": 3}
    assert len(client.posts) == 3
    assert all(url == HOST + self_hosted_h3.ENDPOINT for url, _ in client.posts)
    assert all("SECRET" not in json.dumps(body) for _, body in client.posts)
    assert all(set(body) == {"prompt", "duration", "seed"} for _, body in client.posts)
    assert all(body["duration"] == 10 and body["seed"] == 42 for _, body in client.posts)
    run.generate(execute=True, concurrency=1)
    assert len(client.posts) == 3
    manifests = run.comparison_inputs()
    for pair_id, pair in manifests["original"]["pairs"].items():
        assert manifests["swapped"]["pairs"][pair_id] == judge.swap_pair(pair)
        assert pair["prompt"].startswith("Original prompt")
    add_judgments(run, manifests)
    report = run.report()
    assert report["complete"]
    for reference in ("prompt", "h3_context_ir"):
        overall = report["comparisons"][reference]["overall"]
        assert overall["points"] == 1.5
        assert overall["win_rate_half_split"] == 0.5
        assert (overall["wins"], overall["splits"], overall["losses"]) == (1, 1, 1)
        assert overall["coverage"] == 1
        assert report["comparisons"][reference]["by_source"]["aa_eval"]["win_rate_half_split"] == 0.75
        # Two AA prompts and one MovieGen prompt distinguish the equally
        # weighted dataset average (0.375) from the pooled score (0.5).
        assert report["comparisons"][reference]["scores"] == {"moviegen": 0.0, "aa_eval": 0.75, "average": 0.375}
    assert report["tensorboard"]["step"] == 0
    assert list(Path(report["tensorboard"]["log_dir"]).glob("events.out.tfevents.*"))


def test_reference_benchmark_reuses_exact_clips_and_scores_teacher(prepared, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    source, client, rows = prepared
    benchmark = EvaluationRun.prepare_reference_benchmark(tmp_path / "benchmark", source_run=source.path)
    assert benchmark.status()["planned_new_videos"] == 0
    assert benchmark.status()["planned_judgments"] == 6
    assert "rewriter" not in benchmark.protocol
    manifests = benchmark.comparison_inputs()
    samples = {row["id"]: row for row in rows}
    for pair_id, pair in manifests["original"]["pairs"].items():
        assert pair["prompt"] == samples[pair["prompt_id"]]["prompt"]
        assert {pair["model_A"], pair["model_B"]} == {"prompt", "h3_context_ir"}
        for side in ("A", "B"):
            field = REFERENCE_VIDEO_FIELDS[pair["model_" + side]]
            assert pair["video_path_" + side] == samples[pair["prompt_id"]][field]
        assert manifests["swapped"]["pairs"][pair_id] == judge.swap_pair(pair)
    add_judgments(benchmark, manifests)
    report = benchmark.report()
    assert list(report["comparisons"]) == ["prompt"]
    assert report["comparisons"]["prompt"]["scores"] == {"moviegen": 0.0, "aa_eval": 0.75, "average": 0.375}
    events = EventAccumulator(report["tensorboard"]["log_dir"]).Reload()
    assert events.Scalars("eval/prompt_rewrite/vs_prompt/average")[0].value == 0.375
    with pytest.raises(ValueError, match="existing videos"):
        benchmark.generate(execute=True)
    with pytest.raises(ValueError, match="existing videos"):
        benchmark.rewrite(execute=True)
    assert client.posts == []
    assert not (benchmark.path / "videos").exists()


def test_reference_benchmark_cli_defaults_to_dry_run(prepared, tmp_path, monkeypatch):
    source, client, _ = prepared
    commands = []

    def invoke(command, *, check):
        commands.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(judge.subprocess, "run", invoke)
    assert (
        main(["benchmark-references", "--run-dir", str(tmp_path / "benchmark"), "--source-run", str(source.path)]) == 0
    )
    assert len(commands) == 2
    assert all("--dry-run" in command and "--execute" not in command for command in commands)
    assert all(command[command.index("--concurrency") + 1] == "500" for command in commands)
    assert client.posts == []


@pytest.mark.parametrize("invalid", ["missing", "legacy_path"])
def test_max_references_must_match_dataset_fields(prepared, tmp_path, invalid):
    run, client, rows = prepared
    field = "prompt_h3_max_video"
    if invalid == "missing":
        del rows[0][field]
    else:
        rows[0][field] = rows[0]["prompt_h3_video"]
    dataset = tmp_path / "wrong-references.jsonl"
    write_jsonl(dataset, rows)
    with pytest.raises(ValueError, match="Missing reference video field|Dataset video path differs"):
        EvaluationRun.prepare(
            tmp_path / "invalid",
            dataset=dataset,
            references=Path(run.protocol["reference_manifest"]).parent,
            eval_repo=Path(run.protocol["eval_repo"]),
        )
    assert client.posts == []


def test_saved_run_uses_frozen_generation_settings(prepared, monkeypatch):
    run, client, _ = prepared
    import_outputs(run)
    original_requests = run.generation_tasks()
    monkeypatch.setattr(h3, "H3_SETTINGS", {"model": "a-future-default"})
    monkeypatch.setattr(self_hosted_h3, "SETTINGS", {"duration": 5, "seed": 1})
    run = EvaluationRun(run.path)
    assert run.generation_tasks() == original_requests
    run.generate(execute=True, concurrency=1)
    assert all(body["duration"] == 10 and body["seed"] == 42 for _, body in client.posts)


def test_tensorboard_scores_round_trip_at_training_step(prepared, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    add_judgments(run, run.comparison_inputs())
    log_dir = tmp_path / "training-run"
    report = run.report(tensorboard_dir=log_dir, step=250)
    events = EventAccumulator(str(log_dir)).Reload()
    assert report["tensorboard"]["log_dir"] == str(log_dir)
    assert report["tensorboard"]["step"] == 250
    for reference in ("prompt", "h3_context_ir"):
        prefix = f"eval/prompt_rewrite/vs_{reference}"
        for name, expected in (("moviegen", 0.0), ("aa_eval", 0.75), ("average", 0.375)):
            points = events.Scalars(f"{prefix}/{name}")
            assert len(points) == 1 and points[0].step == 250
            assert points[0].value == pytest.approx(expected)
        assert events.Scalars(f"{prefix}/coverage/moviegen")[0].value == 1
    assert events.Scalars("eval/prompt_rewrite/complete")[0].value == 1


def test_unscored_dataset_has_no_average_or_tensorboard_score(prepared):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    manifests = run.comparison_inputs()
    pair_id = next(
        key
        for key, pair in manifests["original"]["pairs"].items()
        if pair["prompt_id"] == "sample:2" and pair["reference"] == "prompt"
    )
    add_judgments(run, manifests, omit=("swapped", pair_id))
    report = run.report(allow_partial=True)
    assert report["comparisons"]["prompt"]["scores"] == {"moviegen": None, "aa_eval": 0.75, "average": None}
    events = EventAccumulator(report["tensorboard"]["log_dir"]).Reload()
    tags = events.Tags()["scalars"]
    assert "eval/prompt_rewrite/vs_prompt/moviegen" not in tags
    assert "eval/prompt_rewrite/vs_prompt/average" not in tags
    assert events.Scalars("eval/prompt_rewrite/vs_prompt/coverage/moviegen")[0].value == 0
    assert events.Scalars("eval/prompt_rewrite/complete")[0].value == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_saved_prompt_contract_controls_legacy_artifact_identity(prepared, legacy):
    run, _, _ = prepared
    protocol = run.protocol
    protocol["system_prompt"] = "Saved system role"
    if legacy:
        protocol.pop("rewrite_instruction")
    else:
        protocol["rewrite_instruction"] = "Saved task instruction"
    write_json(run.path / "protocol.json", protocol)
    write_json(run.path / "protocol-sha256.json", fingerprint(protocol))
    run = EvaluationRun(run.path)
    expected = []
    for sample in protocol["samples"]:
        content = sample["prompt"] if legacy else f"Saved task instruction\n\nUser request:\n{sample['prompt']}"
        expected.append([{"role": "system", "content": "Saved system role"}, {"role": "user", "content": content}])
    first = protocol["samples"][0]
    rewrite.save_rewrite(run.path, protocol, first, rewrite.RewriteResult("Saved rewrite"))
    record = read_json(rewrite.record_path(run.path, first["id"]))
    assert record["request_sha256"] == fingerprint(
        {"id": first["id"], "messages": expected[0], "rewriter": protocol["rewriter"]}
    )
    for sample in protocol["samples"][1:]:
        rewrite.save_rewrite(run.path, protocol, sample, rewrite.RewriteResult("New rewrite"))
    assert len(rewrite.load_rewrites(run.path, protocol)) == len(protocol["samples"])


def test_legacy_policy_inference_redirects_to_canonical_evaluation(prepared):
    run, _, _ = prepared
    with pytest.raises(ValueError, match="prompt_rewrite.rl evaluate"):
        run.rewrite(execute=True)
    assert not hasattr(rewrite, "VLLMRewriter")
    assert not hasattr(rewrite, "run_rewrites")


@pytest.mark.parametrize("bad", ["empty", "duplicate", "unknown", "wrong_prompt", "truncated", "aborted"])
def test_import_validation_before_any_generation(prepared, bad):
    run, client, _ = prepared
    rows = [{"id": sample["id"], "rewritten_prompt": "rewrite"} for sample in run.protocol["samples"]]
    if bad == "empty":
        rows[0]["rewritten_prompt"] = " "
    elif bad == "duplicate":
        rows.append(rows[0])
    elif bad == "unknown":
        rows[0]["id"] = "wrong-id"
    elif bad == "wrong_prompt":
        rows[0]["prompt"] = "different original"
    elif bad == "truncated":
        rows[0]["finish_reason"] = "length"
    else:
        rows[0]["finish_reason"] = "abort"
    with pytest.raises(ValueError):
        run.import_rewrites(rows)
    assert client.posts == []
    assert run.status()["rewrites_complete"] == 0


def test_unknown_submission_is_never_reposted(prepared):
    run, client, _ = prepared
    import_outputs(run)
    task = run.generation_tasks()[0]
    client.fail_post = True
    args = (task, run.protocol, client, lambda _: MEDIA)
    assert not self_hosted_h3.generate_one(*args)
    assert read_json(Path(task["state"]))["status"] == "submit_unknown"
    client.fail_post = False
    assert not self_hosted_h3.generate_one(*args)
    assert len(client.posts) == 1


def test_poll_failure_resumes_accepted_task_and_missing_download(prepared):
    run, client, _ = prepared
    import_outputs(run)
    task = run.generation_tasks()[0]
    args = (task, run.protocol, client, lambda _: MEDIA)
    client.fail_query = True
    assert not self_hosted_h3.generate_one(*args)
    assert read_json(Path(task["state"]))["task_id"] == str(UUID(int=1))
    client.fail_query = False
    assert self_hosted_h3.generate_one(*args)
    Path(task["output"]).unlink()
    assert self_hosted_h3.generate_one(*args)
    assert len(client.posts) == 1


def test_process_death_during_submission_is_not_reposted(prepared, monkeypatch):
    run, client, _ = prepared
    import_outputs(run)
    task = run.generation_tasks()[0]

    def die(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(client, "submit", die)
    args = (task, run.protocol, client, lambda _: MEDIA)
    with pytest.raises(KeyboardInterrupt):
        self_hosted_h3.generate_one(*args)
    assert read_json(Path(task["state"]))["status"] == "submitting"
    assert not self_hosted_h3.generate_one(*args)


def test_partial_report_does_not_turn_missing_votes_into_half_points(prepared):
    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    manifests = run.comparison_inputs()
    pair_id = next(iter(manifests["original"]["pairs"]))
    add_judgments(run, manifests, omit=("swapped", pair_id))
    with pytest.raises(ValueError, match="lack both judgments"):
        run.report()
    partial = run.report(allow_partial=True)
    assert partial["complete"] is False
    assert len(partial["incomplete"]) == 1
    reference = manifests["original"]["pairs"][pair_id]["reference"]
    assert partial["comparisons"][reference]["overall"]["scored_prompts"] == 2


@pytest.mark.parametrize("fragment", [b'{"status":"completed",', b'{"reason":"\xe2\x82'])
def test_interrupted_judge_ledger_can_resume_without_losing_coverage(prepared, fragment):
    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    manifests = run.comparison_inputs()
    add_judgments(run, manifests)
    path = run.path / "judge/judgments-swapped.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    interrupted = rows.pop()
    write_jsonl(path, rows)
    with path.open("ab") as handle:
        handle.write(fragment)
    incomplete_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="lack both judgments"):
        run.report(log_tensorboard=False)
    partial = run.report(allow_partial=True, log_tensorboard=False)
    assert partial["complete"] is False
    assert len(partial["incomplete"]) == 1
    assert partial["ledger_counts"]["swapped"]["malformed_lines"] == 1
    assert path.read_bytes() == incomplete_bytes

    # The upstream runner keeps the fragment, starts a new line, and retries
    # the missing comparison. Reporting must accept the recovered judgment.
    with path.open("ab") as handle:
        handle.write(b"\n" + json.dumps(interrupted).encode() + b"\n")
    resumed = run.report(log_tensorboard=False)
    assert resumed["complete"] is True
    assert resumed["ledger_counts"]["swapped"] == {"completed": 6, "malformed_lines": 1}
    assert resumed["comparisons"]["prompt"]["scores"] == {"moviegen": 0.0, "aa_eval": 0.75, "average": 0.375}


@pytest.mark.parametrize(
    "field,value",
    [
        ("judge_model", "wrong-model"),
        ("video_processing", "agentic"),
        ("seed", 9),
        ("manifest_sha256", "stale"),
        ("preferred_model", "unknown"),
    ],
)
def test_stale_or_mixed_judgments_are_rejected(prepared, field, value):
    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    manifests = run.comparison_inputs()
    add_judgments(run, manifests)
    path = run.path / "judge/judgments-original.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0][field] = value
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        run.report()


def test_changed_reference_or_dependency_stops_before_api_calls(prepared):
    run, client, _ = prepared
    import_outputs(run)
    dependency = Path(run.protocol["eval_repo"]) / EXTERNAL_FILES[0]
    old = dependency.read_text()
    dependency.write_text("changed implementation")
    with pytest.raises(ValueError, match="External evaluator changed"):
        run.generate(execute=True)
    dependency.write_text(old)
    reference = Path(run.protocol["samples"][0]["references"]["prompt"]["path"])
    reference.write_bytes(b"changed clip")
    with pytest.raises(ValueError, match="Frozen reference video changed"):
        run.generate(execute=True)
    assert client.posts == []


def test_two_training_checkpoints_share_references_not_candidate_artifacts(prepared, tmp_path):
    run, _, _ = prepared
    config = evaluation.RewriteConfig(model="/checkpoints/sft/epoch-1")
    sft = EvaluationRun.prepare(
        tmp_path / "sft",
        name="sft-epoch-1",
        config=config,
        dataset=Path(run.protocol["dataset"]),
        references=Path(run.protocol["reference_manifest"]).parent,
        eval_repo=Path(run.protocol["eval_repo"]),
    )
    assert sft.protocol["samples"] == run.protocol["samples"]
    assert sft.protocol["system_prompt"] == run.protocol["system_prompt"]
    assert sft.protocol["rewrite_instruction"] == run.protocol["rewrite_instruction"]
    import_outputs(sft)
    assert sft.status()["rewrites_complete"] == 3
    assert run.status()["rewrites_complete"] == 0


def test_cli_and_judge_command_are_dry_by_default(prepared):
    run, client, _ = prepared
    with pytest.raises(SystemExit) as error:
        main(["rewrite", "--run-dir", str(run.path)])
    assert error.value.code == 2
    assert run.status()["rewrites_complete"] == 0
    import_outputs(run)
    assert main(["generate", "--run-dir", str(run.path)]) == 0
    assert client.posts == []
    command = judge.command(run.path, run.protocol, "original", False, 8)
    assert "--dry-run" in command and "--execute" not in command
    assert command[command.index("--model") + 1] == "gemini-3.7-flash"
    assert command[command.index("--thinking-level") + 1] == "high"
    assert command[command.index("--video-processing") + 1] == "standard"
    assert "--max-output-tokens" not in command


def test_judge_attempts_swapped_order_after_original_errors(prepared, monkeypatch):
    run, _, _ = prepared
    attempted = []

    def invoke(command, *, check):
        manifest = Path(command[command.index("--manifest") + 1]).name
        attempted.append(manifest)
        return SimpleNamespace(returncode=1 if manifest == "manifest-original.json" else 0)

    monkeypatch.setattr(judge.subprocess, "run", invoke)
    with pytest.raises(RuntimeError, match="Incomplete judge passes: original"):
        judge.run_judge(run.path, run.protocol, execute=True)
    assert attempted == ["manifest-original.json", "manifest-swapped.json"]


def test_blocked_judgments_become_explicit_neutral_fallbacks_without_reposting(prepared, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    manifests = run.comparison_inputs()
    add_judgments(run, manifests)
    originals = {}
    for order in judge.ORDERS:
        path = run.path / "judge" / f"judgments-{order}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0].update(status="error", error_type="HttpFailure", error="HTTP 400: prohibited_content")
        rows[0].pop("preferred_model")
        rows[0].pop("preferred_video")
        write_jsonl(path, rows)
        originals[path] = path.read_bytes()
    monkeypatch.setattr(judge.subprocess, "run", lambda *a, **k: pytest.fail("Completed work was reposted"))
    run.judge(execute=True, neutral_on_failure=True)
    run.judge(execute=True, neutral_on_failure=True)
    report = run.report()
    assert report["complete"] and not report["judging_complete"]
    assert len(report["fallback_comparisons"]) == 1
    overall = report["comparisons"]["prompt"]["overall"]
    assert overall["points"] == 1.0
    assert overall["fallback_prompts"] == 1 and overall["fallback_judgments"] == 2
    assert overall["judged_coverage"] == pytest.approx(2 / 3)
    assert all(path.read_bytes() == data for path, data in originals.items())
    events = EventAccumulator(report["tensorboard"]["log_dir"]).Reload()
    assert events.Scalars("eval/prompt_rewrite/vs_prompt/fallback_prompts/aa_eval")[0].value == 1


def test_judge_hard_timeout_kills_hung_process_and_scores_neutral(prepared, monkeypatch):
    import sys
    import time

    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    monkeypatch.setattr(judge, "command", lambda *a: [sys.executable, "-c", "import time; time.sleep(30)"])
    started = time.monotonic()
    run.judge(execute=True, neutral_on_failure=True, request_timeout_seconds=0.05, pass_timeout_seconds=0.1)
    assert time.monotonic() - started < 5
    report = run.report(log_tensorboard=False)
    assert report["complete"] and not report["judging_complete"]
    for group in report["comparisons"].values():
        assert group["scores"] == {"moviegen": 0.5, "aa_eval": 0.5, "average": 0.5}
        assert group["overall"]["fallback_prompts"] == 3
        assert group["overall"]["judged_coverage"] == 0
        assert group["overall"]["order_disagreement_rate"] is None


def test_one_failed_order_scores_neutral_and_late_real_judgment_takes_precedence(prepared, monkeypatch):
    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    manifests = run.comparison_inputs()
    pair_id = next(iter(manifests["original"]["pairs"]))
    add_judgments(run, manifests, omit=("swapped", pair_id))

    def expire(args, *, check, timeout):
        assert args[args.index("--request-timeout") + 1] == "300"
        assert args[args.index("--file-timeout") + 1] == "300"
        assert args[args.index("--max-attempts") + 1] == "1"
        assert args[args.index("--pair-id") + 1] == pair_id
        assert timeout == 1800
        raise judge.subprocess.TimeoutExpired(args, timeout)

    monkeypatch.setattr(judge.subprocess, "run", expire)
    run.judge(execute=True, neutral_on_failure=True)
    report = run.report(log_tensorboard=False)
    assert report["comparisons"]["prompt"]["overall"]["points"] == 1.0
    assert report["comparisons"]["prompt"]["overall"]["fallback_judgments"] == 1
    add_judgments(run, manifests)
    report = run.report(log_tensorboard=False)
    assert report["judging_complete"] and not report["fallback_comparisons"]
    assert report["comparisons"]["prompt"]["overall"]["points"] == 1.5


def test_fatal_judge_setup_failure_is_not_replaced_with_neutral_scores(prepared, monkeypatch):
    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    monkeypatch.setattr(judge.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="Incomplete judge passes"):
        run.judge(execute=True, neutral_on_failure=True)
    assert not list((run.path / "judge").glob("fallbacks-*.json"))


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_judge_timeout_must_be_finite_and_positive(prepared, timeout):
    run, _, _ = prepared
    with pytest.raises(ValueError, match="timeouts must be positive"):
        judge.run_judge(run.path, run.protocol, execute=True, neutral_on_failure=True, pass_timeout_seconds=timeout)


def test_h3_rejection_scores_neutral_and_judges_only_existing_videos(prepared, monkeypatch):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run, client, _ = prepared
    import_outputs(run)
    get = client.get_json

    def query(url, **kwargs):
        if url.endswith(str(UUID(int=3))):
            return {"id": str(UUID(int=3)), "status": "failed", "error": "generation failed"}
        return get(url, **kwargs)

    monkeypatch.setattr(client, "get_json", query)
    assert run.generate(execute=True, concurrency=1)["video_statuses"] == {"succeeded": 2, "failed": 1}
    task = run.generation_tasks()[-1]
    original_state = Path(task["state"]).read_bytes()
    monkeypatch.setattr(h3.subprocess, "run", lambda *a, **k: pytest.fail("Completed or rejected tasks were reposted"))
    assert main(["generate", "--run-dir", str(run.path), "--execute", "--neutral-on-failure"]) == 0
    status = run.generate(execute=True, neutral_on_failure=True)
    assert status["video_fallbacks"] == 1 and status["videos_accounted_for"] == 3
    assert len(client.posts) == 3 and Path(task["state"]).read_bytes() == original_state
    assert not Path(task["output"]).exists()
    manifests = run.comparison_inputs()
    assert manifests["original"]["metadata"]["prompt_count"] == 2
    assert all(len(manifest["pairs"]) == 4 for manifest in manifests.values())
    assert all(pair["prompt_id"] != task["id"] for pair in manifests["original"]["pairs"].values())
    add_judgments(run, manifests)
    report = run.report(step=138)
    assert report["complete"] and not report["generation_complete"] and not report["judging_complete"]
    assert len(report["generation_fallbacks"]) == 1 and len(report["fallback_comparisons"]) == 2
    for groups in report["comparisons"].values():
        assert groups["scores"] == {"moviegen": 0.5, "aa_eval": 0.75, "average": 0.625}
        assert groups["overall"]["coverage"] == 1 and groups["overall"]["judged_coverage"] == pytest.approx(2 / 3)
        assert groups["overall"]["h3_fallback_prompts"] == 1
        assert groups["overall"]["fallback_judgments"] == 0
    events = EventAccumulator(report["tensorboard"]["log_dir"]).Reload()
    assert events.Scalars("eval/prompt_rewrite/vs_prompt/h3_fallback_prompts/moviegen")[0].value == 1
    # A changed state cannot silently replace a frozen fallback/cohort.
    state = read_json(Path(task["state"]))
    state["task_id"] = "different-job"
    write_json(Path(task["state"]), state)
    with pytest.raises(ValueError, match="state changed"):
        run.report(log_tensorboard=False)


def test_h3_hard_timeout_terminates_hung_worker_and_never_calls_gemini(prepared, monkeypatch, tmp_path):
    import os
    import sys
    import time

    run, client, _ = prepared
    import_outputs(run)
    pid_file = tmp_path / "hung-worker.pid"
    program = (
        "import os,time; from pathlib import Path; Path("
        + repr(str(pid_file))
        + ").write_text(str(os.getpid())); time.sleep(30)"
    )
    monkeypatch.setattr(h3, "worker_command", lambda *a: [sys.executable, "-c", program])
    started = time.monotonic()
    status = run.generate(execute=True, neutral_on_failure=True, pass_timeout_seconds=0.5)
    assert time.monotonic() - started < 5
    assert status["video_fallbacks"] == status["videos_accounted_for"] == 3
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    monkeypatch.setattr(h3.subprocess, "run", lambda *a, **k: pytest.fail("Timed-out work was restarted"))
    run.generate(execute=True, neutral_on_failure=True, pass_timeout_seconds=0.5)
    monkeypatch.setattr(judge, "run_judge", lambda *a, **k: pytest.fail("Missing videos were sent to Gemini"))
    run.judge(execute=True, neutral_on_failure=True)
    report = run.report(log_tensorboard=False)
    assert report["complete"] and not report["generation_complete"]
    assert client.posts == []
    for groups in report["comparisons"].values():
        assert groups["scores"] == {"moviegen": 0.5, "aa_eval": 0.5, "average": 0.5}
        assert groups["overall"]["h3_fallback_prompts"] == 3
        assert groups["overall"]["judged_coverage"] == 0


def test_h3_worker_setup_failure_is_not_scored_neutral(prepared, monkeypatch):
    run, _, _ = prepared
    import_outputs(run)
    monkeypatch.setattr(h3.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="incomplete without a recorded"):
        run.generate(execute=True, neutral_on_failure=True)
    assert not (run.path / "videos/fallbacks.json").exists()


def test_h3_provider_timeout_is_neutral_without_retry(prepared, monkeypatch):
    run, client, _ = prepared
    import_outputs(run)
    tasks = run.generation_tasks()

    get = client.get_json

    def timeout(url, **kwargs):
        if url.endswith("/healthz"):
            return get(url, **kwargs)
        raise TimeoutError("H3 query timed out")

    monkeypatch.setattr(client, "get_json", timeout)
    run.generate(execute=True, concurrency=1)
    assert all(read_json(Path(task["state"]))["error_stage"] == "provider" for task in tasks)
    monkeypatch.setattr(h3.subprocess, "run", lambda *a, **k: pytest.fail("Timeout was retried"))
    status = run.generate(execute=True, neutral_on_failure=True)
    assert status["video_fallbacks"] == 3


def test_h3_generation_worker_completes_and_resumes_in_a_real_subprocess(prepared, tmp_path):
    source, client, _ = prepared
    repo = Path(source.protocol["eval_repo"])
    (repo / "bestofn/normalize.py").write_text(f"def probe(path): return {MEDIA!r}\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, payload):
            assert self.headers.get("Authorization") == "Bearer test-token"
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            assert self.path == "/v1/videos"
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.respond(client.submit("http://local-test", body))

        def do_GET(self):
            self.respond(b"candidate clip" if self.path.endswith("/video") else client.get_json(self.path))

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        run = EvaluationRun.prepare(
            tmp_path / "subprocess-run",
            dataset=Path(source.protocol["dataset"]),
            references=Path(source.protocol["reference_manifest"]).parent,
            eval_repo=repo,
            h3_endpoints=f"http://127.0.0.1:{server.server_port}",
        )
        import_outputs(run)
        status = run.generate(execute=True, concurrency=1, neutral_on_failure=True, pass_timeout_seconds=15)
        assert status["video_statuses"] == {"succeeded": 3} and status["video_fallbacks"] == 0
        run.generate(execute=True, concurrency=1, neutral_on_failure=True, pass_timeout_seconds=15)
        # Recover a missing clip by querying its saved task ID, never by reposting.
        Path(run.generation_tasks()[0]["output"]).unlink()
        run.generate(execute=True, concurrency=1, neutral_on_failure=True, pass_timeout_seconds=15)
        assert len(client.posts) == 3
        assert len(run.comparison_inputs()["original"]["pairs"]) == 6
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_h3_invalid_artifact_is_not_hidden_by_fallback(prepared):
    run, client, _ = prepared
    import_outputs(run)
    task = run.generation_tasks()[0]

    def invalid(_):
        raise ValueError("invalid video container")

    assert not self_hosted_h3.generate_one(task, run.protocol, client, invalid)
    assert read_json(Path(task["state"]))["error_stage"] == "artifact"
    with pytest.raises(ValueError, match="Invalid H3 video artifact"):
        run.generate(execute=True, neutral_on_failure=True)
    assert not (run.path / "videos/fallbacks.json").exists()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_h3_timeout_must_be_finite_and_positive(prepared, timeout):
    run, client, _ = prepared
    with pytest.raises(ValueError, match="timeout must be positive and finite"):
        run.generate(execute=True, neutral_on_failure=True, pass_timeout_seconds=timeout)
    assert client.posts == []


def test_score_is_invariant_to_display_order():
    pair = {
        "prompt_id": "sample",
        "source": "aa_eval",
        "reference": "prompt",
        "model_A": CANDIDATE,
        "model_B": "prompt",
        "video_path_A": "a",
        "video_path_B": "b",
    }
    a, b = {"preferred_model": CANDIDATE}, {"preferred_model": "prompt"}
    first = scoring.score_pair(pair, a, b)
    reversed_result = scoring.score_pair(judge.swap_pair(pair), b, a)
    assert first["score"] == reversed_result["score"] == 0.5
    assert first["candidate_first_win"] == reversed_result["candidate_first_win"]
    assert first["candidate_second_win"] == reversed_result["candidate_second_win"]


def test_duplicate_votes_cannot_inflate_score(prepared):
    run, _, _ = prepared
    import_outputs(run)
    run.generate(execute=True, concurrency=1)
    manifests = run.comparison_inputs()
    add_judgments(run, manifests)
    path = run.path / "judge/judgments-original.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    write_jsonl(path, rows + [copy.deepcopy(rows[0])])
    with pytest.raises(ValueError, match="duplicate completed pair"):
        run.report()
