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
"""Mixed native observations, episode-bound requests, and real evaluator transport without paid calls."""

# Imported pytest fixtures intentionally share names with test parameters.
# ruff: noqa: F811

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from test_gemini_transport_on_cpu import interrupted_gemini, local_runner  # noqa: F401
from test_judge_recovery_on_cpu import external_runner  # noqa: F401
from test_tournament_on_cpu import (  # noqa: F401
    VIDEO_RL,
    TemplateTokenizer,
    candidates,
    loop_factory,
    model_request,
    prepared_rl,
    votes,
)

from examples.video_agents import h3, h3_request, judge, judge_worker, mixed_data, mixed_references, rl, tournament
from examples.video_agents.config import JUDGE_SETTINGS, judging_protocol
from examples.video_agents.image_assets import image_asset, upright_observation
from examples.video_agents.storage import file_sha256, read_json, read_jsonl, write_json, write_jsonl

SCHEMA = read_json(rl.CONFIG_DIR / "sol_h3_unified_tool.json")
CONDITION = {"type": "image", "role": "keyframe", "frame_index": 0, "image_ref": "input_image"}


def make_sample(tmp_path, name="frame", *, kind="PNG", color="red"):
    path = tmp_path / (name + "." + kind.lower())
    Image.new("RGB", (96, 160), color).save(path, kind)
    return {
        "id": name,
        "prompt": "The person waves.",
        "source": "test",
        "task": "i2v",
        "image": image_asset(path),
        "target": {},
    }


def test_normalize_preserves_all_prompts_and_excludes_teacher_text(tmp_path):
    sample = make_sample(tmp_path)
    paths = [tmp_path / f"{i}.jsonl" for i in range(3)]
    write_jsonl(paths[0], [{"id": "i2v", "prompt": "Animate.", "image": sample["image"]["path"]}])
    for i, path in enumerate(paths[1:]):
        write_jsonl(path, [{"id": f"t2v-{i}", "prompt": "Same prompt.", "h3_context_ir": "SECRET TEACHER"}])
    normalized, _ = mixed_data.normalize_sources(paths)
    assert [r["task"] for r in normalized] == ["i2v", "t2v", "t2v"]
    records = rl.dataset_records(normalized, "train", [SCHEMA])
    assert "SECRET TEACHER" not in json.dumps(records)
    assert records[0]["images"][0]["image"] == sample["image"]["path"]
    assert "<image>" in records[0]["prompt"][1]["content"]
    assert all(not row["images"] for row in records[1:])
    assert [r["extra_info"]["id"] for r in records] == [r["id"] for r in normalized]
    with pytest.raises(ValueError, match="image"):
        mixed_data.check_splits(normalized, [{**normalized[0], "id": "different", "prompt": "Different."}])


@pytest.mark.parametrize("ratio", ["auto", *h3_request.ASPECT_RATIOS])
def test_unified_request_binds_image_and_all_worker_geometry(tmp_path, ratio):
    sample = make_sample(tmp_path)
    request = h3_request.tool_request(
        {
            "task": "fl2va",
            "prompt": "Animate.",
            "conditions": [CONDITION],
            "target": {"aspect_ratio": ratio, "duration_seconds": 14.375},
            "seed": 2**63 - 1,
        },
        SCHEMA,
        sample=sample,
    )
    assert h3_request.decode_image(request["conditions"][0]["uri"]) == Path(sample["image"]["path"]).read_bytes()
    assert h3_request.request_output(request)["num_frames"] == 345
    assert request["seed"] == 2**63 - 1
    assert "image_ref" not in request["conditions"][0]


@pytest.mark.parametrize(
    "change",
    [
        {"task": "t2va"},
        {"conditions": []},
        {"conditions": [{**CONDITION, "image_ref": "/tmp/another.png"}]},
        {"conditions": [{**CONDITION, "frame_index": -1}]},
        {"target": {"duration_seconds": 15}},
        {"target": {"short_edge": 1080}},
        {"num_inference_steps": 50},
        {"seed": True},
    ],
)
def test_unified_rejects_wrong_task_image_or_unsupported_controls(tmp_path, change):
    from jsonschema import ValidationError

    sample = make_sample(tmp_path)
    parameters = {"task": "fl2va", "prompt": "Animate.", "conditions": [CONDITION], **change}
    with pytest.raises((ValueError, ValidationError)):
        h3_request.tool_request(parameters, SCHEMA, sample=sample)


def test_unified_defaults_constraints_and_changed_asset(tmp_path):
    sample = {"task": "t2v", "target": {"duration_seconds": 8, "aspect_ratio": "9:16"}}
    request = h3_request.tool_request({"task": "t2va", "prompt": " A scene. ", "seed": None}, SCHEMA, sample=sample)
    assert request == h3_request.video_request("A scene.", duration=8, aspect_ratio="9:16")
    with pytest.raises(ValueError, match="contradicts"):
        h3_request.tool_request(
            {"task": "t2va", "prompt": "Scene.", "target": {"duration_seconds": 10}}, SCHEMA, sample=sample
        )
    sample = make_sample(tmp_path)
    Path(sample["image"]["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen source image"):
        h3_request.tool_request(
            {"task": "fl2va", "prompt": "Animate.", "conditions": [CONDITION]}, SCHEMA, sample=sample
        )


def test_generated_image_payload_stays_out_of_rollout_records(prepared_rl, tmp_path, monkeypatch):
    from test_tournament_on_cpu import MEDIA

    _, original, _ = prepared_rl
    protocol = {**original, "tool_contract": "urn:verl:sol-h3:3"}
    sample = make_sample(tmp_path)
    request = h3_request.video_request("Wave.", image=sample["image"]["path"], aspect_ratio="16:9")
    directory = tmp_path / "generated"

    def generate(*args, **kwargs):
        task = read_json(directory / "generation.json")[0]
        assert task["request"] == request
        Path(task["output"]).write_bytes(b"generated clip")
        write_json(
            Path(task["state"]),
            {
                **h3.identity(task, protocol),
                "status": "succeeded",
                "task_id": "test-job",
                "sha256": file_sha256(Path(task["output"])),
                "bytes": len(b"generated clip"),
                "media": MEDIA,
            },
        )

    monkeypatch.setattr(tournament.subprocess, "run", generate)
    result = tournament.generate_video(directory, protocol, request)
    assert result["request"]["conditions"][0]["sha256"] == sample["image"]["sha256"]
    assert "uri" not in result["request"]["conditions"][0]
    assert read_json(directory / "generation.json")[0]["request"] == request


@pytest.mark.asyncio
async def test_native_parser_executes_nested_unified_tool(loop_factory, model_request):
    make, generated = loop_factory
    text = "<tool_call><function=sol_h3><parameter=task>t2va</parameter><parameter=prompt>A clear scene</parameter>"
    text += '<parameter=target>{"duration_seconds":8,"aspect_ratio":"9:16"}</parameter>'
    text += "<parameter=conditions>[]</parameter><parameter=seed>9223372036854775807</parameter></function></tool_call>"
    loop = make(text)
    shared = loop.tools["sol_h3"]
    shared.protocol = {**shared.protocol, "tool_contract": "urn:verl:sol-h3:3"}
    info = model_request["extra_info"]
    shared.samples[info["split"]][info["id"]] = {"task": "t2v", "image": None, "target": {}}
    model_request["tools"] = [SCHEMA]
    output = await loop.run({}, **model_request)
    assert generated == [h3_request.video_request("A clear scene", duration=8, aspect_ratio="9:16", seed=2**63 - 1)]
    assert output.extra_fields["h3_result"]["status"] == "succeeded"
    assert not hasattr(shared, "bound_sample")  # Concurrent episodes do not mutate the shared tool.


def test_native_dataset_and_token_check_keep_actual_image(tmp_path):
    from omegaconf import OmegaConf

    from verl.utils.dataset.rl_dataset import RLHFDataset

    sample = make_sample(tmp_path)
    records = rl.dataset_records([sample, {**sample, "id": "text", "task": "t2v", "image": None}], "train", [SCHEMA])
    path = tmp_path / "train.jsonl"
    write_jsonl(path, records)
    seen = []

    class Processor:
        image_processor = SimpleNamespace(patch_size=14)

        def apply_chat_template(self, messages, **kwargs):
            seen.append(messages)
            assert kwargs["tools"] == [SCHEMA]
            return "image prompt"

        def __call__(self, *, images, **kwargs):
            assert len(images) == 1 and isinstance(images[0], Image.Image)
            return {"input_ids": [[0] * 987]}

    config = OmegaConf.create(
        {"filter_overlong_prompts": False, "cache_dir": str(tmp_path / "cache"), "image_patch_size": 999}
    )
    dataset = RLHFDataset(path.as_posix(), TemplateTokenizer(), config, processor=Processor())
    native = dataset[0]
    content = native["raw_prompt"][1]["content"]
    assert [item for item in content if item["type"] == "image"][0]["image"] == sample["image"]["path"]
    assert "input_image" in str(content)
    assert isinstance(dataset[1]["raw_prompt"][1]["content"], str)
    count, _ = rl.prompt_tokens(records[0], TemplateTokenizer(), Processor(), config)
    assert count == 987  # Includes processor-expanded image tokens, not text-only tokenization.
    assert seen[0][1]["content"] == content


def test_policy_observation_uses_same_exif_orientation_as_h3(tmp_path):
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[0x0112] = 6
    Image.new("RGB", (160, 96), "red").save(path, exif=exif)
    messages = [{"role": "user", "content": [{"type": "image", "image": str(path)}]}]
    processed = upright_observation(messages)
    assert processed[0]["content"][0]["image"].size == (96, 160)
    assert (image_asset(path)["width"], image_asset(path)["height"]) == (96, 160)
    assert messages[0]["content"][0]["image"] == str(path)


def test_eval_selects_argon_and_four_references_while_training_keeps_rewards(prepared_rl, tmp_path, monkeypatch):
    _, original, _ = prepared_rl
    sample = make_sample(tmp_path)
    references = original["samples"]["val"][0]["references"]
    sample["references"] = {name: references["best_of_4"] for name in mixed_data.REFERENCE_COLUMNS}
    protocol = {
        **original,
        "reference_fields": list(mixed_data.REFERENCE_COLUMNS),
        "eval_judge": {**JUDGE_SETTINGS, "judge_model": "gemini-4-argon"},
    }
    captured = []

    def run(directory, selected, manifests, **kwargs):
        captured.append((selected["judge"], manifests))
        return votes(manifests)

    monkeypatch.setattr(tournament, "run_judgments", run)
    clips = candidates(tmp_path)
    train = tournament.score_group(tmp_path / "train", protocol, sample, clips, validate=False)
    assert [r["score"] for r in train] == pytest.approx([1, 2 / 3, 1 / 3, 0])
    assert captured[-1][0] == JUDGE_SETTINGS
    result = tournament.score_group(tmp_path / "val", protocol, sample, clips[:1], validate=True)
    assert captured[-1][0]["judge_model"] == "gemini-4-argon"
    assert result[0]["judge_comparisons"] == 8
    for order in ("original", "swapped"):
        assert len(captured[-1][1][order]["pairs"]) == 4
        assert all(p["reference_image"] == sample["image"] for p in captured[-1][1][order]["pairs"].values())
    assert protocol["judge"] == JUDGE_SETTINGS


@pytest.mark.parametrize("kind", ["PNG", "JPEG", "WEBP"])
def test_batch_judge_sends_same_first_frame_in_both_orders(local_runner, interrupted_gemini, tmp_path, kind):
    runner, server = local_runner, interrupted_gemini
    server.remaining = 0
    sample = make_sample(tmp_path, kind=kind)
    clips = candidates(tmp_path, group_size=2)
    protocol = {
        "samples": [{**sample, "references": {"ref": clips[1]}}],
        "reference_fields": ["ref"],
        "name": "test",
        "judge": {**JUDGE_SETTINGS, "judge_model": "gemini-4-argon"},
        "rubric": "Compare.",
    }
    manifests = judge.build_manifests(protocol, {sample["id"]: clips[0]})
    pairs = {}
    for order, manifest in manifests.items():
        pairs[order] = next(iter(manifest["pairs"].values()))
    path = tmp_path / "manifest.json"
    write_json(path, {**manifests["original"], "pairs": pairs, "metadata": {"pair_count": 2}})
    with judge_worker.reference_images(runner), judge_worker.parallel_pairs(runner):
        assert (
            runner.main(
                [
                    "--manifest",
                    str(path),
                    "--output",
                    str(tmp_path / "votes.jsonl"),
                    "--all",
                    "--execute",
                    "--model",
                    "gemini-4-argon",
                    "--video-processing",
                    "standard",
                    "--concurrency",
                    "2",
                ]
            )
            == 0
        )
    requests = [json.loads(body) for operation, body in server.calls if operation == "interaction"]
    assert len(requests) == 2
    assert all(r["model"] == "gemini-4-argon" for r in requests)
    assert all(
        [p["type"] for p in r["input"]] == ["text", "text", "image", "text", "video", "text", "video", "text"]
        for r in requests
    )
    assert requests[0]["input"][2] == requests[1]["input"][2]
    assert requests[0]["input"][2]["mime_type"] == sample["image"]["mime_type"]
    assert all("_verl_reference_image_request" not in r for r in requests)
    assert sum(op == "upload" for op, _ in server.calls) == 3
    assert sum(op == "cleanup" for op, _ in server.calls) == 3


def test_streaming_eval_uses_argon_and_shared_source_image(prepared_rl, local_runner, interrupted_gemini, tmp_path):
    from examples.video_agents import streaming_judge

    _, protocol, _ = prepared_rl
    interrupted_gemini.remaining = 0
    sample = make_sample(tmp_path)
    clips = candidates(tmp_path, group_size=5)
    sample["references"] = dict(zip(mixed_data.REFERENCE_COLUMNS, clips[1:], strict=True))
    protocol = {
        **protocol,
        "reference_fields": list(mixed_data.REFERENCE_COLUMNS),
        "eval_judge": {**JUDGE_SETTINGS, "judge_model": "gemini-4-argon"},
    }
    directory = tmp_path / "stream-test"
    root = directory / "judge/stream"
    write_json(root / "context.json", {"protocol": protocol, "sample": sample, "validate": True})
    write_json(root / "candidate-0.json", clips[0])
    write_json(root / "finished.json", True)
    with judge_worker.reference_images(local_runner), judge_worker.parallel_pairs(local_runner):
        streaming_judge.run_stream(directory, local_runner)
    write_json(root / "outcome.json", {"timed_out": False, "returncode": 0})
    manifests = streaming_judge.ready_manifests(protocol, sample, clips[:1], True)
    results = streaming_judge.read_streamed_judgments(
        directory, judging_protocol(protocol, validate=True), manifests, allow_fallback=False
    )
    assert all(len(rows) == 4 for rows in results.values())
    requests = [json.loads(body) for op, body in interrupted_gemini.calls if op == "interaction"]
    assert len(requests) == 8
    assert all(r["model"] == "gemini-4-argon" and r["input"][2]["type"] == "image" for r in requests)
    assert sum(op == "upload" for op, _ in interrupted_gemini.calls) == 6
    assert sum(op == "cleanup" for op, _ in interrupted_gemini.calls) == 6


def test_i2v_reference_accepts_only_documented_cancelled_lanes(tmp_path):
    from test_tournament_on_cpu import MEDIA

    sample = make_sample(tmp_path)
    path = (tmp_path / "exports/videos" / sample["id"] / "best_of_4.mp4").resolve()
    path.parent.mkdir(parents=True)
    path.write_bytes(b"selected clip")
    video = {"path": str(path), "sha256": file_sha256(path), "size": path.stat().st_size, "media": MEDIA}
    winner = "minimax-h3-sol-i2v-repeat1"
    selection = {
        "best_of": 4,
        "available_samples": 3,
        "cancelled_repeats": [4],
        "candidate_model_ids": [f"minimax-h3-sol-i2v-repeat{i}" for i in (1, 2, 3)],
        "winner": winner,
        "selected_repeat": 1,
        "judgments_expected": 6,
        "judgments_completed": 6,
        "video": video,
    }
    generation = {
        "id": sample["id"],
        "prompt": sample["prompt"],
        "image_sha256": sample["image"]["sha256"],
        "repeat": 1,
        "video": video,
    }
    write_json(path.with_suffix(".selection.json"), selection)
    generation_path = path.parent.parent.parent / "raw" / sample["id"] / f"{winner}.generation.json"
    write_json(generation_path, generation)
    result = mixed_references.i2v_reference(sample, path, 4, lambda p: MEDIA)
    assert result["recipe"]["available_samples"] == 3
    del selection["cancelled_repeats"]
    write_json(path.with_suffix(".selection.json"), selection)
    with pytest.raises(ValueError, match="Undocumented"):
        mixed_references.i2v_reference(sample, path, 4, lambda p: MEDIA)
    selection["cancelled_repeats"] = [4]
    write_json(path.with_suffix(".selection.json"), selection)
    generation["image_sha256"] = "wrong-image"
    write_json(generation_path, generation)
    with pytest.raises(ValueError, match="different prompt/image"):
        mixed_references.i2v_reference(sample, path, 4, lambda p: MEDIA)


def test_regenerated_reference_checks_source_and_frozen_request(tmp_path):
    from hashlib import sha256

    from test_tournament_on_cpu import MEDIA

    source = {"sha256": "selected-source", "recipe": {"steps": 4, "best_of": 4}}
    path = (tmp_path / "selected-source/video.mp4").resolve()
    path.parent.mkdir()
    path.write_bytes(b"50-step clip")
    request = {"task": "t2va", "prompt": "Rewritten scene.", "conditions": [], "num_inference_steps": 50}
    completion = {
        "generation_id": source["sha256"],
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "request_sha256": sha256(
            json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
        ).hexdigest(),
        "media": {
            "width": 1344,
            "height": 768,
            "fps": 24,
            "sample_rate": 32000,
            "channels": 2,
            "full_decode_passed": True,
        },
    }
    write_json(path.parent / "request.json", request)
    write_json(path.parent / "completion.json", completion)
    sample = {"id": "test", "task": "t2v"}
    result = mixed_references.regenerated_reference(sample, path, source, lambda p: MEDIA)
    assert result["recipe"]["num_inference_steps"] == 50
    request["prompt"] = "Unrelated request."
    write_json(path.parent / "request.json", request)
    with pytest.raises(ValueError, match="does not match"):
        mixed_references.regenerated_reference(sample, path, source, lambda p: MEDIA)


def test_task_reports_keep_eight_reference_scores_and_separate_coverage():
    from examples.video_agents.agent import evaluation_summary

    reward = {"score": 1.0, "valid_call": 1, **{f"vs_{f}": 1.0 for f in mixed_data.REFERENCE_COLUMNS}}
    records = [
        {"sample_id": "text", "task": "t2v", "dropped": False, "rewards": [reward]},
        {"sample_id": "image", "task": "i2v", "dropped": False, "rewards": [{k: 0.0 for k in reward}]},
    ]
    report = evaluation_summary(1, records, 0.9, fields=mixed_data.REFERENCE_COLUMNS)
    assert report["by_task"]["t2v"]["score"] == 1
    assert report["by_task"]["i2v"]["score"] == 0
    assert report["by_task"]["i2v"]["invalid_calls"] == 1
    assert report["task_macro_score"] == 0.5


def test_mixed_prepare_freezes_native_files_and_separate_judges(prepared_rl, tmp_path, monkeypatch):
    _, legacy, legacy_args = prepared_rl
    (legacy_args.eval_repo / "scripts/run_gemini_pairwise_eval.py").write_text(
        "def build_interaction_request(*, reference_image=None): pass\n"
        "def image_mime_type(path): pass\n"
        "class GeminiRestClient:\n    def upload_image(self, path): pass\n"
    )
    train_image = make_sample(tmp_path, "training", color="red")
    eval_image = make_sample(tmp_path, "evaluation", color="blue")
    train_file, val_file = tmp_path / "mixed-train.jsonl", tmp_path / "mixed-eval.jsonl"
    write_jsonl(
        train_file,
        [
            {"id": "train-i", "prompt": "Animate.", "image": train_image["image"]["path"]},
            {"id": "train-t", "prompt": "Scene.", "h3_context_ir": "SECRET"},
        ],
    )
    write_jsonl(
        val_file,
        [
            {"id": "eval-i", "prompt": "Wave.", "image": eval_image["image"]["path"]},
            {"id": "eval-t", "prompt": "Forest."},
        ],
    )
    ref = legacy["samples"]["val"][0]["references"]["best_of_4"]
    monkeypatch.setattr(
        mixed_references,
        "reference_inventory",
        lambda samples, rows, probe: [
            {**s, "references": {k: ref for k in mixed_data.REFERENCE_COLUMNS}} for s in samples
        ],
    )
    args = rl.parse_args(
        [
            "prepare-mixed",
            "--run-dir",
            str(tmp_path / "mixed-run"),
            "--data-dir",
            str(tmp_path / "mixed-data"),
            "--combined-file",
            str(tmp_path / "combined.jsonl"),
            "--train-files",
            str(train_file),
            "--eval-datasets",
            str(val_file),
            "--eval-repo",
            str(legacy_args.eval_repo),
            "--h3-endpoints",
            "http://gateway:18080",
        ]
    )
    combined = args.combined_file
    for reserved in (args.data_dir / "train.jsonl", args.data_dir / "val.jsonl", args.run_dir / "protocol.json"):
        args.combined_file = reserved
        with pytest.raises(ValueError, match="separate from prepared run artifacts"):
            mixed_data.prepare(args)
        assert not args.run_dir.exists() and not args.data_dir.exists()
    args.combined_file = combined
    summary = mixed_data.prepare(args)
    assert summary["train_by_task"] == {"i2v": 1, "t2v": 1}
    assert summary["judgments_per_evaluation"] == 16 and summary["network_requests"] == 0
    protocol = tournament.load_protocol(args.run_dir / "protocol.json")
    assert protocol["judge"] == JUDGE_SETTINGS
    assert judging_protocol(protocol, validate=True)["judge"]["judge_model"] == "gemini-4-argon"
    assert len(read_jsonl(args.combined_file)) == 2
    assert "SECRET" not in (args.data_dir / "train.jsonl").read_text()
    assert mixed_data.prepare(args) == summary


def test_legacy_evaluator_is_accepted_only_without_image_adapter():
    runner = SimpleNamespace(GeminiRestClient=type("Client", (), {}))
    assert not judge_worker.supports_reference_images(runner)
    with judge_worker.reference_images(runner):
        assert not hasattr(runner, "build_interaction_request")


def test_mixed_gateway_requires_all_workers_on_new_contract():
    health = {
        "workers_ready": 2,
        "api": {
            "version": 2,
            "workers": 2,
            "tasks": ["t2va", "fl2va"],
            "num_frames": {"min": 124, "max": 345, "step": 17},
            "aspect_ratios": ["auto", *h3_request.ASPECT_RATIOS],
            "short_edges": [768],
            "fps": 24,
            "keyframe_frame_indices": [[0]],
            "image_formats": ["png", "jpg", "webp"],
            "max_image_bytes": h3_request.MAX_IMAGE_BYTES,
        },
    }
    h3.check_unified_api(health)
    health["api"]["workers"] = 1
    with pytest.raises(ValueError, match="all ready workers"):
        h3.check_unified_api(health)
