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
"""Natural requests, auditable annotations, and actual H3 argument validation."""

# Imported pytest fixtures intentionally share names with test parameters.
# ruff: noqa: F811

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from test_tournament_on_cpu import loop_factory, model_request, native_worker, prepared_rl, votes  # noqa: F401

from examples.video_agents import control_pilot as pilot
from examples.video_agents import h3_request, mixed_data, rl
from examples.video_agents.control_constraints import FRAME_COUNTS, RATIOS, validate_contract
from examples.video_agents.image_assets import image_asset
from examples.video_agents.storage import fingerprint, read_json, read_jsonl, write_json, write_jsonl

SCHEMA = read_json(rl.CONFIG_DIR / "sol_h3_unified_tool.json")


def sample(tmp_path, *, task="t2v", prompt="A fox walks through snow.", mode="free"):
    asset = None
    if task == "i2v":
        path = tmp_path / "frame.png"
        Image.new("RGB", (96, 160), "blue").save(path)
        asset = image_asset(path)
    return {
        "id": "test-" + task,
        "task": task,
        "source": "test",
        "prompt": prompt,
        "request": mixed_data.natural_request(prompt, image=bool(asset)),
        "image": asset,
        "target": {},
        "pilot_mode": mode,
    }


def annotation(*, task="t2v"):
    return {
        "duration": {"kind": "unspecified", "seconds": None, "max_seconds": None, "evidence": [], "confidence": "high"},
        "aspect_ratio": {"kind": "unspecified", "value": None, "evidence": [], "confidence": "high"},
        "recommended": {
            "duration_seconds": 8,
            "aspect_ratio": "auto" if task == "i2v" else "9:16",
            "reason": "Suitable motion and framing.",
        },
        "feasibility_notes": [],
    }


def materialize(tmp_path, source, value):
    pilot.validate_annotation(value, source)
    root = tmp_path / "annotation"
    write_json(root / "annotation.json", value)
    return pilot.materialize(source, value, root, "gemini-4-argon")


def call(row, *, duration=8, aspect="16:9", omit=()):
    contract = row["allowed_video_gen_args"]
    target = {"duration_seconds": duration, "aspect_ratio": aspect}
    for key in omit:
        target.pop(key)
    return h3_request.tool_request(
        {"task": contract["task"], "conditions": contract["conditions"], "prompt": "A fox walks.", "target": target},
        SCHEMA,
        sample=row,
    )


@pytest.mark.parametrize("task", ["t2v", "i2v"])
def test_generic_requests_do_not_inherit_gemini_recommendations(tmp_path, task):
    source = sample(tmp_path, task=task)
    row = materialize(tmp_path, source, annotation(task=task))
    rules = row["allowed_video_gen_args"]["target"]
    assert rules["duration_seconds"] == {"required": False, "allowed_num_frames": list(FRAME_COUNTS)}
    assert rules["aspect_ratio"]["required"] is False
    assert rules["aspect_ratio"]["allowed_values"] == list(RATIOS)
    for duration, ratio in ((5, "1:1"), (14, "21:9"), (8, "9:16")):
        call(row, duration=duration, aspect=ratio)
    call(row, omit=("duration_seconds", "aspect_ratio"))
    assert row["request"] == source["request"]


def test_complete_timeline_uses_endpoint_and_preserves_original_requirement(tmp_path):
    source = sample(tmp_path, prompt="[0-2s] Look around. [3-10s] Walk away.", mode="both")
    value = annotation()
    value["duration"] = {
        "kind": "exact",
        "seconds": 10,
        "max_seconds": None,
        "evidence": ["[0-2s]", "[3-10s]"],
        "confidence": "high",
    }
    row = materialize(tmp_path, source, value)
    assert row["allowed_video_gen_args"]["target"]["duration_seconds"]["allowed_num_frames"] == [243]
    assert "Keep it about 8 seconds" not in row["request"]
    ratio = row["allowed_video_gen_args"]["target"]["aspect_ratio"]["allowed_values"][0]
    call(row, duration=10.125, aspect=ratio)
    with pytest.raises(ValueError, match="contradicts"):
        call(row, duration=8, aspect=ratio)
    with pytest.raises(ValueError, match="explicitly supply"):
        call(row, duration=10, aspect=ratio, omit=("duration_seconds",))


def test_portrait_allows_multiple_ratios_without_an_exact_hidden_target(tmp_path):
    source = sample(tmp_path, prompt="Make a vertical video of a fox.")
    value = annotation()
    value["aspect_ratio"] = {"kind": "orientation", "value": "portrait", "evidence": ["vertical"], "confidence": "high"}
    row = materialize(tmp_path, source, value)
    assert row["allowed_video_gen_args"]["target"]["aspect_ratio"]["allowed_values"] == ["3:4", "9:16"]
    call(row, aspect="3:4")
    call(row, aspect="9:16")
    with pytest.raises(ValueError, match="aspect ratio violates"):
        call(row, aspect="16:9")
    with pytest.raises(ValueError, match="explicitly supply"):
        call(row, omit=("aspect_ratio",))


def test_landscape_annotation_accepts_all_supported_wide_shapes(tmp_path):
    source = sample(tmp_path, prompt="Use a landscape frame for the fox.")
    value = annotation()
    value["aspect_ratio"] = {
        "kind": "orientation",
        "value": "landscape",
        "evidence": ["landscape"],
        "confidence": "high",
    }
    row = materialize(tmp_path, source, value)
    assert row["allowed_video_gen_args"]["target"]["aspect_ratio"]["allowed_values"] == ["auto", "21:9", "16:9", "4:3"]


def test_i2v_requested_shape_preservation_binds_source_geometry(tmp_path):
    row = materialize(tmp_path, sample(tmp_path, task="i2v", mode="both"), annotation(task="i2v"))
    assert "Keep the image's original aspect ratio" in row["request"]
    assert row["allowed_video_gen_args"]["target"]["aspect_ratio"]["allowed_values"] == ["auto"]
    call(row, aspect="auto")
    with pytest.raises(ValueError, match="contradicts"):
        call(row, aspect="9:16")


def test_short_is_advisory_until_a_numeric_request_is_added(tmp_path):
    source = sample(tmp_path, prompt="Make a short clip of a fox.")
    value = annotation()
    value["duration"].update(kind="qualitative", evidence=["short"])
    row = materialize(tmp_path, source, value)
    assert not row["allowed_video_gen_args"]["target"]["duration_seconds"]["required"]
    source["pilot_mode"] = "duration"
    row = materialize(tmp_path, source, value)
    assert "Keep it about 8 seconds long" in row["request"]
    assert row["allowed_video_gen_args"]["target"]["duration_seconds"]["allowed_num_frames"] == [192]


@pytest.mark.parametrize("seconds", [2, 4, 10])
def test_partial_timing_only_restricts_worker_choices_when_necessary(tmp_path, seconds):
    source = sample(tmp_path, prompt=f"Hide the fox for {seconds} seconds, then reveal it.")
    value = annotation()
    value["duration"].update(kind="minimum", seconds=seconds, evidence=[f"{seconds} seconds"])
    row = materialize(tmp_path, source, value)
    required = seconds == 10
    assert row["allowed_video_gen_args"]["target"]["duration_seconds"]["required"] is required
    if required:
        with pytest.raises(ValueError, match="explicitly supply"):
            call(row, omit=("duration_seconds",))
        with pytest.raises(ValueError, match="duration violates"):
            call(row, duration=5)
    else:
        call(row, omit=("duration_seconds",))
        call(row, duration=5)
    call(row, duration=14)


def test_unsupported_timeline_is_retimed_visibly_instead_of_silently_clamped(tmp_path):
    source = sample(tmp_path, prompt="[0-30s] A fox walks away.")
    value = annotation()
    value["duration"].update(kind="exact", seconds=30, evidence=["[0-30s]"])
    row = materialize(tmp_path, source, value)
    assert row["control_metadata"]["retimed"]
    assert row["control_metadata"]["review_required"]
    assert "Adapt the following scene into a video lasting 8 seconds" in row["request"]
    assert source["prompt"] in row["request"]
    assert row["allowed_video_gen_args"]["target"]["duration_seconds"]["allowed_num_frames"] == [192]


def test_low_confidence_is_flagged_without_hardening_uncertain_labels(tmp_path):
    source = sample(tmp_path, prompt="Show the last 10 seconds.", mode="both")
    value = annotation()
    value["duration"].update(kind="exact", seconds=10, evidence=["last 10 seconds"], confidence="low")
    row = materialize(tmp_path, source, value)
    assert row["control_metadata"]["review_required"]
    assert not row["allowed_video_gen_args"]["target"]["duration_seconds"]["required"]
    assert "Keep it about" not in row["request"]


def test_evidence_and_constraint_identity_are_validated(tmp_path):
    source = sample(tmp_path)
    value = annotation()
    value["duration"].update(kind="exact", seconds=10, evidence=["ten seconds"])
    with pytest.raises(ValueError, match="quote"):
        pilot.validate_annotation(value, source)
    row = materialize(tmp_path, source, annotation())
    with pytest.raises(ValueError, match="visible request"):
        validate_contract({**row, "request": "Different request"}, row["allowed_video_gen_args"])
    contract = deepcopy(row["allowed_video_gen_args"])
    contract["target"]["duration_seconds"]["allowed_num_frames"] = [243]
    with pytest.raises(ValueError, match="Unspecified controls"):
        validate_contract(row, contract)


def test_pilot_output_can_be_normalized_without_leaking_labels(tmp_path):
    rows = [
        materialize(tmp_path, sample(tmp_path, task=task, mode="both"), annotation(task=task))
        for task in ("t2v", "i2v")
    ]
    summary = pilot.save_outputs(tmp_path / "pilot", rows)
    assert summary["samples"] == 2
    loaded, _ = mixed_data.normalize_sources([tmp_path / "pilot/samples.jsonl"])
    assert [r["allowed_video_gen_args"] for r in loaded] == [r["allowed_video_gen_args"] for r in rows]
    for row, record in zip(loaded, read_jsonl(tmp_path / "pilot/train.jsonl"), strict=True):
        user = record["prompt"][1]["content"]
        assert row["request"] in user
        assert not any(
            label in user for label in ("allowed_video_gen_args", "preferred_target", "fl2va", "t2va", "input_image")
        )
        call(
            row,
            duration=row["example_allowed_args"]["target"]["duration_seconds"] if "example_allowed_args" in row else 8,
            aspect=row["allowed_video_gen_args"]["target"]["aspect_ratio"]["allowed_values"][0],
        )


def test_annotation_cache_reuses_real_response_without_another_call(tmp_path):
    source = sample(tmp_path, task="i2v")
    calls = []
    value = annotation(task="i2v")

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def request_json(self, *args, **kwargs):
            calls.append(kwargs["payload"])
            return {"model": "gemini-4-argon", "status": "completed", "output_text": json.dumps(value)}, {}

    runner = SimpleNamespace(
        GeminiRestClient=Client, API_BASE="http://fake", interaction_output_text=lambda r: r["output_text"]
    )
    first, root = pilot.annotate(source, tmp_path, "gemini-4-argon", "Policy", runner, "unused")
    second, _ = pilot.annotate(source, tmp_path, "gemini-4-argon", "Policy", runner, "unused")
    assert first == second == value and len(calls) == 1
    assert calls[0]["input"][-1]["type"] == "image" and calls[0]["input"][-1]["data"]
    assert read_json(root / "response-1.json")["model"] == "gemini-4-argon"


def test_cache_only_never_calls_provider_for_missing_annotations(tmp_path):
    with pytest.raises(ValueError, match="No validated cached annotation"):
        pilot.annotate(sample(tmp_path), tmp_path, "gemini-4-argon", "Policy", None, None, cache_only=True)


def test_cache_only_cli_accounts_for_every_row_and_preserves_refusal(tmp_path, monkeypatch):
    rows = [sample(tmp_path, task=task) for task in ("t2v", "i2v")]
    source, output = tmp_path / "input.jsonl", tmp_path / "output"
    write_jsonl(source, rows)
    identity = pilot.annotation_identity(rows[0], "gemini-4-argon", pilot.POLICY_PATH.read_text())
    write_json(output / "annotations" / fingerprint(identity) / "annotation.json", annotation())
    refusal = {"id": rows[1]["id"], "error": "HTTP 400: prohibited_content"}
    write_json(output / "errors.json", [refusal])
    monkeypatch.setattr(pilot, "load_external", lambda path: pytest.fail("Offline rendering needs no provider"))
    assert pilot.main(["--input", str(source), "--output-dir", str(output), "--per-task", "1", "--render-only"]) == 1
    summary = read_json(output / "summary.json")
    assert summary["selected"] == summary["accounted_for"] == 2
    assert summary["processing_complete"] and not summary["complete"]
    assert summary["samples"] == summary["errors"] == 1
    review = read_jsonl(output / "provider-review.jsonl")
    assert review[0]["annotation_status"] == "provider_refused"
    assert review[0]["error"] == refusal["error"]
    assert review[0]["original_sample"]["prompt"] == rows[1]["prompt"]
    assert review[0]["original_sample"]["image"] == rows[1]["image"]


def test_full_selection_retains_duplicate_weight_and_pilot_modes(tmp_path):
    rows = [sample(tmp_path, task=task) for task in ("t2v", "i2v")]
    rows.append({**rows[0], "id": "repeated-text"})
    limited = pilot.select_samples(rows, 1, 42)
    full = pilot.select_samples(rows, None, 42)
    assert len(limited) == 2 and len(full) == 3
    assert {s["id"] for s in full} == {s["id"] for s in rows}
    assert full[0]["pilot_mode"] == full[1]["pilot_mode"] == limited[0]["pilot_mode"]


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["t2v", "i2v"])
@pytest.mark.parametrize("constrained", [False, True])
async def test_native_controls_reach_published_grpo_rewards(
    tmp_path, loop_factory, model_request, native_worker, monkeypatch, task, constrained
):
    """Use the real parser, executor validation, tournament scorer, and reward publisher."""
    from examples.video_agents import tournament
    from examples.video_agents.storage import file_sha256

    make, generated = loop_factory
    worker, events, published = native_worker
    source = sample(tmp_path, task=task)
    value = annotation(task=task)
    if constrained:
        source["prompt"] = "Make a 10-second vertical video of a fox."
        source["request"] = mixed_data.natural_request(source["prompt"], image=bool(source["image"]))
        value["duration"].update(kind="exact", seconds=10, evidence=["10-second"])
        value["aspect_ratio"].update(kind="orientation", value="portrait", evidence=["vertical"])
    row = materialize(tmp_path, source, value)
    row["id"] = model_request["extra_info"]["id"]
    record = rl.dataset_records([row], "train", [SCHEMA])[0]
    record["raw_prompt"] = record.pop("prompt")
    worker.samples["train"][row["id"]] = row
    worker.protocol["judge_streaming"] = False
    settings = [(10, "3:4"), (10.125, "9:16"), (8, "3:4"), (10, None)]
    outputs = []
    for duration, ratio in settings:
        target = {"duration_seconds": duration, **({"aspect_ratio": ratio} if ratio else {})}
        args = {
            "task": "fl2va" if task == "i2v" else "t2va",
            "conditions": row["allowed_video_gen_args"]["conditions"],
            "prompt": "A fox walks through snow.",
            "target": target,
        }
        text = (
            "<tool_call><function=sol_h3>"
            + "".join(
                f"<parameter={key}>{json.dumps(value) if not isinstance(value, str) else value}</parameter>"
                for key, value in args.items()
            )
            + "</function></tool_call>"
        )
        loop = make(text)
        shared = loop.tools["sol_h3"]
        shared.protocol = {**shared.protocol, "tool_contract": "urn:verl:sol-h3:3"}
        shared.samples["train"][row["id"]] = row
        output = await loop.run({}, **deepcopy(record))
        result = output.extra_fields["h3_result"]
        if result["status"] == "succeeded":
            path = Path(result["path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"Mock video; the external generator and judge are controlled in this test.")
            result["sha256"] = file_sha256(path)
        outputs.append(output)
    assert len(generated) == (2 if constrained else 4)
    assert [o.extra_fields["h3_result"]["status"] for o in outputs] == ["succeeded"] * 2 + [
        "invalid_call" if constrained else "succeeded"
    ] * 2

    def judge(directory, protocol, manifests, **kwargs):
        for pair in manifests["original"]["pairs"].values():
            assert pair["prompt"] == row["request"]
        return votes(manifests)

    monkeypatch.setattr(tournament, "run_judgments", judge)

    async def rollout(*args, session_id, **kwargs):
        return outputs[session_id]

    worker._run_agent_loop = rollout
    prompt = {"uid": "controls", "global_steps": 1, "extra_info": {"id": row["id"]}}
    result = await worker._run_prompt(prompt, {}, {"validate": False})
    expected = [1, 2 / 3, -1, -1] if constrained else [1, 2 / 3, 1 / 3, 0]
    assert [r["score"] for r in result["rewards"]] == pytest.approx(expected)
    assert [o.reward_score for o in outputs] == pytest.approx(expected)
    assert len(published) == 4 and events[-1] == "finished"
