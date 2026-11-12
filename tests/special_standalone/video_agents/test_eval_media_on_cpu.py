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
"""Real TensorBoard media encoding plus stable selection and completion semantics."""

import io
import subprocess

import pytest

from examples.video_agents import eval_media
from examples.video_agents.storage import read_json, write_json

pytest.importorskip("tensorboard")
pytest.importorskip("PIL")


@pytest.fixture
def run(tmp_path):
    from PIL import Image

    run_dir = tmp_path / "run"
    source = run_dir / "images" / "first.png"
    source.parent.mkdir(parents=True)
    Image.new("RGB", (80, 40), "red").save(source)
    samples = [
        {"id": "text-example", "task": "t2v", "prompt": "A ball rolls", "request": "A ball rolls"},
        {
            "id": "image-example",
            "task": "i2v",
            "prompt": "A bird flies",
            "request": "A bird flies",
            "image": {"path": str(source)},
        },
    ]
    write_json(run_dir / "protocol.json", {"samples": {"val": samples}})
    return run_dir, samples


def evaluation(run_dir, samples, step, *, candidate=None, failure=None):
    directory = run_dir / "val" / f"step_{step}"
    write_json(
        directory / "eval-summary.json",
        {
            "policy_step": step,
            "prompts": len(samples),
            "valid": len(samples),
            "dropped": 0,
            "by_task": {
                task: {"prompts": sum(sample["task"] == task for sample in samples)}
                for task in {sample["task"] for sample in samples}
            },
        },
    )
    for number, sample in enumerate(samples):
        group = directory / str(number)
        result = candidate or {"status": "invalid_call", "reason": "No valid tool call"}
        write_json(
            group / "trajectories.json",
            {
                "sample_id": sample["id"],
                "policy_step": step,
                "rollouts": [{"h3_result": result}],
            },
        )
        write_json(group / "inputs.json", {"sample": sample, "candidates": [result]})
        write_json(group / "rewards.json", [{"score": 0.5, "valid_call": int(result["status"] == "succeeded")}])
        if failure and number == 0:
            write_json(
                group / "failure.json",
                {"sample_id": sample["id"], "stage": "gemini", "type": "ProviderError", "error": "judge offline"},
            )
    return directory


def accumulator(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    return EventAccumulator(str(run_dir / "tensorboard" / "eval_examples"), size_guidance={"images": 0}).Reload()


def test_selection_is_order_independent_and_identical_across_resumed_runs():
    samples = [{"id": f"{task}-{number}", "task": task} for task in ("t2v", "i2v") for number in range(12)]
    chosen = eval_media.select_examples(samples, 4)
    assert chosen == eval_media.select_examples(list(reversed(samples)), 4)
    assert [sample["task"] for sample in chosen].count("t2v") == 4
    assert [sample["task"] for sample in chosen].count("i2v") == 4
    assert len({sample["id"] for sample in chosen}) == 8


def test_partial_summary_is_never_published(run):
    run_dir, samples = run
    directory = evaluation(run_dir, samples, 25)
    summary = read_json(directory / "eval-summary.json")
    summary["prompts"] = 1
    write_json(directory / "eval-summary.json", summary)
    report = eval_media.publish_evaluation_media(run_dir, 25)
    assert report["status"] == "incomplete"
    assert not (run_dir / "tensorboard" / "eval_examples").exists()
    summary["prompts"] = 2
    summary["by_task"]["i2v"]["prompts"] = 0
    write_json(directory / "eval-summary.json", summary)
    assert eval_media.publish_evaluation_media(run_dir, 25)["status"] == "incomplete"


def test_failure_panels_are_preserved_and_idempotent(run):
    run_dir, samples = run
    evaluation(run_dir, samples, 25, failure=True)
    report = eval_media.publish_evaluation_media(run_dir, 25)
    assert report["status"] == "published" and report["missing"] == 2
    board = accumulator(run_dir)
    text_tag = "eval_examples/t2v/text-example/details"
    text = board.Tensors(text_tag)[0].tensor_proto.string_val[0].decode()
    assert "Status: dropped" in text and "judge offline" in text
    assert "A ball rolls" in text
    assert board.Tensors(text_tag)[0].step == 25
    assert "eval_examples/i2v/image-example/source" in board.Tags()["images"]
    assert eval_media.publish_evaluation_media(run_dir, 25)["status"] == "already_published"
    assert len(accumulator(run_dir).Tensors(text_tag)) == 1
    assert read_json(run_dir / "tensorboard" / "eval_examples" / "selection.json")["examples"]


def test_real_animated_video_and_relocated_artifact(run):
    from PIL import Image

    run_dir, samples = run
    video = run_dir / "videos" / "clip-id" / "video.mp4"
    video.parent.mkdir(parents=True)
    try:
        executable = eval_media._ffmpeg()
    except RuntimeError:
        pytest.skip("ffmpeg or imageio-ffmpeg is needed for real media encoding")
    subprocess.run(
        [
            executable,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=8:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "1",
            str(video),
        ],
        check=True,
        timeout=20,
    )
    candidate = {
        "status": "succeeded",
        "path": "/old/server/run/videos/clip-id/video.mp4",
        "prompt": "A smoothly rolling ball",
        "request": {"seed": 42, "duration": 1},
    }
    evaluation(run_dir, samples, 50, candidate=candidate)
    report = eval_media.publish_evaluation_media(run_dir, 50, settings={"fps": 4, "max_size": 32})
    assert report["status"] == "published", report
    assert report["published"] == 2 and not report["errors"]
    board = accumulator(run_dir)
    recorded = board.Images("eval_examples/t2v/text-example/video")[0]
    with Image.open(io.BytesIO(recorded.encoded_image_string)) as preview:
        assert max(preview.size) == 32
        assert preview.n_frames == 4
        assert sum(preview.seek(i) or preview.info["duration"] for i in range(preview.n_frames)) == 1000
    details = board.Tensors("eval_examples/t2v/text-example/details")[0].tensor_proto.string_val[0].decode()
    assert '"seed": 42' in details and '"score": 0.5' in details
    assert "A smoothly rolling ball" in details
    assert recorded.step == 50


def test_missing_video_remains_retryable(run):
    run_dir, samples = run
    evaluation(run_dir, samples, 75, candidate={"status": "succeeded", "path": "/missing/video.mp4"})
    report = eval_media.publish_evaluation_media(run_dir, 75)
    assert report["status"] == "pending" and report["missing"] == 2
    assert len(report["errors"]) == 2
    assert not list((run_dir / "tensorboard" / "eval_examples").glob("events.out.tfevents.*"))
    assert read_json(run_dir / "tensorboard" / "eval_examples" / "state.json")["steps"]["75"]["status"] == "pending"
    assert eval_media.publish_evaluation_media(run_dir, 75)["status"] == "pending"


def test_live_reader_discovers_an_older_step_backfilled_later(run):
    run_dir, samples = run
    evaluation(run_dir, samples, 50)
    evaluation(run_dir, samples, 25)
    assert eval_media.publish_evaluation_media(run_dir, 50)["status"] == "published"
    board = accumulator(run_dir)
    tag = "eval_examples/t2v/text-example/details"
    assert [event.step for event in board.Tensors(tag)] == [50]
    assert eval_media.publish_evaluation_media(run_dir, 25)["status"] == "published"
    board.Reload()
    assert sorted(event.step for event in board.Tensors(tag)) == [25, 50]


def test_selection_change_is_rejected_and_disabled_is_dependency_free(run, monkeypatch):
    run_dir, samples = run
    evaluation(run_dir, samples, 0)
    assert eval_media.publish_evaluation_media(run_dir, 0)["status"] == "published"
    assert eval_media.publish_evaluation_media(run_dir, 0, settings={"samples_per_task": 1})["status"] == "error"
    monkeypatch.setattr(eval_media, "_publish", lambda *args: pytest.fail("Disabled publisher must do no work"))
    assert eval_media.publish_evaluation_media(run_dir, 0, settings={"enabled": False})["status"] == "disabled"


def test_cli_backfills_steps_in_numeric_order(run, monkeypatch):
    run_dir, samples = run
    for step in (100, 25, 0):
        evaluation(run_dir, samples, step)
    called = []

    def publish(path, step, protocol_path, **kwargs):
        assert path == run_dir
        called.append(step)
        return {"status": "published"}

    monkeypatch.setattr(eval_media, "publish_evaluation_media", publish)
    assert eval_media.main(["--run-dir", str(run_dir)]) == 0
    assert called == [0, 25, 100]
