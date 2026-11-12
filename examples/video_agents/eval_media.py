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
"""Publish existing held-out video artifacts to TensorBoard without running providers.

The command can backfill saved evaluations or watch a run from a separate process.
Only selected videos are decoded; original MP4s and training artifacts are untouched.
"""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

from .storage import read_json, run_lock, write_json

DEFAULTS = {"enabled": True, "samples_per_task": 4, "fps": 6, "max_size": 384, "timeout": 60}


def select_examples(samples: list[dict], samples_per_task: int) -> list[dict]:
    """Choose a stable, reward-independent panel from a frozen validation cohort."""
    if type(samples_per_task) is not int or samples_per_task < 1:
        raise ValueError("samples_per_task must be a positive integer")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("Validation sample IDs must be unique")
    result = []
    for task in ("t2v", "i2v"):
        choices = [sample for sample in samples if sample.get("task", "t2v") == task]
        choices.sort(key=lambda sample: (hashlib.sha256(sample["id"].encode()).hexdigest(), sample["id"]))
        result.extend({"task": task, "id": sample["id"]} for sample in choices[:samples_per_task])
    return result


def _protocol_path(run_dir: Path, supplied: Path | None) -> Path:
    if supplied is not None:
        return Path(supplied).expanduser().resolve()
    for candidate in (run_dir / "protocol.json", run_dir.parent / "protocol.json"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("No protocol.json found; supply --protocol")


def _complete(summary: dict, step: int, samples: list[dict]) -> bool:
    if summary.get("policy_step") != step or summary.get("prompts") != len(samples):
        return False
    if "by_task" in summary:
        counts = Counter(sample.get("task", "t2v") for sample in samples)
        return all(summary["by_task"].get(task, {}).get("prompts") == count for task, count in counts.items())
    return True


def _resolve_artifact(value: str, run_dir: Path, protocol_dir: Path) -> Path:
    """Find artifacts after copying a whole run without rewriting frozen JSON."""
    path = Path(value).expanduser()
    options = [path] if path.is_absolute() else [run_dir / path, protocol_dir / path]
    for anchor in ("videos", "episodes", "images"):
        if anchor in path.parts:
            relative = Path(*path.parts[path.parts.index(anchor) :])
            options.extend((run_dir / relative, protocol_dir / relative))
    options.extend((run_dir / path.name, protocol_dir / path.name))
    for candidate in options:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Artifact is unavailable locally: {path}")


def _index_groups(step_dir: Path, wanted: set[str], deadline: float) -> dict[str, dict]:
    """Index small trajectory/failure records, never every inputs.json image payload."""
    result = {}
    for directory in sorted(step_dir.iterdir()):
        if not directory.is_dir():
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError("Evaluation media publication exhausted its time budget while indexing")
        trajectory_path, failure_path = directory / "trajectories.json", directory / "failure.json"
        record = read_json(trajectory_path) if trajectory_path.exists() else None
        failure = read_json(failure_path) if failure_path.exists() else None
        sample_id = (record or failure or {}).get("sample_id")
        if sample_id in wanted:
            if sample_id in result:
                raise ValueError(f"Multiple validation groups for selected sample {sample_id}")
            result[sample_id] = {"directory": directory, "trajectory": record, "failure": failure}
    return result


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Install ffmpeg or imageio-ffmpeg to encode TensorBoard video previews") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _video_preview(path: Path, settings: dict, deadline: float, directory: Path) -> tuple[bytes, int, int]:
    from PIL import Image

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Evaluation media publication exhausted its time budget")
    # Scaling preserves aspect ratio and avoids upscaling. FPS sampling retains the
    # complete clip duration, which is needed to inspect motion and timing.
    limit = settings["max_size"]
    scale = f"scale=w='if(gte(iw,ih),min(iw,{limit}),-1)':h='if(gte(iw,ih),-1,min(ih,{limit}))'"
    target = directory / "preview.gif"
    subprocess.run(
        [
            _ffmpeg(),
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(path),
            "-vf",
            f"fps={settings['fps']},{scale}",
            "-an",
            "-loop",
            "0",
            "-y",
            str(target),
        ],
        check=True,
        timeout=remaining,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    encoded = target.read_bytes()
    with Image.open(io.BytesIO(encoded)) as image:
        width, height = image.size
    return encoded, width, height


def _source_preview(path: Path, max_size: int) -> tuple[bytes, int, int]:
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((max_size, max_size))
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue(), image.width, image.height


def _redact_payloads(value):
    if isinstance(value, dict):
        return {key: _redact_payloads(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_payloads(item) for item in value]
    if isinstance(value, str) and (value.startswith("data:") or len(value) > 12000):
        return "<large payload omitted>"
    return value


def _details(sample: dict, candidate: dict, reward: dict, status: str, error: str | None) -> str:
    from .mixed_data import original_request

    fields = [
        f"**{sample.get('task', 't2v')} / {sample['id']}**",
        f"Status: {status}",
        "**Original request**\n\n" + original_request(sample),
        "**Generated prompt**\n\n" + str(candidate.get("prompt", "Unavailable")),
        "**Generation settings**\n\n```json\n"
        + json.dumps(_redact_payloads(candidate.get("request", {})), ensure_ascii=False, indent=2)
        + "\n```",
        "**Evaluation**\n\n```json\n" + json.dumps(reward, ensure_ascii=False, indent=2) + "\n```",
    ]
    if candidate.get("path"):
        fields.append("Original MP4: `" + str(candidate["path"]) + "`")
    if error:
        fields.append("**Failure**\n\n" + error[:2000])
    return "\n\n".join(fields)


def _image_value(tag: str, preview: tuple[bytes, int, int]):
    from tensorboard.compat.proto.summary_pb2 import Summary

    encoded, width, height = preview
    return Summary.Value(
        tag=tag, image=Summary.Image(encoded_image_string=encoded, width=width, height=height, colorspace=3)
    )


def _text_value(tag: str, text: str):
    from tensorboard.compat.proto.summary_pb2 import Summary, SummaryMetadata
    from tensorboard.compat.proto.tensor_pb2 import TensorProto
    from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
    from tensorboard.compat.proto.types_pb2 import DT_STRING

    return Summary.Value(
        tag=tag,
        metadata=SummaryMetadata(plugin_data=SummaryMetadata.PluginData(plugin_name="text")),
        tensor=TensorProto(
            dtype=DT_STRING,
            string_val=[text.encode()],
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
        ),
    )


def _publish(run_dir: Path, step: int, protocol_path: Path | None, settings: dict, media_dir: Path, report: dict):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    deadline = time.monotonic() + settings["timeout"]
    protocol_path = _protocol_path(run_dir, protocol_path)
    samples = read_json(protocol_path)["samples"]["val"]
    step_dir = run_dir / "val" / f"step_{step}"
    summary_path = step_dir / "eval-summary.json"
    if not summary_path.exists() or not _complete(read_json(summary_path), step, samples):
        report["status"] = "incomplete"
        return report
    summary = read_json(summary_path)
    chosen = select_examples(samples, settings["samples_per_task"])
    selection = {"schema_version": 1, "samples_per_task": settings["samples_per_task"], "examples": chosen}
    with run_lock(media_dir):
        selection_path = media_dir / "selection.json"
        if selection_path.exists() and read_json(selection_path) != selection:
            raise ValueError("Evaluation example selection changed; use a new TensorBoard log directory")
        if not selection_path.exists():
            write_json(selection_path, selection)
        state_path = media_dir / "state.json"
        state = read_json(state_path) if state_path.exists() else {"schema_version": 1, "steps": {}}
        event_suffix = f".eval_examples.step_{step:012d}"
        if any(media_dir.glob("events.out.tfevents.*" + event_suffix)):
            report.update(state["steps"].get(str(step), {}))
            report["status"] = "already_published"
            return report
        index = _index_groups(step_dir, {sample["id"] for sample in chosen}, deadline)
        by_id = {sample["id"]: sample for sample in samples}
        dropped = {record["sample_id"]: record for record in summary.get("dropped_examples", [])}
        values = []
        with tempfile.TemporaryDirectory(prefix=".publishing-", dir=media_dir) as temporary:
            temporary = Path(temporary)
            for selected in chosen:
                sample_id, task = selected["id"], selected["task"]
                sample = by_id[sample_id]
                group = index.get(sample_id, {})
                failure = group.get("failure") or dropped.get(sample_id)
                trajectory = group.get("trajectory") or {}
                rollouts = trajectory.get("rollouts", [])
                candidate = rollouts[0].get("h3_result", {}) if rollouts else {}
                reward = {}
                status, error = "missing", None
                prefix = f"eval_examples/{task}/{sample_id}"
                if not group and not failure:
                    report["errors"].append(f"{sample_id}: selected validation group is not available locally yet")
                if group:
                    inputs_path = group["directory"] / "inputs.json"
                    if inputs_path.exists():
                        inputs = read_json(inputs_path)
                        sample = inputs.get("sample", sample)
                        candidates = inputs.get("candidates", [])
                        if candidates:
                            candidate = candidates[0]
                    rewards_path = group["directory"] / "rewards.json"
                    if rewards_path.exists():
                        rewards = read_json(rewards_path)
                        reward = rewards[0] if rewards else {}
                if failure:
                    status = "dropped"
                    error = f"{failure.get('stage', '')}: {failure.get('type', '')}: {failure.get('error', '')}"
                elif candidate:
                    status = candidate.get("status", "missing")
                    error = candidate.get("reason") or candidate.get("error")
                try:
                    if candidate.get("status") == "succeeded":
                        path = _resolve_artifact(candidate["path"], run_dir, protocol_path.parent)
                        values.append(
                            _image_value(prefix + "/video", _video_preview(path, settings, deadline, temporary))
                        )
                        report["published"] += 1
                    else:
                        report["missing"] += 1
                except Exception as exc:
                    status = "preview_error" if not failure else "dropped; preview_error"
                    error = f"{error + '; ' if error else ''}{type(exc).__name__}: {exc}"
                    report["errors"].append(f"{sample_id}: {error}")
                    report["missing"] += 1
                image = sample.get("image")
                if isinstance(image, dict) and image.get("path"):
                    try:
                        source = _resolve_artifact(image["path"], run_dir, protocol_path.parent)
                        values.append(_image_value(prefix + "/source", _source_preview(source, settings["max_size"])))
                    except Exception as exc:
                        report["errors"].append(f"{sample_id}: source image: {type(exc).__name__}: {exc}")
                        error = f"{error + '; ' if error else ''}Source image unavailable: {exc}"
                values.append(_text_value(prefix + "/details", _details(sample, candidate, reward, status, error)))
            if report["errors"]:
                # A complete remote summary can precede artifact synchronization.
                # Do not seal this step until missing files/encoders can be retried.
                report["status"] = "pending"
                state["steps"][str(step)] = report
                write_json(state_path, state)
                return report
            writer_dir = temporary / "events"
            writer = EventFileWriter(str(writer_dir))
            try:
                writer.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=values)))
                writer.flush()
            finally:
                writer.close()
            generated = list(writer_dir.glob("events.out.tfevents.*"))
            if len(generated) != 1:
                raise RuntimeError("Expected one completed TensorBoard media event file")
            # Step suffix deduplicates even if state.json was not yet written.
            # Publication time sorts late backfills after files already watched
            # by TensorBoard's forward-only event directory reader.
            event_path = media_dir / f"events.out.tfevents.{time.time_ns():020d}{event_suffix}"
            os.replace(generated[0], event_path)
        report["status"] = "published"
        state["steps"][str(step)] = report
        write_json(state_path, state)
    return report


def publish_evaluation_media(
    run_dir: Path,
    step: int,
    protocol_path: Path | None = None,
    *,
    settings: dict | None = None,
    log_dir: Path | None = None,
) -> dict:
    """Best-effort publication of one completed evaluation, safe for trainer hooks."""
    run_dir = Path(run_dir).expanduser().resolve()
    media_dir = (
        Path(log_dir).expanduser().resolve() if log_dir is not None else run_dir / "tensorboard"
    ) / "eval_examples"
    report = {"step": step, "status": "error", "published": 0, "missing": 0, "errors": [], "log_dir": str(media_dir)}
    options = {**DEFAULTS, **dict(settings or {})}
    if not options["enabled"]:
        report["status"] = "disabled"
        return report
    try:
        for key in ("fps", "max_size"):
            if type(options[key]) is not int or options[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(options["timeout"], int | float) or options["timeout"] <= 0:
            raise ValueError("timeout must be positive")
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        return _publish(run_dir, step, protocol_path, options, media_dir, report)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--step", type=int, action="append", help="Repeat to backfill particular optimizer steps")
    parser.add_argument("--log-dir", type=Path, help="TensorBoard root; media is a separate eval_examples subrun")
    parser.add_argument("--watch", action="store_true", help="Poll for completed evaluations until interrupted")
    parser.add_argument("--poll-interval", type=float, default=30)
    args = parser.parse_args(argv)
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    seen = set()
    failed = False
    try:
        while True:
            steps = args.step
            if steps is None:
                steps = [
                    int(path.name.removeprefix("step_"))
                    for path in (args.run_dir / "val").glob("step_*")
                    if path.is_dir() and path.name.removeprefix("step_").isdigit()
                ]
            for step in sorted(set(steps) - seen):
                report = publish_evaluation_media(args.run_dir, step, args.protocol, log_dir=args.log_dir)
                print(json.dumps(report, ensure_ascii=False), flush=True)
                if report["status"] in {"published", "already_published", "disabled"}:
                    seen.add(step)
                failed |= report["status"] == "error"
            if not args.watch:
                break
            time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        pass
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
