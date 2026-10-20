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
"""Shared console summaries and TensorBoard metrics for zero-shot, SFT, and RL."""

import time
from pathlib import Path

from .config import SCORE_SOURCES


def print_report(report: dict):
    print(f"Candidate: {report['candidate']} ({'complete' if report['complete'] else 'PARTIAL'})")
    print(f"{'Reference':<18} {'MovieGen':>12} {'AA eval':>12} {'Average':>12}")
    for reference, groups in report["comparisons"].items():
        values = [groups["scores"][name] for name in (*SCORE_SOURCES, "average")]
        formatted = ["unavailable" if value is None else f"{value:.2%}" for value in values]
        print(f"{reference:<18} " + " ".join(f"{value:>12}" for value in formatted))
    print("Average = (MovieGen + AA eval) / 2; equal dataset weights.")
    for reference, groups in report["comparisons"].items():
        coverage = []
        for name, source in SCORE_SOURCES.items():
            group = groups["by_source"].get(source, {})
            coverage.append(f"{name} {group.get('scored_prompts', 0)}/{group.get('expected_prompts', 0)}")
        print(f"Scored vs {reference}: {', '.join(coverage)}")


def tensorboard_scalars(report: dict) -> dict[str, float]:
    """Use the same three scores for console, saved reports, and training curves.

    Rates are fractions in [0, 1]. Coverage and completion are separate metrics;
    unavailable scores are omitted rather than logged as zero.
    """
    metrics = {"eval/prompt_rewrite/complete": float(report["complete"])}
    for reference, groups in report["comparisons"].items():
        prefix = f"eval/prompt_rewrite/vs_{reference}"
        for name, value in groups["scores"].items():
            if value is not None:
                metrics[f"{prefix}/{name}"] = value
        for name, source in SCORE_SOURCES.items():
            metrics[f"{prefix}/coverage/{name}"] = groups["by_source"].get(source, {}).get("coverage", 0.0)
    return metrics


def write_tensorboard(report: dict, log_dir: Path, step: int = 0) -> dict:
    """Write standard scalar events without loading PyTorch or a GPU runtime."""
    if step < 0:
        raise ValueError("TensorBoard step must be nonnegative")
    try:
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
    except ImportError as exc:
        raise RuntimeError(
            "TensorBoard logging requires tensorboard (included in the verl runtime). "
            "For a standalone report, use uv run --no-project --python 3.12 --with tensorboard==2.20.0, "
            "or pass --no-tensorboard for JSON-only output."
        ) from exc
    log_dir = Path(log_dir).expanduser().resolve()
    metrics = tensorboard_scalars(report)
    writer = EventFileWriter(str(log_dir))
    try:
        values = [Summary.Value(tag=tag, simple_value=value) for tag, value in metrics.items()]
        writer.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=values)))
        writer.flush()
    finally:
        writer.close()
    return {"log_dir": str(log_dir), "step": step, "scalar_tags": list(metrics)}
