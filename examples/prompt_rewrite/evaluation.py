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
"""Small orchestration API reused by zero-shot, SFT, and RL evaluation jobs."""

import os
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from . import h3, judge, reporting, rewrite, scoring
from .config import (
    DEFAULT_DATASET,
    DEFAULT_EVAL_REPO,
    DEFAULT_REFERENCES,
    DEFAULT_REWRITE_INSTRUCTION,
    DEFAULT_SYSTEM_PROMPT,
    H3_CONCURRENCY,
    H3_SETTINGS,
    JUDGE_CONCURRENCY,
    JUDGE_SETTINGS,
    REFERENCE_VIDEO_FIELDS,
    ROOT,
    RewriteConfig,
    build_messages,
)
from .dataset import load_samples, reference_inventory, verify_references
from .storage import (
    atomic_text,
    file_sha256,
    fingerprint,
    freeze_json,
    item_key,
    load_external,
    read_json,
    run_lock,
    write_json,
    write_jsonl,
)

EXTERNAL_FILES = (
    "scripts/generate_h3_repeats.py",
    "scripts/run_gemini_pairwise_eval.py",
    "scripts/build_pairwise_manifest.py",
    "bestofn/normalize.py",
)


class EvaluationRun:
    """One checkpoint, one frozen dataset, and independently resumable stages."""

    def __init__(self, run_dir: Path):
        self.path = Path(run_dir).expanduser().resolve()
        self.protocol = read_json(self.path / "protocol.json")
        if read_json(self.path / "protocol-sha256.json") != fingerprint(self.protocol):
            raise ValueError("Frozen evaluation protocol changed; use a new run directory")
        if self.protocol.get("schema_version") != 1:
            raise ValueError("Unsupported evaluation protocol version")

    @classmethod
    def prepare(
        cls,
        run_dir: Path,
        *,
        name: str = "qwen38-zero-shot",
        config: RewriteConfig | None = None,
        dataset: Path = DEFAULT_DATASET,
        references: Path = DEFAULT_REFERENCES,
        eval_repo: Path = DEFAULT_EVAL_REPO,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        rewrite_instruction: str = DEFAULT_REWRITE_INSTRUCTION,
        limit: int | None = None,
    ):
        """Inspect and freeze real reference videos; no inference or network requests."""
        run_dir, dataset = Path(run_dir).expanduser().resolve(), Path(dataset).expanduser().resolve()
        references, eval_repo = Path(references).expanduser().resolve(), Path(eval_repo).expanduser().resolve()
        config = config or RewriteConfig()
        if not name.strip() or not system_prompt.strip():
            raise ValueError("Run name and system prompt must be nonempty")
        with run_lock(run_dir):
            dependencies = {relative: file_sha256(eval_repo / relative) for relative in EXTERNAL_FILES}
            rubric = load_external(eval_repo / "scripts/build_pairwise_manifest.py")
            if rubric.INSTRUCTIONS_VERSION != JUDGE_SETTINGS["instructions_version"]:
                raise ValueError("Expected the existing v4 Gemini comparison rubric")
            probe = load_external(eval_repo / "bestofn/normalize.py").probe
            samples, profile, host = reference_inventory(load_samples(dataset, limit), references, probe)
            protocol = {
                "schema_version": 1,
                "name": name,
                "dataset": str(dataset),
                "dataset_sha256": file_sha256(dataset),
                "reference_manifest": str(references / "manifest.json"),
                "reference_manifest_sha256": file_sha256(references / "manifest.json"),
                "reference_video_fields": REFERENCE_VIDEO_FIELDS,
                "samples": samples,
                "rewriter": asdict(config),
                "system_prompt": system_prompt,
                "rewrite_instruction": rewrite_instruction,
                "chat_template_kwargs": {"enable_thinking": False},
                "h3": H3_SETTINGS,
                "h3_host": host,
                "context_ir_rewriter_enabled": False,
                "media_profile": profile,
                "judge": JUDGE_SETTINGS,
                "rubric": rubric.INSTRUCTIONS,
                "eval_repo": str(eval_repo),
                "dependencies": dependencies,
            }
            freeze_json(run_dir / "protocol.json", protocol)
            freeze_json(run_dir / "protocol-sha256.json", fingerprint(protocol))
            atomic_text(run_dir / "system_prompt.txt", system_prompt + "\n")
            atomic_text(run_dir / "rewrite_instruction.txt", rewrite_instruction + "\n")
        return cls(run_dir)

    @classmethod
    def prepare_reference_benchmark(cls, run_dir: Path, *, source_run: Path, name: str = "h3-context-ir-vs-prompt"):
        """Compare the two saved references from an existing evaluation cohort.

        Context-IR is the candidate, and the original prompt is the reference.
        No rewriter or video-generation stage is involved.
        """
        if not name.strip():
            raise ValueError("Run name must be nonempty")
        source = cls(source_run)
        source.check_dependencies()
        verify_references(source.protocol["samples"])
        protocol = {
            key: value
            for key, value in source.protocol.items()
            if key not in {"rewriter", "system_prompt", "rewrite_instruction", "chat_template_kwargs"}
        }
        protocol.update(
            name=name,
            candidate_field="h3_context_ir",
            reference_fields=["prompt"],
            source_run=str(source.path),
            source_protocol_sha256=fingerprint(source.protocol),
        )
        run_dir = Path(run_dir).expanduser().resolve()
        with run_lock(run_dir):
            freeze_json(run_dir / "protocol.json", protocol)
            freeze_json(run_dir / "protocol-sha256.json", fingerprint(protocol))
        return cls(run_dir)

    def check_dependencies(self):
        # H3 settings belong to the frozen protocol, not today's defaults.
        for relative, digest in self.protocol["dependencies"].items():
            if file_sha256(Path(self.protocol["eval_repo"]) / relative) != digest:
                raise ValueError(f"External evaluator changed since preparation: {relative}. Use a new run.")

    def status(self) -> dict:
        """Read progress without loading models, credentials, or API clients."""
        n = len(self.protocol["samples"])
        summary = {
            "run": str(self.path),
            "candidate": self.protocol["name"],
            "prompts": n,
            "sources": dict(Counter(s["source"] for s in self.protocol["samples"])),
            "reused_reference_videos": 2 * n,
        }
        if "candidate_field" in self.protocol:
            return {**summary, "planned_new_videos": 0, "planned_judgments": 2 * n}
        records = rewrite.load_rewrites(self.path, self.protocol, require_complete=False)
        statuses = Counter()
        for sample in self.protocol["samples"]:
            path = self.path / "videos" / item_key(sample["id"]) / "state.json"
            state = read_json(path) if path.exists() else {}
            status = state.get("status", "pending")
            if status == "succeeded" and not path.with_name("rewritten_prompt.mp4").is_file():
                status = "missing_file"
            statuses[status] += 1
        return {
            **summary,
            "rewrites_complete": len(records),
            "video_statuses": dict(statuses),
            "planned_new_videos": n,
            "planned_judgments": 4 * n,
        }

    def _require_rewriter(self):
        if "rewriter" not in self.protocol:
            raise ValueError("A reference benchmark uses existing videos; only judge, report, and status apply")

    def rewrite(self, *, execute: bool = False, rewriter=None, batch_size: int = 8):
        """Use vLLM or a callable backed by the trainer's loaded policy."""
        self._require_rewriter()
        with run_lock(self.path):
            if not execute:
                return {
                    "rewriter": self.protocol["rewriter"],
                    "chat_template_kwargs": {"enable_thinking": False},
                    "example_messages": build_messages(
                        self.protocol["samples"][0]["prompt"],
                        self.protocol["system_prompt"],
                        self.protocol.get("rewrite_instruction", ""),
                    ),
                    **self.status(),
                }
            backend = (
                rewriter if rewriter is not None else rewrite.VLLMRewriter(RewriteConfig(**self.protocol["rewriter"]))
            )
            rewrite.run_rewrites(self.path, self.protocol, backend, batch_size)
            return self.status()

    def import_rewrites(self, rows: list[dict]):
        """Accept outputs from an SFT/RL rollout without starting another policy process."""
        self._require_rewriter()
        with run_lock(self.path):
            rewrite.import_rewrites(self.path, self.protocol, rows)
            return self.status()

    def generation_tasks(self) -> list[dict]:
        self._require_rewriter()
        records = rewrite.load_rewrites(self.path, self.protocol)
        tasks = []
        for sample in self.protocol["samples"]:
            directory = self.path / "videos" / item_key(sample["id"])
            tasks.append(
                {
                    "id": sample["id"],
                    "request": h3.request_body(records[sample["id"]]["rewritten_prompt"], self.protocol["h3"]),
                    "output": str(directory / "rewritten_prompt.mp4"),
                    "state": str(directory / "state.json"),
                }
            )
        freeze_json(self.path / "generation.json", tasks)
        return tasks

    def generate(self, *, execute: bool = False, concurrency: int = H3_CONCURRENCY, poll_interval: float = 20):
        with run_lock(self.path):
            self.check_dependencies()
            verify_references(self.protocol["samples"])
            tasks = self.generation_tasks()
            if execute:
                repo = Path(self.protocol["eval_repo"])
                client = load_external(repo / "scripts/generate_h3_repeats.py")
                client.load_env()
                host = os.environ.get("H3_HOST", "").rstrip("/")
                token = os.environ.get("H3_TOKEN", "")
                if not host or not token:
                    raise ValueError("Set H3_HOST/H3_TOKEN in the environment or eval repo's arena/.env")
                if host != self.protocol["h3_host"]:
                    raise ValueError("H3 host differs from the frozen reference generation host")
                h3.generate_tasks(
                    tasks,
                    client,
                    host,
                    client.auth_headers(token),
                    load_external(repo / "bestofn/normalize.py").probe,
                    self.protocol["media_profile"],
                    concurrency,
                    poll_interval,
                )
            return self.status()

    def _generated_candidates(self) -> dict:
        """Validate the candidate generation states and their downloaded clips."""
        candidates = {}
        for task in self.generation_tasks():
            state_path, video_path = Path(task["state"]), Path(task["output"])
            if not state_path.exists():
                raise ValueError(f"Candidate video has not been generated: {task['id']}")
            state = read_json(state_path)
            if (
                state.get("status") != "succeeded"
                or state.get("request_sha256") != fingerprint(task["request"])
                or state.get("host") != self.protocol["h3_host"]
                or state.get("endpoint") != h3.H3_ENDPOINT
            ):
                raise ValueError(f"Incomplete or mismatched candidate video: {task['id']}")
            digest = file_sha256(video_path)
            if digest != state.get("sha256") or video_path.stat().st_size != state.get("bytes"):
                raise ValueError(f"Candidate video changed: {video_path}")
            if h3.media_profile(state["media"]) != self.protocol["media_profile"]:
                raise ValueError(f"Candidate video has a different native format: {video_path}")
            candidates[task["id"]] = {"path": str(video_path), "sha256": digest}
        return candidates

    def comparison_inputs(self) -> dict:
        """Verify every clip before preparing or consuming judge results."""
        self.check_dependencies()
        verify_references(self.protocol["samples"])
        if "candidate_field" in self.protocol:
            candidates = {
                sample["id"]: sample["references"][self.protocol["candidate_field"]]
                for sample in self.protocol["samples"]
            }
        else:
            candidates = self._generated_candidates()
        freeze_json(self.path / "judge" / "candidate-videos.json", candidates)
        manifests = judge.build_manifests(self.protocol, candidates)
        judge.freeze_manifests(self.path, manifests)
        return manifests

    def judge(
        self,
        *,
        execute: bool = False,
        concurrency: int = JUDGE_CONCURRENCY,
        neutral_on_failure: bool = False,
        request_timeout_seconds: float = judge.REQUEST_TIMEOUT_SECONDS,
        pass_timeout_seconds: float = judge.PASS_TIMEOUT_SECONDS,
    ):
        with run_lock(self.path):
            self.comparison_inputs()
            judge.run_judge(
                self.path,
                self.protocol,
                execute,
                concurrency,
                neutral_on_failure=neutral_on_failure,
                request_timeout_seconds=request_timeout_seconds,
                pass_timeout_seconds=pass_timeout_seconds,
            )

    def report(
        self,
        *,
        allow_partial: bool = False,
        tensorboard_dir: Path | None = None,
        step: int = 0,
        log_tensorboard: bool = True,
    ) -> dict:
        """Save dataset scores and log them at the supplied training step.

        An explicit log directory takes precedence over TENSORBOARD_DIR, which
        lets SFT/RL evaluations join the trainer's existing TensorBoard run.
        """
        if step < 0:
            raise ValueError("TensorBoard step must be nonnegative")
        with run_lock(self.path):
            manifests = self.comparison_inputs()
            judgments, counts = {}, {}
            for order in judge.ORDERS:
                judgments[order], counts[order] = judge.read_judgments(
                    self.path / "judge" / f"judgments-{order}.jsonl",
                    manifests[order],
                    self.path / "judge" / f"manifest-{order}.json",
                    self.protocol["judge"],
                )
            report, rows = scoring.make_report(self.protocol, manifests, judgments, allow_partial)
            report.update(protocol_sha256=fingerprint(self.protocol), ledger_counts=counts)
            if log_tensorboard:
                log_dir = (
                    tensorboard_dir
                    or os.environ.get("TENSORBOARD_DIR")
                    or (ROOT / "tensorboard_log" / "prompt-rewrite" / self.path.name)
                )
                report["tensorboard"] = reporting.write_tensorboard(report, Path(log_dir), step)
            write_json(self.path / "report.json", report)
            write_jsonl(self.path / "per_prompt_scores.jsonl", rows)
            return report
