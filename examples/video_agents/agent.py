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
"""Thin H3 tool and group-scoring adapters for the native synchronous V1 trainer."""

import asyncio
import json
from collections import Counter
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import ray
import transfer_queue as tq
from jsonschema import Draft202012Validator, ValidationError

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse
from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ, AgentLoopWorkerTQBase
from verl.utils.tensordict_utils import get_non_tensor_data

from .config import REFERENCE_FIELDS
from .h3 import generation_slot
from .observability import ProviderError, failure_record, record_event, timed_stage
from .storage import item_key, write_json
from .tournament import generate_video, load_protocol, runtime_protocol, score_group

# Decoded tokens kept with an invalid call, so the policy's actual output can be read.
RESPONSE_TAIL_TOKENS = 512


def log(message: str):
    """One job-log line per notable event; the referenced files hold the details."""
    print(f"video-agents: {message}", flush=True)


class SolH3Tool(BaseTool):
    """Generate one clip using the existing durable H3 worker."""

    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.protocol_path = Path(config["protocol"])
        self.output_dir = Path(config.get("output_dir", self.protocol_path.parent))
        self.protocol = load_protocol(self.protocol_path)
        self.slots = asyncio.Semaphore(self.protocol["h3_concurrency_per_worker"])

    def get_openai_tool_schema(self):
        # Bootstrap native tool registration from data, not a second prompt source.
        # H3TerminalLoop supplies the current row's raw schema at render time.
        with Path(self.config["dataset"]).open(encoding="utf-8") as handle:
            row = json.loads(next(line for line in handle if line.strip()))
        return OpenAIFunctionToolSchema.model_validate(row["tools"][0])

    @staticmethod
    def build_request(parameters, schema):
        """Validate and apply the defaults advertised in this dataset row."""
        spec = schema["function"]["parameters"]
        validator = Draft202012Validator(spec)
        validator.validate(parameters)
        request = {key: value["default"] for key, value in spec["properties"].items() if "default" in value}
        request.update(parameters)
        validator.validate(request)
        # The gateway uses strict integer fields and strips surrounding prompt
        # whitespace. JSON Schema alone accepts integral floats and blank text.
        if not request["prompt"].strip() or any(type(request[key]) is not int for key in ("duration", "seed")):
            raise ValueError("Sol-H3 requires nonblank prompt text and integer duration/seed")
        request["prompt"] = request["prompt"].strip()
        return request

    async def execute(self, instance_id, parameters, **kwargs):
        agent_data = kwargs["agent_data"]
        request = self.build_request(parameters, agent_data._active_tool_schemas[0])
        directory = self.output_dir / "videos" / item_key(instance_id)
        timing_path = getattr(agent_data, "_video_timing_path", directory / "timings.jsonl")
        with timed_stage(timing_path, "h3_slot_wait", artifact=str(directory)):
            await self.slots.acquire()
        try:
            try:

                def generate():
                    # Keep the shared slot in the executor thread until its H3
                    # subprocess has exited, including if this coroutine is cancelled.
                    with generation_slot(self.output_dir, self.protocol["h3_max_inflight"], timing_path):
                        return generate_video(directory, self.protocol, request)

                result = await asyncio.to_thread(generate)
            except Exception as exc:
                # The native tool loop catches exceptions. Preserve their class so
                # only service failures can be dropped; code/artifact errors abort.
                result = {
                    "status": "infrastructure_error" if isinstance(exc, ProviderError) else "execution_error",
                    "directory": str(directory),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                write_json(directory / "failure.json", failure_record(exc, directory=str(directory)))
        finally:
            self.slots.release()
        agent_data.extra_fields["h3_result"] = result
        return ToolResponse(text=json.dumps({"status": result["status"], "artifact": str(directory)})), None, {}


class H3TerminalLoop(ToolAgentLoop):
    """Use native parsing and token masks, ending immediately after the H3 call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        client = self.server_manager
        self.stop_reason = None

        async def generate(**request):
            output = await client.generate(**request)
            self.stop_reason = output.stop_reason
            if output.stop_reason not in {"completed", "length"} or not output.token_ids:
                raise RuntimeError(f"LLM request did not complete: {output.stop_reason}")
            return output

        # Inspect the native client's completion status before parsing its output.
        # An aborted request can contain partial tokens that look like a bad call.
        self.server_manager = SimpleNamespace(generate=generate)

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        timeout = self.tools["sol_h3"].protocol["timeouts"]["llm"]
        agent_data._video_timing_path = self.timing_path
        with timed_stage(self.timing_path, "llm", **self.timing_fields, timeout_seconds=timeout) as timing:
            try:
                result = await asyncio.wait_for(
                    super()._handle_generating_state(agent_data, sampling_params, ignore_termination),
                    timeout=timeout,
                )
            except TimeoutError as exc:
                # The limit sits above the longest turn max_response_length allows,
                # so only a stalled server reaches it; drop that group like any
                # other service failure. Long outputs are scored, never dropped.
                raise ProviderError("llm", f"LLM generation exceeded {timeout}s") from exc
            timing["response_tokens"] = len(agent_data.response_ids)
            return result

    async def _handle_processing_tools_state(self, agent_data):
        calls, reason = agent_data.tool_calls, None
        if len(calls) != 1:
            reason = f"expected one tool call, got {len(calls)}"
        elif calls[0].name != "sol_h3":
            reason = f"unknown tool {calls[0].name!r}"
        else:
            try:
                self.tools["sol_h3"].build_request(json.loads(calls[0].arguments), agent_data._active_tool_schemas[0])
            except ValidationError as exc:
                reason = f"invalid arguments: {exc.message}"
            except (ValueError, TypeError) as exc:
                reason = f"invalid arguments: {exc}"
        if reason:
            agent_data.extra_fields["h3_result"] = {"status": "invalid_call", "reason": reason[:500]}
            return AgentState.TERMINATED
        await super()._handle_processing_tools_state(agent_data)
        result = agent_data.extra_fields.get("h3_result", {"status": "missing", "error": "the tool returned nothing"})
        if result["status"] == "succeeded":
            return AgentState.TERMINATED
        message = f"H3 {result.get('error_type', result['status'])}: {result.get('error')}"
        if "directory" in result:
            message += f"; see {result['directory']}/failure.json"
        if result["status"] == "infrastructure_error":
            raise ProviderError("h3", message)
        raise RuntimeError(message)

    async def run(self, sampling_params, **kwargs):
        self.timing_path = Path(kwargs.get("_video_timings", Path(self.config.video_rl.output_dir) / "timings.jsonl"))
        self.timing_fields = {key: kwargs[key] for key in ("uid", "session_id") if key in kwargs}
        schemas = kwargs.get("tools")
        template_kwargs = kwargs.get("chat_template_kwargs")
        if (
            not isinstance(schemas, list)
            or len(schemas) != 1
            or schemas[0].get("function", {}).get("name") != "sol_h3"
            or not isinstance(template_kwargs, dict)
        ):
            raise ValueError("Each dataset row must supply sol_h3 tools and chat_template_kwargs")
        # Tool instances are shared across concurrent rollouts. Keep parser metadata
        # local to this loop, while sharing the executor's semaphore and protocol.
        tool = copy(self.tools["sol_h3"])
        tool.output_dir = Path(self.config.video_rl.output_dir)
        tool.protocol = runtime_protocol(tool.protocol, self.config.video_rl)
        tool.tool_schema = OpenAIFunctionToolSchema.model_validate(schemas[0])
        self.tools = {"sol_h3": tool}
        # Render the raw schema so JSON Schema fields such as maxLength survive
        # the native parser's narrower Pydantic model.
        self.tool_schemas = schemas
        self.apply_chat_template_kwargs = dict(template_kwargs)
        self.continuous_token_builder.chat_template_kwargs = self.apply_chat_template_kwargs
        output = await super().run(sampling_params, **kwargs)
        if not output.response_ids:
            raise RuntimeError("The LLM server returned no tokens")
        reason = "no tool call" + (" before the length limit" if self.stop_reason == "length" else "")
        result = output.extra_fields.setdefault("h3_result", {"status": "invalid_call", "reason": reason})
        if result["status"] == "invalid_call":
            result["response_tail"] = self.tokenizer.decode(output.response_ids[-RESPONSE_TAIL_TOKENS:])
        record_event(self.timing_path, "tool_call", "completed", **self.timing_fields, status=result["status"])
        return output


class TournamentWorker(AgentLoopWorkerTQBase):
    """Score all siblings before the native worker publishes their trajectories."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.protocol_path = Path(self.config.video_rl.protocol)
        self.output_dir = Path(self.config.video_rl.output_dir)
        self.protocol = runtime_protocol(load_protocol(self.protocol_path), self.config.video_rl)
        self.samples = {
            split: {sample["id"]: sample for sample in self.protocol["samples"][split]} for split in ("train", "val")
        }
        self.group_slots = asyncio.Semaphore(self.protocol["groups_per_worker"])

    async def _agent_loop_postprocess(self, output, validate, **kwargs):
        # Native _run_agent_loop still supplies templates, parsing, tokens and masks.
        # Delay the native TQ publication until the complete tournament is scored.
        return output

    async def generate_sequences(self, batch):
        """Return one record per prompt group: its rewards, or why it was dropped."""
        await super().generate_sequences(batch)
        # Sync training must observe infrastructure failures before sampling the
        # queue. The native fire-and-forget worker otherwise marks failed groups
        # sampleable, and the trainer may pad their missing trajectories.
        results = await asyncio.gather(*list(self.background_tasks), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results

    async def _run_prompt(self, prompt, sampling_params, trajectory, trace=False):
        partition = "val" if trajectory["validate"] else "train"
        step = int(prompt["global_steps"])
        directory = self.output_dir / partition / f"step_{step}" / item_key(prompt["uid"])
        with timed_stage(directory / "timings.jsonl", "group_slot_wait"):
            await self.group_slots.acquire()
        try:
            # Run-level index: one start/end pair per group, pointing at its files.
            # An unmatched start marks a group that was in flight when the job stopped.
            with timed_stage(
                self.output_dir / "timings.jsonl",
                "group",
                partition=partition,
                policy_step=step,
                uid=prompt["uid"],
                sample_id=prompt["extra_info"]["id"],
                directory=str(directory),
            ) as timing:
                record = await self._run_group(prompt, sampling_params, trajectory, trace, directory)
                if record["dropped"]:
                    timing.update(status="dropped", failure_stage=record["stage"], error_type=record["type"])
                else:
                    timing["scores"] = [reward["score"] for reward in record["rewards"]]
                return record
        finally:
            self.group_slots.release()

    async def _run_group(self, prompt, sampling_params, trajectory, trace, directory):
        uid, validate = prompt["uid"], trajectory["validate"]
        partition = "val" if validate else "train"
        expected_n = 1 if validate else self.protocol["group_size"]
        n = prompt.pop("__rollout_n__", expected_n)
        if n != expected_n or not prompt.pop("__do_sample__", True):
            raise ValueError(
                f"Use {expected_n} sampled rollouts for this {'validation' if validate else 'training'} group"
            )
        sample = self.samples[partition][prompt["extra_info"]["id"]]
        await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "running"})
        try:
            results = await asyncio.gather(
                *[
                    self._run_agent_loop(
                        sampling_params,
                        trajectory=trajectory,
                        trace=trace,
                        session_id=i,
                        _video_timings=str(directory / "timings.jsonl"),
                        **prompt,
                    )
                    for i in range(n)
                ],
                return_exceptions=True,
            )
            # Settle every sibling, and never let a provider error hide a code,
            # artifact, or cancellation error from another sibling.
            errors = [result for result in results if isinstance(result, BaseException)]
            for error in errors:
                if not isinstance(error, ProviderError):
                    raise error
            if errors:
                raise errors[0]
            write_json(
                directory / "trajectories.json",
                {
                    "uid": uid,
                    "policy_step": int(prompt["global_steps"]),
                    "sample_id": sample["id"],
                    "rollouts": [
                        {
                            "prompt_ids": output.prompt_ids,
                            "response_ids": output.response_ids,
                            "response_logprobs": output.response_logprobs,
                            "response_mask": output.response_mask,
                            "h3_result": output.extra_fields["h3_result"],
                        }
                        for output in results
                    ],
                },
            )
            rewards = await asyncio.to_thread(
                score_group,
                directory,
                self.protocol,
                sample,
                [output.extra_fields["h3_result"] for output in results],
                validate=validate,
            )
            for i, (output, reward) in enumerate(zip(results, rewards, strict=True)):
                output.reward_score = reward["score"]
                output.extra_fields["reward_extra_info"] = reward
                await super()._agent_loop_postprocess(output, validate, session_id=i, **prompt)
            await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "finished"})
            return {"dropped": False, "uid": uid, "sample_id": sample["id"], "rewards": rewards}
        except BaseException as exc:
            # A held-out prompt that cannot be scored for any reason is excluded
            # and reported in eval-summary.json; evaluation never stops training.
            drop = isinstance(exc, Exception) and (
                validate
                or (isinstance(exc, ProviderError) and self.config.trainer.v1.sampler.sync_refill_failed_groups)
            )
            failure = failure_record(
                exc,
                dropped=bool(drop),
                validate=validate,
                uid=uid,
                sample_id=sample["id"],
                policy_step=int(prompt["global_steps"]),
            )
            write_json(directory / "failure.json", failure)
            log(
                f"{partition} step {failure['policy_step']} {sample['id']} {'dropped' if drop else 'failed'} "
                f"at {failure['stage']}: {failure['type']}: {failure['error'][:300]} "
                f"(see {directory / 'failure.json'})"
            )
            await tq.async_kv_put(key=uid, partition_id=partition, tag={"status": "failure"})
            if drop:
                # No trajectory or reward is published. Native sync refill replaces
                # a training group, preserving the four-candidate GRPO objective;
                # the validation sampler skips a failed group without refill.
                return failure
            raise


class TournamentManager(AgentLoopManagerTQ):
    """Keep native worker dispatch and enforce synchronous, complete-group rewards."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.config.trainer.use_v1 or self.config.trainer.v1.trainer_mode != "sync":
            raise ValueError("H3 tournaments require the native synchronous V1 trainer")
        group_size = load_protocol(Path(self.config.video_rl.protocol))["group_size"]
        if self.rollout_config.n != group_size or self.rollout_config.val_kwargs.n != 1:
            raise ValueError(f"Use rollout.n={group_size} and rollout.val_kwargs.n=1")
        if self.config.video_rl.max_dropped_groups_per_step < 1:
            raise ValueError("video_rl.max_dropped_groups_per_step must be positive")
        self.reward_loop_worker_handles = None
        self.agent_loop_workers_class = ray.remote(TournamentWorker)
        self._drop_step, self._dropped_groups = None, 0
        self._eval_step, self._eval_records = None, []

    def generate_sequences(self, prompts):
        """Use native dispatch and bound replacement work during a service outage."""
        step = int(get_non_tensor_data(prompts, "global_steps", 0))
        validate = bool(get_non_tensor_data(prompts, "validate", False))
        partition = "val" if validate else "train"
        if self._drop_step != step:
            self._drop_step, self._dropped_groups = step, 0
        timings = Path(self.config.video_rl.output_dir) / "timings.jsonl"
        with timed_stage(timings, "dispatch", policy_step=step, validate=validate, prompts=len(prompts)) as timing:
            chunks = prompts.chunk(len(self.agent_loop_workers))
            results = ray.get(
                [
                    worker.generate_sequences.remote(chunk)
                    for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
                ]
            )
            records = [record for result in results for record in result]
            dropped = Counter(record["stage"] for record in records if record["dropped"])
            timing["dropped"] = dict(dropped)
            if not validate:
                self._dropped_groups += dropped.total()
                timing["total_dropped_groups"] = self._dropped_groups
        log(
            f"{partition} step {step}: {len(records) - dropped.total()}/{len(records)} groups scored"
            + (f"; dropped by stage {dict(dropped)} (see failure.json in each group)" if dropped else "")
        )
        if validate:
            self._record_evaluation(step, records)
            return
        limit = self.config.video_rl.max_dropped_groups_per_step
        if self._dropped_groups > limit:
            raise RuntimeError(
                f"Service failures exceeded {limit} dropped groups at step {step}; "
                "stopping before an optimizer update. Inspect failure.json and timings.jsonl."
            )
        self._check_judge_coverage(step, records)

    def _check_judge_coverage(self, step, records):
        """Bound synthetic judgments over the completed batch, including refill dispatches."""
        if getattr(self, "_judge_step", None) != step:
            self._judge_step = step
            self._judge_groups = self._judge_comparisons = self._judge_fallbacks = 0
        for record in records:
            if not record["dropped"]:
                # Each of the four trajectory rewards repeats these group totals.
                reward = record["rewards"][0]
                self._judge_groups += 1
                self._judge_comparisons += reward["judge_comparisons"]
                self._judge_fallbacks += reward.get("judge_fallback_comparisons", 0)
        fraction = self._judge_fallbacks / self._judge_comparisons if self._judge_comparisons else 0.0
        record_event(
            Path(self.config.video_rl.output_dir) / "timings.jsonl",
            "judge_coverage",
            "summarized",
            policy_step=step,
            groups=self._judge_groups,
            comparisons=self._judge_comparisons,
            fallbacks=self._judge_fallbacks,
            fallback_fraction=fraction,
        )
        limit = self.config.video_rl.get("max_judge_fallback_fraction", 0.75)
        if self._judge_groups >= self.config.data.train_batch_size and fraction > limit:
            raise RuntimeError(
                f"Gemini fallback fraction {fraction:.1%} exceeded {limit:.1%} at step {step}; "
                "stopping before an optimizer update. Inspect judge/fallbacks-*.json and timings.jsonl."
            )

    def _record_evaluation(self, step, records):
        """Summarize held-out prompts; unscorable ones are excluded, never fatal."""
        if self._eval_step != step:
            self._eval_step, self._eval_records = step, []
        self._eval_records.extend(records)
        output_dir = Path(self.config.video_rl.output_dir)
        path = output_dir / "val" / f"step_{step}" / "eval-summary.json"
        summary = evaluation_summary(step, self._eval_records, self.config.video_rl.min_valid_eval_fraction)
        write_json(path, summary)
        record_event(
            output_dir / "timings.jsonl",
            "evaluation",
            "summarized",
            policy_step=step,
            **{key: summary[key] for key in ("prompts", "valid", "dropped", "degraded")},
        )
        log(
            f"val step {step}: score {summary['score']} over {summary['valid']}/{summary['prompts']} valid prompts"
            f"{' (degraded)' if summary['degraded'] else ''}; see {path}"
        )
        if records and all(record["dropped"] for record in records):
            # The native validation loop cannot report an empty batch; this is an outage.
            raise RuntimeError(f"Every held-out prompt in the batch failed at step {step}; inspect {path}")


def evaluation_summary(step: int, records: list[dict], min_valid_fraction: float) -> dict:
    """Valid-only held-out means; invalid policy calls stay in the denominator as 0."""
    valid = [record["rewards"][0] for record in records if not record["dropped"]]
    fraction = len(valid) / len(records) if records else 0.0

    def mean(key):
        return sum(reward[key] for reward in valid) / len(valid) if valid else None

    return {
        "policy_step": step,
        "prompts": len(records),
        "valid": len(valid),
        "dropped": len(records) - len(valid),
        "valid_fraction": fraction,
        "degraded": fraction < min_valid_fraction,
        "score": mean("score"),
        **{f"vs_{reference}": mean(f"vs_{reference}") for reference in REFERENCE_FIELDS},
        "invalid_calls": sum(reward["valid_call"] == 0 for reward in valid),
        "dropped_examples": sorted(
            (
                {
                    "sample_id": record["sample_id"],
                    "stage": record["stage"],
                    "type": record["type"],
                    "error": record["error"][:500],
                }
                for record in records
                if record["dropped"]
            ),
            key=lambda row: row["sample_id"],
        ),
    }
