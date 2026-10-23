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
from .observability import ProviderError, record_event, timed_stage
from .storage import item_key, write_json
from .tournament import GROUP_SIZE, generate_video, load_protocol, runtime_protocol, score_group


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
                result = await asyncio.to_thread(generate_video, directory, self.protocol, request)
            except Exception as exc:
                # The native tool loop catches exceptions. Preserve their class so
                # only service failures can be dropped; code/artifact errors abort.
                result = {
                    "status": "infrastructure_error" if isinstance(exc, ProviderError) else "execution_error",
                    "directory": str(directory),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
                write_json(directory / "failure.json", result)
        finally:
            self.slots.release()
        agent_data.extra_fields["h3_result"] = result
        return ToolResponse(text=json.dumps({"status": result["status"], "artifact": str(directory)})), None, {}


class H3TerminalLoop(ToolAgentLoop):
    """Use native parsing and token masks, ending immediately after the H3 call."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        client = self.server_manager

        async def generate(**request):
            output = await client.generate(**request)
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
                # A slow-tail generation drops its group like a provider failure.
                raise ProviderError("llm", f"LLM generation exceeded {timeout}s") from exc
            timing["response_tokens"] = len(agent_data.response_ids)
            return result

    async def _handle_processing_tools_state(self, agent_data):
        calls = agent_data.tool_calls
        valid = len(calls) == 1 and calls[0].name == "sol_h3"
        if valid:
            try:
                self.tools["sol_h3"].build_request(json.loads(calls[0].arguments), agent_data._active_tool_schemas[0])
            except (ValueError, TypeError, ValidationError):
                valid = False
        if not valid:
            agent_data.extra_fields["h3_result"] = {"status": "invalid_call"}
            return AgentState.TERMINATED
        await super()._handle_processing_tools_state(agent_data)
        result = agent_data.extra_fields.get("h3_result", {})
        if result.get("status") == "infrastructure_error":
            raise ProviderError("h3", f"H3 execution failed: {result}")
        if result.get("status") != "succeeded":
            raise RuntimeError(f"H3 execution failed: {result}")
        return AgentState.TERMINATED

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
        if (kwargs.get("extra_info") or {}).get("tool_selection"):
            raise ValueError("The dataset tools field defines the complete H3 tool selection")
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
        output.extra_fields.setdefault("h3_result", {"status": "invalid_call"})
        record_event(
            self.timing_path,
            "tool_call",
            "completed",
            **self.timing_fields,
            status=output.extra_fields["h3_result"]["status"],
        )
        if not output.response_ids:
            raise RuntimeError("The LLM server returned no tokens")
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
        await super().generate_sequences(batch)
        # Sync training must observe infrastructure failures before sampling the
        # queue. The native fire-and-forget worker otherwise marks failed groups
        # sampleable, and the trainer may pad their missing trajectories.
        tasks = list(self.background_tasks)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return [result for result in results if result is not None]

    async def _run_prompt(self, prompt, sampling_params, trajectory, trace=False):
        partition = "val" if trajectory["validate"] else "train"
        directory = self.output_dir / partition / f"step_{prompt['global_steps']}" / item_key(prompt["uid"])
        with timed_stage(directory / "timings.jsonl", "group_slot_wait"):
            await self.group_slots.acquire()
        try:
            with timed_stage(directory / "timings.jsonl", "group", uid=prompt["uid"]) as timing:
                result = await self._run_group(prompt, sampling_params, trajectory, trace)
                if result and result["dropped"]:
                    timing["status"] = "dropped"
                return result
        finally:
            self.group_slots.release()

    async def _run_group(self, prompt, sampling_params, trajectory, trace):
        uid, validate = prompt["uid"], trajectory["validate"]
        partition = "val" if validate else "train"
        directory = self.output_dir / partition / f"step_{prompt['global_steps']}" / item_key(uid)
        expected_n = 1 if validate else GROUP_SIZE
        n = prompt.pop("__rollout_n__", expected_n)
        if n != expected_n or not prompt.pop("__do_sample__", True):
            raise ValueError("Use four sampled training rollouts and one held-out rollout")
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
            if validate:
                return {"dropped": False, "validate": True, "uid": uid, "sample_id": sample["id"], "reward": rewards[0]}
        except BaseException as exc:
            # A held-out prompt that cannot be scored for any reason is excluded
            # and reported in eval-summary.json; evaluation never stops training.
            drop = isinstance(exc, Exception) and (
                validate
                or (isinstance(exc, ProviderError) and self.config.trainer.v1.sampler.sync_refill_failed_groups)
            )
            failure = {
                "type": type(exc).__name__,
                "error": str(exc),
                "stage": getattr(exc, "stage", "internal"),
                "dropped": bool(drop),
                "validate": validate,
                "uid": uid,
                "sample_id": sample["id"],
                "policy_step": int(prompt["global_steps"]),
            }
            write_json(directory / "failure.json", failure)
            record_event(
                directory / "timings.jsonl",
                "group",
                "dropped" if drop else "failed",
                provider_stage=failure["stage"],
                **{key: value for key, value in failure.items() if key not in {"stage", "error"}},
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
        if self.rollout_config.n != GROUP_SIZE or self.rollout_config.val_kwargs.n != 1:
            raise ValueError("Use rollout.n=4 and rollout.val_kwargs.n=1")
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
        if self._drop_step != step:
            self._drop_step, self._dropped_groups = step, 0
        timings = Path(self.config.video_rl.output_dir) / "timings.jsonl"
        with timed_stage(timings, "dispatch", policy_step=step, validate=validate, prompts=len(prompts)):
            chunks = prompts.chunk(len(self.agent_loop_workers))
            results = ray.get(
                [
                    worker.generate_sequences.remote(chunk)
                    for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
                ]
            )
        records = [record for result in results for record in result]
        if validate:
            self._record_evaluation(step, records)
            return
        dropped = sum(record["dropped"] for record in records)
        self._dropped_groups += dropped
        if dropped:
            record_event(
                timings,
                "group_replacement",
                "requested",
                policy_step=step,
                groups=dropped,
                total_dropped_groups=self._dropped_groups,
            )
        limit = self.config.video_rl.max_dropped_groups_per_step
        if self._dropped_groups > limit:
            raise RuntimeError(
                f"Service failures exceeded {limit} dropped groups at step {step}; "
                "stopping before an optimizer update. Inspect failure.json and timings.jsonl."
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
        if records and all(record["dropped"] for record in records):
            # The native validation loop cannot report an empty batch; this is an outage.
            raise RuntimeError(f"Every held-out prompt in the batch failed at step {step}; inspect {path}")


def evaluation_summary(step: int, records: list[dict], min_valid_fraction: float) -> dict:
    """Valid-only held-out means; invalid policy calls stay in the denominator as 0."""
    valid = [record["reward"] for record in records if not record["dropped"]]
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
