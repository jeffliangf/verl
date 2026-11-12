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
"""Native Qwen tool parsing, continuous tokens and policy masks for inspection RL."""

import asyncio
import json
from copy import copy
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from jsonschema import ValidationError
from omegaconf import OmegaConf

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .agent import TournamentManager, TournamentWorker
from .h3 import generation_slot
from .image_assets import image_bytes, upright_observation
from .inspection_episode import InspectionEpisode
from .inspection_worker import inspect_video
from .mixed_data import original_request
from .observability import ProviderError, timed_stage
from .policy_cache import EpisodePolicy
from .storage import read_json, read_jsonl, write_json
from .tournament import generate_video, load_protocol, runtime_protocol

_PROTOCOLS = {}


def episode_protocol(path, settings):
    """Load a frozen protocol once per worker process; episodes only read it.

    Mixed protocols are tens of megabytes. Loading one per episode would block the
    event loop for about a second and retain a separate copy per concurrent episode.
    """
    key = (str(path), json.dumps(OmegaConf.to_container(settings, resolve=True), sort_keys=True, default=str))
    if key not in _PROTOCOLS:
        _PROTOCOLS[key] = runtime_protocol(load_protocol(Path(path)), settings)
    return _PROTOCOLS[key]


def recorded_messages(messages, source_image):
    """Record an upright policy image, which is a PIL object, by its frozen source path."""
    if not source_image:
        return messages
    return [
        {
            **message,
            "content": [
                {**part, "image": source_image["path"]}
                if part.get("type") == "image" and not isinstance(part.get("image"), str)
                else part
                for part in message["content"]
            ],
        }
        if isinstance(message.get("content"), list)
        else message
        for message in messages
    ]


async def replayed_inspection(directory, identity, call, *, required=False):
    """Persist a lookahead episode's inspection outcome; replays must observe the same result.

    Later policy turns are cached against the exact observation, so a resumed
    episode reuses the saved report (or failure) instead of asking Gemini again.
    """
    path = Path(directory) / "outcome.json"
    if path.exists():
        saved = read_json(path)
        if saved["identity"] != identity:
            raise ValueError(f"Saved inspection outcome belongs to another video or request: {path}")
        if saved["status"] == "failed":
            raise ProviderError("inspection", saved["error"])
        return saved["report"]
    if required:
        raise RuntimeError(f"Checkpointed inspection outcome is missing: {path}")
    try:
        report = await call()
    except ProviderError as exc:
        if exc.stage == "inspection":
            write_json(path, {"identity": identity, "status": "failed", "error": str(exc)})
        raise
    write_json(path, {"identity": identity, "status": "succeeded", "report": report})
    return report


class InspectionActionTool(BaseTool):
    """Register actual native actions; state belongs to AgentData, never a shared tool."""

    def get_openai_tool_schema(self):
        row = read_jsonl(Path(self.config["dataset"]))[0]
        return OpenAIFunctionToolSchema.model_validate(
            next(tool for tool in row["tools"] if tool["function"]["name"] == self.config["name"])
        )

    async def execute(self, instance_id, parameters, **kwargs):
        response = await kwargs["agent_data"]._inspection_episode.execute(self.name, parameters)
        return ToolResponse(text=json.dumps(response, ensure_ascii=False)), None, {}


class InspectionLoop(ToolAgentLoop):
    """The environment enforces limits independently of prompts and reward values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_reason = None
        self.policy = None
        client = self.server_manager

        async def generate(**request):
            # Lookahead cohorts cache every turn from one behavior version.
            output = (
                await self.policy.generate(client, request)
                if self.policy is not None
                else await client.generate(**request)
            )
            self.stop_reason = output.stop_reason
            if output.stop_reason not in {"completed", "length"} or not output.token_ids:
                raise ProviderError("llm", f"LLM request did not complete: {output.stop_reason}")
            return output

        self.server_manager = SimpleNamespace(generate=generate)

    async def _handle_pending_state(self, agent_data, sampling_params):
        agent_data._inspection_episode = self.episode
        self.agent_data = agent_data
        return await super()._handle_pending_state(agent_data, sampling_params)

    async def _handle_generating_state(self, agent_data, sampling_params, ignore_termination=False):
        # Reserve a little space for native chat delimiters, never for silent truncation.
        remaining = (
            min(
                self.response_length - len(agent_data.response_mask),
                self.rollout_config.max_model_len - len(agent_data.prompt_ids),
            )
            - 64
        )
        if remaining <= 0 or agent_data.assistant_turns >= 3:
            self.episode.invalid("Episode context or action-turn limit reached")
            return AgentState.TERMINATED
        params = {**sampling_params, "max_tokens": min(self.config.inspection_rl.turn_max_tokens, remaining)}
        with timed_stage(self.episode.directory / "timings.jsonl", "llm") as timing:
            try:
                state = await asyncio.wait_for(
                    super()._handle_generating_state(agent_data, params), timeout=self.protocol["timeouts"]["llm"]
                )
            except TimeoutError as exc:
                raise ProviderError("llm", "Policy server exceeded the per-turn service deadline") from exc
            timing["response_tokens"] = len(agent_data.response_ids)
        if self.stop_reason == "length":
            self.episode.invalid("Policy turn exhausted its token budget")
            return AgentState.TERMINATED
        if state == AgentState.TERMINATED:
            self.episode.invalid("Expected a single available tool action")
        return state

    async def _handle_processing_tools_state(self, agent_data):
        try:
            if len(agent_data.tool_calls) != 1:
                raise ValueError("Expected exactly one tool action per turn")
            call = agent_data.tool_calls[0]
            self.episode.validate_action(call.name, json.loads(call.arguments))
        except (ValueError, TypeError, KeyError, ValidationError) as exc:
            self.episode.invalid(str(exc))
            return AgentState.TERMINATED
        if self.policy is not None and (
            call.name == "finish_video" or (call.name == "sol_h3" and self.episode.generation_calls == 1)
        ):
            # Both actions end the episode. Mark its policy work complete first so a
            # second render and the judging can overlap the next learner update.
            self.policy.complete(call.name)
        # Keep the native token merger and zero masks for observations. Override
        # only dispatch below so unhandled service errors propagate and JSON is not cut.
        state = await super()._handle_processing_tools_state(agent_data)
        if self.episode.final_result is not None:
            return AgentState.TERMINATED
        if state == AgentState.TERMINATED:
            self.episode.invalid("Complete tool observation does not fit the episode context")
        return state

    async def _call_tool(self, tool_call, tools_kwargs, agent_data):
        tool = self.tools[tool_call.name]
        # All actions use episode state; they need no global create/release state.
        return await tool.execute(agent_data.request_id, json.loads(tool_call.arguments), agent_data=agent_data)

    async def run(self, sampling_params, **kwargs):
        self.protocol = episode_protocol(self.config.video_rl.protocol, self.config.video_rl)
        schemas = kwargs["tools"]
        if [s["function"]["name"] for s in schemas] != ["sol_h3", "inspect_video", "finish_video"]:
            raise ValueError("Prepared inspection rows must supply the three native actions in order")
        self.tools = {name: copy(self.tools[name]) for name in ("sol_h3", "inspect_video", "finish_video")}
        for schema in schemas:
            self.tools[schema["function"]["name"]].tool_schema = OpenAIFunctionToolSchema.model_validate(schema)
        self.tool_schemas = schemas
        self.apply_chat_template_kwargs = dict(kwargs["chat_template_kwargs"])
        self.continuous_token_builder.chat_template_kwargs = self.apply_chat_template_kwargs
        info = kwargs["extra_info"]
        sample = next(sample for sample in self.protocol["samples"][info["split"]] if sample["id"] == info["id"])
        # As in H3TerminalLoop: the unified contract binds the row's task, controls and first frame.
        bound = sample if self.protocol.get("tool_contract") == "urn:verl:sol-h3:3" else None
        source_image = bound.get("image") if bound else None
        if source_image:
            await asyncio.to_thread(image_bytes, source_image)
            kwargs["raw_prompt"] = await asyncio.to_thread(upright_observation, kwargs["raw_prompt"])
        # The inspector sees the same original request as the judge.
        original = original_request(sample)
        # T2V calls keep the original inspector signature.
        image_kwargs = {"source_image": source_image} if source_image else {}
        output_dir = Path(self.config.video_rl.output_dir)
        directory = output_dir / "episodes" / uuid4().hex
        if "cohort_step" in kwargs:
            # A lookahead episode may resume after a restart. Keep its artifacts in
            # its group's durable directory so renders and inspections are reused.
            group = Path(kwargs["_video_timings"]).parent
            session = int(kwargs["session_id"])
            directory = group / "episodes" / str(session)
            self.policy = EpisodePolicy(
                group / "policy" / str(session),
                kwargs["behavior_version"],
                {
                    "uid": kwargs["uid"],
                    "session": session,
                    "protocol": self.protocol["protocol_sha256"],
                    "sample": info,
                    "tools": schemas,
                },
                required=kwargs.get("require_cached_policy", False),
            )

        async def generate(request, directory):
            def work():
                with generation_slot(output_dir, self.protocol["h3_max_inflight"], directory / "timings.jsonl"):
                    return generate_video(directory, self.protocol, request)

            try:
                return await asyncio.to_thread(work)
            finally:
                self.episode.save()

        async def inspect(video, original_request, directory):
            def call():
                return asyncio.to_thread(
                    inspect_video,
                    video,
                    original_request,
                    directory,
                    self.protocol,
                    dict(self.config.inspection_rl),
                    output_dir,
                    **image_kwargs,
                )

            try:
                if self.policy is None:
                    return await call()
                identity = {"video": video, "original_request": original_request}
                return await replayed_inspection(directory, identity, call, required=self.policy.required)
            finally:
                self.episode.save()

        self.episode = InspectionEpisode(original, schemas, directory, generate, inspect, sample=bound)
        self.episode.save()
        output = await super().run(sampling_params, **kwargs)
        if self.episode.final_result is None:
            self.episode.invalid("Episode ended without a final video")
        if self.policy is not None:
            # Invalid actions, length stops and inspection fallbacks end here.
            self.policy.complete(self.episode.terminal_reason)
        output.extra_fields["h3_result"] = self.episode.final_result
        summary = self.episode.summary()
        summary.update(
            policy_tokens=sum(output.response_mask),
            observation_tokens=len(output.response_mask) - sum(output.response_mask),
            prompt_tokens=len(output.prompt_ids),
            assistant_turns=self.agent_data.assistant_turns,
        )
        output.extra_fields["episode"] = summary
        write_json(
            directory / "trajectory.json",
            {
                **summary,
                "messages": recorded_messages(self.agent_data.messages, source_image),
                "prompt_ids": output.prompt_ids,
                "response_ids": output.response_ids,
                "response_mask": output.response_mask,
                "response_logprobs": output.response_logprobs,
            },
        )
        return output


class InspectionWorker(TournamentWorker):
    def reward_metadata(self, output):
        episode = output.extra_fields["episode"]
        return {f"inspection/{key}": value for key, value in episode.items() if isinstance(value, int | float)}


class InspectionManager(TournamentManager):
    worker_class = InspectionWorker
