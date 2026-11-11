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
"""Durable native tool state; the policy alone selects a final artifact."""

import copy
import json
import re
import time
from pathlib import Path

from jsonschema import Draft202012Validator

from examples.video_agents.storage import read_json, write_json

from .feedback import validate_feedback

POLICY_PATH = Path(__file__).with_name("policy.txt")
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "sol_h3",
            "description": "Generate a new 10-second video with synchronized audio. "
            "Returns video_id and resource usage. "
            "Does not inspect the video. Each call consumes one generation from the episode budget.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "prompt": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 8000,
                        "description": "Complete video/audio description preserving the original goal. "
                        "Context-IR sections "
                        "integrated_multimodal_description, overall_soundscape and non_diegetic_music may help.",
                    },
                    "duration": {"type": "integer", "enum": [10], "default": 10},
                    "seed": {"type": "integer", "enum": [42], "default": 42},
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_video",
            "description": "Inspect one video produced in this episode. Returns observed action sequence, strengths to "
            "preserve, grounded defects, up to two repair priorities, and explicit earlier-issue rechecks. "
            "Original goal and actual generation prompt are supplied by the environment. No acceptance "
            "verdict, score, comparison, or stopping recommendation is returned. You decide whether another "
            "generation is worth its cost. Repeating inspection of the same video returns its cached report.",
            "parameters": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "video_id": {
                        "type": "string",
                        "minLength": 1,
                        "description": "An exact video_id returned by sol_h3 in this episode.",
                    }
                },
                "required": ["video_id"],
            },
        },
    },
]
FINAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "final_video_id": {"type": "string", "minLength": 1},
        "decision_summary": {"type": "string", "minLength": 1},
    },
    "required": ["final_video_id", "decision_summary"],
}
TOOLS.append(
    {
        "type": "function",
        "function": {
            "name": "finish_video",
            "description": "Finish the episode by selecting an existing video_id and briefly explaining "
            "the cost/quality "
            "decision. This terminal action makes no model/service call and consumes no generation or "
            "inspection budget. Use this action when you are done, including when budgets are exhausted.",
            "parameters": FINAL_SCHEMA,
        },
    }
)


def parse_final_text(text):
    """Accept an explicit final object, including a single Markdown-fenced object."""
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        blocks = list(re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE))
        if len(blocks) != 1:
            raise ValueError("Use finish_video or one unambiguous final JSON object") from None
        block = blocks[0]
        outside = text[: block.start()] + text[block.end() :]
        if re.search(r'"final_video_id"\s*:', outside):
            raise ValueError("Conflicting final objects outside the Markdown block") from None
        value = json.loads(block.group(1))
    Draft202012Validator(FINAL_SCHEMA).validate(value)
    return value


class VideoSession:
    """Shared tool implementation for inference and a future multi-turn RL adapter.

    Backends perform media generation/inspection only. They never select a final video.
    Exactly one action can be pending per episode. Durable backend caches make restart safe.
    """

    def __init__(self, directory, sample, generate, inspect, max_generations=3, max_inspections=3):
        self.directory = Path(directory)
        self.path = self.directory / "session.json"
        self.generate_backend = generate
        self.inspect_backend = inspect
        if self.path.exists():
            self.state = read_json(self.path)
            if self.state["id"] != sample["id"] or self.state["original_request"] != sample["original_request"]:
                raise ValueError("Episode identity changed")
            if (self.state["max_generations"], self.state["max_inspections"]) != (max_generations, max_inspections):
                raise ValueError("Episode resource limits changed")
        else:
            self.state = {
                "id": sample["id"],
                "source": sample["source"],
                "original_request": sample["original_request"],
                "messages": [
                    {"role": "system", "content": POLICY_PATH.read_text()},
                    {"role": "user", "content": sample["messages"][-1]["content"]},
                ],
                "max_generations": max_generations,
                "max_inspections": max_inspections,
                "generation_calls": 0,
                "inspection_calls": 0,
                "tool_calls": 0,
                "policy_turns": 0,
                "policy_input_tokens": 0,
                "policy_output_tokens": 0,
                "videos": {},
                "inspections": {},
                "events": [],
                "pending": None,
                "terminal": False,
                "final": None,
                "finish_reason": None,
                "created_at": time.time(),
            }
            self.save()

    def save(self):
        write_json(self.path, self.state)

    def resources(self):
        return {
            "generation_calls_used": self.state["generation_calls"],
            "generation_calls_remaining": self.state["max_generations"] - self.state["generation_calls"],
            "inspection_calls_used": self.state["inspection_calls"],
            "inspection_calls_remaining": self.state["max_inspections"] - self.state["inspection_calls"],
        }

    def begin_call(self, name, arguments, content=""):
        if self.state["terminal"] or self.state["pending"]:
            raise ValueError("An episode must be active with no pending call")
        self.state["tool_calls"] += 1
        call_id = f"call_{self.state['tool_calls']}"
        self.state["messages"].append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
            }
        )
        self.state["pending"] = {"id": call_id, "name": name, "arguments": arguments}
        self.save()

    def _generate(self, pending):
        args = pending["arguments"]
        request = {"duration": 10, "seed": 42, **args}
        if not request["prompt"].strip() or any(type(request[k]) is not int for k in ("seed", "duration")):
            raise ValueError("Nonblank prompt and integer duration/seed required")
        if not pending.get("reserved"):
            if self.state["generation_calls"] >= self.state["max_generations"]:
                raise ValueError("Generation budget exhausted; inspect an existing video or select a final video")
            self.state["generation_calls"] += 1
            pending["generation_index"] = self.state["generation_calls"]
            pending["reserved"] = True
            self.save()
        number = pending["generation_index"]
        video_id = f"video_{number}"
        result = self.generate_backend(self.directory / "generations" / video_id, request)
        if result["status"] != "succeeded":
            raise RuntimeError("Video generation did not succeed")
        video = {**result, "video_id": video_id, "generation_index": number, "request": request}
        self.state["videos"][video_id] = video
        return {"status": "succeeded", "video_id": video_id, "duration_seconds": 10.125, "inspection_available": True}

    def _inspect(self, pending):
        video_id = pending["arguments"]["video_id"]
        if video_id not in self.state["videos"]:
            raise ValueError("Unknown video_id; inspect only an artifact returned in this episode")
        if video_id in self.state["inspections"]:
            return {
                "status": "succeeded",
                "video_id": video_id,
                "cached": True,
                "feedback": self.state["inspections"][video_id]["feedback"],
            }
        video = self.state["videos"][video_id]
        if not pending.get("reserved"):
            if self.state["inspection_calls"] >= self.state["max_inspections"]:
                raise ValueError("Inspection budget exhausted; select a final video or use remaining generation budget")
            self.state["inspection_calls"] += 1
            previous = [
                record
                for key, record in self.state["inspections"].items()
                if self.state["videos"][key]["generation_index"] < video["generation_index"]
            ]
            previous.sort(key=lambda record: self.state["videos"][record["video_id"]]["generation_index"])
            prior = []
            if previous:
                record = previous[-1]
                prior = [
                    {**finding, "id": record["video_id"] + ":" + finding["id"]}
                    for finding in record["feedback"]["findings"]
                ]
                unresolved = {
                    check["prior_issue_id"]
                    for check in record["feedback"]["prior_issue_checks"]
                    if check["state"] == "unassessable"
                }
                prior.extend(finding for finding in record.get("prior_findings", []) if finding["id"] in unresolved)
            pending["prior_findings"] = prior
            pending["reserved"] = True
            self.save()
        prior = pending["prior_findings"]
        feedback = self.inspect_backend(
            video, self.state["original_request"], prior, self.directory / "inspections" / video_id
        )
        validate_feedback(feedback, prior)
        self.state["inspections"][video_id] = {"video_id": video_id, "feedback": feedback, "prior_findings": prior}
        return {"status": "succeeded", "video_id": video_id, "cached": False, "feedback": feedback}

    def execute_pending(self):
        pending = self.state["pending"]
        if not pending:
            raise ValueError("No pending tool call")
        start = time.monotonic()
        try:
            schema = next(
                (t["function"]["parameters"] for t in TOOLS if t["function"]["name"] == pending["name"]), None
            )
            if schema is None:
                raise ValueError("Unknown action; available actions are sol_h3, inspect_video, and finish_video")
            Draft202012Validator(schema).validate(pending["arguments"])
            if pending["name"] == "finish_video":
                self._select_final(pending["arguments"])
                result = {
                    "status": "succeeded",
                    "final_video_id": self.state["final"]["final_video_id"],
                    "episode_finished": True,
                }
            else:
                result = self._generate(pending) if pending["name"] == "sol_h3" else self._inspect(pending)
        except Exception as error:
            result = {"status": "error", "error_type": type(error).__name__, "message": str(error)[:1600]}
        observation = {"tool": pending["name"], **result, "resources": self.resources()}
        elapsed = time.monotonic() - start
        self.state["messages"].append(
            {
                "role": "tool",
                "name": pending["name"],
                "tool_call_id": pending["id"],
                "content": json.dumps(observation, ensure_ascii=False),
            }
        )
        self.state["events"].append(
            {
                "call_id": pending["id"],
                "tool": pending["name"],
                "arguments": copy.deepcopy(pending["arguments"]),
                "status": result["status"],
                "elapsed_seconds": elapsed,
                "at": time.time(),
                "result": observation,
            }
        )
        self.state["pending"] = None
        self.save()
        return observation

    def _select_final(self, decision):
        Draft202012Validator(FINAL_SCHEMA).validate(decision)
        if self.state["terminal"]:
            raise ValueError("Episode already ended")
        if decision["final_video_id"] not in self.state["videos"]:
            raise ValueError("Final selection must identify a successfully generated video in this episode")
        if not decision["decision_summary"].strip():
            raise ValueError("Explain the final decision briefly")
        self.state.update(final=decision, terminal=True, finish_reason="policy_final", finished_at=time.time())

    def finalize(self, decision):
        if self.state["pending"]:
            raise ValueError("A tool call is pending")
        self._select_final(decision)
        self.state["messages"].append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False)})
        self.save()

    def format_error(self, message, raw_public_text=""):
        self.state["format_errors"] = self.state.get("format_errors", 0) + 1
        self.state["messages"].append({"role": "assistant", "content": raw_public_text[:2500]})
        self.state["messages"].append(
            {
                "role": "user",
                "content": "Environment format error: "
                + message
                + ". Call one available tool, or call finish_video with final_video_id and decision_summary. "
                "Only already generated video IDs are valid final selections.",
            }
        )
        self.save()

    def terminate_error(self, reason):
        # A guardrail failure is never silently converted into a policy-selected video.
        self.state.update(terminal=True, finish_reason=reason, finished_at=time.time())
        self.save()
