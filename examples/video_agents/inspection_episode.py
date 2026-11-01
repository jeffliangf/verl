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
"""Per-trajectory state and hard service-call limits for inspection RL."""

import time
from pathlib import Path

from jsonschema import Draft202012Validator

from .storage import write_json


class InvalidAction(ValueError):
    """A policy action that must not execute a service request."""


class InspectionEpisode:
    """Only G→finish, G→I→finish and G→I→G are successful trajectories."""

    def __init__(self, original_request, schemas, directory, generate, inspect):
        self.original_request = original_request
        self.schemas = {schema["function"]["name"]: schema for schema in schemas}
        self.directory = Path(directory)
        self.generate, self.inspect = generate, inspect
        self.videos, self.events = {}, []
        self.generation_calls = self.inspection_calls = 0
        self.final_result = None
        self.terminal_reason = None

    def environment(self):
        actions = (
            []
            if self.final_result is not None
            else (
                ["sol_h3"]
                if not self.videos
                else ["sol_h3", "finish_video"]
                if self.inspection_calls
                else ["inspect_video", "finish_video"]
            )
        )
        return {
            "generation_calls_remaining": 2 - self.generation_calls,
            "inspection_calls_remaining": 1 - self.inspection_calls,
            "available_actions": actions,
            "second_generation_requires_inspection": True,
            "second_generation_finishes_task": True,
        }

    def validate_action(self, name, parameters):
        if name not in self.schemas:
            raise InvalidAction(f"Unknown tool: {name}")
        Draft202012Validator(self.schemas[name]["function"]["parameters"]).validate(parameters)
        if name not in self.environment()["available_actions"]:
            raise InvalidAction(f"Action {name} is unavailable in this state")
        if name == "sol_h3":
            spec = self.schemas[name]["function"]["parameters"]["properties"]
            request = {key: field["default"] for key, field in spec.items() if "default" in field}
            request.update(parameters)
            if not request["prompt"].strip() or any(type(request[k]) is not int for k in ("duration", "seed")):
                raise InvalidAction("Generation requires a nonblank prompt and integer duration/seed")
            request["prompt"] = request["prompt"].strip()
            return request
        key = "video_id" if name == "inspect_video" else "final_video_id"
        if parameters[key] not in self.videos:
            raise InvalidAction("Video ID does not belong to this episode")
        if name == "finish_video" and not parameters["decision_summary"].strip():
            raise InvalidAction("The finish decision summary must be nonblank")
        return parameters

    async def execute(self, name, parameters):
        parameters = self.validate_action(name, parameters)
        start = time.monotonic()
        event = {"tool": name, "parameters": parameters, "status": "started"}
        self.events.append(event)
        self.save()
        if name == "sol_h3":
            self.generation_calls += 1
            video_id = f"video_{self.generation_calls}"
            result = await self.generate(parameters, self.directory / video_id)
            if result["status"] != "succeeded":
                raise RuntimeError("The generation backend returned an unsuccessful artifact")
            self.videos[video_id] = result
            if self.generation_calls == 2:
                self.final_result, self.terminal_reason = result, "second_generation"
            response = {"status": "succeeded", "video_id": video_id, "generation_request": parameters}
        elif name == "inspect_video":
            self.inspection_calls += 1
            report = await self.inspect(
                self.videos[parameters["video_id"]], self.original_request, self.directory / "inspection"
            )
            response = {"video_id": parameters["video_id"], "report": report}
        else:
            self.final_result = self.videos[parameters["final_video_id"]]
            self.terminal_reason = "finish_after_inspection" if self.inspection_calls else "finish_after_generation"
            response = {"status": "finished", **parameters}
        response["environment"] = self.environment()
        event.update(status="completed", seconds=time.monotonic() - start, response=response)
        self.save()
        return response

    def invalid(self, reason):
        self.final_result = {"status": "invalid_call", "reason": str(reason)[:500]}
        self.terminal_reason = "invalid_call"
        self.save()

    def summary(self):
        return {
            "generation_calls": self.generation_calls,
            "inspection_calls": self.inspection_calls,
            "revised": int(self.generation_calls == 2),
            "terminal_reason": self.terminal_reason,
            "tool_seconds": sum(e.get("seconds", 0) for e in self.events),
            "directory": str(self.directory),
        }

    def save(self):
        write_json(
            self.directory / "episode.json",
            {
                "original_request": self.original_request,
                "videos": self.videos,
                "events": self.events,
                "final_result": self.final_result,
                **self.summary(),
            },
        )
