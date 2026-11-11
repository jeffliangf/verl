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
"""Thin verl BaseTool adapters sharing the same episode implementation as inference.

A multi-turn training loop must supply its VideoSession to both tools' create calls.
The existing H3TerminalLoop is intentionally not modified; it is a one-call trainer.
"""

import asyncio
import json

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from .session import TOOLS


class SessionTool(BaseTool):
    tool_name = None

    def __init__(self, config, tool_schema=None):
        schema = tool_schema or OpenAIFunctionToolSchema.model_validate(
            next(t for t in TOOLS if t["function"]["name"] == self.tool_name)
        )
        super().__init__(config, schema)
        self.sessions = {}

    async def create(self, instance_id=None, **kwargs):
        instance_id, response = await super().create(instance_id, **kwargs)
        self.sessions[instance_id] = kwargs["video_session"]
        return instance_id, response

    async def execute(self, instance_id, parameters, **kwargs):
        session = self.sessions[instance_id]
        session.begin_call(self.tool_name, parameters)
        result = await asyncio.to_thread(session.execute_pending)
        return ToolResponse(text=json.dumps(result, ensure_ascii=False)), None, {"tool": self.tool_name}

    async def release(self, instance_id, **kwargs):
        self.sessions.pop(instance_id, None)


class SolH3Tool(SessionTool):
    tool_name = "sol_h3"


class InspectVideoTool(SessionTool):
    tool_name = "inspect_video"


class FinishVideoTool(SessionTool):
    tool_name = "finish_video"
