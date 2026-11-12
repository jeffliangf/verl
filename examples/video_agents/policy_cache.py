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
"""Durable complete policy responses (one turn or a whole episode), before H3 work or a learner update."""

import math
from pathlib import Path

from .storage import fingerprint, freeze_json, read_json, write_json


def validate_response(output, version):
    if output.stop_reason not in {"completed", "length"} or not output.token_ids:
        raise RuntimeError("A cohort requires a complete policy response")
    if output.routed_experts is not None:
        raise ValueError("Cohort response recovery does not support router replay")
    if output.log_probs is None or len(output.log_probs) != len(output.token_ids):
        raise RuntimeError("Cohort responses require actual behavior log probabilities for every token")
    if not all(math.isfinite(value) for value in output.log_probs):
        raise RuntimeError("Nonfinite behavior log probabilities")
    versions = [output.extra_fields.get(key) for key in ("min_global_steps", "max_global_steps")]
    if any(type(value) is not int or value != version for value in versions):
        raise RuntimeError(f"Expected complete response from policy {version}, got versions {versions}")


class PolicyCache:
    def __init__(self, directory, version, identity, *, required=False):
        self.directory, self.version = Path(directory), int(version)
        self.identity, self.required = identity, required

    async def generate(self, client, request):
        from verl.workers.rollout.replica import TokenOutput

        identity = {
            "identity": self.identity,
            "policy_version": self.version,
            "prompt_ids": request["prompt_ids"],
            "sampling_params": request["sampling_params"],
        }
        freeze_json(self.directory / "request.json", identity)
        path = self.directory / "response.json"
        if path.exists():
            saved = read_json(path)
            if saved["sha256"] != fingerprint(saved["output"]) or saved["request_sha256"] != fingerprint(identity):
                raise ValueError("Cached policy response identity/checksum mismatch")
            output = TokenOutput.model_validate(saved["output"])
        else:
            if self.required:
                raise RuntimeError(f"Checkpointed policy response is missing: {path}")
            output = await client.generate(**request)
            validate_response(output, self.version)
            value = output.model_dump(mode="json")
            write_json(path, {"request_sha256": fingerprint(identity), "output": value, "sha256": fingerprint(value)})
        validate_response(output, self.version)
        return output


class EpisodePolicy:
    """Durable turns of a multi-turn episode, all from one behavior version.

    Each policy turn is a ``PolicyCache`` under ``turn-<k>``. ``response.json`` in
    this directory marks the episode's policy work complete, as a single cached
    response does for the terminal H3 loop, so the cohort trainer can drain,
    checkpoint and replay inspection episodes under the same contract.
    """

    def __init__(self, directory, version, identity, *, required=False):
        self.directory, self.version = Path(directory), int(version)
        self.identity, self.required = identity, required
        self.turns, self.completed = 0, False

    async def generate(self, client, request):
        if self.completed:
            raise RuntimeError("A completed episode policy cannot generate another turn")
        cache = PolicyCache(
            self.directory / f"turn-{self.turns}",
            self.version,
            {**self.identity, "turn": self.turns},
            required=self.required,
        )
        output = await cache.generate(client, request)
        self.turns += 1
        return output

    def complete(self, last_action):
        """Call once no further policy turn can follow; replays must reach the same point."""
        if self.completed:
            return
        freeze_json(
            self.directory / "response.json",
            {
                "policy_version": self.version,
                "turns": self.turns,
                "last_action": last_action,
                "identity_sha256": fingerprint(self.identity),
            },
        )
        self.completed = True
