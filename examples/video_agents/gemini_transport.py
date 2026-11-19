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
"""Route Argon and Hydrogen directly to Vertex with google-genai.

The frozen evaluator builds Interactions requests. This adapter translates them
to generateContent while retaining its result validation and bounded retries.
It never falls back to an API key, another provider, or another model.

Credentials come from google.auth.default(): the service-account key named by
GOOGLE_APPLICATION_CREDENTIALS when set (it does not expire, so long runs should
use it), otherwise the gcloud application-default login, which the organization
ends after 16 hours.
"""

import base64
import importlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

VERTEX_MODELS = frozenset({"gemini-4-argon", "gemini-4-hydrogen"})
DEFAULT_PROJECT = "utopai-labs"
DEFAULT_LOCATION = "global"
BACKEND = "vertex"
API = "generateContent"


def uses_vertex(model):
    return model in VERTEX_MODELS


def vertex_settings():
    return {
        "project": os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or DEFAULT_PROJECT,
        "location": os.environ.get("VERTEX_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or DEFAULT_LOCATION,
    }


def transport_identity(model):
    if not uses_vertex(model):
        return {}
    settings = vertex_settings()
    return {
        "backend": BACKEND,
        "vertex_api": API,
        "vertex_project": settings["project"],
        "vertex_location": settings["location"],
    }


def read_api_key(runner, model, env_file):
    if uses_vertex(model):
        return None  # Vertex uses Google credentials; do not read either provider's API key.
    key = os.environ.get("GEMINI_API_KEY") or runner.read_env_file(Path(env_file)).get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(f"GEMINI_API_KEY is required in the environment or {env_file}")
    return key


def resolved_model(runner, args):
    return (
        args.model
        or os.environ.get("GEMINI_EVAL_MODEL")
        or runner.read_env_file(args.env_file).get("GEMINI_EVAL_MODEL")
        or runner.DEFAULT_MODEL
    )


def create_vertex_client(*, project, location, timeout=600):
    import google.auth
    from google import genai
    from google.auth.exceptions import DefaultCredentialsError
    from google.genai import types

    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    except DefaultCredentialsError:
        raise RuntimeError(
            "Vertex credentials are unavailable. Set GOOGLE_APPLICATION_CREDENTIALS to a service-account key "
            f"file, or run: gcloud auth application-default login --project {project}"
        ) from None
    # Explicit credentials ensure GEMINI_API_KEY cannot select API-key auth.
    # Only the evaluator owns retries, so its configured budget is not multiplied.
    return genai.Client(
        enterprise=True,
        credentials=credentials,
        project=project,
        location=location,
        http_options=types.HttpOptions(timeout=round(timeout * 1000), retry_options=types.HttpRetryOptions(attempts=1)),
    )


def vertex_request(payload):
    """Translate supported fields explicitly; reject unrepresentable settings."""
    from google.genai import types

    allowed = {
        "model",
        "system_instruction",
        "input",
        "response_format",
        "generation_config",
        "store",
        "background",
        "stream",
    }
    if set(payload) - allowed or not uses_vertex(payload.get("model")):
        raise ValueError("Unsupported Vertex request or model")
    if any(payload.get(key) for key in ("store", "background", "stream")):
        raise ValueError("Vertex requires synchronous, non-streaming requests")
    parts = []
    for part in payload["input"]:
        kind = part["type"]
        if kind == "text" and set(part) == {"type", "text"}:
            parts.append(types.Part.from_text(text=part["text"]))
            continue
        if kind not in {"image", "video"} or set(part) - {
            "type",
            "uri",
            "data",
            "mime_type",
            "processing",
            "resolution",
        }:
            raise ValueError("Unsupported Vertex media part")
        mime = part["mime_type"]
        if bool(part.get("uri")) == bool(part.get("data")):
            raise ValueError("Supply exactly one inline media URI or base64 data value")
        uri = part.get("uri") or f"data:{mime};base64,{part['data']}"
        prefix = f"data:{mime};base64,"
        if not mime.startswith(kind + "/") or not uri.startswith(prefix):
            raise ValueError("Vertex requires inline media, not Gemini Files API URIs")
        media = types.Part.from_bytes(data=base64.b64decode(uri[len(prefix) :], validate=True), mime_type=mime)
        if "processing" in part:
            processing = part["processing"]
            if (
                kind != "video"
                or not isinstance(processing, dict)
                or set(processing) != {"type", "fps"}
                or processing["type"] != "static"
            ):
                raise ValueError(
                    "Vertex generateContent supports standard video or explicit static fps; "
                    "agentic steps are unavailable"
                )
            media.media_processing = types.MediaProcessing.STATIC
            media.video_metadata = types.VideoMetadata(fps=processing["fps"])
        if "resolution" in part:
            media.media_resolution = types.PartMediaResolution(level="MEDIA_RESOLUTION_" + part["resolution"].upper())
        parts.append(media)
    response_format = payload["response_format"]
    if (
        set(response_format) != {"type", "mime_type", "schema"}
        or response_format["type"] != "text"
        or response_format["mime_type"] != "application/json"
    ):
        raise ValueError("Vertex requires a JSON response schema")
    generation = payload.get("generation_config", {})
    if set(generation) - {"thinking_level", "seed", "max_output_tokens"}:
        raise ValueError("Unsupported Vertex generation setting")
    config = {
        "response_mime_type": "application/json",
        "response_json_schema": response_format["schema"],
    }
    if payload.get("system_instruction"):
        config["system_instruction"] = payload["system_instruction"]
    if generation.get("thinking_level") is not None:
        config["thinking_config"] = types.ThinkingConfig(thinking_level=generation["thinking_level"].upper())
    for key in ("seed", "max_output_tokens"):
        if generation.get(key) is not None:
            config[key] = generation[key]
    return {
        "model": payload["model"],
        "contents": [types.Content(role="user", parts=parts)],
        "config": types.GenerateContentConfig(**config),
    }


def _network_errors():
    errors = [TimeoutError, ConnectionError, OSError]
    for name in ("httpx", "httpx2"):
        try:
            errors.append(importlib.import_module(name).TransportError)
        except ImportError:
            pass
    return tuple(errors)


def client_type(runner, model):
    original = runner.GeminiRestClient
    if not uses_vertex(model) or getattr(original, "vertex_genai", False):
        return original

    class VertexGenAIClient(original):
        vertex_genai = True

        def __init__(self, api_key=None, **kwargs):
            super().__init__("vertex-adc", **kwargs)
            self.vertex_settings = vertex_settings()
            self.sdk = create_vertex_client(**self.vertex_settings, timeout=self.request_timeout)

        def _inline(self, path, mime_for):
            path = Path(path)
            if not path.stat().st_size:
                raise ValueError("Vertex media is empty")
            mime = mime_for(path)
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return {"name": "", "uri": f"data:{mime};base64,{data}", "mime_type": mime}

        def upload_video(self, path):
            return self._inline(path, runner.video_mime_type)

        def upload_image(self, path):
            return self._inline(path, runner.image_mime_type)

        def delete_file(self, name):
            return None  # Inline media creates no remote files.

        def _request_once(self, method, url, *, data=None, **kwargs):
            from google.auth.exceptions import RefreshError, TransportError
            from google.genai.errors import APIError

            if method != "POST" or url != self.interactions_url:
                raise ValueError("Vertex transport only supports inference requests")
            payload = json.loads(data)
            if payload.get("model") != model:
                raise ValueError("Vertex request model differs from the configured client model")
            request = vertex_request(payload)
            try:
                response = self.sdk.models.generate_content(**request)
            except APIError as exc:
                raise runner.HttpFailure(int(exc.code), str(exc.message)[:4000]) from None
            except RefreshError:
                raise RuntimeError(
                    "Vertex credential refresh failed; check the GOOGLE_APPLICATION_CREDENTIALS service-account "
                    "key, or run gcloud auth application-default login again"
                ) from None
            except (TransportError, *_network_errors()):
                raise runner.NetworkFailure("Vertex network request failed") from None
            candidates = response.candidates or []
            if len(candidates) != 1 or candidates[0].finish_reason != "STOP" or not candidates[0].content:
                raise runner.EvalError("Vertex returned an incomplete or invalid completion")
            text = "".join(p.text for p in candidates[0].content.parts or [] if p.text and not p.thought)
            if not text.strip():
                raise runner.EvalError("Vertex returned no final text")
            usage = response.usage_metadata
            normalized = {
                "id": response.response_id,
                "model": model,
                "status": "completed",
                "steps": [{"type": "model_output", "content": [{"type": "text", "text": text}]}],
                "usage": {
                    "total_input_tokens": (usage.prompt_token_count or 0) if usage else 0,
                    "total_output_tokens": (usage.candidates_token_count or 0) if usage else 0,
                    "total_thought_tokens": (usage.thoughts_token_count or 0) if usage else 0,
                },
                "vertex_response": response.model_dump(mode="json", exclude_none=True),
                **transport_identity(model),
            }
            return {}, json.dumps(normalized).encode()

        def evaluate(self, request_payload, *, require_agentic_steps):
            if require_agentic_steps:
                raise ValueError("Vertex generateContent requires video_processing=standard for judging")
            response, _ = self.request_json(
                "POST",
                self.interactions_url,
                payload=request_payload,
                retry_network=False,
                context=f"Vertex {model} judgment",
            )
            judgment, diagnostics = runner.extract_interaction_result(response, require_agentic_steps=False)
            return judgment, {
                **diagnostics,
                **transport_identity(model),
                "served_model": response["vertex_response"].get("model_version"),
            }

    return VertexGenAIClient


def make_client(runner, model, env_file, **kwargs):
    return client_type(runner, model)(read_api_key(runner, model, env_file), **kwargs)


def validate_backend(row, model):
    if uses_vertex(model) and any(row.get(k) != v for k, v in transport_identity(model).items()):
        raise ValueError(f"{model} judgment uses a different backend; use a new output/run directory for direct Vertex")


@contextmanager
def vertex_runner(runner, model):
    """Use the evaluator's Vertex factory hook without requiring a Gemini key."""
    if not uses_vertex(model):
        yield
        return
    if not hasattr(runner, "VertexRestClient"):
        raise RuntimeError("Vertex routing requires an evaluator with the Vertex client hook")
    names = ("GeminiRestClient", "VertexRestClient", "parse_args", "append_jsonl", "load_completed_keys")
    saved = {name: getattr(runner, name) for name in names}
    clients = []
    base = client_type(runner, model)

    class ManagedClient(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clients.append(self)

    def parse_args(argv=None):
        args = saved["parse_args"](argv)
        if resolved_model(runner, args) != model:
            raise ValueError("Cannot mix models in the same routed evaluator")
        args.backend = "vertex"
        return args

    def vertex_client(*args, max_attempts=5, request_timeout=600, **kwargs):
        # Resolve the current class so timing, reference images and shared
        # uploads remain active in batch and streaming workers.
        return runner.GeminiRestClient(None, max_attempts=max_attempts, request_timeout=request_timeout)

    def append_jsonl(path, row):
        if uses_vertex(row.get("judge_model")):
            row = {**row, **transport_identity(model)}
        return saved["append_jsonl"](path, row)

    def load_completed_keys(path):
        if Path(path).exists():
            for line in Path(path).read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("status") == "completed" and uses_vertex(row.get("judge_model")):
                    validate_backend(row, model)
        return saved["load_completed_keys"](path)

    runner.GeminiRestClient = ManagedClient
    runner.VertexRestClient = vertex_client
    runner.parse_args, runner.append_jsonl, runner.load_completed_keys = parse_args, append_jsonl, load_completed_keys
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(runner, name, value)
        for client in clients:
            client.sdk.close()
