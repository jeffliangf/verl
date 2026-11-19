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
"""Small direct Vertex connectivity check (service-account key or gcloud ADC), independent of the eval checkout."""

import argparse
import json
import sys

from .gemini_transport import create_vertex_client, vertex_settings


def main(argv=None):
    from google.auth.exceptions import RefreshError
    from google.genai import errors, types

    defaults = vertex_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="*")
    parser.add_argument("--project", default=defaults["project"])
    parser.add_argument("--location", default=defaults["location"], choices=("global", "us", "eu"))
    parser.add_argument("--model", default="gemini-4-argon")
    parser.add_argument("--system")
    parser.add_argument("--thinking", default="MEDIUM", choices=("LOW", "MEDIUM", "HIGH"))
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--show-thinking", action="store_true", help="Print returned thought summaries to stderr")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args(argv)
    prompt = " ".join(args.prompt) or (sys.stdin.read().strip() if not sys.stdin.isatty() else "")
    if not prompt:
        parser.error("Supply a prompt as arguments or on stdin")
    config = types.GenerateContentConfig(
        system_instruction=args.system,
        thinking_config=types.ThinkingConfig(thinking_level=args.thinking, include_thoughts=args.show_thinking),
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    try:
        with create_vertex_client(project=args.project, location=args.location) as client:
            response = client.models.generate_content(model=args.model, contents=prompt, config=config)
    except (RuntimeError, RefreshError, errors.APIError) as exc:
        print(f"Vertex request failed: project={args.project} location={args.location}: {exc}", file=sys.stderr)
        return 1
    if args.raw:
        print(json.dumps(response.model_dump(mode="json", exclude_none=True), ensure_ascii=False, indent=2))
        return 0
    if args.show_thinking:
        for candidate in response.candidates or []:
            for part in (candidate.content.parts or []) if candidate.content else []:
                if part.text and part.thought:
                    print(part.text, file=sys.stderr)
    if response.text:
        print(response.text.strip())
        return 0
    print("Vertex returned no text; use --raw to inspect the response.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
