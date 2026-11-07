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
"""Request-bound control constraints, kept outside the model's user message."""

from hashlib import sha256

from .h3_request import ASPECT_RATIOS, canvas_size, request_output

FRAME_COUNTS = tuple(range(124, 346, 17))
RATIOS = ("auto", *ASPECT_RATIOS)
FIRST_FRAME = {"type": "image", "role": "keyframe", "frame_index": 0, "image_ref": "input_image"}


def ratio_canvas(sample, ratio):
    """Resolve model-visible ratio choices to this episode's real worker geometry."""
    if ratio == "auto":
        image = sample.get("image")
        shape = (image["width"], image["height"]) if image else (16, 9)
    else:
        shape = tuple(map(int, ratio.split(":")))
    return canvas_size(*shape)


def allowed_arguments(sample, *, frames=None, ratios=None, require_duration=False, require_aspect=False):
    """Refine the shared tool schema with only requirements visible in the request."""
    frames = sorted(FRAME_COUNTS if frames is None else frames)
    ratios = [ratio for ratio in RATIOS if ratios is None or ratio in ratios]
    result = {
        "version": 1,
        "tool_schema_id": "urn:verl:sol-h3:3",
        "request_sha256": sha256(sample["request"].encode()).hexdigest(),
        "task": "fl2va" if sample["task"] == "i2v" else "t2va",
        "conditions": [dict(FIRST_FRAME)] if sample["image"] else [],
        "image_sha256": sample["image"]["sha256"] if sample["image"] else None,
        "target": {
            "duration_seconds": {"required": require_duration, "allowed_num_frames": frames},
            "aspect_ratio": {
                "required": require_aspect,
                "allowed_values": ratios,
                "allowed_canvases": [
                    {"width": width, "height": height}
                    for width, height in sorted({ratio_canvas(sample, ratio) for ratio in ratios})
                ],
            },
            "short_edge": 768,
        },
    }
    validate_contract(sample, result)
    return result


def validate_contract(sample, contract):
    """Reject stale labels, impossible constraints, and restrictions on free choices."""
    expected_task = "fl2va" if sample["task"] == "i2v" else "t2va"
    expected_image = sample["image"]["sha256"] if sample["image"] else None
    if (
        contract.get("version") != 1
        or contract.get("tool_schema_id") != "urn:verl:sol-h3:3"
        or contract.get("request_sha256") != sha256(sample["request"].encode()).hexdigest()
        or contract.get("task") != expected_task
        or contract.get("image_sha256") != expected_image
        or contract.get("conditions") != ([FIRST_FRAME] if sample["image"] else [])
    ):
        raise ValueError("Allowed video arguments do not match the visible request and source image")
    target = contract["target"]
    duration, aspect = target["duration_seconds"], target["aspect_ratio"]
    frames, ratios = duration["allowed_num_frames"], aspect["allowed_values"]
    if (
        type(duration["required"]) is not bool
        or type(aspect["required"]) is not bool
        or not frames
        or any(type(n) is not int or n not in FRAME_COUNTS for n in frames)
        or len(set(frames)) != len(frames)
        or not ratios
        or any(ratio not in RATIOS for ratio in ratios)
        or len(set(ratios)) != len(ratios)
        or target.get("short_edge") != 768
    ):
        raise ValueError("Unsupported allowed video arguments")
    if (not duration["required"] and set(frames) != set(FRAME_COUNTS)) or (
        not aspect["required"] and set(ratios) != set(RATIOS)
    ):
        raise ValueError("Unspecified controls must allow every supported value")
    canvases = [{"width": w, "height": h} for w, h in sorted({ratio_canvas(sample, ratio) for ratio in ratios})]
    if aspect.get("allowed_canvases") != canvases:
        raise ValueError("Allowed canvases disagree with the source image and aspect choices")


def validate_call(sample, parameters, request):
    """Check emitted controls against hidden labels without filling them from those labels."""
    contract = sample.get("allowed_video_gen_args")
    if contract is None:
        return
    validate_contract(sample, contract)
    rules = contract["target"]
    supplied = parameters.get("target", {})
    for key in ("duration_seconds", "aspect_ratio"):
        if rules[key]["required"] and key not in supplied:
            raise ValueError(f"Tool call must explicitly supply requested output setting: {key}")
    output = request_output(request)
    if output["num_frames"] not in rules["duration_seconds"]["allowed_num_frames"]:
        raise ValueError("Tool duration violates the visible request")
    if request["target"]["aspect_ratio"] not in rules["aspect_ratio"]["allowed_values"]:
        raise ValueError("Tool aspect ratio violates the visible request")
