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
"""Build Sol-H3's SGLang requests and derive their expected output geometry.

Matches Sana's models/minimax_h3/Sol-H3/API.md (2026-11-05). Pillow is only
needed for first frames; text-to-video workers use the standard library.
"""

import base64
import binascii
import io
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

ASPECT_RATIOS = ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16")
MAX_IMAGE_BYTES = 20 * 2**20
IMAGE_MIMES = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}


def frame_count(duration):
    """Apply SGLang's rounding to the supported 17n+5 frame counts at 24 fps."""
    if type(duration) not in (int, float) or not 4 <= duration <= 15:
        raise ValueError("Sol-H3 duration must be a finite number between 4 and 15 seconds")
    frames = round(duration * 24)
    frames += (5 - frames) % 17
    if not 124 <= frames <= 345:
        raise ValueError("Sol-H3 duration must resolve to 124–345 frames at 24 fps")
    return frames


def canvas_size(width, height):
    """Match MiniMax-H3's short edge, area cap, and multiples of 32."""
    ratio = width / height
    if not 0.25 <= ratio <= 4:
        raise ValueError("Sol-H3 auto aspect ratio must be between 1:4 and 4:1; set an explicit aspect_ratio")
    width, height = (768 * ratio, 768) if ratio >= 1 else (768, 768 / ratio)
    if width * height > 1344 * 768:
        scale = (1344 * 768 / (width * height)) ** 0.5
        width, height = width * scale, height * scale
    return round(width / 32) * 32, round(height / 32) * 32


def decode_image(uri):
    """Decode the two inline image forms accepted by the service, without fetching URLs."""
    if not isinstance(uri, str):
        raise ValueError("Sol-H3 needs a base64 image URI")
    if uri.startswith("data:"):
        header, separator, payload = uri.partition(",")
        if not separator or not header[5:].lower().startswith("image/") or ";base64" not in header:
            raise ValueError("Sol-H3 needs a base64 image data URI")
    elif uri.startswith("base64://"):
        payload = uri[len("base64://") :]
    else:
        raise ValueError("Sol-H3 conditions require inline base64 images, not URLs or file paths")
    if len(payload) > ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 4:
        raise ValueError("Sol-H3 first frames must be at most 20 MiB")
    payload = payload.strip().replace("-", "+").replace("_", "/")
    try:
        data = base64.b64decode(payload + "=" * (-len(payload) % 4), validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Sol-H3 image base64 is invalid") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Sol-H3 first frames must be nonempty and at most 20 MiB")
    return data


def image_info(data):
    """Check image limits before decoding and report the EXIF-upright dimensions."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in IMAGE_MIMES:
                raise ValueError("Sol-H3 first frames must be PNG, JPEG, or WebP")
            width, height = image.size
            if not (32 <= width <= 8192 and 32 <= height <= 8192) or width * height > 40_000_000:
                raise ValueError("Sol-H3 image sides must be 32–8192 pixels, with at most 40 million pixels")
            image.load()
            if image.getexif().get(0x0112, 1) in (5, 6, 7, 8):
                width, height = height, width
            return IMAGE_MIMES[image.format], width, height
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise ValueError("Sol-H3 first frame could not be decoded") from None


def image_uri(image):
    """Freeze a local first frame or inline URI as validated base64 image bytes."""
    if isinstance(image, str) and image.startswith(("data:", "base64://")):
        data = decode_image(image)
    else:
        if not isinstance(image, str | Path) or not str(image).strip() or "://" in str(image):
            raise ValueError("Sol-H3 image must be a local PNG/JPEG/WebP path or a base64 image URI")
        try:
            with Path(image).expanduser().open("rb") as handle:
                data = handle.read(MAX_IMAGE_BYTES + 1)
        except OSError:
            raise ValueError("Sol-H3 first frame is not a readable local file") from None
        if not data or len(data) > MAX_IMAGE_BYTES:
            raise ValueError("Sol-H3 first frames must be nonempty and at most 20 MiB")
    mime, _, _ = image_info(data)
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def request_output(request):
    """Resolve both legacy and SGLang requests independently of the reference profile."""
    if "task" not in request:
        frames = {5: 124, 10: 243}[request["duration"]]
        width, height = 1344, 768
    else:
        task = request["task"].strip().lower()
        conditions, target = request.get("conditions", []), request.get("target", {})
        if task not in {"t2va", "fl2va"} or (task == "t2va" and conditions):
            raise ValueError("Sol-H3 supports t2va without conditions or fl2va with one first frame")
        image_size = None
        if task == "fl2va":
            keyframe = {"type": "image", "role": "keyframe", "frame_index": 0}
            if len(conditions) != 1 or any(conditions[0].get(key) != value for key, value in keyframe.items()):
                raise ValueError("Sol-H3 fl2va requires exactly one image keyframe at frame_index 0")
            _, *image_size = image_info(decode_image(conditions[0]["uri"]))
        if target.get("short_edge", 768) != 768:
            raise ValueError("Sol-H3 only supports a 768-pixel target short_edge")
        ratio = target.get("aspect_ratio")
        if ratio is None:
            ratio = "auto" if task == "fl2va" else "16:9"
        if ratio == "auto":
            width, height = canvas_size(*(image_size or (16, 9)))
        elif ratio in ASPECT_RATIOS:
            width, height = canvas_size(*map(int, ratio.split(":")))
        else:
            raise ValueError("Unsupported Sol-H3 aspect_ratio")
        duration = target.get("duration_seconds")
        if duration is None:
            duration = request.get("seconds")
        if duration is None:
            duration = 5
        frames = frame_count(duration)
    return {"width": width, "height": height, "fps": 24, "num_frames": frames, "duration_seconds": frames / 24}


def video_request(prompt, *, duration=10, seed=42, image=None, aspect_ratio=None):
    """Build the SGLang API body; an optional first frame selects fl2va and auto geometry."""
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 8000:
        raise ValueError("Sol-H3 needs a nonempty prompt of at most 8000 characters")
    if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
        raise ValueError("Sol-H3 seed must be an integer from 0 through 9223372036854775807")
    frame_count(duration)
    if aspect_ratio is not None and aspect_ratio not in (*ASPECT_RATIOS, "auto"):
        raise ValueError("Unsupported Sol-H3 aspect_ratio")
    conditions = []
    if image is not None:
        conditions.append({"type": "image", "role": "keyframe", "frame_index": 0, "uri": image_uri(image)})
    request = {
        "task": "fl2va" if conditions else "t2va",
        "prompt": prompt.strip(),
        "conditions": conditions,
        "target": {
            "short_edge": 768,
            "aspect_ratio": aspect_ratio or ("auto" if conditions else "16:9"),
            "duration_seconds": duration,
        },
        "seed": seed,
    }
    request_output(request)
    return request


def request_summary(request):
    """Describe generation inputs to the policy without echoing base64 image payloads."""
    summary = deepcopy(request)
    for condition in summary.get("conditions", []):
        data = decode_image(condition.pop("uri"))
        condition.update(bytes=len(data), sha256=sha256(data).hexdigest())
    return summary


def tool_request(parameters, schema, *, sample=None):
    """Apply row-specific tool defaults, preserving the contract of frozen legacy rows."""
    from jsonschema import Draft202012Validator

    spec = schema["function"]["parameters"]
    validator = Draft202012Validator(spec)
    validator.validate(parameters)
    request = {key: deepcopy(value["default"]) for key, value in spec["properties"].items() if "default" in value}
    request.update(parameters)
    validator.validate(request)
    if spec.get("$id") == "urn:verl:sol-h3:3":
        from .image_assets import image_bytes

        if sample is None or sample.get("task") not in {"t2v", "i2v"}:
            raise RuntimeError("The unified H3 tool requires a bound dataset sample")
        expected = "fl2va" if sample["task"] == "i2v" else "t2va"
        if request["task"] != expected:
            raise ValueError(f"This sample requires task {expected}")
        conditions = request["conditions"]
        if (expected == "fl2va" and len(conditions) != 1) or (expected == "t2va" and conditions):
            raise ValueError("I2V requires its supplied first frame; T2V requires empty conditions")
        strict_controls = "request" in sample
        required = sample.get("target", {})
        if strict_controls and (missing := required.keys() - request["target"].keys()):
            raise ValueError(
                f"Tool call must explicitly supply requested output settings: {', '.join(sorted(missing))}"
            )
        image = None
        if expected == "fl2va":
            asset = sample["image"]
            image = f"data:{asset['mime_type']};base64," + base64.b64encode(image_bytes(asset)).decode("ascii")
        target_spec = spec["properties"]["target"]["properties"]
        target = {key: deepcopy(value["default"]) for key, value in target_spec.items() if "default" in value}
        if not strict_controls:
            target.update(required)
        target.update(request["target"])
        if not strict_controls and any(target.get(key) != value for key, value in required.items()):
            raise ValueError("Tool target contradicts the sample's requested output settings")
        if target.get("short_edge", 768) != 768:
            raise ValueError("Sol-H3 only supports short_edge 768")
        result = video_request(
            request["prompt"],
            duration=target["duration_seconds"],
            aspect_ratio=target.get("aspect_ratio"),
            image=image,
            seed=42 if request["seed"] is None else request["seed"],
        )
        if strict_controls and required:
            actual = request_output(result)
            expected_output = request_output({**result, "target": {**result["target"], **required}})
            fields = []
            if "duration_seconds" in required:
                fields.append("num_frames")
            if "aspect_ratio" in required:
                fields.extend(("width", "height"))
            if any(actual[key] != expected_output[key] for key in fields):
                raise ValueError("Tool target contradicts the request's required output settings")
        if "allowed_video_gen_args" in sample:
            from .control_constraints import validate_call

            validate_call(sample, parameters, result)
        return result
    if "aspect_ratio" in spec["properties"]:
        return video_request(**request)
    # Old prepared datasets remain byte-compatible with their persisted jobs.
    if not request["prompt"].strip() or any(type(request[key]) is not int for key in ("duration", "seed")):
        raise ValueError("Sol-H3 requires nonblank prompt text and integer duration/seed")
    request["prompt"] = request["prompt"].strip()
    return request
