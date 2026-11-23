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
"""First-frame (I2V) requests for the step-300 rewriter.

s300 was trained on text-to-video only, so an I2V request gets exactly the T2V policy input from policy.py (same
system prompt, sol_h3 tool, user wording and chat-template kwargs) with the first frame placed at the start of the
user turn. Nothing else changes: the answer is the same T2V sol_h3 call, parsed by policy.parse, and the gateway
returns its prompt behind FL2VA_START, the line that opens Sol-H3 first-frame prompts (bestofn's I2V_RAW_PREFIX).

prompt_ids() checks every input it builds: with the image block removed, the token ids must equal
policy.prompt_ids() for the same text. The observation follows the joint T2V/I2V RL runs: EXIF-upright, then
qwen_vl_utils.fetch_image with at most 768 x 768 pixels in 32-pixel tiles (image_patch_size 16).
"""

import base64
import binascii
import hashlib
import io

import policy

MAX_IMAGE_BYTES = 20 * 2**20  # decoded size, as Sol-H3's own limit
MAX_SOURCE_PIXELS = 64_000_000  # refuse larger sources before decoding them
OBSERVATION_MAX_PIXELS = 768 * 768  # the joint runs' policy observation (mixed_data.OBSERVATION_MAX_PIXELS)
IMAGE_PATCH_SIZE = 16
FL2VA_START = (
    "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
)

_processor = None


class ImageRejected(ValueError):
    """The caller's image cannot be used (HTTP 400)."""


def processor():
    global _processor
    if _processor is None:
        from transformers import AutoProcessor

        _processor = AutoProcessor.from_pretrained(policy.MODEL_DIR)
    return _processor


def decode_data_url(url):
    """Return the image bytes of a base64 data: URL. Remote URLs are refused: the gateway never fetches them."""
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ImageRejected("the image must be a base64 data: URL (PNG, JPEG or WebP)")
    header, comma, payload = url.partition(",")
    if not comma or not header.endswith(";base64"):
        raise ImageRejected("the image data: URL must be base64-encoded")
    if len(payload) > (MAX_IMAGE_BYTES * 4) // 3 + 8:
        raise ImageRejected(f"the image must be at most {MAX_IMAGE_BYTES >> 20} MiB")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError):
        raise ImageRejected("the image data: URL is not valid base64") from None
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ImageRejected(f"the image must be 1 byte to {MAX_IMAGE_BYTES >> 20} MiB")
    if not (
        data.startswith(b"\x89PNG\r\n\x1a\n")
        or data.startswith(b"\xff\xd8\xff")
        or (data[:4] == b"RIFF" and data[8:12] == b"WEBP")
    ):
        raise ImageRejected("the image must be PNG, JPEG or WebP")
    return data


def observation(data):
    """The policy's view of the image: EXIF-upright RGB, resized like the joint runs' policy inputs."""
    from PIL import Image, ImageOps
    from qwen_vl_utils import fetch_image

    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.width * img.height > MAX_SOURCE_PIXELS:
                raise ImageRejected(f"the image has more than {MAX_SOURCE_PIXELS:,} pixels")
            img.load()
            upright = ImageOps.exif_transpose(img).convert("RGB")
    except ImageRejected:
        raise
    except Exception as exc:  # noqa: BLE001 - any decoder failure is the caller's bad image
        raise ImageRejected(f"the image cannot be decoded: {exc}") from None
    try:
        return fetch_image({"image": upright, "max_pixels": OBSERVATION_MAX_PIXELS}, image_patch_size=IMAGE_PATCH_SIZE)
    except Exception as exc:  # noqa: BLE001 - e.g. an extreme aspect ratio
        raise ImageRejected(f"the image cannot be resized for the policy: {exc}") from None


def prepare(url):
    """Decode and resize a caller's image. Returns (sha256 of the caller's bytes, observation, PNG data URL of the
    observation for the backend, source (width, height))."""
    from PIL import Image

    data = decode_data_url(url)
    obs = observation(data)
    with Image.open(io.BytesIO(data)) as img:
        source_size = img.size
    buf = io.BytesIO()
    obs.save(buf, format="PNG")
    return (
        hashlib.sha256(data).hexdigest(),
        obs,
        "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
        source_size,
    )


def messages(raw_prompt):
    """policy.messages() with the first frame in front of the user's text."""
    return [
        {"role": "system", "content": policy.SYSTEM},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": policy.USER_PREFIX + raw_prompt}]},
    ]


def strip_image(ids):
    """Remove every <|vision_start|> ... <|vision_end|> block."""
    tok = policy.tokenizer()
    start, end = tok.convert_tokens_to_ids("<|vision_start|>"), tok.convert_tokens_to_ids("<|vision_end|>")
    out, inside = [], False
    for t in ids:
        if t == start:
            inside = True
        elif t == end and inside:
            inside = False
        elif not inside:
            out.append(t)
    return out


def prompt_ids(raw_prompt, obs):
    """Token ids of the I2V input with the image placeholder expanded for `obs`. Raises RuntimeError unless the ids
    are the T2V ids plus one image block."""
    proc = processor()
    text = policy.tokenizer().apply_chat_template(
        messages(raw_prompt), tools=policy.TOOLS, add_generation_prompt=True, tokenize=False, **policy.TEMPLATE_KWARGS
    )
    out = proc(text=[text], images=[obs], return_tensors="np")
    ids = [int(t) for t in out["input_ids"][0]]
    image_tokens = int(out["image_grid_thw"].prod()) // proc.image_processor.merge_size**2
    pad = policy.tokenizer().convert_tokens_to_ids("<|image_pad|>")
    if strip_image(ids) != policy.prompt_ids(raw_prompt) or ids.count(pad) != image_tokens:
        raise RuntimeError("I2V input is not the T2V input plus one image block")
    return ids, image_tokens


def render_ready(prompt):
    """The rewrite as a Sol-H3 first-frame prompt: FL2VA_START, a blank line, then the policy's prompt."""
    body = prompt.strip()
    if body.startswith(FL2VA_START):
        return body, False
    return f"{FL2VA_START}\n\n{body}", True
