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
"""Bind policy observations, generation, and judging to one immutable first frame."""

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

from .h3_request import MAX_IMAGE_BYTES, image_info


def image_asset(path):
    """Validate an original image and freeze its content identity, without resizing it."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as handle:
        data = handle.read(MAX_IMAGE_BYTES + 1)
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"First frame must be nonempty and at most 20 MiB: {path}")
    mime, width, height = image_info(data)
    return {
        "path": str(path),
        "sha256": sha256(data).hexdigest(),
        "bytes": len(data),
        "mime_type": mime,
        "width": width,
        "height": height,
    }


def image_bytes(asset):
    """Fail as an artifact error, not a policy error, if the frozen image changes."""
    with Path(asset["path"]).open("rb") as handle:
        data = handle.read(MAX_IMAGE_BYTES + 1)
    if len(data) != asset["bytes"] or sha256(data).hexdigest() != asset["sha256"]:
        raise RuntimeError(f"Frozen source image changed: {asset['path']}")
    return data


def reference_image_fields(sample):
    """Include the content hash in manifests, hence in every cached judge identity."""
    image = sample.get("image")
    return {"reference_image": image} if isinstance(image, dict) else {}


def upright_observation(messages):
    """Match H3's EXIF orientation before qwen_vl_utils resizes policy images.

    The native vision loader does not apply EXIF transpose. Keep original bytes
    for generation/judging and replace only rotated policy inputs with upright PIL images.
    """
    from PIL import Image, ImageOps

    messages = deepcopy(messages)
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image" or not isinstance(part.get("image"), str):
                continue
            with Image.open(part["image"]) as image:
                if image.getexif().get(0x0112, 1) != 1:
                    part["image"] = ImageOps.exif_transpose(image).copy()
    return messages
