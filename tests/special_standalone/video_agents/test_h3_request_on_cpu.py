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
"""Check the SGLang request contract, image transport, and requested media geometry."""

import base64
import copy
import io
from pathlib import Path

import pytest
from PIL import Image

from examples.video_agents import h3, h3_request
from examples.video_agents.inspection_episode import InspectionEpisode
from examples.video_agents.storage import file_sha256, fingerprint, read_json, write_json

SCHEMA = read_json(Path(h3.__file__).with_name("configs") / "sol_h3_tool.json")


def first_frame(width=64, height=96, kind="PNG", **save):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buffer, kind, **save)
    return buffer.getvalue()


def test_text_request_uses_sglang_fields_and_frozen_row_defaults():
    schema = copy.deepcopy(SCHEMA)
    schema["function"]["parameters"]["properties"]["duration"]["default"] = 8
    schema["function"]["parameters"]["properties"]["seed"]["default"] = 7
    assert h3_request.tool_request({"prompt": " A fox. "}, schema) == {
        "task": "t2va",
        "prompt": "A fox.",
        "conditions": [],
        "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 8},
        "seed": 7,
    }


def test_frozen_legacy_tool_schema_keeps_its_request_identity():
    schema = copy.deepcopy(SCHEMA)
    properties = schema["function"]["parameters"]["properties"]
    for key in ("image", "aspect_ratio"):
        properties.pop(key)
    properties["duration"] = {"type": "integer", "enum": [5, 10], "default": 5}
    assert h3_request.tool_request({"prompt": " A fox. "}, schema) == {"prompt": "A fox.", "duration": 5, "seed": 42}
    with pytest.raises(ValueError, match="integer"):
        h3_request.tool_request({"prompt": "A fox.", "duration": 5.0}, schema)


@pytest.mark.parametrize("kind,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_first_frame_bytes_are_embedded_and_detached_from_source_file(tmp_path, kind, mime):
    path = tmp_path / "first_frame"
    data = first_frame(kind=kind)
    path.write_bytes(data)
    request = h3_request.video_request("A fox.", image=path, duration=8)
    path.write_bytes(b"changed after submission")
    assert request["task"] == "fl2va"
    assert request["conditions"] == [
        {
            "type": "image",
            "role": "keyframe",
            "frame_index": 0,
            "uri": f"data:{mime};base64," + base64.b64encode(data).decode(),
        }
    ]
    assert h3_request.request_output(request) == {
        "width": 768,
        "height": 1152,
        "fps": 24,
        "num_frames": 192,
        "duration_seconds": 8,
    }
    uri = "base64://" + base64.urlsafe_b64encode(data).decode().rstrip("=")
    assert h3_request.video_request("A fox.", image=uri, duration=8) == request


def test_auto_canvas_respects_exif_rotation_and_explicit_ratio_stretches():
    exif = Image.Exif()
    exif[0x0112] = 6
    data = first_frame(96, 64, "JPEG", exif=exif.tobytes())
    uri = "base64://" + base64.b64encode(data).decode()
    auto = h3_request.request_output(h3_request.video_request("A fox.", image=uri))
    assert (auto["width"], auto["height"]) == (768, 1152)
    stretched = h3_request.request_output(h3_request.video_request("A fox.", image=uri, aspect_ratio="16:9"))
    assert (stretched["width"], stretched["height"]) == (1344, 768)


@pytest.mark.parametrize(
    "ratio,size",
    [
        ("21:9", (1536, 672)),
        ("16:9", (1344, 768)),
        ("4:3", (1024, 768)),
        ("1:1", (768, 768)),
        ("3:4", (768, 1024)),
        ("9:16", (768, 1344)),
        ("auto", (1344, 768)),
    ],
)
def test_canvas_matches_service_geometry(ratio, size):
    output = h3_request.request_output(h3_request.video_request("A fox.", aspect_ratio=ratio))
    assert (output["width"], output["height"]) == size


def test_every_supported_frame_count_and_nonintegral_durations():
    for frames in range(124, 346, 17):
        assert h3_request.frame_count(frames / 24) == frames
    assert [h3_request.frame_count(seconds) for seconds in (4.5, 5, 5.5, 8, 8.5, 10)] == [124, 124, 141, 192, 209, 243]


def test_raw_sglang_defaults_and_duration_precedence():
    request = {"task": " T2VA ", "prompt": "A fox.", "seconds": None}
    assert h3_request.request_output(request)["num_frames"] == 124
    request["seconds"] = 8
    assert h3_request.request_output(request)["num_frames"] == 192
    request["target"] = {"duration_seconds": 10}
    assert h3_request.request_output(request)["num_frames"] == 243


@pytest.mark.parametrize("duration", [4, 4.4, 14.4, 15, float("nan"), float("inf"), True, "10"])
def test_invalid_duration_rejected_before_submission(duration):
    with pytest.raises(ValueError, match="duration"):
        h3_request.video_request("A fox.", duration=duration)


@pytest.mark.parametrize("seed", [True, -1, 2**63, 42.0, "42"])
def test_strict_seed_validation(seed):
    with pytest.raises(ValueError, match="seed"):
        h3_request.video_request("A fox.", seed=seed)


@pytest.mark.parametrize(
    "image", ["https://example.com/frame.png", "file:///frame.png", "base64://***", "data:image/png,abc"]
)
def test_rejected_images_do_not_echo_image_payload(image):
    with pytest.raises(ValueError) as error:
        h3_request.video_request("A fox.", image=image)
    assert image not in str(error.value)


@pytest.mark.parametrize("data", [b"not an image", first_frame(16, 32), first_frame(kind="GIF"), first_frame()[:40]])
def test_bad_image_data_rejected(data):
    with pytest.raises(ValueError):
        h3_request.video_request("A fox.", image="base64://" + base64.b64encode(data).decode())


def test_image_byte_limit_and_auto_ratio(tmp_path, monkeypatch):
    path = tmp_path / "frame.png"
    path.write_bytes(first_frame())
    monkeypatch.setattr(h3_request, "MAX_IMAGE_BYTES", 32)
    with pytest.raises(ValueError, match="20 MiB"):
        h3_request.video_request("A fox.", image=path)
    with pytest.raises(ValueError, match="20 MiB"):
        h3_request.decode_image("base64://" + base64.b64encode(path.read_bytes()).decode())
    monkeypatch.undo()
    path.write_bytes(first_frame(320, 64))
    with pytest.raises(ValueError, match="aspect ratio"):
        h3_request.video_request("A fox.", image=path)
    assert h3_request.video_request("A fox.", image=path, aspect_ratio="16:9")["task"] == "fl2va"


def test_media_validation_checks_requested_canvas_duration_frames_and_audio():
    request = h3_request.video_request("A fox.", duration=8, aspect_ratio="9:16")
    media = {
        "duration_s": 8,
        "video": {"width": 768, "height": 1344, "fps": 24, "codec": "avc1", "frames": 192},
        "audio": {"sample_rate": 32000, "channels": 2, "codec": "mp4a"},
    }
    reference = {**h3.media_profile(media, duration=8), "width": 1344, "height": 768}
    h3.validate_media(media, request, reference)
    for field, change in (
        ("video", {"width": 1344, "height": 768}),
        ("video", {"frames": 243}),
        ("video", {"fps": 30}),
        ("audio", {"sample_rate": 48000}),
        ("audio", {"channels": 1}),
    ):
        with pytest.raises(ValueError):
            h3.validate_media({**media, field: {**media[field], **change}}, request, reference)
    with pytest.raises(ValueError):
        h3.validate_media({**media, "duration_s": 10.125}, request, reference)


@pytest.mark.asyncio
async def test_inspection_generation_uses_the_same_image_without_echoing_bytes(tmp_path):
    path = tmp_path / "frame.png"
    path.write_bytes(first_frame())
    generated = []

    async def generate(request, directory):
        generated.append(request)
        return {"status": "succeeded", "request": request}

    episode = InspectionEpisode("A fox.", [SCHEMA], tmp_path, generate, None)
    parameters = {"prompt": "A fox.", "image": str(path), "duration": 8, "aspect_ratio": "1:1"}
    response = await episode.execute("sol_h3", parameters)
    assert generated == [h3_request.video_request(**parameters)]
    condition = response["generation_request"]["conditions"][0]
    assert "uri" not in condition
    assert condition["sha256"] == file_sha256(path) and condition["bytes"] == path.stat().st_size
    assert episode.events[0]["parameters"] == response["generation_request"]
    assert "uri" in generated[0]["conditions"][0]


@pytest.mark.parametrize("duration", [5, 14.375])
def test_inspector_checks_feedback_against_requested_duration(tmp_path, duration):
    from examples.video_agents import inspection_worker
    from examples.video_agents.inspection_agent import feedback

    path = tmp_path / "video.mp4"
    path.write_bytes(b"test video")
    request = h3_request.video_request("A fox.", duration=duration)
    video = {"path": str(path), "sha256": file_sha256(path), "request": request}
    template = feedback.build_request("MEDIA_URI", "A fox.", request["prompt"], ())
    identity = {"video_sha256": video["sha256"], "request": template}
    result_path = tmp_path / fingerprint(identity) / "result.json"
    report = {
        "schema_version": "native-inspection-feedback-v1",
        "status": "ok",
        "assessment_note": "Visible.",
        "audio_assessed": True,
        "observed_sequence": [{"phase": "ending", "observation": "The fox pauses.", "start_s": 13, "end_s": 14}],
        "successful_elements_to_preserve": [],
        "findings": [],
        "primary_repair_targets": [],
        "prior_issue_checks": [],
        "uncertain_observations": [],
    }
    write_json(result_path, report)
    if duration == 5:
        with pytest.raises(ValueError, match="within the media"):
            inspection_worker.inspect_generation(video, "A fox.", tmp_path, tmp_path)
    else:
        assert inspection_worker.inspect_generation(video, "A fox.", tmp_path, tmp_path) == report
        with pytest.raises(ValueError, match="within the media"):
            feedback.validate_feedback(report)  # The frozen inspector retains its original default.
