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
"""Reproducible phrasing for training requests without changing their controls."""

from collections import Counter
from pathlib import Path

from .storage import file_sha256, fingerprint, read_json

LIBRARY_PATH = Path(__file__).with_name("configs") / "request_wording.json"
LIBRARY = read_json(LIBRARY_PATH)
LIBRARY_SHA256 = file_sha256(LIBRARY_PATH)
RENDERER_SHA256 = file_sha256(Path(__file__))
NUMBER_WORDS = {5: "five", 8: "eight", 10: "ten", 12: "twelve", 14: "fourteen"}


def wording_config(seed=42):
    """Bind a prepared corpus to the exact phrase library, renderer, and seed."""
    return {
        "version": LIBRARY["version"],
        "seed": seed,
        "library_sha256": LIBRARY_SHA256,
        "renderer_sha256": RENDERER_SHA256,
    }


def render_request(sample, *, duration_seconds=None, aspect=None, retime_seconds=None, seed=42):
    """Vary only the synthetic wording; retain original scene text and custom requests."""
    from .mixed_data import natural_request

    metadata = {**wording_config(seed), "choices": {}}
    choices = metadata["choices"]

    def pick(name, values):
        index = int(fingerprint([LIBRARY["version"], seed, sample["id"], name]), 16) % len(values)
        choices[name] = index
        return values[index]

    def seconds(value, name):
        words = pick(name, (False, True))
        return NUMBER_WORDS[value] if words and value in NUMBER_WORDS else f"{value:g}"

    fragments = []
    duration_text = None
    if duration_seconds is not None:
        template = pick("duration", LIBRARY["duration"])
        duration_text = seconds(duration_seconds, "duration_number")
        fragments.append(template.format(seconds=duration_text))
    if aspect is not None:
        if aspect["kind"] == "orientation":
            kind = aspect["value"]
        elif aspect["kind"] == "preserve_source":
            kind = "preserve_source"
        else:
            kind = "square" if aspect["value"] == "1:1" else "ratio"
        fragments.append(pick(kind, LIBRARY[kind]).format(ratio=aspect["value"]))
        metadata["aspect_wording"] = kind
    reverse_controls = len(fragments) == 2 and pick("control_order", (False, True))
    if reverse_controls:
        fragments.reverse()
    settings = pick("joiner", (" and ", ", ")).join(fragments) if fragments else ""
    # Most requests integrate controls into the opening sentence; others put them
    # before or after the scene, as a user might add a separate instruction.
    layout = pick("layout", ("inline", "inline", "inline", "prefix", "suffix")) if settings else "none"
    original_request = sample.get("request", natural_request(sample["prompt"], image=bool(sample["image"])))
    custom = original_request != natural_request(sample["prompt"], image=bool(sample["image"]))
    retiming = ""
    if retime_seconds is not None:
        retiming = pick("retiming", LIBRARY["retiming"]).format(seconds=seconds(retime_seconds, "retiming_number"))
    if custom:
        # A supplied request may contain instructions absent from the scene text.
        # Do not replace it with one of the generic openings.
        body = original_request
        if retiming:
            body = retiming + "\n\n" + body
        if layout == "inline":
            layout = "prefix"
    else:
        intro = pick(sample["task"], LIBRARY[sample["task"]])
        intro = intro.format(settings=", " + settings if layout == "inline" else "")
        body = "\n\n".join(part for part in (intro, retiming, sample["prompt"]) if part)
    if settings and layout in {"prefix", "suffix"}:
        sentences = []
        if duration_text is not None:
            sentences.append(pick("duration_sentence", LIBRARY["duration_sentence"]).format(seconds=duration_text))
        if aspect is not None:
            sentences.append(pick(kind + "_sentence", LIBRARY[kind + "_sentence"]).format(ratio=aspect["value"]))
        if reverse_controls:
            sentences.reverse()
        sentence = " ".join(sentences)
        body = sentence + "\n\n" + body if layout == "prefix" else body + "\n\n" + sentence
    metadata.update(layout=layout, custom_request_preserved=custom)
    return body, metadata


def wording_summary(samples):
    """Report actual phrase coverage without treating the preview as a distribution."""
    labels = [s["control_metadata"]["wording"] for s in samples if "wording" in s["control_metadata"]]
    return {
        "version": LIBRARY["version"],
        "library_sha256": LIBRARY_SHA256,
        "intro_variants_available": {task: len(LIBRARY[task]) for task in ("t2v", "i2v")},
        "intro_variants_used": {
            task: len({label["choices"][task] for label in labels if task in label["choices"]})
            for task in ("t2v", "i2v")
        },
        "layouts": dict(Counter(label["layout"] for label in labels)),
    }
