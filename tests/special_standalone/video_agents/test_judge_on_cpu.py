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
"""Keep interrupted, stale, or fabricated judge results out of agent rewards."""

import json

import pytest

from examples.video_agents import judge, scoring
from examples.video_agents.config import CANDIDATE, JUDGE_SETTINGS
from examples.video_agents.storage import file_sha256, read_jsonl, write_jsonl


@pytest.fixture
def ledger(tmp_path):
    protocol = {
        "name": "test-agent",
        "judge": JUDGE_SETTINGS,
        "rubric": "Compare the videos against the original prompt.",
        "samples": [
            {
                "id": "sample",
                "prompt": "The original request",
                "source": "aa_eval",
                "references": {name: {"path": name + ".mp4"} for name in ("h3_context_ir", "best_of_4")},
            }
        ],
    }
    manifests = judge.build_manifests(protocol, {"sample": {"path": "candidate.mp4"}})
    judge.freeze_manifests(tmp_path, manifests)
    manifest = manifests["original"]
    manifest_path = tmp_path / "judge/manifest-original.json"
    rows = []
    for pair_id, pair in manifest["pairs"].items():
        rows.append(
            {
                "status": "completed",
                "pair_id": pair_id,
                "prompt_id": pair["prompt_id"],
                "model_A": pair["model_A"],
                "model_B": pair["model_B"],
                "preferred_model": CANDIDATE,
                "preferred_video": "A" if pair["model_A"] == CANDIDATE else "B",
                "manifest_sha256": file_sha256(manifest_path),
                "evaluation_signature": judge.evaluation_signature(pair, manifest["instructions_how_to_compare"]),
                **JUDGE_SETTINGS,
            }
        )
    path = tmp_path / "judge/judgments-original.jsonl"
    write_jsonl(path, rows)
    return path, manifest, manifest_path, JUDGE_SETTINGS


@pytest.mark.parametrize("fragment", [b'{"status":"completed",', b'{"reason":"\xe2\x82'])
def test_interrupted_ledger_keeps_completed_votes_and_can_resume(ledger, fragment):
    path, manifest, _, _ = ledger
    rows = read_jsonl(path)
    missing = rows.pop()
    write_jsonl(path, rows)
    with path.open("ab") as handle:
        handle.write(fragment)
    before = path.read_bytes()
    completed, counts = judge.read_judgments(*ledger)
    assert len(completed) == 1 and counts["malformed_lines"] == 1
    assert path.read_bytes() == before
    assert set(completed) < set(manifest["pairs"])
    with path.open("ab") as handle:
        handle.write(b"\n" + json.dumps(missing).encode() + b"\n")
    completed, counts = judge.read_judgments(*ledger)
    assert set(completed) == set(manifest["pairs"])
    assert counts == {"completed": 2, "malformed_lines": 1}


@pytest.mark.parametrize(
    "field,value",
    [
        ("pair_id", "unknown"),
        ("prompt_id", "another-prompt"),
        ("model_A", "another-model"),
        ("judge_model", "another-judge"),
        ("video_processing", "agentic"),
        ("seed", 9),
        ("manifest_sha256", "stale"),
        ("evaluation_signature", "stale"),
        ("preferred_model", "unknown"),
        ("preferred_video", "tie"),
    ],
)
def test_stale_or_misoriented_judgments_are_rejected(ledger, field, value):
    path = ledger[0]
    rows = read_jsonl(path)
    rows[0][field] = value
    write_jsonl(path, rows)
    with pytest.raises(ValueError):
        judge.read_judgments(*ledger)


def test_duplicate_votes_cannot_inflate_rewards(ledger):
    path = ledger[0]
    rows = read_jsonl(path)
    write_jsonl(path, rows + [rows[0]])
    with pytest.raises(ValueError, match="duplicate completed pair"):
        judge.read_judgments(*ledger)


def test_failure_reports_keep_the_latest_error_for_each_pending_pair(ledger):
    path = ledger[0]
    rows = read_jsonl(path)
    first, second = rows[0]["pair_id"], rows[1]["pair_id"]
    errors = [
        {"status": "error", "pair_id": first, "stage": "upload", "error": "HTTP 503"},
        {"status": "error", "pair_id": first, "stage": "interaction", "error": "HTTP 400: prohibited_content"},
        {"status": "error", "pair_id": second, "stage": "interaction", "error": "read timed out"},
    ]
    write_jsonl(path, rows + errors)
    assert judge.judgment_errors(path, {first}) == {first: "interaction: HTTP 400: prohibited_content"}
    assert judge.judgment_errors(path, set()) == {}


def test_pair_score_is_invariant_to_display_order(ledger):
    pair = next(iter(ledger[1]["pairs"].values()))
    win = {"status": "completed", "preferred_model": CANDIDATE}
    loss = {"status": "completed", "preferred_model": pair["reference"]}
    first = scoring.score_pair(pair, win, loss)
    reversed_result = scoring.score_pair(judge.swap_pair(pair), loss, win)
    assert first["score"] == reversed_result["score"] == 0.5
    assert first["candidate_first_win"] == reversed_result["candidate_first_win"]
    assert first["candidate_second_win"] == reversed_result["candidate_second_win"]


@pytest.mark.parametrize("status", ["error", "fallback", None])
def test_pair_score_requires_real_judgments_in_both_orders(ledger, status):
    pair = next(iter(ledger[1]["pairs"].values()))
    win = {"status": "completed", "preferred_model": CANDIDATE}
    with pytest.raises(ValueError, match="completed judgment in both orders"):
        scoring.score_pair(pair, win, {**win, "status": status})
