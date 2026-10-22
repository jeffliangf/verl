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
"""Score a video pair from the judge's two presentation orders."""

from .config import CANDIDATE


def score_pair(pair: dict, original: dict, swapped: dict, *, candidate: str = CANDIDATE) -> dict:
    """Each order contributes half a point; disagreement is a split, not a judge tie."""
    if any(row.get("status") != "completed" for row in (original, swapped)):
        raise ValueError("Scoring requires a completed judgment in both orders")
    winners = [original["preferred_model"], swapped["preferred_model"]]
    if any(winner not in (candidate, pair["reference"]) for winner in winners):
        raise ValueError("Unexpected winner in a candidate/reference comparison")
    first, second = (original, swapped) if pair["model_A"] == candidate else (swapped, original)
    score = sum(winner == candidate for winner in winners) / 2
    return {
        "id": pair["prompt_id"],
        "source": pair["source"],
        "reference": pair["reference"],
        "score": score,
        "outcome": {0.0: "loss", 0.5: "split", 1.0: "win"}[score],
        "candidate_first_win": first["preferred_model"] == candidate,
        "candidate_second_win": second["preferred_model"] == candidate,
        "original_winner": winners[0],
        "swapped_winner": winners[1],
        "original_reason": original.get("reason"),
        "swapped_reason": swapped.get("reason"),
    }
