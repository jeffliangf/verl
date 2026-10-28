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


def score_pair(
    pair: dict, original: dict, swapped: dict, *, candidate: str = CANDIDATE, allow_fallback: bool = False
) -> dict:
    """Average both orders; an explicitly allowed missing judgment contributes 0.5."""
    winners, values = [], []
    for row in (original, swapped):
        if row.get("status") == "completed":
            winner = row["preferred_model"]
            if winner not in (candidate, pair["reference"]):
                raise ValueError("Unexpected winner in a candidate/reference comparison")
            winners.append(winner)
            values.append(float(winner == candidate))
        elif (
            allow_fallback
            and row.get("status") == "fallback"
            and row.get("fallback_policy") == "neutral-v1"
            and row.get("fallback_score") == 0.5
            and row.get("preferred_model") is None
        ):
            winners.append(None)
            values.append(0.5)
        else:
            raise ValueError("Scoring requires a completed judgment in both orders")
    fallback_count = winners.count(None)
    first, second = (original, swapped) if pair["model_A"] == candidate else (swapped, original)
    score = sum(values) / 2
    return {
        "id": pair["prompt_id"],
        "source": pair["source"],
        "reference": pair["reference"],
        "score": score,
        "outcome": "fallback" if fallback_count else {0.0: "loss", 0.5: "split", 1.0: "win"}[score],
        "fallback_count": fallback_count,
        "candidate_first_win": first["preferred_model"] == candidate if first["status"] == "completed" else None,
        "candidate_second_win": second["preferred_model"] == candidate if second["status"] == "completed" else None,
        "original_winner": winners[0],
        "swapped_winner": winners[1],
        "original_reason": original.get("reason"),
        "swapped_reason": swapped.get("reason"),
    }
