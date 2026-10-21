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
"""Pure per-prompt scoring, independent of inference and video generation."""

import random
from collections import Counter

from .config import CANDIDATE, REFERENCE_FIELDS, SCORE_SOURCES
from .storage import item_key


def score_pair(pair: dict, original: dict, swapped: dict, *, candidate: str = CANDIDATE) -> dict:
    """Each order contributes half a point; disagreement is a split, not a judge tie."""
    fallback_orders = [
        order for order, row in (("original", original), ("swapped", swapped)) if row.get("status") == "fallback"
    ]
    if fallback_orders:
        return {
            "id": pair["prompt_id"],
            "source": pair["source"],
            "reference": pair["reference"],
            "score": 0.5,
            "outcome": "fallback",
            "fallback_stage": "gemini",
            "fallback_orders": fallback_orders,
            "candidate_first_win": None,
            "candidate_second_win": None,
            "original_winner": original.get("preferred_model"),
            "swapped_winner": swapped.get("preferred_model"),
            "original_reason": original.get("reason"),
            "swapped_reason": swapped.get("reason"),
        }
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


def summarize(rows: list[dict], expected: int, bootstrap_rounds: int = 2000, seed: int = 20260925) -> dict:
    """Bootstrap prompts, keeping their two correlated orientation votes together."""
    if bootstrap_rounds < 100:
        raise ValueError("Use at least 100 bootstrap rounds")
    n = len(rows)
    counts = Counter(row["outcome"] for row in rows)
    result = {
        "expected_prompts": expected,
        "scored_prompts": n,
        "incomplete_prompts": expected - n,
        "coverage": n / expected if expected else 0.0,
        "judged_prompts": n - counts["fallback"],
        "judged_coverage": (n - counts["fallback"]) / expected if expected else 0.0,
        "fallback_prompts": counts["fallback"],
        "h3_fallback_prompts": sum(row.get("fallback_stage") == "h3" for row in rows),
        "gemini_fallback_prompts": sum(row.get("fallback_stage") == "gemini" for row in rows),
        "fallback_judgments": sum(len(row.get("fallback_orders", [])) for row in rows),
        "wins": counts["win"],
        "splits": counts["split"],
        "losses": counts["loss"],
        "points": sum(row["score"] for row in rows),
        "win_rate_half_split": None,
        "prompt_bootstrap95": None,
        "order_disagreement_rate": None,
        "candidate_first_win_rate": None,
        "candidate_second_win_rate": None,
    }
    if not n:
        return result
    scores = [row["score"] for row in rows]
    judged = [row for row in rows if row["outcome"] != "fallback"]
    rng = random.Random(seed)
    bootstrap = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(bootstrap_rounds))
    result.update(
        win_rate_half_split=sum(scores) / n,
        prompt_bootstrap95=[
            bootstrap[int(0.025 * (bootstrap_rounds - 1))],
            bootstrap[int(0.975 * (bootstrap_rounds - 1))],
        ],
        order_disagreement_rate=counts["split"] / len(judged) if judged else None,
        candidate_first_win_rate=sum(row["candidate_first_win"] for row in judged) / len(judged) if judged else None,
        candidate_second_win_rate=sum(row["candidate_second_win"] for row in judged) / len(judged) if judged else None,
    )
    return result


def dataset_scores(by_source: dict) -> dict[str, float | None]:
    """The default three scores, with equal dataset weights in the average.

    If either dataset has no completed comparisons, its score and the average
    are unavailable. Partial coverage within a dataset remains explicit in the
    report's source summaries.
    """
    scores = {name: by_source.get(source, {}).get("win_rate_half_split") for name, source in SCORE_SOURCES.items()}
    scores["average"] = (
        sum(scores.values()) / len(scores) if all(value is not None for value in scores.values()) else None
    )
    return scores


def make_report(
    protocol: dict,
    manifests: dict,
    judgments: dict,
    allow_partial: bool = False,
    *,
    generation_fallbacks: dict | None = None,
) -> tuple[dict, list]:
    """Report both references separately, overall and for each data source."""
    rows, incomplete = [], []
    generation_fallbacks = generation_fallbacks or {}
    if set(generation_fallbacks) - {sample["id"] for sample in protocol["samples"]}:
        raise ValueError("Unknown H3 fallback prompt")
    for sample in protocol["samples"]:
        for reference in protocol.get("reference_fields", REFERENCE_FIELDS):
            pair_id = item_key(sample["id"]) + "__vs_" + reference
            pair = manifests["original"]["pairs"].get(pair_id)
            if sample["id"] in generation_fallbacks:
                fallback = generation_fallbacks[sample["id"]]
                if pair is not None or fallback.get("score") != 0.5 or fallback.get("stage") != "h3":
                    raise ValueError("H3 fallback conflicts with a candidate video or neutral policy")
                rows.append(
                    {
                        "id": sample["id"],
                        "source": sample["source"],
                        "reference": reference,
                        "score": 0.5,
                        "outcome": "fallback",
                        "fallback_stage": "h3",
                        "fallback_orders": [],
                        "candidate_first_win": None,
                        "candidate_second_win": None,
                        "original_winner": None,
                        "swapped_winner": None,
                        "original_reason": fallback["reason"],
                        "swapped_reason": fallback["reason"],
                    }
                )
            elif pair is None or pair_id not in judgments["original"] or pair_id not in judgments["swapped"]:
                incomplete.append({"id": sample["id"], "reference": reference})
            else:
                rows.append(
                    score_pair(
                        pair,
                        judgments["original"][pair_id],
                        judgments["swapped"][pair_id],
                        candidate=protocol.get("candidate_field", CANDIDATE),
                    )
                )
    if incomplete and not allow_partial:
        raise ValueError(f"{len(incomplete)} comparisons lack both judgments; resume judging or use --allow-partial")
    sources = Counter(sample["source"] for sample in protocol["samples"])
    comparisons = {}
    for reference in protocol.get("reference_fields", REFERENCE_FIELDS):
        matched = [row for row in rows if row["reference"] == reference]
        by_source = {
            source: summarize([row for row in matched if row["source"] == source], count)
            for source, count in sorted(sources.items())
        }
        comparisons[reference] = {
            "scores": dataset_scores(by_source),
            "overall": summarize(matched, len(protocol["samples"])),
            "by_source": by_source,
        }
    return {
        "candidate": protocol["name"],
        "complete": not incomplete,
        "judging_complete": not incomplete and all(row["outcome"] != "fallback" for row in rows),
        "generation_complete": not generation_fallbacks,
        "generation_fallbacks": [
            {key: row[key] for key in ("id", "status", "reason")} for row in generation_fallbacks.values()
        ],
        "judge": protocol["judge"],
        "scoring": (
            "1 for two wins, 0.5 for split orders or an explicit failure fallback, "
            "0 for two losses; missing pairs unscored"
        ),
        "fallback_comparisons": [
            {key: row[key] for key in ("id", "reference", "fallback_orders", "fallback_stage")}
            for row in rows
            if row["outcome"] == "fallback"
        ],
        "average_definition": "(moviegen + aa_eval) / 2; equal dataset weights",
        "comparisons": comparisons,
        "incomplete": incomplete,
    }, rows
