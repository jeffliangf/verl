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
"""Expand the frozen 2K I2V set with public pairs, without length/domain quotas.

Reuse verified API reviews, screen unused public images and prompts, and preserve
the original export. The only model calls are quality/overlap reviews and text
embeddings; this workflow does not generate or rewrite any prompt.
"""

import argparse
import asyncio
import copy
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import prepare_i2v_data as base

ORIGINAL = base.OUT
OUT = base.ROOT / "data/i2v_public_expansion"
SNAPSHOT = base.ROOT / "data/prompt_rewrite_i2v_rl_original_2000.jsonl"
EXPORT = base.ROOT / "data/prompt_rewrite_i2v_rl_expanded.jsonl"
FINGERPRINT_FIELDS = ("image_sha256", "pixel_sha256", "phash", "width", "height")


def find_call(phase, key):
    for folder in (OUT, ORIGINAL):
        path = folder / "calls" / phase / f"{key}.json"
        if path.exists():
            return base.read(path)
    raise FileNotFoundError(f"Missing {phase} provider record {key}")


def bound_review(row, review, fingerprint):
    call = find_call("review", review["review_call"])
    request = next(item for item in call["request"]["payload"] if item["id"] == row["id"])
    image = next(item for item in call["request"]["images"] if item["id"] == row["id"])
    response = next(item for item in call["parsed"]["items"] if item["id"] == row["id"])
    assert request["prompt"] == row["prompt"], "Stale quality text"
    assert image["sha256"] == fingerprint["image_sha256"], "Stale quality image"
    assert response == {key: value for key, value in review.items() if key != "review_call"}
    assert call["response_id"] and call["model_returned"], "Missing provider provenance"
    return review["usable"] and min(review["grounding"], review["coherence"]) >= 4


def prepare(args):
    OUT.mkdir(parents=True, exist_ok=True)
    current = base.ROOT / "data/prompt_rewrite_i2v_rl.jsonl"
    expected = base.read(ORIGINAL / "validation.json")["dataset_sha256"]
    if not SNAPSHOT.exists():
        assert base.digest(current.read_bytes()) == expected, "Original export changed"
        shutil.copy2(current, SNAPSHOT)
    assert base.digest(SNAPSHOT.read_bytes()) == expected
    original = base.rows(SNAPSHOT)
    pool = base.rows(ORIGINAL / "public_pool.jsonl")
    assert all(count <= 2000 for count in Counter(row["source"] for row in pool).values())
    old_reviews = {row["id"]: row for row in base.read(ORIGINAL / "public_reviews.json")}
    used_image_ids = {row["metadata"].get("image_id", row["id"]) for row in original}
    used_prompts = {base.normalized(row["prompt"]) for row in original}
    eval_rows = base.rows(base.EVAL)
    eval_prompts = {base.normalized(row["prompt"]) for row in eval_rows}
    exclusions = []
    candidates = []
    for row in pool:
        reason = None
        if row["id"] in used_image_ids:
            reason = "Image already retained in original training snapshot"
        elif base.normalized(row["prompt"]) in used_prompts | eval_prompts:
            reason = "Exact normalized prompt overlap"
        elif row["id"] in old_reviews:
            review = old_reviews[row["id"]]
            if not review["usable"] or min(review["grounding"], review["coherence"]) < 4:
                reason = "Existing image/prompt quality review failed"
        if reason:
            exclusions.append({"id": row["id"], "reason": reason})
        else:
            candidates.append(row)
    with zipfile.ZipFile(ORIGINAL / "raw/aigc_images.zip") as archive:
        for row in candidates:
            path = Path(row["image"])
            if row["source"] == "aigcbench" and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(archive.read(row["metadata"]["member"]))
                temporary.replace(path)

    def fetch(row):
        try:
            if not Path(row["image"]).exists():
                base.download(row["url"], row["image"])
            return {"id": row["id"], "image": row["image"], **base.image_info(row["image"])}
        except Exception as error:
            return {"id": row["id"], "error": str(error)[:300]}

    images = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for index, value in enumerate(executor.map(fetch, candidates), 1):
            if "error" in value:
                exclusions.append({"id": value["id"], "reason": "Image unavailable", "error": value["error"]})
            else:
                images[value["id"]] = value
            if index % 100 == 0:
                print(f"Prepared {index}/{len(candidates)} public images", flush=True)
    eval_fps = base.read(ORIGINAL / "audit/eval_images.json")
    retained_fps = [row["metadata"] for row in original]
    allowed = []
    for row in candidates:
        if row["id"] not in images:
            continue
        fp = images[row["id"]]
        value = int(fp["phash"], 16)
        reason = None
        if min(fp["width"], fp["height"]) < 256:
            reason = "Image shorter side below 256 pixels"
        elif any(
            fp["image_sha256"] == other["image_sha256"]
            or fp["pixel_sha256"] == other["pixel_sha256"]
            or (value ^ int(other["phash"], 16)).bit_count() <= 6
            for other in eval_fps
        ):
            reason = "Evaluation image overlap"
        elif any(
            fp["image_sha256"] == other["image_sha256"]
            or fp["pixel_sha256"] == other["pixel_sha256"]
            or (value ^ int(other["phash"], 16)).bit_count() <= 4
            for other in retained_fps
        ):
            reason = "Original training image overlap"
        if reason:
            exclusions.append({"id": row["id"], "reason": reason})
        else:
            allowed.append(row)
    base.dump(OUT / "image_inventory.json", images)
    base.dump_rows(OUT / "candidates.jsonl", allowed)
    base.dump(OUT / "prepare_exclusions.json", exclusions)
    base.dump(
        OUT / "protocol.json",
        {
            "original_sha256": expected,
            "evaluation_sha256": base.digest(base.EVAL.read_bytes()),
            "public_pool": len(pool),
            "candidates": len(allowed),
            "source_cap": 2000,
            "purpose": "Broader coverage while preserving all original 2000 pairs; no length or topic quotas.",
            "generation": "None; all added prompts are public originals retained verbatim.",
        },
    )
    print("Prepared candidates", len(allowed), "excluded", len(exclusions), flush=True)


async def review(args):
    candidates = base.rows(OUT / "candidates.jsonl")
    images = base.read(OUT / "image_inventory.json")
    old = {row["id"]: row for row in base.read(ORIGINAL / "public_reviews.json")}
    results, pending, unavailable = [], [], []
    for row in candidates:
        if row["id"] in old:
            bound_review(row, old[row["id"]], images[row["id"]])
            results.append(old[row["id"]])
        else:
            pending.append(row)
    print("Quality reviews reused", len(results), "new pairs", len(pending), flush=True)
    async with base.Gemini(args.concurrency) as api:

        async def batch(items):
            try:
                schema = copy.deepcopy(base.REVIEW_SCHEMA)
                schema["properties"]["items"].update(minItems=len(items), maxItems=len(items))
                schema["properties"]["items"]["items"]["properties"]["id"]["enum"] = [row["id"] for row in items]
                record = await api.call(
                    "review",
                    base.REVIEW_INSTRUCTION,
                    [{"id": row["id"], "prompt": row["prompt"]} for row in items],
                    schema,
                    images=items,
                )
                values = record["parsed"]["items"]
                if Counter(row["id"] for row in values) != Counter(row["id"] for row in items):
                    raise ValueError("Review IDs differ")
                return [{**row, "review_call": record["request_key"]} for row in values]
            except (base.EmptyProviderResponse, ValueError) as error:
                if len(items) > 1:
                    values = await asyncio.gather(*(batch([row]) for row in items))
                    return [row for group in values for row in group]
                unavailable.append({"id": items[0]["id"], "error": str(error)[:300]})
                return []

        for values in await asyncio.gather(*(batch(pending[i : i + 4]) for i in range(0, len(pending), 4))):
            results.extend(values)
    lookup = {row["id"]: row for row in results}
    qualified = []
    for row in candidates:
        if row["id"] in lookup and bound_review(row, lookup[row["id"]], images[row["id"]]):
            qualified.append({**row, "review": lookup[row["id"]]})
    base.dump(OUT / "quality_reviews.json", results)
    base.dump(OUT / "quality_unavailable.json", unavailable)
    base.dump_rows(OUT / "qualified.jsonl", qualified)
    print("Quality-qualified additions", len(qualified), "unavailable", len(unavailable), flush=True)


async def screen(args):
    from sklearn.feature_extraction.text import TfidfVectorizer

    candidates = base.rows(OUT / "qualified.jsonl")
    original, evaluation = base.rows(SNAPSHOT), base.rows(base.EVAL)
    all_train = original + candidates
    texts = [row["prompt"] for row in evaluation + all_train]
    # Reuse exact-text embedding caches; no source annotations are changed.
    (OUT / "embeddings").mkdir(exist_ok=True)
    for text in texts:
        destination = base.embedding_path(text)
        prior = ORIGINAL / "embeddings" / destination.name
        if prior.exists() and not destination.exists():
            shutil.copy2(prior, destination)
    async with base.Gemini(args.concurrency) as api:
        await base.embed_texts(api, texts)
        embeddings = np.stack([np.load(base.embedding_path(text)) for text in texts])
        offset = len(evaluation) + len(original)
        eval_semantic = embeddings[offset:] @ embeddings[: len(evaluation)].T
        train_semantic = embeddings[offset:] @ embeddings[len(evaluation) :].T
        for i in range(len(candidates)):
            train_semantic[i, len(original) + i] = -1
        tfidf = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True, max_features=60000
        ).fit_transform(texts)
        eval_lexical = (tfidf[offset:] @ tfidf[: len(evaluation)].T).toarray()
        payloads, neighbors = [], []
        for i, row in enumerate(candidates):
            eval_indices = sorted(
                set(np.argsort(-eval_semantic[i])[:3].tolist() + np.argsort(-eval_lexical[i])[:2].tolist())
            )
            train_indices = np.argsort(-train_semantic[i])[:3].tolist()
            references = [
                {"id": evaluation[j]["id"], "kind": "eval", "prompt": evaluation[j]["prompt"]} for j in eval_indices
            ]
            references += [
                {"id": all_train[j]["id"], "kind": "train", "prompt": all_train[j]["prompt"]} for j in train_indices
            ]
            payloads.append({"id": row["id"], "prompt": row["prompt"], "references": references})
            neighbors.append(
                {
                    "id": row["id"],
                    "eval_ids": [evaluation[j]["id"] for j in eval_indices],
                    "train_ids": [all_train[j]["id"] for j in train_indices],
                    "max_eval_semantic": float(eval_semantic[i].max()),
                }
            )
        base.dump(OUT / "neighbors.json", neighbors)
        unavailable = []

        async def batch(items):
            schema = copy.deepcopy(base.OVERLAP_SCHEMA)
            schema["properties"]["items"].update(minItems=len(items), maxItems=len(items))
            props = schema["properties"]["items"]["items"]["properties"]
            props["id"]["enum"] = [row["id"] for row in items]
            for field, kind in [("eval_duplicates", "eval"), ("training_duplicates", "train")]:
                props[field]["items"]["enum"] = sorted(
                    {r["id"] for item in items for r in item["references"] if r["kind"] == kind}
                )
            try:
                record = await api.call("overlap", base.OVERLAP_INSTRUCTION, items, schema)
                values = record["parsed"]["items"]
                if Counter(r["id"] for r in values) != Counter(r["id"] for r in items):
                    raise ValueError("Overlap IDs differ")
                for value in values:
                    item = next(r for r in items if r["id"] == value["id"])
                    for field, kind in [("eval_duplicates", "eval"), ("training_duplicates", "train")]:
                        if not set(value[field]) <= {r["id"] for r in item["references"] if r["kind"] == kind}:
                            raise ValueError("Invalid duplicate reference ID")
                return [{**row, "overlap_call": record["request_key"]} for row in values]
            except (base.EmptyProviderResponse, ValueError) as error:
                if len(items) > 1:
                    groups = await asyncio.gather(*(batch([row]) for row in items))
                    return [row for group in groups for row in group]
                unavailable.append({"id": items[0]["id"], "error": str(error)[:300]})
                return []

        results = await asyncio.gather(*(batch(payloads[i : i + 4]) for i in range(0, len(payloads), 4)))
    base.dump(OUT / "overlap_reviews.json", [row for group in results for row in group])
    base.dump(OUT / "overlap_unavailable.json", unavailable)
    print("Overlap reviewed", sum(map(len, results)), "unavailable", len(unavailable), flush=True)


def finalize(args):
    original = base.rows(SNAPSHOT)
    candidates = base.rows(OUT / "qualified.jsonl")
    images = base.read(OUT / "image_inventory.json")
    overlaps = {row["id"]: row for row in base.read(OUT / "overlap_reviews.json")}
    # Create undirected conflict edges so a judgment in either direction counts.
    conflicts = {row["id"]: set() for row in original + candidates}
    prior_overlaps = [
        row["metadata"].get("overlap_review", {"id": row["id"], "training_duplicates": []}) for row in original
    ]
    for row in [*overlaps.values(), *prior_overlaps]:
        for other in row["training_duplicates"]:
            if other in conflicts:
                conflicts[row["id"]].add(other)
                conflicts[other].add(row["id"])
    selected, rejected = [], []
    retained_ids = {row["id"] for row in original}
    retained_norms = {base.normalized(row["prompt"]) for row in original}
    retained_fps = [row["metadata"] for row in original]
    source_counts = Counter(row["metadata"].get("image_source", row["source"]) for row in original)
    # Highest image-grounding/coherence first, deterministic tie breaking. No
    # constraint rewards matching the evaluation length or topic distribution.
    candidates.sort(
        key=lambda row: (-row["review"]["grounding"], -row["review"]["coherence"], base.digest([20261106, row["id"]]))
    )
    for row in candidates:
        fp = images[row["id"]]
        overlap = overlaps.get(row["id"])
        reason = None
        if overlap is None:
            reason = "No complete overlap audit"
        elif overlap["eval_duplicates"]:
            reason = "Semantic evaluation duplicate"
        elif conflicts[row["id"]] & retained_ids:
            reason = "Semantic training duplicate"
        elif base.normalized(row["prompt"]) in retained_norms:
            reason = "Exact normalized training duplicate"
        elif source_counts[row["source"]] >= 2000:
            reason = "Source cap"
        elif any(
            fp["pixel_sha256"] == other["pixel_sha256"]
            or (int(fp["phash"], 16) ^ int(other["phash"], 16)).bit_count() <= 4
            for other in retained_fps
        ):
            reason = "Training image duplicate"
        if reason:
            rejected.append({"id": row["id"], "reason": reason})
            continue
        value = {key: row[key] for key in ("id", "prompt", "image", "source")}
        value["metadata"] = {
            **row["metadata"],
            **{key: fp[key] for key in FINGERPRINT_FIELDS},
            "quality_review": row["review"],
            "overlap_review": overlap,
            "curation_pass": "public_coverage_expansion",
        }
        selected.append(value)
        retained_ids.add(row["id"])
        retained_norms.add(base.normalized(row["prompt"]))
        retained_fps.append(fp)
        source_counts[row["source"]] += 1
    base.dump_rows(OUT / "additions.jsonl", selected)
    base.dump_rows(EXPORT, original + selected)
    base.dump(OUT / "selection_exclusions.json", rejected)
    base.dump(
        OUT / "selection.json",
        {
            "original_count": len(original),
            "added_count": len(selected),
            "statistics": base.statistics(original + selected),
            "image_sources": dict(source_counts),
            "added_domains": dict(Counter(row["metadata"]["quality_review"]["domain"] for row in selected)),
            "added_media": dict(Counter(row["metadata"]["quality_review"]["medium"] for row in selected)),
        },
    )
    print("Exported", len(original) + len(selected), "pairs; added", len(selected), flush=True)


def validate(args):
    original, final = base.rows(SNAPSHOT), base.rows(EXPORT)
    protocol = base.read(OUT / "protocol.json")
    assert base.digest(SNAPSHOT.read_bytes()) == protocol["original_sha256"]
    assert final[: len(original)] == original, "Original snapshot altered"
    assert len(final) > len(original), "No expansion"
    assert base.digest(base.EVAL.read_bytes()) == protocol["evaluation_sha256"]
    ids = {row["id"] for row in final}
    assert len(ids) == len(final), "Duplicate IDs"
    norms = {base.normalized(row["prompt"]) for row in final}
    assert len(norms) == len(final), "Duplicate prompts"
    assert not norms & {base.normalized(row["prompt"]) for row in base.rows(base.EVAL)}
    source_counts = Counter(row["metadata"].get("image_source", row["source"]) for row in final)
    assert max(source_counts.values()) <= 2000, "Source cap exceeded"
    pool = {row["id"]: row for row in base.rows(ORIGINAL / "public_pool.jsonl")}
    frozen_eval = {row["id"]: row for row in base.read(ORIGINAL / "audit/eval_images.json")}
    eval_images = []
    for row in base.rows(base.EVAL):
        fp = base.image_info(row["image"])
        assert fp["image_sha256"] == frozen_eval[row["id"]]["image_sha256"]
        eval_images.append(fp)
    fps = []
    for index, row in enumerate(final):
        meta = row["metadata"]
        fp = base.image_info(row["image"])
        assert all(fp[key] == meta[key] for key in FINGERPRINT_FIELDS)
        assert bound_review(row, meta["quality_review"], fp)
        if index >= len(original):
            assert row["prompt"] == pool[row["id"]]["prompt"]
            assert row["image"] == pool[row["id"]]["image"]
        overlap = meta["overlap_review"]
        call = find_call("overlap", overlap["overlap_call"])
        request = next(item for item in call["request"]["payload"] if item["id"] == row["id"])
        response = next(item for item in call["parsed"]["items"] if item["id"] == row["id"])
        assert request["prompt"] == row["prompt"]
        assert response == {key: value for key, value in overlap.items() if key != "overlap_call"}
        assert call["response_id"] and call["model_returned"]
        assert not overlap["eval_duplicates"]
        assert not set(overlap["training_duplicates"]) & ids
        fps.append(fp)
    assert len({fp["image_sha256"] for fp in fps}) == len(final)
    assert len({fp["pixel_sha256"] for fp in fps}) == len(final)
    min_eval, min_train = 64, 64
    train_values = [int(fp["phash"], 16) for fp in fps]
    eval_values = [int(fp["phash"], 16) for fp in eval_images]
    for i, value in enumerate(train_values):
        min_eval = min(min_eval, min((value ^ other).bit_count() for other in eval_values))
        if i:
            min_train = min(min_train, min((value ^ other).bit_count() for other in train_values[:i]))
    assert min_eval > 6 and min_train > 4
    assert not {fp["image_sha256"] for fp in fps} & {fp["image_sha256"] for fp in eval_images}
    assert not {fp["pixel_sha256"] for fp in fps} & {fp["pixel_sha256"] for fp in eval_images}
    result = {
        "passed": True,
        "dataset": str(EXPORT),
        "dataset_sha256": base.digest(EXPORT.read_bytes()),
        "original_count": len(original),
        "count": len(final),
        "added_count": len(final) - len(original),
        "original_preserved": True,
        "evaluation_unchanged": True,
        "all_images_decoded": len(final),
        "image_sources": dict(source_counts),
        "min_eval_phash_distance": min_eval,
        "min_train_phash_distance": min_train,
        "exact_prompt_overlap": 0,
        "exact_image_overlap": 0,
        "quality_reviewed": len(final),
        "overlap_reviewed": len(final),
        "limits": [
            "Automated retrieval and review do not prove zero semantic contamination.",
            "Image-domain coverage still requires visual inspection; labels are approximate.",
        ],
    }
    base.dump(OUT / "validation.json", result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "review", "screen", "finalize", "validate", "all"])
    parser.add_argument("--concurrency", type=int, default=12)
    args = parser.parse_args()
    base.OUT = OUT
    actions = {"prepare": prepare, "review": review, "screen": screen, "finalize": finalize, "validate": validate}
    for stage in actions if args.stage == "all" else [args.stage]:
        result = actions[stage](args)
        if asyncio.iscoroutine(result):
            asyncio.run(result)


if __name__ == "__main__":
    main()
