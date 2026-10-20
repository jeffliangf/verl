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
"""Export SFT conversations as Parquet, readable JSONL, and text previews."""

import argparse
import json
from pathlib import Path

from .config import DEFAULT_DATASET, DEFAULT_REWRITE_INSTRUCTION, DEFAULT_SYSTEM_PROMPT, ROOT, build_messages
from .storage import atomic_text, file_sha256, fingerprint, read_jsonl, write_json, write_jsonl

TARGET_KEY = "h3_context_ir"
LEGACY_TARGET_KEY = "h3_context_ir, "


def _convert_rows(path: Path, split: str) -> tuple[list[dict], int, set[str]]:
    """Preserve text and row order, repairing only the known target-key typo."""
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"Empty {split} dataset: {path}")
    converted, seen_ids, prompts, repaired = [], set(), set(), 0
    for index, row in enumerate(rows):
        if LEGACY_TARGET_KEY in row:
            if TARGET_KEY in row and row[TARGET_KEY] != row[LEGACY_TARGET_KEY]:
                raise ValueError(f"Conflicting target fields in {path}, row {index}: {row.get('id')}")
            row[TARGET_KEY] = row.pop(LEGACY_TARGET_KEY)
            repaired += 1
        for field in ("id", "prompt", TARGET_KEY):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing nonempty {field} in {path}, row {index}: {row.get('id')}")
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate ID in {path}: {row['id']}")
        seen_ids.add(row["id"])
        prompts.add(row["prompt"].strip())
        source = row.get("source", "unknown")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Invalid source in {path}: {row['id']}")
        converted.append(
            {
                "id": row["id"],
                "source": source,
                "split": split,
                "messages": build_messages(row["prompt"]) + [{"role": "assistant", "content": row[TARGET_KEY]}],
                "enable_thinking": False,
            }
        )
    return converted, repaired, prompts


def prepare_sft(train_file: Path, val_file: Path, output_dir: Path) -> dict:
    """Validate both cohorts before writing SFT files; never modify source JSONL."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    train_file, val_file, output_dir = (path.expanduser().resolve() for path in (train_file, val_file, output_dir))
    train, train_repaired, train_prompts = _convert_rows(train_file, "train")
    val, val_repaired, val_prompts = _convert_rows(val_file, "val")
    shared_ids = {row["id"] for row in train} & {row["id"] for row in val}
    shared_prompts = train_prompts & val_prompts
    if shared_ids or shared_prompts:
        raise ValueError(
            f"Train/validation overlap: {len(shared_ids)} shared IDs, {len(shared_prompts)} shared prompts"
        )

    schema = pa.schema(
        [
            ("id", pa.string()),
            ("source", pa.string()),
            ("split", pa.string()),
            ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
            ("enable_thinking", pa.bool_()),
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "messages_key": "messages",
        "apply_chat_template_kwargs": {"enable_thinking": False},
        "system_prompt_sha256": fingerprint(DEFAULT_SYSTEM_PROMPT),
        "rewrite_instruction_sha256": fingerprint(DEFAULT_REWRITE_INSTRUCTION),
        "splits": {},
    }
    for split, source, rows, repaired, prompts in (
        ("train", train_file, train, train_repaired, train_prompts),
        ("val", val_file, val, val_repaired, val_prompts),
    ):
        destination = output_dir / f"{split}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination)
        jsonl_path = output_dir / f"{split}.jsonl"
        write_jsonl(jsonl_path, rows)
        preview_path = output_dir / f"{split}.preview.txt"
        preview_rows = rows[:3]
        preview = [
            f"{split}: first {len(preview_rows)} of {len(rows)} examples",
            "Full message text, without truncation.",
        ]
        for index, row in enumerate(preview_rows, 1):
            preview.append(
                f"{'=' * 80}\nExample {index}\n"
                f"id: {row['id']}\nsource: {row['source']}\n"
                f"enable_thinking: {json.dumps(row['enable_thinking'])}"
            )
            for message in row["messages"]:
                preview.append(f"[{message['role']}]\n{message['content']}")
        atomic_text(preview_path, "\n\n".join(preview) + "\n")
        manifest["splits"][split] = {
            "input": str(source),
            "input_sha256": file_sha256(source),
            "output": str(destination),
            "jsonl_output": str(jsonl_path),
            "preview_output": str(preview_path),
            "rows": len(rows),
            "repaired_target_keys": repaired,
            "duplicate_prompt_rows": len(rows) - len(prompts),
        }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main():
    """Prepare the existing SFT training set and the established 200-example validation set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=ROOT / "data/prompt_rewrite_sft.jsonl")
    parser.add_argument("--val-file", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/prompt_rewrite_sft")
    args = parser.parse_args()
    print(json.dumps(prepare_sft(args.train_file, args.val_file, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
