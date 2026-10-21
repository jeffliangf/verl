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
"""Text-only SFT reader with the exact inference prefix and assistant-only loss."""

from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class PromptRewriteSFTDataset(Dataset):
    """Render whole conversations for templates that reject isolated system turns."""

    def __init__(self, parquet_files, tokenizer, config, processor=None, max_samples=-1):
        if config.get("pad_mode") != "no_padding" or config.get("truncation") != "error":
            raise ValueError("Prompt rewrite SFT requires no_padding and truncation=error")
        paths = [parquet_files] if isinstance(parquet_files, str | Path) else parquet_files
        self.rows = [row for path in paths for row in pq.read_table(Path(path).expanduser()).to_pylist()]
        if max_samples > 0:
            self.rows = self.rows[:max_samples]
        if not self.rows:
            raise ValueError("Empty prompt rewrite SFT dataset")
        self.tokenizer, self.max_length = tokenizer, config.get("max_length", 8192)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        messages = row["messages"]
        if row.get("enable_thinking") is not False or [m["role"] for m in messages] != ["system", "user", "assistant"]:
            raise ValueError(f"Expected a non-thinking system/user/assistant conversation: {row['id']}")
        kwargs = {"tokenize": True, "return_dict": False, "enable_thinking": False}
        full = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)
        prefix = self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, **kwargs)
        if full[: len(prefix)] != prefix or len(full) <= len(prefix):
            raise ValueError(f"Assistant target does not follow the exact inference prefix: {row['id']}")
        if len(full) > self.max_length:
            raise ValueError(f"Conversation exceeds max_length={self.max_length}: {row['id']} ({len(full)} tokens)")
        input_ids = torch.tensor(full, dtype=torch.long)
        loss_mask = torch.zeros_like(input_ids)
        loss_mask[len(prefix) :] = 1
        return {"input_ids": input_ids, "loss_mask": loss_mask, "position_ids": torch.arange(len(full))}
