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
"""Native video RL data with a stable schema for optional source images."""

import json

import datasets
import pyarrow as pa

from verl.utils.dataset.rl_dataset import RLHFDataset


def video_features(first_record):
    """Keep the frozen row's tool schema, explicitly typing even empty image lists."""
    if "images" not in first_record:
        raise ValueError("Mixed video records require an images list, including [] for T2V")
    features = datasets.Features.from_arrow_schema(pa.Table.from_pylist([first_record]).schema)
    features["images"] = datasets.List({"image": datasets.Value("string"), "max_pixels": datasets.Value("int64")})
    return features


class VideoRLDataset(RLHFDataset):
    """Use native tokenization and batching with explicit mixed JSON features."""

    def _load_data_file(self, data_file):
        if not data_file.endswith(".jsonl"):
            return super()._load_data_file(data_file)
        with open(data_file, encoding="utf-8") as stream:
            first_record = next((json.loads(line) for line in stream if line.strip()), None)
        if first_record is None:
            raise ValueError(f"Empty mixed video dataset: {data_file}")
        return datasets.load_dataset("json", data_files=data_file, features=video_features(first_record))["train"]
