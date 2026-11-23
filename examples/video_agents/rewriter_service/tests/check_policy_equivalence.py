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
"""Check that the service builds the same token ids as the eval for every AA prompt (sha256 stored per record),
and that its parser recovers the eval's rewrite from every stored output."""

import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import policy  # noqa: E402

EVAL = Path("/home/jeff/rl-benchmarks/rl-rewriter-vs-context-ir-candidate8-20261112")
ids_ok = ids_bad = parse_ok = parse_bad = 0
for f in sorted(glob.glob(str(EVAL / "rewrites/*/sampled-*.json"))):
    rec = json.loads(Path(f).read_text())
    raw = rec["input"]["messages"][1]["content"][len(policy.USER_PREFIX) :]
    if f.endswith("sampled-0.json"):
        ids = policy.prompt_ids(raw)
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest() == rec["input"]["prompt_ids_sha256"]:
            ids_ok += 1
        else:
            ids_bad += 1
            print("ids mismatch", f)
    request, _ = policy.parse(rec["output"]["text"])
    if (request["prompt"] if request else None) == rec["rewrite"]:
        parse_ok += 1
    else:
        parse_bad += 1
        print("parse mismatch", f)
print(
    json.dumps(
        {
            "prompt_ids_match": ids_ok,
            "prompt_ids_mismatch": ids_bad,
            "parse_match": parse_ok,
            "parse_mismatch": parse_bad,
        }
    )
)
sys.exit(1 if ids_bad or parse_bad else 0)
