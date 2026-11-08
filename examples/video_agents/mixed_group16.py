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
"""Prepare mixed GRPO by adapting the existing, verified group-16 launcher."""

import argparse
import json
import re
import shlex
import shutil
import socket
import subprocess
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .config import ROOT
from .rl import tool_config
from .storage import atomic_text, file_sha256, fingerprint, freeze_json, read_json, read_jsonl, write_json
from .tournament import load_protocol

EXPECTED_COUNTS = {"train": {"t2v": 5046, "i2v": 3242}, "val": {"t2v": 200, "i2v": 200}}


def mixed_protocol(source, token_file, data_dir):
    """Keep annotated samples unchanged and give this run its own native files."""
    result = deepcopy(source)
    result.update(
        group_size=16,
        groups_per_worker=4,
        h3_concurrency_per_worker=16,
        h3_endpoints=["http://127.0.0.1:28080"],
        h3_token_file=str(token_file),
    )
    for split, spec in result["datasets"].items():
        destination = data_dir / f"{split}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(spec["path"], destination)
        if file_sha256(destination) != spec["sha256"]:
            raise ValueError(f"Frozen {split} dataset changed during copying")
        spec["path"] = str(destination)
    return result


def mixed_overrides(previous, h3_max_inflight=128):
    """Retain training settings and apply the deployed fleet's shared H3 limit."""
    if type(h3_max_inflight) is not int or h3_max_inflight < 1:
        raise ValueError("The shared H3 limit must be a positive integer")
    key = "video_rl.max_judge_fallback_fraction"
    image = "actor_rollout_ref.rollout.engine_kwargs.vllm.limit_mm_per_prompt.image"
    h3_limit = "video_rl.h3_max_inflight"
    settings = dict(value.split("=", 1) for value in previous)
    required = {
        "trainer.nnodes": "2",
        "trainer.n_gpus_per_node": "8",
        "data.train_batch_size": "32",
        "actor_rollout_ref.rollout.n": "16",
        "actor_rollout_ref.actor.ppo_mini_batch_size": "32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": "1",
        "actor_rollout_ref.rollout.agent.num_workers": "8",
        "video_rl.h3_max_inflight": "128",
        "video_rl.judge_concurrency": "500",
        "video_rl.judge_streaming": "true",
    }
    if any(settings.get(k) != v for k, v in required.items()):
        raise ValueError("Previous launcher is not the approved two-node, 32 x 16 configuration")
    result = [value for value in previous if value.split("=", 1)[0] not in {key, image, h3_limit}]
    result += [f"{key}=0.25", f"{image}=1", f"{h3_limit}={h3_max_inflight}"]
    return result


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence of launcher adaptation: {old[:100]}")
    return text.replace(old, new, 1)


def adapt_evaluation_monitor(text):
    """Keep existing completion checks, deriving cohort size from frozen samples."""
    text = replace_once(
        text,
        "def evaluation_status(root, step, expected=200, min_valid_fraction=0.9):",
        "def evaluation_status(root, step, expected=None, min_valid_fraction=0.9):",
    )
    return replace_once(
        text,
        "    root = Path(root)\n"
        '    status = {"step": step, "complete": False, "valid": None, "dropped": None, "degraded": False}',
        "    root = Path(root)\n"
        "    if expected is None:\n"
        '        expected = len(json.loads((root / "protocol.json").read_text())["samples"]["val"])\n'
        '    status = {"step": step, "complete": False, "valid": None, "dropped": None, "degraded": False}',
    )


GEMINI_PREFLIGHT = '''"""Check both frozen judge model IDs from each allocated policy node."""
import json
import os
import socket
import time
import urllib.request
from pathlib import Path
from examples.video_agents.judge_worker import gemini_address_family
from examples.video_agents.storage import load_external, write_json
from examples.video_agents.tournament import load_protocol

root = Path(os.environ["RL_RUN_DIR"])
protocol = load_protocol(root / "protocol.json")
runner = load_external(Path(protocol["eval_repo"]) / "scripts/run_gemini_pairwise_eval.py")
key = os.environ.get("GEMINI_API_KEY") or runner.read_env_file(
    Path(protocol["eval_repo"]) / "arena/.env"
).get("GEMINI_API_KEY", "")
assert key, "Gemini credential is missing"
checks = []
for model in sorted({protocol["judge"]["judge_model"], protocol["eval_judge"]["judge_model"]}):
    request = urllib.request.Request(runner.API_BASE + "/models/" + model, headers={"x-goog-api-key": key})
    started = time.monotonic()
    with gemini_address_family("ipv4"), urllib.request.urlopen(request, timeout=30) as response:
        assert response.status == 200
        checks.append({"model": model, "status": response.status, "seconds": time.monotonic() - started})
result = {"host": socket.gethostname(), "checks": checks}
write_json(root / f"gemini-preflight-{socket.gethostname()}.json", result)
print(json.dumps(result), flush=True)
'''


def require_h3_pool(status, expected_workers):
    """Reject partial or incompatible deployments before allocating policy work."""
    if not status["all_ready"] or len(status["endpoints"]) != 1:
        raise ValueError("The mixed H3 gateway is not ready")
    endpoint = status["endpoints"][0]
    if endpoint["health"]["workers_ready"] < expected_workers:
        raise ValueError("Wait for the full H3 fleet")
    if endpoint["settings_check"]["status"] == "mismatch":
        raise ValueError("H3 reported incompatible generator settings")


H3_PREFLIGHT = '''"""Require all 32 deployed workers to advertise the complete mixed API."""
import json
import os
from pathlib import Path
from examples.video_agents.h3 import server_status
from examples.video_agents.mixed_group16 import require_h3_pool
from examples.video_agents.tournament import load_protocol

root = Path(os.environ["RL_RUN_DIR"])
manifest = json.loads((root / "launch.json").read_text())
status = server_status(load_protocol(root / "protocol.json"))
print(json.dumps(status, indent=2), flush=True)
require_h3_pool(status, manifest["expected_h3_workers"])
'''


def adapt_helpers(previous, output, checkout, connection):
    old_manifest = read_json(previous / "grpo/launch.json")
    old_checkout = old_manifest["code_checkout"]
    originals = {
        p.name: p for p in (previous / "common").iterdir() if p.is_file() and p.suffix in {".py", ".sh", ".sbatch"}
    }
    provenance = {}
    for name, source in sorted(originals.items()):
        text = source.read_text().replace(str(previous), str(output)).replace(old_checkout, str(checkout))
        if name == "canary_ops.py":
            text = adapt_evaluation_monitor(text)
        elif name == "gemini-preflight.py":
            text = GEMINI_PREFLIGHT
        elif name == "ray-node.sh":
            command = shlex.join(
                [
                    "ssh",
                    "-p",
                    str(connection["ssh_port"]),
                    "-i",
                    connection["identity_file"],
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=yes",
                    "-o",
                    "ConnectTimeout=15",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "ServerAliveInterval=15",
                    "-o",
                    "ServerAliveCountMax=3",
                    "-N",
                    "-L",
                    "127.0.0.1:28080:" + connection["gateway_target"],
                    connection["ssh_host"],
                ]
            )
            text, count = re.subn(
                r"ssh -p 44222 .*?jeff@157\.66\.255\.187", lambda _, command=command: command, text, flags=re.DOTALL
            )
            if count != 1:
                raise ValueError("Expected the original H3 tunnel command exactly once")
            text = replace_once(
                text,
                '-m examples.video_agents.rl status --run-dir "$RL_RUN_DIR"',
                '"$RL_SHARED/common/h3-preflight.py"',
            )
        destination = output / "common" / name
        atomic_text(destination, text)
        provenance[name] = {
            "source": str(source),
            "source_sha256": file_sha256(source),
            "prepared_sha256": file_sha256(destination),
            "byte_for_byte": source.read_text() == text,
        }
    atomic_text(output / "common/h3-preflight.py", H3_PREFLIGHT)
    return provenance


def prepare(args):
    source_dir, previous, output = (
        p.expanduser().resolve() for p in (args.source_run, args.previous_root, args.output_root)
    )
    if output.exists():
        raise ValueError("Choose a fresh output root; preserve prior launches")
    source = load_protocol(source_dir / "protocol.json")
    counts = {split: dict(Counter(s["task"] for s in samples)) for split, samples in source["samples"].items()}
    if counts != EXPECTED_COUNTS or source.get("tool_contract") != "urn:verl:sol-h3:3":
        raise ValueError(f"Expected the approved mixed controls dataset, got {counts}")
    for split, spec in source["datasets"].items():
        path = Path(spec["path"])
        if file_sha256(path) != spec["sha256"]:
            raise ValueError(f"Frozen {split} dataset changed")
        rows = read_jsonl(path)
        if [r["extra_info"]["id"] for r in rows] != [s["id"] for s in source["samples"][split]]:
            raise ValueError("Frozen sample and dataset identities disagree")
    data_dir = output / "data"
    overrides = mixed_overrides(read_json(previous / "common-overrides.json"), args.h3_max_inflight)
    checkout = ROOT.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=checkout):
        raise ValueError("Prepare from a clean isolated runtime checkout")
    token_file = args.token_file.expanduser().resolve()
    if not token_file.is_file():
        raise ValueError("The existing H3 credential file is required")
    run = output / "grpo"
    run.mkdir(parents=True)
    connection = {
        "ssh_host": args.ssh_host,
        "ssh_port": args.ssh_port,
        "gateway_target": args.gateway_target,
        "identity_file": str(args.identity_file.expanduser().resolve()),
    }
    protocol = mixed_protocol(source, token_file, data_dir)
    freeze_json(run / "protocol.json", protocol)
    freeze_json(run / "protocol-sha256.json", fingerprint(protocol))
    freeze_json(run / "tools.json", tool_config(run, data_dir / "train.jsonl"))
    freeze_json(output / "common-overrides.json", overrides)
    freeze_json(output / "gateway-connection.json", connection)
    provenance = adapt_helpers(previous, output, checkout, connection)
    freeze_json(output / "infrastructure-provenance.json", provenance)
    old = read_json(previous / "grpo/launch.json")
    keys = (
        "variant",
        "model_revision",
        "nodes",
        "gpus_per_node",
        "total_training_steps",
        "target_step",
        "save_freq",
        "test_freq",
        "evaluate_initial",
        "evaluate_final",
        "checkpoint_retention",
        "train_batch_size",
        "norm_adv_by_std_in_grpo",
        "judge_streaming",
        "h3_phase_timeouts",
        "automatic_recovery",
        "bulk_storage",
        "requested_duration_hours",
        "finish_reserve_seconds",
        "group_size",
        "h3_max_inflight",
        "judge_concurrency",
        "h3_concurrency_per_worker",
        "groups_per_worker",
        "agent_workers",
        "base_model",
        "base_model_source",
        "initial_checkpoint_step",
        "source_checkpoint",
    )
    manifest = {key: deepcopy(old[key]) for key in keys}
    manifest.update(
        data_dir=str(data_dir),
        requested_nodes=args.nodes,
        experiment_name=output.name,
        run_dir=str(run),
        job_name=output.name,
        code_checkout=str(checkout),
        code_commit=commit,
        common_launcher=str(output / "common/train.py"),
        common_overrides=str(output / "common-overrides.json"),
        batch_script=str(output / "common/train.sbatch"),
        source_configuration=str(previous / "grpo/training-config.json"),
        source_configuration_sha256=file_sha256(previous / "grpo/training-config.json"),
        initialization="Fresh Qwen3.8 base weights; new optimizer and data loader; resume disabled",
        evaluation_references=protocol["reference_fields"],
        training_judge=protocol["judge"]["judge_model"],
        evaluation_judge=protocol["eval_judge"]["judge_model"],
        max_judge_fallback_fraction=0.25,
        h3_max_inflight=args.h3_max_inflight,
        expected_h3_workers=32,
        sample_counts=counts,
        source_protocol=str(source_dir / "protocol.json"),
        source_protocol_sha256=file_sha256(source_dir / "protocol.json"),
        controller_host=socket.gethostname(),
        watchdog_service=f"video-agents-{output.name}.service",
        budget_authorization="Reuse prior 72-hour allocation and two-hour finish reserve for the approved mixed run",
    )
    inputs = [
        *sorted((output / "common").glob("*")),
        output / "common-overrides.json",
        output / "gateway-connection.json",
        run / "protocol.json",
        run / "protocol-sha256.json",
        run / "tools.json",
        *[Path(spec["path"]) for spec in protocol["datasets"].values()],
    ]
    manifest["input_sha256"] = {str(p): file_sha256(p) for p in inputs if p.is_file()}
    freeze_json(run / "launch.json", manifest)
    write_json(
        output / "preparation-summary.json",
        {
            "status": "prepared_not_submitted",
            "source_run": str(source_dir),
            "previous_run": str(previous),
            "runtime_commit": commit,
            "sample_counts": counts,
            "native_datasets_reused_unchanged": True,
            "videos_per_training_update": 512,
            "judgments_per_training_update": 7680,
            "videos_per_evaluation": 400,
            "judgments_per_evaluation": 3200,
            "judge_concurrency": 500,
            "h3_max_inflight": args.h3_max_inflight,
            "max_judge_fallback_fraction": 0.25,
            "run_dir": str(run),
            "common_overrides": str(output / "common-overrides.json"),
        },
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--nodes", nargs=2, default=["gmicloud-loki-gpu-001", "gmicloud-loki-gpu-006"])
    parser.add_argument("--ssh-host", required=True)
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--gateway-target", required=True)
    parser.add_argument(
        "--h3-max-inflight", type=int, default=32, help="Shared H3 admission slots across both policy nodes"
    )
    parser.add_argument("--identity-file", type=Path, default=Path("/home/jeff/.ssh/gmi"))
    parser.add_argument("--token-file", type=Path, default=Path("/home/jeff/.config/video-agents/h3.token"))
    args = parser.parse_args()
    manifest = prepare(args)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("run_dir", "code_commit", "sample_counts", "judge_concurrency", "max_judge_fallback_fraction")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
