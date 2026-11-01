"""Apply the tested timeout fix after this run's first verified checkpoint."""

import argparse
import fcntl
import json
import shutil
import time
from pathlib import Path

import recover
from canary_ops import atomic_json


def read(path):
    return json.loads(path.read_text())


def ready_checkpoint(root, minimum):
    marker = root / "checkpoints/latest_checkpointed_iteration.txt"
    verified = root / "checkpoints/verified.json"
    if not marker.exists() or not verified.exists():
        return None
    step = int(marker.read_text())
    return step if step >= minimum and read(verified)["step"] == step else None


def updated_config(config, budgets):
    result = json.loads(json.dumps(config))
    if result["video_rl"]["timeouts"]["h3"] != budgets["generation"]:
        raise ValueError("Rendering budget must preserve this experiment's H3 limit")
    result["inspection_rl"].update(
        h3_queue_timeout_seconds=budgets["queue"],
        h3_download_timeout_seconds=budgets["download"],
    )
    return result


def deploy(root):
    plan_path = root / "queue-timeout-deployment.json"
    plan = read(plan_path)
    if plan["status"] in {"submitted", "error"}:
        return plan
    with (root / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(root / "launch.json")
        if (
            manifest["job_id"] != plan["source_job"]
            or manifest["code_commit"] != plan["source_commit"]
            or manifest["code_checkout"] != plan["code_checkout"]
            or manifest["job_name"] != "rl-inspection-v1-1-16h100"
        ):
            raise ValueError("Allocation or code changed; inspect before applying this deployment")
        step = ready_checkpoint(root, plan["minimum_checkpoint_step"])
        smoke = root / "queue-timeout-smoke/result.json"
        if step is None or not smoke.exists():
            return {"status": "waiting", "checkpoint_step": step, "smoke_complete": smoke.exists()}
        if read(smoke)["status"] != "succeeded":
            raise ValueError("Live H3 validation has not succeeded")
        if read(root / "queue-timeout-smoke/protocol.json")["h3_phase_timeouts"] != plan["budgets"]:
            raise ValueError("Live H3 validation used different budgets")
        if recover.verify_launch(root, manifest) != step:
            raise ValueError("Checkpoint changed during verification")
        code = Path(plan["code_checkout"])
        staged = Path(plan["staged_checkout"])
        command = recover.command
        if command(["git", "rev-parse", "HEAD"], cwd=staged) != plan["target_commit"]:
            raise ValueError("Staged fix no longer matches the tested commit")
        if command(["git", "status", "--porcelain"], cwd=staged):
            raise ValueError("Staged fix is dirty")
        command(["git", "merge-base", "--is-ancestor", plan["source_commit"], plan["target_commit"]], cwd=code)
        job = str(plan["source_job"])
        details = command(["scontrol", "show", "job", job, "--oneliner"]) + " "
        required = [f"JobId={job} ", f"JobName={manifest['job_name']} ", f"Command={root / 'train.sbatch'} "]
        if not all(value in details for value in required):
            raise ValueError("Slurm allocation identity does not match the deployment")
        before_config = read(root / "training-config.json")
        after_config = updated_config(before_config, plan["budgets"])
        disabled = root / "recovery-disabled.json"
        if disabled.exists():
            raise ValueError("Another operation already disabled recovery")
        archive = root / "attempts" / job / "before-queue-timeout-fix"
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("training-config.json", "launch.json", "run-budget.json"):
            shutil.copy2(root / name, archive / name)
        plan.update(status="stopping", checkpoint_step=step, started_at_epoch=time.time())
        atomic_json(plan_path, plan)
        atomic_json(disabled, {"reason": "Deploy tested H3 phase budgets from a verified checkpoint"})
        try:
            command(["systemctl", "--user", "stop", manifest["watchdog_service"]])
            if recover.scheduler(job)["state"] not in recover.TERMINAL:
                command(["scancel", job])
            expires = time.monotonic() + 180
            while recover.scheduler(job)["state"] not in recover.TERMINAL:
                if time.monotonic() >= expires:
                    raise TimeoutError("Previous allocation did not terminate within 180 seconds")
                time.sleep(5)
            command(["git", "merge", "--ff-only", plan["target_commit"]], cwd=code)
            atomic_json(root / "training-config.json", after_config)
            manifest["code_commit"] = plan["target_commit"]
            manifest["h3_phase_timeouts"] = plan["budgets"]
            manifest.setdefault("configuration_revisions", []).append(
                {
                    "reason": "Exclude shared H3 queue and download time from rendering timeout",
                    "checkpoint_step": step,
                    "source_commit": plan["source_commit"],
                    "target_commit": plan["target_commit"],
                    "budgets": plan["budgets"],
                    "archive": str(archive),
                }
            )
            atomic_json(root / "launch.json", manifest)
            result = recover.submit(
                root,
                "Apply verified H3 queue/render/download budgets from the run's own checkpoint; "
                "preserve original deadline",
                automatic=False,
            )
            plan.update(status="submitted", new_job=result["job_id"], submitted_at_epoch=time.time())
            disabled.unlink()
        except Exception as error:
            plan.update(status="error", error=f"{type(error).__name__}: {error}")
            # Keep automatic recovery disabled if this transaction is incomplete.
            raise
        finally:
            atomic_json(plan_path, plan)
        return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(deploy(args.run_dir)), flush=True)
