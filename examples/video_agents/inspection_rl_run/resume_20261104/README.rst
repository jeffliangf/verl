Inspection continuation configuration, 2026-11-04
================================================

This snapshot records Slurm job ``4142`` for
``inspection-v1-1-group8-16h100-20261101-resume-20261102``. It launched on
2026-11-04 using code ``4af40b373592161f169d75d6d32704651d6abdfe`` and restored
checkpoint 75, including model, optimizer, scheduler, RNG and dataloader state.
The first resumed training step is 76; initial evaluation is disabled.

The previous allocation failed during the step-80 old-log-probability pass.
This continuation enables the existing Torch output-head kernels::

    actor_rollout_ref.model.use_fused_kernels=true
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch

The kernels compute log probabilities and entropy in chunks. The configuration
keeps ``use_remove_padding=false``, batch 32, group 8, two nodes with eight H100s
each, and the 32,768-token context (4,096 prompt plus 28,672 continuation tokens).
It uses FSDP2, PPO microbatch size 1 per GPU, rollout tensor parallelism 4 and
rollout GPU memory utilization 0.45. H3 concurrency is 32; judge concurrency is
100; the inspector deadline is 120 seconds. The resolved configuration contains
the remaining model, optimizer, sampling, service and evaluation settings.

The checkpoint interval is five steps and the evaluation interval is 25 steps.
Retention preserves the latest two complete checkpoints and the best evaluated
checkpoint. ``run-budget.json`` preserves the existing 72-hour clock, including
recovery downtime: the deadline is 2026-11-05 18:49:13 UTC, with a two-hour
reserve for finishing and final evaluation.

Files and dependencies
----------------------

* ``common-overrides.json`` is the exact override list read by the deployed
  launcher for this inspection allocation.
* ``overrides.json`` is the complete argument list passed after ``--`` to
  ``inspection_rl.py train``. It includes the model cache, experiment name,
  checkpoint path, initial-evaluation choice and worker Python environment.
* ``training-config.json`` is the exact resolved configuration written by the
  running trainer, copied without rewriting its paths.
* ``run.json`` records code/model revisions, allocation settings, frozen-input
  hashes, source paths, snapshot hashes and validation provenance.
* ``run-budget.json`` is the exact allocation-budget record.

The absolute paths refer to the original host. Datasets, the frozen protocol and
tools, the model cache and checkpoint shards remain external inputs. The launch
also requires the recorded ``resume_callback.ResumeCheckpoint`` module and its
``run_budget.py`` dependency on ``PYTHONPATH``. That callback handles the existing
deadline and final evaluation; both source paths and hashes are in ``run.json``.
This snapshot does not install launchers or services. Repository-wide and
inspection-preset defaults are unchanged.

Preview and validation
----------------------

On the original host, the following prints a launch preview using the saved
arguments. It does not execute training::

    uv run python - <<'PY'
    import json
    from pathlib import Path
    import subprocess
    import sys

    snapshot = Path("examples/video_agents/inspection_rl_run/resume_20261104")
    run = json.loads((snapshot / "run.json").read_text())
    overrides = json.loads((snapshot / "overrides.json").read_text())
    subprocess.run([
        sys.executable, "-m", "examples.video_agents.inspection_rl", "train",
        "--run-dir", run["run_dir"], "--data-dir", run["data_dir"],
        "--", *overrides,
    ], check=True)
    PY

Before committing, composing the saved arguments reproduced the entire live
configuration after normalizing only the checkout-root path. The deployed
16-rank GPU preflight restored checkpoint 75 and passed log-probability/entropy
and PPO-update checks for the previously failing 19,242-token trajectory and a
32,768-token synthetic trajectory. Preflight optimizer updates were discarded;
the production trainer independently restored checkpoint 75.
