Inspection RL experiment snapshot
================================

This directory preserves the launch, checkpoint, monitoring, recovery and
diagnostic sources used for ``inspection-v1-1-group8-16h100-20261101``. It is a
historical snapshot with machine-specific paths, not a new active allocation.
``snapshot.json`` records each original path and SHA256. The sources and frozen
configuration are copied byte-for-byte, including their original formatting.

The user stopped Slurm job 3573 on 2026-11-01 and paused its goal. Automatic
recovery, checkpoint archiving and timeout deployment services were disabled.
Step-zero evaluation completed on all 200 prompts (199 scored, one provider
failure), and one optimizer update completed. No training checkpoint was saved
before cancellation. Logs, generated videos and datasets remain in the original
experiment directory, whose location is recorded in ``snapshot.json``.

The running snapshot used code ``366e2af0``. The phase-timeout fix ``4098862d``
was staged and tested but was not deployed. Its live H3 smoke test was cancelled
while waiting for local admission; it did not validate the staged worker against
a fresh real generation. ``EXPERIMENT.rst`` records the historical configuration
and findings; its pre-cancellation deployment plan is superseded by this stop
record and ``grpo/user-stop-request.json``.

Directory contents
------------------

* ``grpo/train.sh``, ``train.sbatch`` and ``cluster-env.sh``: two-node launch and
  exact training overrides, including batch 32, group 8 and context 32,768.
* ``grpo/run_callback.py`` and ``run_budget.py``: verified checkpoint recovery,
  retention and the original three-day deadline.
* ``watchdog.py``, ``check30.py``, ``recover.py`` and ``archive_checkpoint.py``:
  monitoring, bounded recovery and backup of this experiment's checkpoints.
* ``grpo/storage.py`` and storage probes: the SSHFS/node-local storage setup.
* ``deploy_queue_timeouts.py`` and its test: the now-disabled checkpoint-gated
  timeout deployment procedure.
* ``systemd/``: inspection-specific service and timer definitions. Copying these
  files into this repository does not install or enable them.

The original ``grpo-group8-16h100-20261031`` experiment is separate. None of these
archived scripts are imported by its ``rl.py`` or by the inspection policy loop.
The shared TensorBoard service is documented in ``EXPERIMENT.rst``; its definition
also serves other runs and is not part of this inspection-specific snapshot.

To prepare a future experiment, use the repository's ``inspection_rl.py`` with
new run/data directories and a reviewed launcher. These archived launchers refer
to the cancelled allocation's checkout, mounts, paths, hashes and budget.

The isolated operational tests can be run without starting jobs::

    uv run python -m pytest examples/video_agents/inspection_rl_run/test_watchdog.py \
      examples/video_agents/inspection_rl_run/test_recover.py \
      examples/video_agents/inspection_rl_run/test_archive_checkpoint.py \
      examples/video_agents/inspection_rl_run/test_deploy_queue_timeouts.py -q
