Group-16 GRPO run configuration
===============================

This directory records the configuration of Slurm job 4138,
``grpo-group16-16h100-20261104``. It started fresh from
``Qwen/Qwen3.8-27B`` revision
``1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`` using verl commit
``4af40b373592161f169d75d6d32704651d6abdfe``. Checkpoint resume was disabled.

The user cancelled the job during step 6 at 20:24:53 UTC on November 4, 2026.
Its allocation was released and its monitor stopped. Checkpoint 5 completed
at 20:05:26 UTC, with 49 required archives verified (305.8 GiB). Checkpoint
tensors remain in the original run directory; the verification record here
checks archive structure, not tensor CRCs or a GPU restore.

Settings
--------

* Two nodes with eight H100 GPUs each; rollout tensor parallelism 4.
* Group size 16 and 32 prompts per batch: 512 trajectories and up to 7,680
  ordered pairwise judgments per training batch, before retries or refills.
* PPO minibatch 32 prompts, microbatch 1 per GPU, and one PPO epoch. The trainer
  expands the configured minibatch by the rollout group size.
* H3 maximum inflight 128, with eight agent workers and 16 H3 slots per worker.
  Each worker admits four prompt groups.
* Judge concurrency 500 per group, with streaming judging enabled.
* Learning rate ``1e-6``, KL coefficient ``0.01``, and the inherited judge
  fallback-fraction limit ``0.75``.
* Save every five updates and evaluate every 25, including initial and final
  evaluation. Retain the latest two complete checkpoints and the best evaluated
  checkpoint.
* A new 72-hour allocation budget, starting November 4 at 18:08:57 UTC and
  ending November 7 at 18:08:57 UTC, with the same two-hour finish reserve.
  The user cancellation ended this run before that deadline.

Other trainer settings match the group-8 resume run. ``config-validation.json``
records the exact differences, including the new paths and disabled resume.

Files
-----

* ``grpo/training-config.json`` is the complete resolved configuration actually
  used by the trainer. ``common-overrides.json`` records the shared overrides.
* ``grpo/protocol-settings.json`` contains every frozen protocol field except
  the ``samples`` payload. It retains generator and judge revisions, the rubric,
  worker concurrency, dataset paths and hashes, and evaluator dependency hashes.
  ``grpo/protocol-sha256.json`` identifies the original full protocol, not this
  reduced settings file.
* ``grpo/launch.json``, ``submission.json`` and ``run-budget.json`` preserve the
  model initialization, exact submission command, allocation and budget.
  ``runtime-settings.json`` extracts scheduler, environment, Ray and watchdog
  settings from the deployment files. Shell export values remain unexpanded.
* ``grpo/user-stop-request.json`` and ``grpo/checkpoints/`` record cancellation
  and the last completed save. They contain metadata only.
* ``systemd/`` preserves the run-specific monitor definition; ``tensorboard.json``
  identifies this run and both GPU telemetry streams in the existing dashboard.
* ``snapshot.json`` records original paths and SHA256 values. Files marked
  ``byte-for-byte`` are exact copies; derived files describe their transformation.
  The configuration and startup validation reports preserve checks from launch.

The original directory is
``/home/jeff/rl-benchmarks/grpo-group16-16h100-20261104``. Datasets, videos,
checkpoint tensors, credentials and event logs remain external runtime assets,
as in the main video-agent example. Launcher helper sources remain there too;
their hashes are included for provenance. No helper code or service is activated
by this snapshot.

These historical machine paths, stopped-state records and budget are not a
ready-to-submit experiment. A future run needs a new directory, a full protocol
prepared from the external datasets, and an appropriate launcher and allocation.
