Mixed T2V and I2V group-16 launch
===============================

``python -m examples.video_agents.mixed_group16`` prepares a new run from the
approved controls-v2 dataset and the existing two-node group-16 launcher.
It requires a clean runtime checkout and a fresh output directory. It does not
submit a Slurm job. Pass ``--source-run``, ``--previous-root``, ``--output-root``,
``--ssh-host``, ``--ssh-port``, and ``--gateway-target`` after confirming the
deployment's SSH endpoint and private gateway address. Credential files remain
outside the checkout and the launch manifest.

The preparation preserves all 8,288 training requests, hidden argument checks,
and 400 evaluation requests. Native JSONL files are copied byte for byte into
the new run so token checks write previews without changing the approved data
directory. The source protocol and dataset hashes are recorded in the manifest.
Evaluation retains 10-second requests, four reference sets, and Gemini 4 Argon;
training retains Gemini 3.8 Flash and the source image for I2V comparisons.

The previous Slurm, Ray, NCCL, model staging, tunnel supervision, watchdog,
checkpoint verification, retention, and allocation-budget helpers are reused.
``infrastructure-provenance.json`` records each original and prepared file hash.
Changes to those helpers are limited to run and checkout paths, the deployed
gateway connection, checking both judge models, requiring 32 compatible H3
workers, and deriving evaluation size from the frozen protocol.

Training keeps 32 prompts with 16 rollouts, eight agent workers, four prompt
groups per worker, and 16 H3 slots per worker. The shared H3 admission limit
defaults to 32 for the newly validated fleet; ``--h3-max-inflight`` sets this
limit across both policy nodes. The previous run used 128 shared slots. The judge
concurrency is 500 and the maximum fallback fraction is 0.25. The model can
observe one source image. The optimizer, offloading, tensor parallelism,
checkpoint cadence, and 72-hour budget with a two-hour finish reserve follow
the previous experiment. Selection of the best mixed checkpoint averages the
T2V and I2V scores with equal weight.

Before submission, verify the composed configuration and full-dataset token
limits, the deployed gateway's complete API-v2 contract, and all 32 workers.
The reused batch script also checks Gemini access, 16-rank NCCL communication,
the H3 gateway from each policy node, and a two-node Ray cluster with 16 GPUs.
The run's allocation budget starts when Slurm allocates the nodes.

Mixed JSONL files use ``VideoRLDataset``, which assigns an explicit type to
``images`` even when the first JSON chunks contain only T2V rows. This preserves
the existing native dataset behavior and all frozen records.

Run ``python -m examples.video_agents.data_preflight --run-dir RUN`` with the
production training Python environment on a policy node, before allocating GPUs.
It loads the complete train and evaluation datasets through the actual trainer
loader, checks every record and image binding, exercises mixed collation, and
checks all prompt lengths. The resulting report is tied to the runtime commit,
configuration, protocol, datasets, and Python environment.

Submit with ``python -m examples.video_agents.submit_mixed --run-dir RUN`` after
the gateway tunnel is available. Missing, partial, or stale preflight reports
prevent submission. For a recovery attempt, pass ``--budget-from`` pointing to
the original run's ``run-budget.json``; the original deadline is retained.
