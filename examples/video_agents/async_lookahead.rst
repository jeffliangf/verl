Complete-response lookahead
===========================

``rl train --async-lookahead`` enables one extra training batch while sharing
the existing policy/learner GPUs. It also moves Gemini file deletion off the
reward path. The default launch remains synchronous.

For a prepared group-16 run, retain its existing model, batch, GPU and provider
overrides and add the flag::

   python -m examples.video_agents.rl train \
     --run-dir /path/to/prepared/run --data-dir /path/to/prepared/data \
     --async-lookahead --execute -- \
     actor_rollout_ref.rollout.n=16 data.train_batch_size=32 \
     actor_rollout_ref.actor.ppo_mini_batch_size=32 \
     trainer.nnodes=2 trainer.n_gpus_per_node=8

Use a new prepared run when changing a frozen launch configuration. For an
existing lookahead run, restart with the same configuration and
``trainer.resume_mode=auto``. This change does not deploy or modify Sana workers.

Scheduling and the learning contract
-----------------------------------

The first update admits batch A, completes its policy responses, and admits
batch B while A's videos and comparisons finish. Before the optimizer starts,
B's policy responses also finish and are written durably. B's H3 and Gemini
work can continue while the shared GPUs compute log probabilities and update
the actor. After the weight update, batch C starts under the new weights while
B's remaining external work finishes. The final update admits no extra batch.

At most two batches of groups are admitted. A batch's members are fixed until
a provider failure exhausts recovery; only that failed group is replaced.
Each update consumes exactly the configured prompt count and every sibling
once. Faster groups do not displace slower groups. Existing invalid-call
penalties, pairwise scoring, fallback limits, and successful-sibling recovery
are retained.

The trainer checks the actual min/max policy versions reported by vLLM, not
the dispatch step. A response must come entirely from one weight version;
at consumption it may be zero or one optimizer update old. Every response
must include finite per-token behavior log probabilities. The launcher enables
native decoupled token importance correction (threshold defaults to 2.0),
with bypass mode disabled. It does not use partial-response abort/resume.

Inference and actor training still alternate on the shared GPUs. The overlap
is between external video/judge work and policy/learner computation. Waiting
for complete lookahead responses can add latency when policy generation is
the bottleneck. This is a bounded scheduling change, not a measured throughput
guarantee or a separate rollout GPU pool.

Multi-turn inspection episodes (``inspection_rl.py train --async-lookahead``) keep
this contract by caching every policy turn; see ``inspection_rl.rst``.

Checkpoint and crash recovery
-----------------------------

Native synchronous model/optimizer/data saves also persist TransferQueue and
``video-cohorts.json``. The latter records the unconsumed batch's membership
and actual behavior versions before the native save publishes its latest-step
pointer. A checkpoint callback verifies matching policy/cohort/queue steps.

``train/groups/<uid-hash>/policy/<sibling>/response.json`` holds the complete
tokens, behavior log probabilities, version and request checksum. On restart,
the trainer rebuilds only unconsumed trajectories from these caches and the
existing H3/pair ledgers. It clears partially published queue rows first.
Missing or changed policy caches fail explicitly; they cannot be regenerated
using a newer policy. A synchronous checkpoint with no pending queue can seed
the first lookahead batch.

Keep the run directory, frozen inputs, group artifacts and checkpoints together
on shared storage. Moving only the actor checkpoint does not preserve pending
lookahead work. Completed response caches remain in the run directory; retention
of old run artifacts is separate from model-checkpoint retention.

Gemini cleanup and admission
---------------------------

``video_rl.judge_background_cleanup=true`` can also be used with synchronous
streaming training. The lookahead flag enables it automatically. An independent
reaper deletes only files in the existing ownership journals, after the last
group reader exits. Reader and admission locks are released by the kernel on
process death. Streaming judges and local H3 clients also exit when their parent
worker dies. The remote Sana job stays recoverable through its saved identity.

The durable registry lives under ``provider-recovery/cleanup/active``; ready
cleanup tasks live under ``pending``. Active readers do not count against the
deletion backlog. Failed deletions remain queued across restarts. Starting or
closing another streaming judge starts a reaper if none owns the leader lock.
An idle reaper exits after 60 seconds. The queue can be drained manually with::

   python -m examples.video_agents.cleanup /path/to/run/provider-recovery/cleanup

Run-wide streaming limits default to 256 simultaneous comparisons, 16 uploads,
8 deletions, and 128 pending cleanup groups. Set them using
``video_rl.judge_limits.{comparisons,uploads,deletions,cleanup_backlog}``.
The existing per-group comparison limit still applies, capped at 32 waiting
comparison threads when shared limits are enabled. Uploads pause at the cleanup
backlog limit; deletion has independent slots and remains able to make progress.
The storage-quota recovery gate also applies to training streams.

These are run-wide concurrency limits and a group-count backlog limit. They
are not account-wide request-rate or upload-byte quotas; independent runs that
share a Gemini project still need an aggregate budget. Locks require shared
storage with working cross-node ``flock`` semantics. Verify that property
before a multi-node rollout.

Observability and validation
----------------------------

Track ``training/cohort/policy_lag_mean``, ``policy_lag_max``,
``replaced_groups``, and ``prefetched_groups``. Native timers include
``cohort_policy_drain`` and ``cohort_reward_wait``. ``timings.jsonl`` contains
group events and ``cohort_update`` boundaries; each group's existing LLM, H3,
comparison and upload journals remain available for overlap analysis.

The CPU suites cover full and incomplete groups, actual policy versions,
replacement, response replay, native TransferQueue persistence, callback
publication, independent validation dispatch, killed readers, and recovery of
provider slots after process death::

   python -m pytest tests/special_standalone/video_agents tests/trainer/ppo/v1 -q

Before increasing the production workload, validate GPU sleep/update/wake and
checkpoint restart with real providers. Compare warmed completed valid groups
per hour and held-out quality at the same allocation, batch and group size.
Device utilization alone is insufficient. DAPO filtering, custom samplers,
multi-update weight synchronization, router replay, and asynchronous actor
checkpoint writes are outside this implementation's supported contract.
