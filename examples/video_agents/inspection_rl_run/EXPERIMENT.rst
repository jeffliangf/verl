Inspection RL V1-1: experiment record
====================================

The authoritative current job ID, code commit, and file hashes are in
``grpo/launch.json``. The allocation budget began on 2026-11-01 at 21:26:21 UTC
and ends on 2026-11-04 at 21:26:21 UTC, including recovery downtime. The original
video-only experiment on nodes 001/006 is separate and must remain untouched.

The runtime checkout is ``/home/jeff/verl-inspection-v1-1-20261101``. The requested
entry point is also available in the main checkout at
``/home/jeff/verl/examples/video_agents/inspection_rl.py``. The runtime starts from
the original streaming/group-eight experiment commit, then adds the native
inspection loop. It starts from base Qwen3.8-27B revision
``1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0``, with a fresh optimizer/reference policy.

Configuration
-------------

* Nodes 004/005, eight H100s each; four rollout replicas with tensor parallelism 4.
* The same 2,046 training goals and ordered 200 evaluation goals as the reference.
* Batch 32, group 8: 256 policy trajectories per initial sampled training batch.
* Native actions: ``sol_h3``, ``inspect_video(video_id)``, and ``finish_video``.
* Hard limits: two generations, one inspection; generation two requires inspection
  and ends the trajectory automatically. Inspection is optional.
* Generator schema is unchanged, including configurable seed and 5/10-second duration.
* V1 feedback prompt/schema/settings are unchanged: Gemini 3.8 Flash, 4 FPS, high
  resolution, medium thinking. The RL adapter deletes its own temporary uploads.
* Original system prompt plus the agreed short cost/benefit and finish paragraph.
* 4,096 initial prompt tokens; 28,672 continuation tokens including observations;
  32,768 model context; at most 8,192 generated tokens per policy turn.
* Complete generated reasoning stays in context. Only policy output tokens receive
  policy loss; tool responses and inserted delimiters are masked out.
* Final-video quality reward only. Calls, time, and tokens are logged separately.
* H3 concurrency 20; inspection ceiling 500; comparison concurrency 8 per active group.
* Step-zero evaluation before any update; evaluation every 25 updates and at finish.
  Comparisons use the unchanged Context-IR, best-of-4 and best-of-16 references.
* Full checkpoints every five updates, keeping two verified saves. Budget-aware
  stopping reserves time for the final checkpoint/evaluation.

Storage and monitoring
----------------------

The ``grpo/episodes`` mount uses node 004's local
``/tmp/jeff-inspection-v1-1-20261101/episodes``; ``grpo/checkpoints`` uses node 005's
local ``/tmp/jeff-inspection-v1-1-20261101/checkpoints``. These paths are exposed
through SSHFS on both compute nodes and the controller. Startup checks the mount
identities; monitoring rejects missing mounts or exhausted reserves. Large media
and checkpoints therefore do not consume shared ``/home`` capacity.

The controller archive service asynchronously copies the latest verified full
checkpoint to ``/home/jeff/rl-benchmarks/inspection-v1-1-group8-16h100-20261101/checkpoint-backup``, retaining its
prior copy until the replacement verifies. Its status is ``grpo/checkpoint-backup.json``.
This shared-filesystem backup is independent of the training node's disk; it is not a policy tool.

The user services begin with ``video-agents-inspection-v1-1-16h100-``: ``watchdog``,
``check`` (30-minute timer), and ``archive``. Recovery is restricted to this run,
verified checkpoints, classified transient errors, and the original deadline.

Read ``grpo/watchdog-state.json`` for phase/health, ``grpo/val/step_0`` for initial
evaluation artifacts, ``grpo/metrics-<job>.jsonl`` for optimizer metrics, and
``grpo/launch.json`` for the current job's Slurm log. Group ``trajectories.json``
files include episode accounting; episode directories preserve both videos,
inspection reports, requests, native tokens, masks, and log probabilities.

TensorBoard is served by the existing ``video-agents-tensorboard.service`` on
controller port 8792 (bound to localhost). Its run selector includes
``inspection-v1-1-group8-16h100-20261101`` alongside the original experiment.
Forward port 8792 over SSH when viewing remotely. GPU utilization and memory
are recorded every 15 seconds under the two ``system`` sub-runs; evaluation
and optimizer metrics appear when their respective phases complete.

Validation before training included 260 inherited tournament/judge tests, 19
inspection-loop tests, 55 monitor/recovery tests, two backup tests, actual Qwen
tokenizer/mask verification, live inspection/upload cleanup, and cross-client
storage checks. The frozen legacy V1 source retains its existing style/license
hook exceptions; the new code passes lint/format and the remaining applicable hooks.

Step-zero evaluation
--------------------

The full 200-goal evaluation completed before training. There are 199 scored
examples (including four policy failures scored as zero) and one provider-blocked
inspection. The combined preference scores are 42.71% against Context-IR,
33.17% against best-of-4, and 18.84% against best-of-16; the three-reference mean
is 31.57%.

Of the 195 episodes returning valid final videos, all chose to inspect;
101 stopped after inspection and 94 generated a revision (48.21%). Inspection
was available as an optional action throughout. The scored episodes used 290
generation calls and 196 inspection calls; the excluded provider failure used
one additional generation and one attempted inspection.

On the 198 prompts scored in both experiments, the three-reference mean is
31.57% for this inspection loop and 30.30% for the original base policy. The
paired prompt-bootstrap 95% interval for the difference is -3.45 to +5.98
percentage points (10,000 resamples, seed 20261101). This does not establish a
clear zero-shot quality gain. It is the baseline for measuring later RL changes.
Complete accounting and the matched comparison are in ``grpo/step-zero-audit.json``.

H3 timeout correction (staged 2026-11-01)
----------------------------------------

The original H3 150-second subprocess deadline included gateway queueing,
rendering, and artifact transfer. Both experiments share a gateway; their
independent admission caps are 16 and 20. Four groups in training batch 1 were
replaced after H3 timeouts, although the gateway subsequently reported all four
videos successful. Gateway queue times were 34.77--61.58 seconds and actual
render times were 31.65--44.56 seconds. Client traces also showed local/transfer
delays; the losses cannot be attributed solely to gateway queueing.

Staged commit ``4098862d5f727ff4e223cc52fecc7757c9829ddf`` adds separate budgets:
600 seconds for the gateway queue, 150 seconds for rendering, and 120 seconds
for download/validation, plus a final 930-second subprocess guard. Queue and
render accounting uses gateway timestamps and persists across backend retries.
Local admission waiting already happens outside the H3 subprocess. The initial
30-minute queue proposal was reduced to 10 minutes after discussing the measured
load. The original video-only run retains its existing timeout behavior.

All 294 H3/native-inspection/tournament tests passed, including synthetic long
queues and timeout recovery, and Ruff checks passed. Six deployment tests verify
checkpoint gating, allocation identity, configuration preservation, and archive
retention. The deployment timer waits for the live H3 smoke result and this run's
first verified checkpoint at step 5 before stopping allocation 3573 and resuming
with the new code. The original 72-hour deadline and completed step-zero eval
are preserved. State is recorded in ``grpo/queue-timeout-deployment.json`` and
``queue-timeout-deployment.log``. This is a staged change until that state reports
``submitted`` and the resumed allocation is verified.

Training update 1 completed with finite loss and gradients (gradient norm
0.4314). All 32 accepted groups had eight valid trajectories and complete judge
comparisons with no fallback; 43.75% generated a revision after inspection.
