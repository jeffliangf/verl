Inspection RL V1-1
==================

``inspection_rl.py`` reuses a frozen video-only RL experiment. Preparation copies
its ordered train/evaluation rows and generator schema, adds the V1 inspector and
``finish_video``, and appends one paragraph to the original system prompt. It does
not change the user's goal, references, Gemini comparison rubric, or GRPO reward.

The environment permits three successful trajectories::

    sol_h3 -> finish_video
    sol_h3 -> inspect_video -> finish_video
    sol_h3 -> inspect_video -> sol_h3  (automatic termination)

Limits are checked before backend execution. Each trajectory has at most two
generation calls and one inspection; a second generation requires inspection.
Invalid policy actions retain the original tournament's invalid-call treatment.
Inspection provider failures finish the episode with the inspected video. The
tool returns ``status: failed`` and ``Inspection tool failed. The previous video
is final.`` with no available actions; no further policy turn or generation is
allowed. The existing video receives its normal final-video quality reward, so
the other seven trajectories are retained. Generation and policy-server failures
still use the existing complete-group replacement mechanism. Invalid actions,
corrupt local artifacts and programming errors retain their existing handling.
Backend retries are distinct from policy calls and retain the original inputs.

The inspection subprocess deadline is 120 seconds, configurable with
``inspection_rl.timeout_seconds``. This replaces the former 900-second default:
across 540 observed calls from the original and resumed first steps, the resumed
p95 was 37.6 seconds and maximum 49.7 seconds; only one original call exceeded
120 seconds (124.8 seconds). Timeout cleanup runs in a bounded background pool,
and a report already saved before a cleanup timeout remains usable. The separate
inspection quota wait precedes the subprocess deadline. Each fallback is saved
in the episode trace and reported as
``training/reward_extra/inspection/fallback/mean`` in TensorBoard.

H3 queueing has a separate 600-second budget. The existing 150-second H3 limit
applies to rendering, measured from the gateway's worker-start timestamp to its
finish timestamp. Queue and render time are accumulated across backend retries;
neither budget is reset by resuming a saved job. Artifact download and validation
have a separate 120-second deadline. A final subprocess guard is the sum of these
budgets plus 60 seconds for control operations. Both entry points inherit these
defaults from ``video_rl.h3_phase_timeouts`` in ``configs/rl.yaml``; its generation
value references ``video_rl.timeouts.h3``. The inspection preset has no separate
H3 limits. Set ``video_rl.h3_phase_timeouts=null`` to reproduce the legacy total
deadline. Both launch previews print the resolved budgets and process limit.

The generator keeps its configurable seed (default 42, valid range 0 through
9223372036854775807) and 5/10-second duration (default 10). ``inspect_video`` accepts
only a registered ``video_id``. The server supplies the immutable original goal
and the actual generation prompt to the unchanged V1 Gemini feedback backend.
The policy receives the complete feedback JSON plus environment counters and
available actions. Gemini receives no stopping budget and supplies no overall
verdict or stopping decision. ``finish_video`` is a terminal control action,
not a paid service.

The continuation allowance is 28,672 tokens, including observations, after a
4,096-token initial prompt allowance, with a 32,768-token model context. Each
policy turn is capped at 8,192 tokens and the remaining episode budget. Native
continuous token histories preserve generated reasoning and log probabilities.
Only policy-generated tokens have loss mask 1; tool observations and inserted
chat delimiters have loss mask 0. Oversized observations are never silently
truncated, and exhausted trajectories do not silently select an earlier video.

Rewards remain final-video quality only. Generation/inspection counts, elapsed
tool time, policy and observation token counts, revision rate, and stopping paths
are recorded separately. The cost instruction in the system prompt is therefore
not an explicit training penalty; learning a calibrated cost tradeoff is deferred.

Preparation and execution::

    uv run python -m examples.video_agents.inspection_rl prepare \
      --reference-run /path/to/original/grpo \
      --run-dir /path/to/new/grpo --data-dir /path/to/new/data

    uv run python -m examples.video_agents.inspection_rl train \
      --run-dir /path/to/new/grpo --data-dir /path/to/new/data --check-tokens

    # Add --execute to launch the reviewed configuration.

The prepared protocol freezes the inspector prompt, schema, implementation,
settings, source datasets, and evaluation references. Three-day allocation,
checkpoint storage, retention, monitoring, and recovery belong to the experiment's
launch files; they are not part of the policy's tool interface.

Compatibility with video-only RL
--------------------------------

The inspection implementation builds on commit ``74262494`` used by
``grpo-group8-16h100-20261031``. The generator schema, datasets, original system
prompt and judge implementation retain that baseline. Inspection uses its own
entry point, configuration overrides and loop class. H3 provider limits are
shared by both entry points; new launches now use separate phase budgets by
default, while explicit null selects the old total deadline.

Three shared files provide integration points. In ``agent.py``, the default
worker class is still ``TournamentWorker`` and the default reward metadata hook
returns an empty mapping. Episode metadata is written only for inspection
trajectories. In ``h3.py`` and ``tournament.py``, phase accounting is enabled only
when the runtime protocol supplies ``h3_phase_timeouts``. The launch settings
always override any frozen phase settings, including when disabling them.
Regression tests exercise both native loops, shared overrides, explicit legacy
mode, group-eight dispatch, reward publication and token masks. No inspection
action is registered by the video-only entry point.

For a pair of experiments, use clean checkouts pinned to the same commit and
record that commit in both launch manifests. Use each entry point's own run/data
directories and policy settings. Old launch scripts in ``inspection_rl_run/``
are historical snapshots, not the shared source of current timeout defaults.
Existing runs require a recorded resolved-config revision when adopting the new
provider limits; their frozen datasets and evaluation protocols stay unchanged.

The original running experiment uses a separate checkout at
``/home/jeff/verl-grpo-streaming-20261031`` pinned to ``74262494``. Updating this
branch does not update that checkout or its running processes. Historical launch
and monitoring sources for the stopped inspection experiment are preserved in
``inspection_rl_run/``; their embedded paths and allocation IDs describe that
specific experiment.
