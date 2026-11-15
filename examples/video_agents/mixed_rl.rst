Unified T2V and I2V GRPO
=======================

The mixed preset trains one policy with one ``sol_h3`` tool. It uses the existing
synchronous V1 GRPO trainer and pairwise tournament rewards. Prompt groups are
shuffled by the native sampler; siblings always share one original request and,
for I2V, one source image.

Preparation
-----------

Run in the existing Python environment with the video-agent dependencies:

.. code-block:: bash

   python -m examples.video_agents.rl prepare-mixed \
     --run-dir data/prompt_rewrite_mixed_rl_v4/run \
     --h3-endpoints http://127.0.0.1:28080

The default inputs are:

* ``data/prompt_rewrite_i2v_diverse_v1_controls_train_2900.jsonl``: 2,900 I2V
  samples, the diverse I2V set without the 100 rows held out for evaluation.
* ``data/prompt_rewrite_rl.jsonl``: 2,046 T2V samples.
* ``data/prompt_rewrite_sft.jsonl``: 3,000 T2V samples. Only the original prompts
  enter RL; teacher rewrites are excluded.

Preparation writes ``data/prompt_rewrite_mixed_rl_v4.jsonl`` with all 7,946 normalized
samples, plus native ``train.jsonl`` and ``val.jsonl`` inside
``data/prompt_rewrite_mixed_rl_v4``. Each normalized sample has ``id``, ``task``
(``t2v``/``i2v``), original ``prompt``, natural ``request``, ``source``, ``image``
(null or a path/checksum/media record), ``target`` (required controls), and
source-file provenance. Only the natural request and attached image enter the
policy user message; task and control labels remain backend metadata.
The natural mixture is approximately 64% T2V and 36% I2V. All requested rows are
retained, including ten repeated T2V prompt occurrences under distinct IDs.

The command validates images, checks exact train/evaluation overlap by ID,
trimmed prompt, and image hash, and verifies the reference exports of every
evaluation sample. It makes no network requests. Output directories are immutable:
use fresh paths when changing data, tools, judge settings, or reference exports.
Use ``--train-files``, ``--eval-datasets``, ``--data-dir``, and ``--combined-file``
to prepare another collection. ``--group-size`` must match the launch override
``actor_rollout_ref.rollout.n`` when changing the existing default of four.

Tool contract
-------------

``configs/sol_h3_unified_tool.json`` is the model-visible v3 tool schema. The
original ``configs/sol_h3_tool.json`` remains available for legacy preparations;
old frozen runs keep their original request semantics.

.. list-table:: Original RL tool versus unified tool
   :header-rows: 1

   * - Capability
     - Original RL
     - Unified mixed RL
   * - Task
     - Text only
     - ``t2va`` or first-frame ``fl2va``
   * - Conditioning
     - None
     - One episode-bound ``input_image`` shown to the policy
   * - Duration
     - 5/10-second presets
     - Numeric duration validated against all supported frame counts
   * - Aspect ratio
     - Landscape
     - ``auto``, ``21:9``, ``16:9``, ``4:3``, ``1:1``, ``3:4``, ``9:16``
   * - Prompt format
     - Context-IR guidance
     - The same Context-IR guidance

The ``prompt`` description repeats the original tool's sentence word for word:
``integrated_multimodal_description`` (visuals over time), ``overall_soundscape``
(sounds), and ``non_diegetic_music`` (music) are suggested sections inside the one
prompt string, used when helpful; they are not separate arguments. Preparations
made before 2026-11-15 froze the earlier free-form wording ("any clear descriptive
format") in their rows and keep it.

The intermediate flat schema already supports ``image``, ``duration``, and
``aspect_ratio``. The unified schema organizes these as ``task``, ``conditions``,
and ``target`` and connects them to the full RL data/execution/judging path.

.. code-block:: json

   {
     "task": "fl2va",
     "prompt": "The woman turns and smiles. Quiet cafe ambience.",
     "conditions": [
       {"type": "image", "role": "keyframe", "frame_index": 0, "image_ref": "input_image"}
     ],
     "target": {"duration_seconds": 10, "aspect_ratio": "auto"},
     "seed": 42
   }

T2V uses ``task: t2va`` and empty conditions. The executor verifies the frozen
image checksum and replaces the handle with the worker's base64 ``uri``. An I2V
sample cannot discard or replace its supplied first frame. The policy observes
the image through verl's native multimodal input path; observation resizing is
capped at 768x768 pixels of area and EXIF rotation is applied before processing.
Generation and judging use the original image. Image files must remain readable
at their recorded paths on every rollout host.

Duration defaults to 10 seconds and aspect ratio defaults to 16:9 for T2V or
``auto`` for I2V. These defaults apply to free choices. For explicit sample
``target`` constraints, the policy must emit the corresponding arguments;
missing or conflicting controls are invalid. Effective frame/canvas equivalence
is accepted. The executor does not insert hidden ground-truth controls.
A null or omitted
seed resolves to 42; integer seeds range from 0 through 2^63-1.

See ``control_targets.rst`` for the implemented Gemini annotation pass, completed
100-row pilot, full-corpus command, and request-bound ``allowed_video_gen_args``.
The native executor validates these hidden constraints without adding labels to
policy inputs. Generic requests allow every supported duration and aspect choice.
Benchmark evaluation rows use explicit 10-second requests and additionally require
the 1344x768 canvas for T2V; held-out training-domain rows keep their own contract.

The deployed worker supports 124 + 17*k frames for k=0..13 at 24 fps, approximately
5.167 through 14.375 seconds. It rounds a requested duration to a supported frame
count: 10 seconds produces 243 frames (10.125 seconds). The outer API range of
4..15 seconds does not make every value valid; 4 and 15 fail frame-count checks.
The only resolution setting is ``short_edge: 768``, with a 1344x768 area cap and
dimensions rounded to multiples of 32. The actual short edge can therefore be
smaller, such as 672 for 21:9.

The service fixes four sampling steps. It does not accept guidance, negative
prompts, multiple keyframes, or last-frame conditioning. First-frame conditioning
through the FastH3 adapter remains experimental according to Sana's API contract.

Rewards and evaluation
----------------------

Training uses the ``--judge-model`` reward judge (default ``gemini-3.8-flash``)
with the existing v4 preference rubric; only the model name can change. I2V
adds the required source image before both videos; T2V requests are unchanged.
The evaluator uses the existing image-input helper and shares image uploads across
concurrent comparisons. Image hashes enter manifest signatures, and the frozen
judge model remains part of result identity.

Both presentation orders are judged. Their scores are averaged and then averaged
over sibling opponents. GRPO advantages, invalid-call penalties, and the existing
training fallback policy are unchanged. Evaluation requires real judgments.

Evaluation uses the ``--eval-judge-model`` judge (default ``gemini-4-argon``) for
both:

* T2V: ``data/prompt_rewrite_aa_moviegen_eval_200.jsonl`` (200 samples).
* I2V: ``data/i2v_eval_batch2_diverse_200.jsonl`` (200 samples): the 100 AA batch2
  rows of ``data/aa_i2v_eval_200.jsonl`` and 100 rows held out of the diverse I2V
  training set (source ``i2v_diverse_heldout``; 25 short and 25 long eval-format
  rows, 50 broad rows).

The 200 T2V and 100 batch2 requests explicitly ask for a nominal 10-second video,
including MovieGen. The wording asks to adapt any storyboard timing to this duration.
The backend requires the emitted duration to resolve to 243 frames at 24 fps
(10.125 seconds). T2V requests also explicitly require 16:9 landscape format,
which this worker resolves to 1344x768; I2V has no additional benchmark aspect
constraint.

The held-out training-domain rows keep their own natural request and control
contract (``allowed_video_gen_args``), exactly as in training: exact durations of
5-14 seconds, ranges, minimums, qualitative or unspecified durations, and
requested aspect wording. A call that violates the contract is invalid and scores 0.
In general, an evaluation row that has ``allowed_video_gen_args`` is evaluated as
written; every other row receives the 10-second benchmark request. The quality
judge receives the same natural request as the policy.

Each candidate is compared against these exact columns:

* ``best_of_4_h3_self_hosted_video``
* ``best_of_16_h3_self_hosted_video``
* ``best_of_4_h3_50step`` (T2V only)
* ``best_of_16_h3_50step`` (T2V only)

The 50-step columns are used only when an evaluation row has them, and every
sample of one task must have the same columns; preparation records them in
``reference_fields_by_task``. The held-out I2V rows have no 50-step regenerations,
so both I2V halves use the two 4-step references (the batch2 50-step videos remain
in ``data/aa_i2v_eval_200.jsonl``). Their references follow the AA I2V recipe:
16 image-conditioned Context-IR expansions rendered by Sol-H3 at four steps and
picked by a Gemini 3.8 Flash round robin. They start from the row's natural
request and render at its preferred duration with ``auto`` aspect.

The saved 50-step videos regenerate the selected self-hosted winners with their
recorded prompts/seeds; they are not a new selection over N fresh 50-step clips.
Preparation checks that each regeneration belongs to its selected source, verifies
its own recipe and media, and preserves that distinction in provenance.
One batch2 best-of-16 export has 15 available candidates because repeat 16
was explicitly cancelled. The saved video remains in the requested comparison;
the exception is included in preparation and evaluation reports.

A complete evaluation generates 400 policy clips and makes 2,400 Gemini requests
(200 T2V prompts x four references and 200 I2V prompts x two references, each in
two orders), excluding retries. It reuses the
saved baseline videos but obtains fresh judgments from the evaluation judge.

Results under ``val/step_N`` include ``eval-summary.json``,
``eval-summary-t2v.json``, and ``eval-summary-i2v.json``. They report each task's
reference win rates, invalid calls, dropped prompts, judge model, and baseline
exceptions. Task reports also break these down by source (``by_source``), such as
``aa-i2v-batch2`` and ``i2v_diverse_heldout``. ``train/step_N/task-summary.json`` reports reward and judge coverage
separately for each task without changing the objective.

Launch preview and checks
-------------------------

.. code-block:: bash

   python -m examples.video_agents.rl train \
     --run-dir data/prompt_rewrite_mixed_rl_v4/run \
     --data-dir data/prompt_rewrite_mixed_rl_v4

   python -m examples.video_agents.rl status \
     --run-dir data/prompt_rewrite_mixed_rl_v4/run

   python -m examples.video_agents.rl evaluate \
     --run-dir data/prompt_rewrite_mixed_rl_v4/run \
     --data-dir data/prompt_rewrite_mixed_rl_v4 \
     --output-dir data/prompt_rewrite_mixed_rl_v4/evaluation-checkpoint \
     --model /path/to/exported-vision-policy

Train/evaluate commands preview configuration and cost until ``--execute`` is
provided. ``--check-tokens`` checks the actual model processor, including image
tokens and tool definitions. Execution always runs that check and rejects a
text-only checkpoint for mixed data. Model overrides still use the existing
Hydra arguments, e.g. ``-- actor_rollout_ref.model.path=/path/to/vision-policy``.

Mixed runs require one stable Sol-H3 gateway with the full v2 contract on every
ready worker. For the documented localhost endpoint, maintain the SSH tunnel on
each Ray worker host. Supply credentials through ``H3_SELF_HOSTED_TOKEN_FILE``
or preparation's ``--h3-token-file``; credentials are never embedded in datasets.
Confirm connectivity and run a small live multimodal canary before a full launch.
