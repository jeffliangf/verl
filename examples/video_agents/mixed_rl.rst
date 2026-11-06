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
     --run-dir data/prompt_rewrite_mixed_rl/run \
     --h3-endpoints http://127.0.0.1:28080

The default inputs are:

* ``data/prompt_rewrite_i2v_rl_dedup.jsonl``: 3,245 I2V samples.
* ``data/prompt_rewrite_rl.jsonl``: 2,046 T2V samples.
* ``data/prompt_rewrite_sft.jsonl``: 3,000 T2V samples. Only the original prompts
  enter RL; teacher rewrites are excluded.

Preparation writes ``data/prompt_rewrite_mixed_rl.jsonl`` with all 8,291 normalized
samples, plus native ``train.jsonl`` and ``val.jsonl`` inside
``data/prompt_rewrite_mixed_rl``. Each normalized sample has ``id``, ``task``
(``t2v``/``i2v``), ``prompt``, ``source``, ``image`` (null or a path/checksum/media
record), ``target`` (explicit user constraints), and source-file provenance.
The natural mixture is approximately 61% T2V and 39% I2V. All requested rows are
retained, including ten repeated T2V prompt occurrences under distinct IDs.

The command validates images, checks exact train/evaluation overlap by ID,
trimmed prompt, and image hash, and verifies the four reference exports per
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
     - General visual, motion, camera, style, speech, sound, and music description

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
``auto`` for I2V. These are defaults, not forced overrides. Explicit sample
``target`` constraints take precedence over policy choices. A null or omitted
seed resolves to 42; integer seeds range from 0 through 2^63-1.

See ``control_targets.rst`` for the proposed follow-up on extracting explicit
controls, annotating suggestions, mixing fixed and free choices, and measuring
instruction compliance. Those annotations and benchmark control assignments are
not applied to this prepared dataset.

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

Training keeps ``gemini-3.8-flash`` and the existing v4 preference rubric. I2V
adds the required source image before both videos; T2V requests are unchanged.
The evaluator uses the existing image-input helper and shares image uploads across
concurrent comparisons. Image hashes enter manifest signatures, and the frozen
judge model remains part of result identity.

Both presentation orders are judged. Their scores are averaged and then averaged
over sibling opponents. GRPO advantages, invalid-call penalties, and the existing
training fallback policy are unchanged. Evaluation requires real judgments.

Evaluation uses ``gemini-4-argon`` for both:

* T2V: ``data/prompt_rewrite_aa_moviegen_eval_200.jsonl`` (200 samples).
* I2V: ``data/aa_i2v_eval_200.jsonl`` (200 samples).

Each candidate is compared against these exact columns:

* ``best_of_4_h3_self_hosted_video``
* ``best_of_16_h3_self_hosted_video``
* ``best_of_4_h3_50step``
* ``best_of_16_h3_50step``

The saved 50-step videos regenerate the selected self-hosted winners with their
recorded prompts/seeds; they are not a new selection over N fresh 50-step clips.
Preparation checks that each regeneration belongs to its selected source, verifies
its own recipe and media, and preserves that distinction in provenance.
One existing I2V best-of-16 export has 15 available candidates because repeat 16
was explicitly cancelled. The saved video remains in the requested comparison;
the exception is included in preparation and evaluation reports.

A complete evaluation generates 400 policy clips and makes 3,200 Gemini requests
(400 prompts x four references x two orders), excluding retries. It reuses the
saved baseline videos but obtains fresh Argon judgments.

Results under ``val/step_N`` include ``eval-summary.json``,
``eval-summary-t2v.json``, and ``eval-summary-i2v.json``. They report all eight
task/reference win rates, invalid calls, dropped prompts, judge model, and baseline
exceptions. ``train/step_N/task-summary.json`` reports reward and judge coverage
separately for each task without changing the objective.

Launch preview and checks
-------------------------

.. code-block:: bash

   python -m examples.video_agents.rl train \
     --run-dir data/prompt_rewrite_mixed_rl/run \
     --data-dir data/prompt_rewrite_mixed_rl

   python -m examples.video_agents.rl status \
     --run-dir data/prompt_rewrite_mixed_rl/run

   python -m examples.video_agents.rl evaluate \
     --run-dir data/prompt_rewrite_mixed_rl/run \
     --data-dir data/prompt_rewrite_mixed_rl \
     --output-dir data/prompt_rewrite_mixed_rl/evaluation-checkpoint \
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
