Natural requests and control supervision
========================================

Implemented in the v3 prepared dataset
-------------------------------------

New preparations store a natural ``request`` separately from the original
``prompt``. Only that request and any attached image enter the policy's user
message. The ordinary system message and tool definition remain available.
``task``, ``target``, and ``control_metadata`` are backend labels for validation,
rewards, and reporting; they are not appended to the user message.

Generic training requests say "Make a video of the following scene: ..." or
"Animate this image: ...". They have no duration/aspect matching requirement.
Documented tool defaults and worker capability validation still apply.

All 400 evaluation requests now explicitly ask for a nominal 10-second video,
including AA T2V, AA I2V, and MovieGen. The natural wording asks to adapt the scene
to that duration, so a pasted storyboard with a different timeline is deliberately
retimed for this benchmark. The original prompt remains unchanged for provenance.
The 200 T2V requests also explicitly ask for 16:9 landscape format, which this
worker resolves to 1344x768. I2V evaluation leaves aspect ratio unconstrained.

.. code-block:: json

   {
     "id": "example-001",
     "task": "t2v",
     "prompt": "A fox crosses a snowy clearing.",
     "image": null,
     "request": "Create a 10-second video in 16:9 landscape format, adapting the following scene to fit that duration and format:\n\nA fox crosses a snowy clearing.",
     "target": {"duration_seconds": 10, "aspect_ratio": "16:9"},
     "control_metadata": {
       "duration_seconds": {
         "source": "benchmark_request",
         "effective_num_frames": 243
       }
     }
   }

The policy must emit a requested control. The executor checks its arguments
instead of supplying the answer from a hidden target. For example, omitting
duration or choosing 8 seconds fails the evaluation request; 10 and 10.125 seconds
both satisfy it because the worker produces 243 frames at 24 fps. Exact aspect
requirements, when present, compare the resolved worker canvas rather than an
ideal mathematical ratio. Old frozen rows without a natural ``request`` field
retain their previous defaults and validation semantics.

The policy and Gemini quality judge receive the same natural request. Existing
quality scoring, GRPO, and invalid-call accounting remain unchanged. Evaluation
invalid calls receive zero wins. Generic training samples can choose any supported
duration/aspect ratio without a control-matching penalty.

Training annotation and allowed arguments
-----------------------------------------

``control_pilot.py`` runs Gemini 4 Argon on the original scene and, for I2V, the
actual first-frame image. It separates evidence-backed requirements from suggested
settings. The annotation policy is ``configs/control_annotation.txt``. Cache
identity includes the original text, image checksum, model, policy, response
schema, and generation settings. Raw provider responses and validated annotations
are retained; rebuilding natural requests reuses those annotations.

The completed 100-row pilot has 50 T2V and 50 I2V samples. Its deterministic
augmentation modes are 50 without added controls, 20 duration-only, 20 aspect-only,
and 10 combined. Existing requirements are retained in every mode, including
timeline endpoints. The result has 36 duration-constrained and 30 aspect-constrained
requests; three unsupported timelines are visibly retimed and flagged for review.
There were no failed annotations. No policy inference or videos were generated
by this annotation pass.

Run the pilot, or annotate all input rows, with:

.. code-block:: bash

   python -m examples.video_agents.control_pilot --execute

   python -m examples.video_agents.control_pilot --all \
     --output-dir data/prompt_rewrite_controls_8291 \
     --annotation-cache data/prompt_rewrite_control_annotations \
     --concurrency 32 --execute

Omit ``--execute`` to preview. A repeated command resumes the immutable manifest
and cached annotations. ``--all`` retains repeated prompt rows and their training
weight, while requesting only one annotation per unique prompt/image identity.
Copy the pilot's ``annotations`` directory into the shared cache to reuse its
responses before a full pass. The full pass does not change the frozen pilot.
Use ``--render-only`` to rebuild from existing annotations without provider calls;
missing annotations remain reported failures, retaining their original reasons.

Original scene text is preserved inside short natural prefixes/postfixes. Exact
requirements are stored in ``target``; advisory values in ``preferred_target``;
evidence, origins, confidence, and review flags in ``control_metadata``. The
``allowed_video_gen_args`` field additionally records every acceptable duration
frame count and aspect value/canvas, whether each argument must be emitted, the
required task and image condition, and checksums binding these labels to the
visible request and source image. It refines the shared tool schema, which still
validates prompt, seed, and worker capability limits.

For example, "Keep it about 5 seconds long. Use a vertical frame." has this
backend control rule:

.. code-block:: json

   {
     "duration_seconds": {"required": true, "allowed_num_frames": [124]},
     "aspect_ratio": {
       "required": true,
       "allowed_values": ["3:4", "9:16"],
       "allowed_canvases": [
         {"width": 768, "height": 1024},
         {"width": 768, "height": 1344}
       ]
     },
     "short_edge": 768
   }

An unspecified control has ``required: false`` and allows all supported values;
the validator rejects hidden restrictions on such a field. Suggestions become
requirements only when added to the visible request. Low-confidence annotations
are flagged and withheld from hard constraints. Recommendations cover nominal
5, 8, 10, 12, and 14 seconds and all six T2V ratios, without forcing uniform
coverage. I2V normally recommends ``auto``; preservation is required only when
the request says so. Different explicit ratios stretch the worker's first frame.

Each output directory contains:

* ``samples.jsonl``: original scenes, natural requests, images, allowed arguments,
  advisory targets, annotation provenance, and an example valid control choice.
* ``train.jsonl``: native verl inputs, with labels kept out of user messages.
* ``t2v.input.json`` and ``i2v.input.json``: real prepared messages, tools, attached
  image paths, and chat-template options for two examples.
* ``t2v.allowed-args.json`` and ``i2v.allowed-args.json``: their backend constraints.
* ``summary.json``, ``errors.json``, and ``manifest.json``: counts, failures, and
  frozen selection/provenance. ``complete`` must be true before using the full set.
* ``provider-review.jsonl``: failed rows with their original sample and failure
  reason, preserving coverage without inventing annotations. ``accounted_for``
  and ``processing_complete`` include these rows; ``complete`` counts only
  successfully annotated rows. A processed batch can therefore still need review.

The example controls are checked with the actual request builder; they are not
model-generated rewrites. ``prepare-mixed --train-files .../samples.jsonl`` carries
the hidden constraints into the frozen protocol used by the native executor.
Always use fresh run/data/combined paths when preparing annotated inputs.

.. code-block:: bash

   python -m examples.video_agents.rl prepare-mixed \
     --train-files data/prompt_rewrite_controls_8291/samples.jsonl \
     --run-dir data/prompt_rewrite_mixed_rl_controls_v1/run \
     --data-dir data/prompt_rewrite_mixed_rl_controls_v1 \
     --combined-file data/prompt_rewrite_mixed_rl_controls_v1.jsonl \
     --h3-endpoints http://127.0.0.1:28080

The pilot was also rendered with the cached Qwen3.8-27B tokenizer and vision
processor through verl's native multimodal path. All 100 inputs fit the configured
4,096-token limit (maximum 1,705 T2V and 2,085 I2V tokens). The pilot directory's
``*.rendered-input.txt`` files show the actual templates; ``native-input-validation.json``
records the checkpoint revision, processor, and counts. This renders inputs and
does not perform Qwen inference.

Interpret only what the user actually requests
----------------------------------------------

* "Make a video of ..." has no exact duration or aspect requirement.
* "Make it vertical" constrains orientation; both 3:4 and 9:16 can satisfy it.
  Do not secretly require 9:16.
* "Make it square" supports a 1:1 requirement.
* "A short clip" does not establish a unique numeric duration.
* "Hide the subject for two seconds" establishes a lower bound for a sub-action.
  When every supported worker duration satisfies that bound, omitting an overall
  duration argument is valid too; there is no duration matching penalty.
* A complete output timeline "[0-2s] ... [3-10s] ..." specifies a nominal
  10-second video. Use the final endpoint, not the sum of interval lengths:
  the gap from 2 to 3 seconds does not make it a 9-second clip.

Store the kind and evidence of coarse constraints separately instead of inventing
an exact target. An incomplete timeline may establish only a lower bound.
Timestamps into an existing source video, independently timed alternatives, and
historical dates are not automatically output duration. Ambiguous cases need
semantic annotation; uncertainty must not become an unjustified exact penalty.

Do not silently clamp explicit durations outside the worker's supported range.
Keep the original row and record a resolution: flag it for review or deliberately
construct a retimed request. The current training corpus includes long multi-shot
timelines as well as historical eras that a naive seconds regex would misread.
The fixed 10-second evaluation benchmark explicitly requests retiming; this does
not establish a general rule for the training set.

Reward for annotated training controls
--------------------------------------

Continue using the existing tournament quality reward Q:

.. code-block:: text

   reward = Q   for a valid call satisfying all visible requirements
   reward = -1  for an invalid call or a violated requirement

Unspecified controls automatically pass this gate. Inferred suggestions never
create a matching penalty. Worker failures remain execution failures rather than
evidence that the policy chose the wrong arguments.

For exact duration requests, compare effective frame counts. For exact aspect
requests, compare effective canvas dimensions, including source-image geometry
for ``auto``. For qualitative constraints, evaluate the actual meaning:
orientation for "vertical", rather than an arbitrary exact ratio. Gemini keeps
the quality rubric and evaluates the same natural request and source image.

The CPU integration tests pass annotated samples through verl's native tool parser,
executor, tournament scorer, and reward publisher with controlled external
generation/judging responses. Wrong or missing controls become ``invalid_call``
before generation and receive -1; valid free choices retain ordinary quality
scoring. These checks verify reward plumbing, not the quality of a learned policy.

Future reports should distinguish applicable constraints, duration compliance,
aspect compliance, missing arguments, and quality. Compute compliance rates only
over applicable requirements. All-invalid sibling groups have equal penalties
and no relative control-learning signal; monitor their frequency before deciding
whether a supervised warm-up is needed.

Evaluation policy
-----------------

The fixed-duration benchmark contains 100 AA T2V, 100 MovieGen T2V, and 200 AA I2V
samples. All use explicit nominal 10-second requests and Gemini 4 Argon, with the
four saved references per sample. T2V additionally requests 16:9; all 800 saved
T2V reference slots have a 1344x768 canvas. The 1,600 reference slots contain
243 video frames at 24 fps, or 10.125 seconds.

MovieGen duration suggestions may still be annotated for analysis or a future
variable-duration benchmark. They do not change this agreed 10-second evaluation.
Reference lengths establish the baselines' generation settings, not the original
prompts' intended duration. Any future benchmark with different requested controls
must disclose mismatches or regenerate references under those same controls.
