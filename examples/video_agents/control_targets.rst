Natural requests and control supervision
========================================

Implemented in the v2 prepared dataset
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
There is no additional benchmark aspect-ratio constraint.

.. code-block:: json

   {
     "id": "example-001",
     "task": "t2v",
     "prompt": "A fox crosses a snowy clearing.",
     "image": null,
     "request": "Create a 10-second video, adapting the following scene to fit that duration:\n\nA fox crosses a snowy clearing.",
     "target": {"duration_seconds": 10},
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

Training annotation: proposed next step
---------------------------------------

No Gemini annotation pass, synthetic training-control curriculum, or semantic
timeline/orientation extractor has been implemented or run yet.

An offline Gemini pass over unique training examples can provide two distinct
outputs: requirements supported by the original request, and suggested settings
for otherwise unspecified controls. Cache by prompt/image content, annotation
model, and annotation version. Include the source image for I2V when interpreting
its contents; its dimensions already give a deterministic shape preference.

Preserve the original scene text. Use short, varied prefixes/postfixes to construct
natural requests, for example "Make this about eight seconds long", "Use a portrait
format", or "Animate this image, keeping its original shape." Preserve existing
instructions, and do not manufacture requirements from vague language.

Keep required exact values in ``target``, advisory values in a proposed
``preferred_target``, and evidence/origin/confidence in ``control_metadata``.
If a Gemini suggestion is deliberately turned into a visible request for eight
seconds, eight becomes a requirement of that constructed request. If the request
stays generic, the suggestion remains advisory and never produces a matching
penalty. Annotation reasoning is not policy input.

A simple initial mixture could put explicit controls in approximately half the
otherwise unspecified samples and keep half free-choice. Include duration-only,
aspect-only, and combined requests, with ordinary phrasing as well as precise
numbers. All siblings in a GRPO group share the same final request. The exact
mixture is a proposal, not an applied training setting.

Cover supported nominal durations such as 5, 8, 10, 12, and 14 seconds, and the
six T2V aspect ratios. For I2V, prefer the source image's shape through ``auto``;
make preservation a hard requirement only when the visible request asks for it.
The worker stretches a first frame for a different explicit aspect ratio, so
shape-changing requests need to reflect that operation.

Interpret only what the user actually requests
----------------------------------------------

* "Make a video of ..." has no exact duration or aspect requirement.
* "Make it vertical" constrains orientation; both 3:4 and 9:16 can satisfy it.
  Do not secretly require 9:16.
* "Make it square" supports a 1:1 requirement.
* "A short clip" does not establish a unique numeric duration.
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

Future reports should distinguish applicable constraints, duration compliance,
aspect compliance, missing arguments, and quality. Compute compliance rates only
over applicable requirements. All-invalid sibling groups have equal penalties
and no relative control-learning signal; monitor their frequency before deciding
whether a supervised warm-up is needed.

Evaluation policy
-----------------

The fixed-duration benchmark contains 100 AA T2V, 100 MovieGen T2V, and 200 AA I2V
samples. All use explicit nominal 10-second requests and Gemini 4 Argon, with the
four saved references per sample. The 1,600 reference slots also contain 243 video
frames at 24 fps, or 10.125 seconds.

MovieGen duration suggestions may still be annotated for analysis or a future
variable-duration benchmark. They do not change this agreed 10-second evaluation.
Reference lengths establish the baselines' generation settings, not the original
prompts' intended duration. Any future benchmark with different requested controls
must disclose mismatches or regenerate references under those same controls.
