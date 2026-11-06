Duration and aspect-ratio supervision: proposed next step
=======================================================

This is a design for extending the implemented mixed-run format. The current
implementation accepts explicit ``target`` constraints, binds source images, and
uses a flexible generation instruction. It does not yet infer targets, synthesize
control requests, or assign fixed evaluation durations. Those annotation and
reward changes should be introduced as a new prepared experiment.

Separate requirements from suggestions
-------------------------------------

A prompt can require a duration/shape, suggest one, or leave the decision to the
policy. An inferred suitable duration is not ground truth about the user's intent.
Keep required values in the existing ``target`` object. Add ``preferred_target``
for advisory annotations and ``control_metadata`` for their origin and evidence.
An absent required field means the policy is free to choose any supported value.

.. code-block:: json

   {
     "id": "example-001",
     "task": "t2v",
     "prompt": "A red fox crosses a snowy clearing.",
     "image": null,
     "target": {
       "duration_seconds": 8,
       "aspect_ratio": "9:16"
     },
     "preferred_target": {},
     "control_metadata": {
       "duration_seconds": {
         "source": "synthetic_request",
         "effective_num_frames": 192
       },
       "aspect_ratio": {
         "source": "synthetic_request"
       },
       "annotation_version": 1
     }
   }

Render the policy's user message deterministically from the required fields:
"Generate an approximately 8-second video in 9:16 portrait format of: ...".
For I2V, include the source image and its ``input_image`` handle. With no required
fields, render "Generate a video of: ... Choose a suitable duration and aspect
ratio." Required settings must also be visible in the original request supplied
to the judge. Teacher rewrites and annotation reasoning stay out of the policy
observation.

An inferred-only example would have ``target: {}``,
``preferred_target: {"duration_seconds": 8}``, and metadata recording
``source: gemini_inference``, the model ID, evidence, confidence, and annotation
version. It remains a free-choice task. Suggestions may help choose plausible
synthetic requests and support analysis; they do not create a hard reward penalty.

How to assign controls
----------------------

1. Extract explicit requests and timelines first. Preserve direct duration and
   aspect-ratio instructions, recording the matching text as evidence. Distinguish
   clip duration from action timestamps, object counts, frame rates, and historical
   eras such as "1980s". A list of timestamps needs a timeline interpretation,
   not a naive numeric maximum.
2. Use a cached Gemini annotation pass only for ambiguous requests or optional
   suggestions. Require structured output separating ``explicit``, ``inferred``,
   ``unspecified``, and ``unsupported``. Check evidence against the original text
   and validate every proposed value against the worker. The annotation cache
   should include prompt/image hashes, model ID, and annotation version. No Gemini
   call is needed during RL to determine the requested controls.
3. For otherwise unspecified training samples, start with a simple seeded 50/50
   split between synthetic explicit requests and free-choice requests. Keep each
   sample's assignment fixed in the prepared dataset. This changes the rendered
   request, not the GRPO sampler or objective.
4. Cover nominal durations 5, 8, 10, 12, and 14 seconds in the synthetic half, and
   all six explicit T2V aspect ratios. Preserve I2V's source-image shape with
   ``auto`` initially; use a different I2V ratio when a shape change is explicitly
   requested. The worker stretches a first frame for a different explicit ratio,
   so automatic random shape changes should not masquerade as image-preserving
   animation requests.

Never overwrite an explicit original constraint with a sampled or inferred one.
Explicit durations outside the worker's range need a recorded resolution. Keep
such rows in the master corpus and flag them for review; either omit them from the
first strict-control training view or prepare clearly labeled retimed requests,
such as "Adapt this storyboard into a 10-second clip, preserving event order."
Silently clamping an original 30-second requirement to 10 seconds would produce
contradictory supervision.

The current training files contain multi-shot prompts with timelines extending
beyond this worker's maximum duration. They also contain era references that a
simple seconds regex would misread. Explicit extraction and inference therefore
need separate provenance, even if both are produced by one annotation pass.

Reward signal
-------------

Keep the existing Gemini quality preference and GRPO computation. Treat required
duration/aspect settings as part of tool-call correctness, using the existing
invalid-call penalty rather than introducing a weighted reward sum.

For training, let Q be the existing group tournament reward:

.. code-block:: text

   required_controls_ok = duration_ok AND aspect_ratio_ok
   reward = Q       if the tool call is valid and required_controls_ok
   reward = -1      for an invalid or noncompliant policy call

When a control is unspecified, its correctness flag is automatically true.
Preferred/inferred settings never produce the -1 penalty. Existing neutral
fallback accounting, provider-failure handling, and GRPO advantages stay intact.
Noncompliant calls can be rejected before H3 generation, saving generation and
judging work. In evaluation, retain the existing invalid-call behavior of zero
wins and report compliance separately.

Compare effective worker settings rather than raw floats:

* Duration: compare ``frame_count(actual_seconds)`` with
  ``frame_count(required_seconds)``. Both 10 and 10.125 produce 243 frames and
  should satisfy a nominal 10-second request. The current implementation checks
  explicit target values literally; this equivalence check is a proposed refinement.
* Aspect ratio: resolve the requested and actual settings through the worker's
  canvas calculation, including the source image for ``auto``. For example,
  16:9 maps to 1344x768 under this deployment. Do not penalize that output for
  failing an ideal mathematical 16:9 pixel ratio.

Record ``duration_ok``, ``aspect_ratio_ok``, requested/effective frames, requested/
effective canvas, and ``valid_call`` as diagnostics. Keep Q separately available
in reports so a quality improvement can be distinguished from better instruction
following. A worker returning the wrong geometry after a compliant call is an
execution failure, not evidence that the policy chose the wrong tool arguments.

Evaluation policy and observed evidence
--------------------------------------

The current evaluation files contain:

* 100 ``aa_eval`` and 100 ``movie_gen_video_bench`` T2V prompts.
* 100 ``aa-i2v`` and 100 ``aa-i2v-batch2`` I2V prompts.

All 1,600 saved reference slots contain 243 video frames at 24 fps, or 10.125
seconds. The MovieGen prompts have no numeric seconds expressions in the local
audit; absence of explicit timing does not establish an intended shorter duration.

For the requested comparisons with the existing four baselines, set a required
nominal duration of 10 seconds on all AA samples. Also use 10 seconds on MovieGen
for this fixed-baseline benchmark, recording
``source: benchmark_reference_profile`` rather than inventing a user-specified
duration. This makes the main comparison consistent with the saved reference
videos. Preserve explicit aspect requirements; otherwise record the benchmark
shape policy explicitly (16:9 T2V, source-image ``auto`` I2V), and disclose any
shape difference from the saved landscape reference exports.

Gemini-inferred MovieGen durations can be retained as suggestions and evaluated
in a separate variable-duration/control suite. To claim a controlled comparison
against the same baselines at those new durations, regenerate the reference
videos under the same requested settings and freeze that as a separate benchmark.
Keep those results separate from the existing 200-prompt T2V comparison.

Use Gemini 4 Argon for both evaluation tasks as already requested. Report quality
win rates against each of the four references together with duration compliance,
aspect compliance, invalid calls, and service/judge coverage. This adds a precise
instruction-following signal while leaving the Gemini quality rubric unchanged.
