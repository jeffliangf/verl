Qwen tool-call audit without video generation
===========================================

``tool_call_audit.py`` runs one Qwen completion per prepared training and
evaluation observation, extracts calls with verl's ``qwen3_coder`` parser,
and checks the arguments with the same ``tool_request`` used by the RL
executor. It does not instantiate the H3 tool, contact a video worker, or
call a Gemini judge. This measures tool use and argument compliance;
it cannot measure generated video quality.

Run inside an allocated GPU environment with the repository's vLLM extra::

    uv run --no-sync python -m examples.video_agents.tool_call_audit \
      --protocol data/prompt_rewrite_mixed_rl_controls_v2/run/protocol.json \
      --model /path/to/Qwen3.8-27B/snapshot \
      --output-dir data/qwen_tool_call_audit \
      --tensor-parallel-size 4 --batch-size 32

The default is greedy decoding, one completion, seed 42, and an 8,192-token
response limit. Each row supplies its unchanged system message, natural user
request, tool schema, and chat-template options. I2V includes the actual first
frame, with the native EXIF handling and observation resizing. Hidden control
labels are used only after generation. The runner verifies that vLLM's prompt
token IDs exactly match the native processor's output. A length-limited response
is retained and parsed, as it is in the native RL loop.

This is a greedy base-model diagnostic, rather than the four sampled
sibling rollouts used during GRPO. The held-out set is processed first, and
T2V/I2V observations are interleaved. There are no automatic retries of policy
failures or repairs to model arguments. Completed responses are saved immediately,
and the queue is refilled while longer responses continue. ``--batch-size``
sets the maximum active sequences; at most twice that many inputs are queued.
Infrastructure or input-integrity
failures stop the run instead of being counted as policy mistakes.

Metrics and denominators
------------------------

``summary.json`` reports the following for the entire run and each split,
task, and source:

* ``parsed_sol_h3_call``: exactly one call named ``sol_h3`` was extracted.
* ``schema_valid``: its raw arguments satisfy the complete JSON Schema.
* ``task_correct`` and ``first_frame_correct``: the inferred task and supplied
  image handle match the observation.
* ``duration_supported`` and ``aspect_supported``: emitted values or permitted
  defaults are supported by the worker.
* ``requested_duration_correct`` and ``requested_aspect_correct``: explicitly
  supplied values match the request's saved allowed arguments. Only requests
  with the corresponding constraint enter these denominators. A missing call
  counts as a failure on constrained requests; unconstrained requests are
  reported as not applicable. Duration matching uses the worker's real frame
  quantization. Existing evaluation canvas-equivalence rules are preserved.
* ``worker_compatible``: schema, task, image binding, and worker settings pass
  without checking desired controls.
* ``all_args_valid``: the unchanged native validator accepts the complete call
  against the actual sample, including its instruction constraints.

The main rates use all applicable inputs, including failures to invoke a tool.
``among_parsed_calls`` also supplies conditional argument accuracy among inputs
that successfully produced one ``sol_h3`` call. Each rate includes its numerator,
denominator, and percentage. A zero denominator produces ``null``, not 100%.

Artifacts and resumption
------------------------

``manifest.json`` freezes the model metadata, source code, package versions,
decoding settings, protocol, and complete input identity. ``inputs.json`` stores
each observation alongside its validation labels. Per-sample files under
``results/<split>/`` retain the rendered prompt, image identity, raw completion,
token IDs, parser output, actual arguments, resolved worker geometry, and
validation errors. Inline image bytes are not copied into results.

Re-running the identical command resumes only missing samples. A changed
manifest requires a new output directory. ``--limit N`` limits the amount of
pending work performed by that process without changing the full experiment's
expected sample count; its summary remains incomplete until every input has a
result.

An optional ``--supplemental FILE.jsonl`` accepts separately accounted original
requests outside the prepared dataset. Each line contains ``split``, ``record``
(the native observation), and ``sample`` (the original validation metadata).
For example, unannotated source rows can be tested under ``source_review`` with
no inferred control labels. They must not be silently added to the prepared
training denominator or given fabricated desired controls.
