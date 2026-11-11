Native video generation and inspection
======================================

This package separates two callable tools from the policy's final decision.
``sol_h3`` returns a registered video ID. ``inspect_video`` accepts that ID and
returns evidence, strengths to preserve, repair priorities, and explicit rechecks
of earlier findings. The environment supplies the immutable original request and
the actual generation prompt. Inspection does not happen automatically.

Feedback contains no acceptance verdict, numerical quality score, comparison, or
stopping advice. ``status`` describes whether media could be assessed, not its
quality. The inspector never receives remaining resources or the cost objective.
Individual findings still carry minor/major severity and evidential confidence.

The policy may finish after any successful generation, with or without inspection,
by calling ``finish_video`` with ``final_video_id`` and ``decision_summary``. This
terminal control action makes no model/service call and consumes no media budget.
A plain or single Markdown-fenced final JSON object is also accepted. An earlier video
may be selected. All costs, native messages, raw policy outputs, media identities,
and final decisions are recorded. Context or turn-limit failures have no implicit
final video and are reported separately.

The initial evaluation caps the episode at three generation calls and three
uncached inspection attempts. Failed service attempts consume their call budget;
repeated successful inspections return cached feedback without another backend
request. Deterministic seed 42, ten-second generation, and the existing H3 backend
are retained. The policy is prompted to balance expected benefit and call cost;
this zero-shot run does not train or establish an optimal cost penalty.

Files
-----

* ``session.py``: shared episode state, schemas, dispatch and final selection.
* ``feedback.py`` / ``inspector_prompt.txt``: Gemini evidence contract and transport.
* ``policy.txt``: policy objective and repair instructions.
* ``verl_tools.py``: ``BaseTool`` adapters for the same session implementation.

Training integration
--------------------

All three action adapters must receive the same episode-local ``VideoSession`` in their
``create(video_session=...)`` calls. Tool calls are sequential within an episode.
The training loop must retain native messages and masks, permit multiple tools,
parse the final selection, and reward the selected artifact plus its measured
resource costs. These adapters do not replace those training-loop responsibilities.
The existing one-call ``H3TerminalLoop`` and the active GRPO run are unchanged.

The isolated inference runner is under
``outputs/prompt_rewrite/qwen38-native-inspection-200-20261101-v1/``. It freezes the
tool definitions, prompts, model/settings, source hashes and 200 prompt identities.
Eight balanced smoke cases must finish successfully with working native inspection
and no tool errors before the remaining 192 are dispatched. No pairwise comparison
tool is exposed. A separate offline evaluator measures the frozen final selections
and consecutive generated rounds after inference has ended.

Verification
------------

Run the behavioral tests with the existing environment and checkout::

  PYTHONPATH=/home/jeff/verl/examples/video_agents:/home/jeff/verl-grpo-streaming-20261031 \
    uv run --no-project --python /home/jeff/.venvs/video-agents-h100-f169e786/bin/python \
    python -m pytest -q tests/special_standalone/video_agents/test_native_inspection_agent_on_cpu.py

The tests cover native separation, optional inspection, early and earlier-video
selection, resource boundaries, durable pending calls, original-goal ownership,
feedback identity, persistent/unassessable issue tracking, and verdict exclusion.
