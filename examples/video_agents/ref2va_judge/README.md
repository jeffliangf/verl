# Ref2VA judge (rubric R1): specification and wiring plan

**Status: specification only. Nothing in the RL code uses this yet.** This folder freezes what the Ref2VA judge is,
exactly as validated on 2026-11-25. It also lists what to change to wire it into the video-agents RL run.

Evidence and every experiment are in `~/rl-benchmarks/ref2va-judge-study-20261125/README.md`. The best-of-4 picks
made with this judge are in `~/verl/data/pai_r2v_eval200_smd_h3_raw_vs_cir/bestof4/judge-argon-r1/`.

| File | What it is |
|---|---|
| `instructions_r1.txt` | The rubric, byte-exact as validated (484 words). The sha256 of the text without its trailing newline is `57de1b0ad540d419a0b6e98b8cd08b70742a05e14038ea66652e5ed16181179b`. |
| `judgment_schema.json` | The response JSON schema: `preferred_video` (A/B), `reason`, `confidence` (low/medium/high). It has the same three keys as the T2V/I2V v4 judge, so the runner's `validate_judgment` needs no change. |
| `golden_requests.json` | Three complete example requests (image-only, image+audio, image+video+audio). Text parts are verbatim, except the user prompt: it is given as its eval200 row id plus the sha256 of the full `Original generation prompt:` part, so user content is not published. Media parts give kind, mime, sha256 of the bytes sent, source file and media resolution. Use it as a golden test when wiring. |

## 1. What the judge is, exactly

One **pairwise forced-choice** call per ordered pair (A, B), on Vertex `generateContent`:
- **Model:** `gemini-4-hydrogen` for training rewards; `gemini-4-argon` for evals and the best-of-4 selection.
- **Settings:** `thinking_level` HIGH, `seed` 0, `response_mime_type` `application/json`, `response_json_schema` =
  `judgment_schema.json`, no `max_output_tokens`.
- **Orders:** every pair is judged in both orders. The reward is the same tournament win rate as T2V/I2V.

The request is a single user turn with parts in this order. It is the existing T2V/I2V runner layout
(`run_gemini_pairwise_eval.build_interaction_request`) with N labelled references instead of one image:

```
text   "Original generation prompt:\n" + <original request text>
text   "Reference media given to the generator, in the order it received them. Each item is introduced by the label the prompt uses for it: [ImageN], [VideoN], [AudioN]. Some prompts instead write Image N, 图N, 图片N or @图片N for the same numbering, or give a reference a name."
for each reference, in the order the generator received them:
  text   "<label> (<noun>):"
  media  the reference
text   "Video A is the first video input and follows immediately."
video  candidate A                                   media_resolution HIGH
text   "Video B is the second video input and follows immediately."
video  candidate B                                   media_resolution HIGH
text   "Comparison instructions:\n" + <instructions_r1.txt>
```

- **Original request text.** Use the user's prompt as written in the dataset, with list-valued prompts joined by
  `"\n"`. Never use the policy's rewrite or the Context-IR expansion. The study sent the prompt alone, with no
  "Requested output settings" line. `mixed_data.original_request(sample)` returns exactly that when the sample has a
  `request` field.
- **Reference order and labels.** All images, then all videos, then all audio clips, each in dataset order. This is
  the order the H3 Ref2VA worker receives them.
  - Labels count per kind from 1: `[Image1]`, `[Image2]`, …, `[Video1]`, …, `[Audio1]`, … (no space).
  - These are the dataset's Seedance-style tags, so they match the prompt text. They are not H3's `<Picture N>`
    labels.
  - Nouns, exactly as validated:
    - image: `reference image`
    - video: `reference video (with its own soundtrack, if any)`. The doubled parentheses in the rendered line
      `[Video1] (reference video (with its own soundtrack, if any)):` are what was tested; tidying them is cosmetic
      and untested.
    - audio: `reference audio`
- **Media preparation, as validated** (study `scripts/media.py`):
  - Reference images: decode by content (extensions lie in this dataset), apply EXIF orientation, convert to RGB,
    resize so the longest side is at most 1536 px, encode as JPEG quality 92. Send at the default media resolution,
    which for images is HIGH, about 1.1k tokens each.
  - Reference audio: ffmpeg to 16 kHz mono PCM WAV (`-ac 1 -ar 16000 -c:a pcm_s16le`), mime `audio/wav`, as an audio
    part. Some dataset files are MP3 inside WAV, and Gemini resamples audio to 16 kHz mono anyway. About 32 tokens per
    second.
  - Reference videos: ffmpeg to H.264 + AAC MP4, at most 1280 px on the long side, rotation applied (dataset refs
    include HEVC, MOV, 60 fps and rotated phone clips). Sent at the **default** media resolution, not HIGH.
  - Candidate videos: the rendered MP4 as is. Re-encode only if larger than 14 MB; no H3 render was. Sent at
    `media_resolution` HIGH, about 297 tokens per second including audio.
- **Typical size:** about 12k input tokens per judgment (two 15-s candidates ≈ 9k, three images ≈ 3.3k, text ≈ 1.5k)
  and about 1–1.4k output+thinking tokens. Latency is about 13–14 s with Hydrogen and about 27 s with Argon.

## 2. Why this rubric (one-paragraph summary of the evidence)

- **Swap test.** One reference was replaced by a SigLIP look-alike: paired AUC 0.97, with no inversions in 180 swaps.
  Voice swaps scored 0.90–0.98.
- **Decoy renders.** These are the RL sibling case: same prompt and seed, one character's image swapped. R1 picks the
  faithful video 66/68 times; the v4 rubric with references attached gets 58/68 and trades identity for smoothness.
- **Stability.** Order flips on a close matchup: R1 4/48, v4 + refs 10/48. First-shown wins 50.3% pooled.
- **Robustness.**
  - Pasted reference stills or pasted intros never win (156/156).
  - A late identity switch is caught 50/50 times when the look-alike is on screen.
  - Synthetic morph artifacts are caught 40/40.
- **Variants rejected.**
  - A 272-word version flipped more (9/48).
  - The old I2V quality text pasted in verbatim detected defects identically but flipped 13/48 and leaned to the first
    video (57%).

## 3. Wiring plan for the RL run (code as on `origin/feat/prompt-rewrite-eval` at `c6a7c26b`)

Do these in order. All file references are under `examples/video_agents/`.

1. **Data rows** (`mixed_data.py`, dataset builders). A Ref2VA sample needs:
   - `task: "ref2va"`
   - `request`: the original prompt text, for the judge and the policy observation
   - `target`
   - `references: [{"kind": "image"|"video"|"audio", "path": ..., "sha256": ...}]` in generator order

   Labels are derived from that order, never stored separately. The rollout must send the same references in the
   same order to the H3 Ref2VA gateway (`task: "ref2va"`, `conditions` images → videos → audio), so the judge's
   `[ImageN]` and the generator's `<Picture N>` name the same file.
2. **Rubric per task** (protocol builders in `mixed_data.py` and `rl.py`). Today both load the v4 rubric from
   `<eval_repo>/scripts/build_pairwise_manifest.py`, check `INSTRUCTIONS_VERSION == 4`, and freeze one
   `protocol["rubric"]`.
   - Add a frozen per-task map: `t2v`/`i2v` → v4 text (unchanged), `ref2va` → `instructions_r1.txt`, checked by the
     sha256 above.
   - Give each task its own instructions version, e.g. `ref2va-r1`.
   - Keep `protocol["rubric"]` for old protocols.
3. **Manifests.** `tournament.tournament_manifests` (training) and `judge.build_manifests` (eval) currently add
   `**reference_image_fields(sample)` (`image_assets.py`) and `instructions_how_to_compare = protocol["rubric"]`.
   - For `ref2va`, add `references: [{label, kind, path, sha256}]` to each pair instead.
   - Use the task's rubric and version. The pair dict and the instructions feed `judge.evaluation_signature`, so the
     sha256s make every cached vote content-addressed, like the I2V image today.
4. **Request building.** `judge_worker.reference_images(runner)` wraps the external runner's
   `build_interaction_request(reference_image=...)`; that function takes one image and one source video. Either:
   - (a) extend the runner in `video-generation-eval` with a `references` list and the parts above (then update the
     frozen `dependencies` hash in the protocol), or
   - (b) add a `reference_media(runner)` context manager next to `reference_images`. It would prepare each
     reference (section 1), upload or inline it, and rebuild the payload with the labelled parts.

   `streaming_judge.py` enters `reference_images(runner)` and `reference_images(secondary_runner)` around judging, so
   enter the new manager in the same place.
5. **Transport.** `gemini_transport.vertex_request` rejects any media part other than image/video. Add
   `"audio"` → `types.Part.from_bytes(..., mime_type="audio/...")`, with no resolution. Give the Vertex client an
   inline `upload_audio` next to `upload_image`/`upload_video`, and make the runner's `media_part` accept `"audio"`.
6. **Media resolution.** `gemini_transport.video_media_part` sets the run's `video_resolution` on every video part,
   which would include reference videos. The validated setup is HIGH for the two candidates and default for reference
   videos. Restrict the patch to candidate videos. Leaving refs at HIGH is untested: it costs about 3k more tokens per
   10-s reference video, on about 10% of rows.
7. **Secondary (fallback) judge.** Flash currently goes through the Gemini API (Interactions). Check that it accepts
   inline audio parts.
   - Ranking only: Flash on Vertex worked in the study (look-alike AUC 0.95), but its absolute scores are lenient.
   - Without audio support, route Ref2VA failures to the neutral 0.5 fallback or to Flash on Vertex. Note that Flash
     on Vertex shares the Vertex quota, which is what the fallback exists to avoid.
8. **Reward and eval.** Leave the tournament reward unchanged: all pairs, both orders, win rate, neutral fallback.
   - Eval uses Argon with the same rubric and request.
   - Reference ladder columns already exist in
     `~/verl/data/pai_r2v_eval200_smd_h3_raw_vs_cir/eval200.jsonl`:
     - `h3_context_ir_h3_smd_ref2va_video` (one Context-IR render)
     - `best_of_4_h3_context_ir_h3_smd_ref2va_video` (Argon R1 picks)
     - `seedance_video_720p_path`
     - `prompt_h3_video` (hosted H3)
   - Ref2VA rewards are new, so there is no older baseline to stay comparable with. Freeze the rubric before launch.
9. **Checks before launch:**
   - **Golden test.** Build the request for the three rows in `golden_requests.json` and compare: text parts exactly
     (the prompt part by its sha256), media parts by sha256 after the section-1 preparation, and the resolutions.
   - **Unit tests.** Audio part in `vertex_request`; label numbering with mixed kinds; a pair with no images
     (video-only rows exist).
   - **Reproduction smoke.** Rejudge about 20 eval200 raw-vs-Context-IR pairs in both orders with Hydrogen. Compare
     with the study cache (`~/rl-benchmarks/ref2va-judge-study-20261125/judge/pairs-hydrogen/rawcir-R1-*.json`);
     seed 0 with identical inputs should reproduce about 95% of votes.
   - **Cost and quota.** A group of 8 is 28 pairs × 2 orders = 56 judgments of ~12k tokens, about 1.8× a T2V
     judgment. Watch for Vertex 429s while other studies run.

## 4. Known limits

- Gemini samples video at 1 fps. A 1.5-s freeze was missed about 30% of the time (mostly order splits). Subtle
  late-clip morphing (a logo, a fish) was sometimes missed. 2 fps would be the lever, at about twice the video tokens,
  and is untested.
- Voice-timbre checks were validated with random-speaker swaps only. True voice references score as clearly matched
  only 57–64% in absolute terms.
- When a storyboard contradicts a character sheet, the right answer is ambiguous for any judge.
- Hydrogen and Argon have been swapped server-side before (11-14, 11-17). Keep a determinism canary in the run.
