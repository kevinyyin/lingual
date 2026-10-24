# Two-Way ASL MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On one laptop, a Deaf signer and a hearing person converse both ways. ASL (~150 everyday signs plus fingerspelling) becomes expressive English speech, and English speech becomes a VRM avatar signing with facial grammar and tone.

**Architecture:** One browser page runs MediaPipe Tasks on the webcam and streams landmarks (never video) over a WebSocket to a Python FastAPI backend. The page also renders the three.js VRM avatar. The backend owns everything else: segmentation, sign and fingerspelling classifiers, the grammar and emotion readers, the Claude calls, Cartesia Sonic playback and Cartesia Ink transcription on the laptop's own speaker and mic, turn-taking, and the turn log. Vocabulary lives in one YAML file used by training, the classifier labels, the LLM prompts, the validator and the avatar.

**Tech Stack:**
- Python 3.12 with uv: FastAPI, uvicorn, pydantic, numpy, PyTorch (MPS for training, CPU for inference), mediapipe (Tasks API), anthropic, websockets, sounddevice, PyYAML, python-dotenv, pytest.
- Web: Vite, TypeScript, `@mediapipe/tasks-vision`, three, `@pixiv/three-vrm`, kalidokit, vitest.

**Spec:** `docs/superpowers/specs/2026-10-16-two-way-asl-design.md`

## Global Constraints

- All new code lives in `app/`. Leave `latency-bench/` untouched.
- Only landmarks, blendshapes and head angles leave the browser. Never send video or images. Store no audio or video. `app/data/ours/` holds landmark arrays only.
- API keys (`ANTHROPIC_API_KEY`, `CARTESIA_API_KEY`) are read only from `app/.env` by the backend and never sent to the browser.
- Use the same landmark models (MediaPipe Tasks hand, pose and face `.task` files, same versions) for training extraction in Python and at runtime in the browser. Never mirror frames given to MediaPipe in either place; mirror the preview only with CSS.
- `app/vocab/vocab.yaml` is the single source of truth for signs. No gloss string is hard-coded anywhere else, except in test fixtures.
- Latency: ≤ 2 s from hands at rest to first audio, and ≤ 2 s from the speaker stopping to the avatar starting. If the picker appears, the first clock restarts at the tap.
- Facial grammar uses brows and head yaw. The emotion reader must never read any `brow*` blendshape.
- One direction at a time. Mic input is ignored while the voice plays or the signer's turn is active, and signer turns are ignored while the avatar plays.
- LLM: model `claude-haiku-5-5` for both language steps. Cartesia: Sonic for TTS (streaming, with emotion, speed and volume controls) and Ink for STT. Verify exact model IDs and message shapes against current Cartesia docs in Task 9 before coding against them.
- If time runs short, the only allowed cuts, in order: (1) vocabulary 150 → ~75 signs (set `active: false` in `vocab.yaml`), (2) natural pace → brief hold per sign (raise the segmenter's pause threshold). Nothing else is cut.

## Review Focus

1. **A hand leaves the frame or no face is detected mid-sentence.** Expect no crash: missing hands are zeros plus a presence flag, and a missing face means grammar `statement` and emotion `neutral`, never a guess. (Tests in Tasks 3, 11 and 12.)
2. **Both people at once.** If the hearing person talks while the signer is signing or the voice is playing, that audio is dropped, not merged into a turn. A signer starting while the avatar plays is ignored until it finishes. (Test in Task 4.)
3. **Claude returns malformed JSON, an out-of-vocabulary gloss, or times out.** Retry once, then fall back: the gloss side fingerspells the unknown token, and the compose side speaks the glosses joined as plain words. Never hang a turn. (Tests in Tasks 9 and 13.)
4. **Empty turns.** Hands raised and dropped with no sign, or mic noise that transcribes to an empty or whitespace string, produce nothing spoken or signed, and the session returns to ready. (Test in Task 4.)
5. **Cartesia drops mid-utterance.** The caption or "you said" text is still shown, an error state is logged in the turn log, and the session returns to ready instead of sticking in `speaking` or `listening`. (Test in Task 10.)

---

## File Structure

```
app/
  pyproject.toml            uv project; scripts: `asl-server`, `asl-train`
  .env.example              ANTHROPIC_API_KEY=, CARTESIA_API_KEY=
  .gitignore                data/, models/, logs/, .venv, web/node_modules
  vocab/vocab.yaml          the sign list (Task 2)
  server/
    __init__.py
    protocol.py             pydantic message models, browser <-> backend (Task 1)
    vocab.py                load/query vocab.yaml (Task 2)
    features.py             raw Frame -> normalized feature vector (Task 3)
    session.py              orchestrator state machine + turn log (Task 4)
    main.py                 FastAPI app, WS /ws, wiring (Task 4)
    segmenter.py            sign/fingerspell segments + end of turn (Task 7)
    classifier.py           sign model def + SignPredictor (Task 6)
    fingerspell.py          letter model + word decoder (Task 15)
    grammar.py              facial grammar reader (Task 11)
    emotion.py              emotion reader (Task 12)
    compose.py              Claude: signs -> English (Task 9)
    voice.py                emotion -> Sonic controls; Sonic streaming playback (Task 10)
    stt.py                  mic capture + Ink streaming + turn end (Task 13)
    gloss.py                Claude: English -> sign plan + validator (Task 13)
  training/
    landmarks.py            video file -> list[Frame] via MediaPipe Tasks (Task 5)
    extract_asl_citizen.py  ASL Citizen videos for active glosses -> .npz (Task 5)
    train_signs.py          train + save models/signs.pt (Task 6)
    train_fingerspell.py    train + save models/letters.pt (Task 15)
    build_motion_library.py reference videos -> web/public/signs/*.json (Task 14)
    eval.py                 subcommands: signs | grid | spell | latency (Task 18)
  scripts/probe_cartesia.py live Sonic/Ink API shape check (Task 9)
  tests/                    pytest, one file per server module
  web/
    package.json, vite.config.ts (proxy /ws -> :8000), index.html
    src/main.ts             boot, mode switch (converse | record)
    src/ws.ts               typed socket client
    src/camera.ts           MediaPipe Tasks -> Frame messages; calibration
    src/ui.ts               captions, "you said", picker, state badge
    src/record.ts           recorder mode (Task 8)
    src/avatar/vrm.ts       load VRM, apply Kalidokit rig from landmark frame
    src/avatar/player.ts    sequence clips/letters, blend, timing
    src/avatar/face.ts      grammar + tone overlays on VRM expressions/head
    src/**/*.test.ts        vitest
    public/models/          MediaPipe .task files (downloaded in Task 3)
    public/avatar.vrm       a CC0/permissive VRM (Task 14)
    public/signs/*.json     motion clips (Task 14)
    public/letters/*.json   26 handshape clips (Task 15)
  data/                     gitignored: asl_citizen/, landmarks/, ours/
  models/                   gitignored: signs.pt, letters.pt
  logs/                     gitignored: turns.jsonl
```

## Shared Data Shapes (defined in Task 1, used everywhere)

- **`Frame`** (browser → backend, ~30/s): `t: float` (ms, `performance.now()`); `left: list[list[float]] | None` (21×[x,y,z]); `right: …` (same); `pose: list[list[float]]` (33×[x,y,z,visibility]); `blend: dict[str, float] | None` (the 52 ARKit-named face blendshapes); `head: tuple[float,float,float] | None` (yaw, pitch, roll in degrees, from the facial transformation matrix). Training `.npz` files store the same fields as arrays.
- **Browser → backend messages:** `{"type":"frame", ...Frame}`, `{"type":"calibrate"}`, `{"type":"pick","segment":int,"gloss":str}`, `{"type":"avatar_done"}`, `{"type":"record","signer":str,"gloss":str,"frames":[Frame]}`.
- **Backend → browser messages:**
  - `{"type":"state","state":"calibrating|ready|signing|picking|speaking|listening|avatar"}`
  - `{"type":"pick_request","segment":int,"options":[{"gloss":str,"english":str,"p":float}]}` (3 options)
  - `{"type":"you_said","text":str,"emotion":str}`
  - `{"type":"caption","text":str,"final":bool}`
  - `{"type":"sign_plan","tokens":[{"kind":"sign","gloss":str}|{"kind":"spell","word":str}],"grammar":"yn|wh|neg|statement","tone":"warm|apologetic|urgent|neutral","english":str}`
- **Reader outputs:** `Grammar = {"question": "yn"|"wh"|None, "negation": bool}`; `Emotion = {"label": "happy"|"sad"|"angry"|"neutral", "intensity": float 0..1}`.

## Day Plan

| Day | Tasks | End-of-day proof |
|---|---|---|
| 1 | 1–5 | Stub conversation both ways in the browser (text only); ASL Citizen extraction running |
| 2 | 6–10 | Signing real vocabulary produces spoken English with emotion-controlled Sonic |
| 3 | 11–14 | Grammar and emotion change the speech; speaking produces a validated sign plan; motion clips exist |
| 4 | 15–17 | Avatar signs with face grammar and tone; fingerspelling works both ways; picker works |
| 5 | 18 | Targets measured; tuned; five clean conversations |

---

### Task 1: App scaffold and message protocol

**Files:**
- Create: `app/pyproject.toml`, `app/.env.example`, `app/.gitignore`, `app/server/__init__.py`, `app/server/protocol.py`
- Create: `app/web/` via Vite vanilla-ts template; `app/web/src/ws.ts`
- Test: `app/tests/test_protocol.py`

**Interfaces:**
- Produces: pydantic models `Frame`, `FrameMsg`, `CalibrateMsg`, `PickMsg`, `AvatarDoneMsg`, `RecordMsg`, `StateMsg`, `PickRequestMsg`, `YouSaidMsg`, `CaptionMsg`, `SignPlanMsg`, `SignToken`, and `parse_client(raw: str) -> ClientMsg` (a discriminated union on `type`). `ws.ts` exports `connect(onMsg: (m: ServerMsg) => void): { send(m: ClientMsg): void }` with TS types mirroring the above.

- [ ] **Step 1:** `uv init` the project in `app/` with Python `==3.12.*` and add the dependencies from Tech Stack (dev group: pytest). `npm create vite@latest web -- --template vanilla-ts` in `app/`, then `npm i @mediapipe/tasks-vision three @pixiv/three-vrm kalidokit` and `npm i -D vitest @types/three`. Set the Vite proxy so `/ws` goes to `ws://localhost:8000`.
- [ ] **Step 2: Write failing tests** in `test_protocol.py`:
  - `test_parse_frame_with_missing_hand`: a frame JSON with `"left": null` parses, and `msg.left is None`.
  - `test_parse_rejects_unknown_type`: `parse_client('{"type":"nope"}')` raises `ValueError`.
  - `test_sign_plan_roundtrip`: a `SignPlanMsg` with one sign token and one spell token survives `model_dump_json` → `model_validate_json` unchanged.
- [ ] **Step 3:** Run `cd app && uv run pytest tests/test_protocol.py -v`. Expected: FAIL (import error).
- [ ] **Step 4:** Implement `protocol.py` per Shared Data Shapes, wrapping pydantic `ValidationError` as `ValueError` in `parse_client`. Write `ws.ts` with matching types.
- [ ] **Step 5:** Run the tests. Expected: 3 passed. Run `cd app/web && npx tsc --noEmit`. Expected: no errors.
- [ ] **Step 6: Commit** `feat(app): scaffold backend/web and message protocol`.

### Task 2: Vocabulary file and loader

**Files:**
- Create: `app/vocab/vocab.yaml`, `app/server/vocab.py`
- Test: `app/tests/test_vocab.py`

**Interfaces:**
- Produces: `Sign` dataclass (`gloss: str`, `english: str`, `category: str`, `asl_citizen: str`, `active: bool`, `avatar_ok: bool`). `load_vocab(path=DEFAULT) -> Vocab`, where `Vocab` has `.active: list[Sign]` (stable order, the classifier label order), `.index(gloss) -> int`, `.get(gloss) -> Sign | None`, `.glosses: set[str]` (active only), `.prompt_table() -> str` (one line per active sign `GLOSS — english`, for LLM prompts).

- [ ] **Step 1:** Write `vocab.yaml` with ~150 general everyday signs covering the spec's categories: greetings and social (HELLO, THANK-YOU, PLEASE, SORRY, NICE-MEET-YOU…), people (I/ME, YOU, HE-SHE, FRIEND, FAMILY…), question words (WHAT, WHERE, WHEN, WHO, WHY, HOW, WHICH, HOW-MUCH), wants and needs (WANT, NEED, HAVE, LIKE, HELP, FINISH…), yes/no/not (YES, NO, NOT, NONE), numbers ONE–TEN, time (NOW, TODAY, TOMORROW, LATER, TIME…), places (HOME, WORK, SCHOOL, BATHROOM, STORE…), food and drink (COFFEE, WATER, TEA, MILK, EAT, DRINK…), and feelings (HAPPY, SAD, TIRED, SICK, GOOD, BAD…). Each entry's `asl_citizen` field is the matching ASL Citizen gloss string. Pick only glosses present in ASL Citizen; verify against its gloss list once downloaded (Task 5), and fix or replace any missing ones. Set all to `active: true, avatar_ok: true`.
- [ ] **Step 2: Write failing tests:**
  - `test_active_order_stable`: two loads give identical `[s.gloss for s in active]`.
  - `test_no_duplicate_glosses`.
  - `test_size_in_range`: `75 <= len(active) <= 160`.
  - `test_inactive_excluded`: a temp YAML with one `active: false` entry leaves it out of `.active` and `.glosses`.
- [ ] **Step 3:** Run `uv run pytest tests/test_vocab.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement `vocab.py`.
- [ ] **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6: Commit** `feat(app): everyday vocabulary and loader`.

### Task 3: Camera client and feature normalization

**Files:**
- Create: `app/web/src/camera.ts`, `app/server/features.py`, and the `.task` files in `app/web/public/models/` (hand_landmarker, pose_landmarker_lite, face_landmarker)
- Test: `app/tests/test_features.py`

**Interfaces:**
- Consumes: `Frame` (Task 1).
- Produces: `camera.ts`: `startCamera(video: HTMLVideoElement, onFrame: (f: Frame) => void): Promise<void>`. It runs HandLandmarker (2 hands), PoseLandmarker and FaceLandmarker with `outputFaceBlendshapes: true` and `outputFacialTransformationMatrixes: true` in VIDEO mode, and converts the matrix to yaw/pitch/roll degrees.
- Produces: `features.py`: `FEAT_DIM: int`, `frame_features(f: Frame) -> np.ndarray[FEAT_DIM]`, `sequence_features(frames: list[Frame], T: int = 32) -> np.ndarray[T, FEAT_DIM]` (linear time resample).
- Feature layout (fixed): for each hand, 21×3 coordinates relative to the wrist and divided by shoulder width, plus 1 presence flag; then the wrist position relative to the shoulder midpoint divided by shoulder width (2×3); then pose points for the nose, shoulders, elbows and wrists relative to the shoulder midpoint divided by shoulder width (7×3). Missing hand means zeros and flag 0. **No blendshapes in sign features.**

- [ ] **Step 1: Write failing tests:**
  - `test_missing_hand_zero_and_flag`: a frame with `right=None` gives that hand's 63 values == 0 and its flag == 0, and no exception.
  - `test_scale_invariant`: scaling every x,y by 2 around the shoulder midpoint leaves `frame_features` equal within 1e-5.
  - `test_resample_shape`: 17 frames → shape `(32, FEAT_DIM)`; 1 frame → `(32, FEAT_DIM)`.
  - `test_zero_shoulder_width_safe`: coincident shoulders produce finite output, with no NaN.
- [ ] **Step 2:** Run `uv run pytest tests/test_features.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `features.py` per the layout. Clamp shoulder width to ≥ 1e-3.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Implement `camera.ts`. Show the preview mirrored with CSS `transform: scaleX(-1)` only. Drop frames when the socket is backpressured (`bufferedAmount > 256 KB`).
- [ ] **Step 6:** Manual check: with `npm run dev`, the console logs ~25–30 frames/s, and `blend.browInnerUp` rises visibly when you raise your brows.
- [ ] **Step 7: Commit** `feat(app): camera landmarks client and feature normalization`.

### Task 4: Orchestrator skeleton with stubs, end to end

**Files:**
- Create: `app/server/session.py`, `app/server/main.py`, `app/web/src/ui.ts`, `app/web/src/main.ts` (converse mode), `app/web/index.html`
- Test: `app/tests/test_session.py`

**Interfaces:**
- Consumes: protocol (Task 1), `Frame` (Task 3).
- Produces: `class Session` with injected collaborators so later tasks swap stubs for real parts:
  - `Session(send: Callable[[ServerMsg], Awaitable[None]], segmenter, recognizer, readers, composer, voice, gloss, log_path)`
  - `async on_client(msg: ClientMsg)`
  - `async on_transcript(text: str, final: bool)`: called by STT
  - `state: str`
  - Each turn writes one JSONL line to `logs/turns.jsonl`: `{"turn_id", "direction": "sign2speech"|"speech2sign", "stamps": {stage: ms}, "result": ..., "error": str|None}`. Stage names for sign2speech: `hands_rest`, `recognized`, `picked`, `composed`, `first_audio`. For speech2sign: `speech_end`, `planned`, `avatar_start`.
- Calibration: `Baseline` dataclass (`rest_y: float` = the mean wrist y over the calibration window plus a 0.05 margin, `brow: dict[str, float]` = mean of each `brow*` blendshape, `motion_floor: float` = mean wrist speed at rest, `speed: float` and `size: float` filled from the first signing turn's mean wrist speed and wrist bounding-box area, used by Task 12). `compute_baseline(frames: list[Frame]) -> Baseline` collects 2 s (~60 frames) of neutral face with hands at rest after `CalibrateMsg`.
- Collaborator protocols (stubs live in `session.py` until replaced):
  - `Segmenter.push(f: Frame) -> SegEvent | None` (`SegEvent` = `TurnStart`, or `TurnEnd(segments: list[Segment])`)
  - `Recognizer.recognize(seg: Segment) -> list[tuple[str, float]]` (top-3)
  - `Readers.read(frames: list[Frame], baseline: Baseline) -> tuple[Grammar, Emotion]`
  - `Composer.compose(signs: list[str], spelled: list[str], grammar, emotion) -> str`
  - `Voice.speak(text, emotion) -> AsyncIterator[str]` (yields `"first_audio"` then `"done"`; raises on failure)
  - `Gloss.plan(english: str) -> SignPlanMsg`
- State machine: `calibrating → ready`. On `ready` + `TurnStart` → `signing`. On `TurnEnd`: if any segment's top-1 p < `PICK_THRESHOLD` (0.6, tuned in Task 18) → `picking`, otherwise compose → `speaking` → `ready`. On `ready` + final transcript → `listening` → `avatar` → `ready` on `avatar_done`. Transcripts outside `ready`/`listening` are dropped; frames during `avatar`/`speaking` are fed to nothing.

- [ ] **Step 1: Write failing tests** with fake collaborators and a list-collecting `send`:
  - `test_sign_turn_speaks_and_logs`: frames trigger TurnStart then TurnEnd with confident segments; `send` receives `you_said` and states `signing`, `speaking`, `ready`; the log line has all 5 sign2speech stamps.
  - `test_low_confidence_requests_pick`: one segment at p=0.4 → `pick_request` with 3 options, and no `you_said` until a `PickMsg` arrives.
  - `test_transcript_during_signing_dropped`: while `signing`, `on_transcript("hello", True)` sends no `sign_plan`. *(Review Focus 2)*
  - `test_transcript_during_speaking_dropped`. *(Review Focus 2)*
  - `test_empty_sign_turn_returns_ready`: TurnEnd with `segments=[]` sends no `you_said`, and the state goes back to `ready`. *(Review Focus 4)*
  - `test_blank_transcript_ignored`: `on_transcript("   ", True)` → no `sign_plan`. *(Review Focus 4)*
  - `test_voice_failure_still_shows_text`: Voice raises mid-stream → `you_said` was sent, the log has `error`, and the state is `ready`. *(Review Focus 5)*
  - `test_speech_turn_plans_and_waits_for_avatar`: final transcript → `caption` final, `sign_plan`, state `avatar`; after `AvatarDoneMsg` the state is `ready`.
  - `test_calibration_sets_baseline`: `CalibrateMsg` plus 60 resting frames → state `ready` and `baseline.rest_y` > the mean resting wrist y.
- [ ] **Step 2:** Run `uv run pytest tests/test_session.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `session.py` with stubs:
  - `StubSegmenter`: TurnStart when either wrist rises above the calibrated rest line, TurnEnd after the wrists stay below it for 20 frames, as one segment.
  - `StubRecognizer`: `[("HELLO", .9), …]`.
  - `StubReaders`: statement/neutral.
  - `StubComposer`: joins glosses.
  - `StubVoice`: prints.
  - `StubGloss`: one spell token per word.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Implement `main.py`: FastAPI `WS /ws` creates one `Session` per connection, plus a dev-only text box in `ui.ts` that posts typed text as a transcript (removed in Task 13). Add the console script `asl-server = server.main:run` (uvicorn on :8000).
- [ ] **Step 6:** Manual check: run `uv run asl-server` and `npm run dev`. Calibrate, raise and lower your hands, and see "you said: HELLO". Type "hi there" and see a sign plan with spell tokens in the UI.
- [ ] **Step 7: Commit** `feat(app): orchestrator skeleton with stub pipeline`.

### Task 5: Landmark extraction from ASL Citizen

**Files:**
- Create: `app/training/landmarks.py`, `app/training/extract_asl_citizen.py`
- Test: `app/tests/test_landmarks.py`

**Interfaces:**
- Consumes: `Frame` (Task 1), `load_vocab` (Task 2).
- Produces: `video_to_frames(path: Path) -> list[Frame]` (MediaPipe Tasks Python with the **same `.task` files** as `web/public/models/`); `save_npz(frames, path)` / `load_npz(path) -> list[Frame]`. On disk: `data/landmarks/asl_citizen/{split}/{gloss}/{video_id}.npz`, where `split` is ASL Citizen's own train/val/test split (signer-disjoint).

- [ ] **Step 1:** Download ASL Citizen to `data/asl_citizen/` (a long download; start it first and keep it running in the background). Read its license and record its terms in one line in `app/data/README.md`, which is committed while the data stays gitignored. Check every `asl_citizen` gloss in `vocab.yaml` exists in the dataset's gloss list, and fix the vocabulary.
- [ ] **Step 2: Write failing test** `test_npz_roundtrip_preserves_missing_hand`: frames with `left=None` survive `save_npz` → `load_npz`.
- [ ] **Step 3:** Run it. Expected: FAIL. Implement `landmarks.py`. Run again. Expected: PASS.
- [ ] **Step 4:** Implement `extract_asl_citizen.py`: for each active gloss and each split, extract all its videos with a process pool, skipping files already done (resumable). Print counts per split.
- [ ] **Step 5:** Run `uv run python -m training.extract_asl_citizen` in the background. Expected when done: every active gloss has ≥ 15 train videos. Print any that don't and swap those signs in `vocab.yaml`.
- [ ] **Step 6: Commit** `feat(training): ASL Citizen landmark extraction`.

### Task 6: Sign classifier

**Files:**
- Create: `app/training/train_signs.py`, `app/server/classifier.py`
- Test: `app/tests/test_classifier.py`

**Interfaces:**
- Consumes: `sequence_features` and `FEAT_DIM` (Task 3); `Vocab.active` (Task 2); the npz files (Task 5); our recordings under `data/ours/{signer}/{gloss}/*.npz` (Task 8; optional at first train).
- Produces: `SignNet(n_classes, feat_dim=FEAT_DIM, T=32)` (torch module) and `SignPredictor(model_path="models/signs.pt")` with `.topk(frames: list[Frame], k=3) -> list[tuple[str, float]]` (softmax probabilities, glosses from the checkpoint's saved label list). The checkpoint stores `{"state_dict", "labels": [gloss...], "feat_dim", "T"}`. `SignPredictor` implements the Task 4 `Recognizer` protocol.

- [ ] **Step 1: Write failing tests:**
  - `test_predictor_output_shape`: an untrained `SignNet` saved with 5 labels → `topk(frames, 3)` returns 3 pairs with probabilities summing ≤ 1 and descending.
  - `test_labels_come_from_checkpoint`: the predictor uses the checkpoint's labels even if `vocab.yaml` order changed.
- [ ] **Step 2:** Run. Expected: FAIL. Implement `classifier.py`: a small temporal model (2-layer bidirectional GRU or 4-layer transformer encoder, hidden 256) with dropout, inferring on CPU. Run. Expected: PASS.
- [ ] **Step 3:** Implement `train_signs.py`. It trains on the ASL Citizen train split plus the `data/ours` signers listed in `--train-signers`, validates on the ASL Citizen val split, and holds out the test split and `--holdout-signer`.
  - Augmentation: random scale ±10%, rotation ±10°, time warp 0.8–1.2×, random frame drop 10%, and left/right hand swap with x-mirror.
  - AdamW with cosine schedule on `mps`, early stopping on val top-1.
  - Prints test top-1/top-3 at the end.
- [ ] **Step 4:** Train: `uv run python -m training.train_signs`. Expected: ASL Citizen test top-1 ≥ 0.80 (spec target). If below, first try more epochs and augmentation. If still below at the end of day 2, apply cut 1 (vocabulary → ~75) and retrain.
- [ ] **Step 5:** Wire `SignPredictor` into `main.py` in place of `StubRecognizer`.
- [ ] **Step 6: Commit** `feat(app): sign classifier trained on ASL Citizen`. Don't commit `models/`.

### Task 7: Segmenter

**Files:**
- Create: `app/server/segmenter.py`
- Test: `app/tests/test_segmenter.py`

**Interfaces:**
- Consumes: `Frame`; `Baseline` (Task 4).
- Produces: `Segmenter(baseline)` implementing `push(f) -> SegEvent | None`, plus `Segment(frames: list[Frame], kind: Literal["sign", "spell"])`, `TurnStart` and `TurnEnd(segments, frames)` (`frames` = the whole turn, for the readers).
- Rules: motion energy = smoothed speed of both wrists plus fingertip spread change.
  - The turn starts when a wrist rises above `rest_y`, and ends after both wrists stay below it for ≥ `REST_FRAMES` = 15.
  - Inside the turn, a pause is energy below `motion_floor × PAUSE_K` (1.5) for ≥ `PAUSE_FRAMES` (4; raise to ~10 for cut 2), and that splits segments.
  - Drop segments under 6 frames.
  - A segment is `spell` when exactly one hand is present, the wrist stays inside a box of 0.6 shoulder widths for the whole segment, and fingertip motion is high relative to wrist motion; otherwise it's `sign`.

- [ ] **Step 1: Write failing tests** using a synthetic frame builder in the test file (wrist trajectories plus finger jitter):
  - `test_two_signs_with_pause_make_two_segments`.
  - `test_hands_rest_ends_turn`: TurnEnd fires within `REST_FRAMES + 1` frames of the hands dropping.
  - `test_hands_never_raised_no_turn`: 100 resting frames → no event. *(Review Focus 4)*
  - `test_raise_and_drop_without_motion_gives_empty_segments`. *(Review Focus 4)*
  - `test_still_wrist_busy_fingers_is_spell`.
  - `test_hand_drops_out_mid_sign_no_crash`: `right=None` for 5 frames in the middle → still one segment. *(Review Focus 1)*
- [ ] **Step 2:** Run. Expected: FAIL. Implement. Run. Expected: PASS.
- [ ] **Step 3:** Replace `StubSegmenter` in `main.py`. Manual check: sign two vocabulary signs with a short pause and see two recognized glosses spoken.
- [ ] **Step 4: Commit** `feat(app): motion-energy segmenter`.

### Task 8: Recorder mode (our own signing)

**Files:**
- Create: `app/web/src/record.ts`; modify `app/server/main.py` to handle `RecordMsg`
- Test: `app/tests/test_record.py`

**Interfaces:**
- Consumes: `RecordMsg` (Task 1), `save_npz` (Task 5), `Vocab` (Task 2).
- Produces: a `?mode=record&signer=NAME` page. It shows the next gloss plus its English, and a link to the ASL Citizen reference for that gloss. The signer presses Space to start and drops their hands to stop, and the server writes `data/ours/{signer}/{gloss}/{n}.npz`. It also has an "emotion grid" sub-mode (`&grid=1`) that prompts {yn, wh, statement} × {happy, sad, angry, neutral} × a short question sentence and saves under `data/ours/{signer}/_grid/{qtype}_{emotion}_{n}.npz`.

- [ ] **Step 1: Write failing tests:**
  - `test_record_writes_npz_to_signer_gloss_dir` (with tmp data root).
  - `test_record_rejects_unknown_gloss`.
  - `test_record_rejects_path_traversal_signer`: signer `"../x"` → rejected and nothing written.
- [ ] **Step 2:** Run. Expected: FAIL. Implement. Run. Expected: PASS.
- [ ] **Step 3:** Record: Kevin and 1–2 teammates each do 5 takes per active sign, plus 3 takes per grid cell. Pick one person as the holdout signer and write their name in `app/data/README.md`. Retrain Task 6 with `--train-signers` (everyone except the holdout).
- [ ] **Step 4: Commit** `feat(app): landmark recorder and emotion-grid capture`.

### Task 9: English composer (signs → English)

**Files:**
- Create: `app/server/compose.py`, `app/scripts/probe_cartesia.py` (a doc-verified smoke probe for Tasks 10 and 13)
- Test: `app/tests/test_compose.py`

**Interfaces:**
- Consumes: `Grammar`, `Emotion`, `Vocab.get`.
- Produces: `Composer(client: anthropic.AsyncAnthropic)` with `async compose(signs: list[str], spelled: list[str], grammar: Grammar, emotion: Emotion) -> str`. It uses Claude `claude-haiku-5-5`, `max_tokens` 200, timeout 4 s, and asks for JSON `{"text": str}`.
- Prompt rules (system):
  - Write one natural first-person English sentence the signer means.
  - Keep ASL meaning, not word order.
  - `question=yn` → yes/no question ending "?"; `wh` → wh-question ending "?"; `negation` → the sentence is negated.
  - Use spelled words verbatim as names.
  - Don't add facts not signed.
  - Emotion may color word choice lightly but must not change meaning.

- [ ] **Step 1: Write failing tests** with a fake client:
  - `test_yn_question_passes_grammar_to_prompt`: the request includes `question: yn`.
  - `test_bad_json_retries_once_then_falls_back`: the fake returns `"oops"` twice → the result is `"coffee want"` (each sign's `english`, joined) and no exception. *(Review Focus 3)*
  - `test_timeout_falls_back`: the fake raises `anthropic.APITimeoutError` → fallback text. *(Review Focus 3)*
  - `@pytest.mark.live test_live_compose`: (skipped without a key) `["COFFEE","WANT"]`, yn, neutral → the text ends with "?" and contains "coffee".
- [ ] **Step 2:** Run `uv run pytest tests/test_compose.py -v -m "not live"`. Expected: FAIL. Implement. Run. Expected: PASS. Then run `-m live`. Expected: PASS.
- [ ] **Step 3:** Write `probe_cartesia.py`. It opens a Sonic TTS WebSocket and an Ink STT WebSocket using the current documented model IDs and message shapes, prints the event types received and the time to first audio, and saves nothing. Run it, then update the Global Constraints line with the verified model IDs.
- [ ] **Step 4:** Wire `Composer` into `main.py`.
- [ ] **Step 5: Commit** `feat(app): Claude English composer with fallback`.

### Task 10: Expressive voice

**Files:**
- Create: `app/server/voice.py`
- Test: `app/tests/test_voice.py`

**Interfaces:**
- Consumes: `Emotion`; verified Sonic API (Task 9 probe).
- Produces: `voice_controls(e: Emotion) -> dict` returning `{"emotion": list[str], "speed": float, "volume": float}` in Sonic's `generation_config` terms, and `Voice(api_key, voice_id)` implementing `speak(text, emotion) -> AsyncIterator[str]`. It streams PCM from the Sonic WebSocket into a `sounddevice.RawOutputStream` on the default output, yields `"first_audio"` when the first chunk is written, and `"done"` after the drain.
- Mapping (starting values, tuned by the listening test in Task 18). Intensity scales between the neutral and full values:
  - happy: brighter emotion, speed +, volume +
  - sad: sadness emotion, speed −, volume −
  - angry: anger emotion, speed +, volume ++
  - neutral: no emotion tag, speed 1.0, volume 1.0
- Default voice: one of Cartesia's documented most-expressive voices, set as `DEFAULT_VOICE_ID` in `voice.py`.

- [ ] **Step 1: Write failing tests:**
  - `test_neutral_is_plain`: neutral → speed 1.0, volume 1.0, no emotion tag.
  - `test_intensity_monotonic`: sad at 0.9 is slower and quieter than sad at 0.3.
  - `test_controls_within_api_bounds`: every label × intensity {0, 0.5, 1} stays inside the documented speed/volume ranges.
  - `test_socket_drop_raises_after_first_audio`: a fake socket closes mid-stream → `speak` yields `"first_audio"`, then raises `VoiceError`. *(Review Focus 5; Task 4 already pins the session behavior)*
- [ ] **Step 2:** Run. Expected: FAIL. Implement. Run. Expected: PASS.
- [ ] **Step 3:** Wire into `main.py`. Manual check: sign a sentence and hear it within ~2 s. Check that `turns.jsonl` `first_audio - hands_rest` ≤ 2000.
- [ ] **Step 4: Commit** `feat(app): Cartesia Sonic expressive voice`.

### Task 11: Facial grammar reader

**Files:**
- Create: `app/server/grammar.py`
- Test: `app/tests/test_grammar.py`

**Interfaces:**
- Consumes: `Frame` list for the turn, `Baseline.brow` (Task 7).
- Produces: `read_grammar(frames, baseline) -> Grammar`.
  - brow_up = mean over frames of (`browInnerUp` + mean(`browOuterUpLeft`, `browOuterUpRight`))/2 minus baseline.
  - brow_down = mean(`browDownLeft`, `browDownRight`) minus baseline.
  - Use only the middle 80% of the turn, which skips lead-in and lead-out.
  - `yn` if brow_up > `UP_T` (0.15); `wh` if brow_down > `DOWN_T` (0.15) and not yn; negation if head yaw changes direction ≥ 2 times with peak-to-peak ≥ 8° inside the turn.
  - Frames without `blend`/`head` are skipped; if < 30% of frames have a face → `{"question": None, "negation": False}`.
  - Thresholds tuned in Task 18.

- [ ] **Step 1: Write failing tests:**
  - `test_raised_brows_yn`.
  - `test_lowered_brows_wh`.
  - `test_headshake_negation`: yaw oscillating ±10° three times.
  - `test_slow_head_turn_not_negation`: one 20° turn → negation False.
  - `test_baseline_subtracted`: a user whose resting `browInnerUp` is 0.2 and holds 0.25 → not yn.
  - `test_no_face_defaults_statement`: all `blend=None` → `None`/False. *(Review Focus 1)*
- [ ] **Step 2:** Run. Expected: FAIL. Implement. Run. Expected: PASS.
- [ ] **Step 3: Commit** `feat(app): facial grammar reader`.

### Task 12: Emotion reader

**Files:**
- Create: `app/server/emotion.py`; modify `session.py` to add a `RealReaders` that runs `read_grammar` and `read_emotion` concurrently (`asyncio.to_thread`)
- Test: `app/tests/test_emotion.py`

**Interfaces:**
- Consumes: `Frame` list, `Baseline`.
- Produces: `read_emotion(frames, baseline) -> Emotion`, using only these blendshapes: `mouthSmileLeft/Right`, `mouthFrownLeft/Right`, `cheekSquintLeft/Right`, `eyeSquintLeft/Right`, `eyeWideLeft/Right`, `mouthPressLeft/Right`, `noseSneerLeft/Right`, `jawOpen`, `mouthPucker`. It also uses signing speed (mean wrist speed) and signing size (wrist bounding-box area), both relative to baseline.
  - Rule scoring: happy = smile + cheekSquint; sad = frown + low speed + small size; angry = sneer + press + eyeSquint + high speed + large size.
  - The highest score above `EMO_T` (0.2) wins, otherwise neutral. Intensity = that score clipped to 0..1.
  - < 30% face frames → neutral at 0.

- [ ] **Step 1: Write failing tests:**
  - `test_smile_is_happy`.
  - `test_frown_slow_small_is_sad`.
  - `test_sneer_fast_big_is_angry`.
  - `test_brows_ignored`: two turns identical except all `brow*` blendshapes set to 1.0 in one → identical `Emotion`. *(Anti-goal: questions read as emotion)*
  - `test_no_face_neutral`. *(Review Focus 1)*
- [ ] **Step 2:** Run. Expected: FAIL. Implement. Run. Expected: PASS.
- [ ] **Step 3:** Wire `RealReaders` into `main.py`. Manual check: sign the same question smiling, then frowning. The question mark stays, and the voice changes.
- [ ] **Step 4: Commit** `feat(app): emotion reader independent of brows`.

### Task 13: Speech to sign plan (Ink transcription and English → gloss)

**Files:**
- Create: `app/server/stt.py`, `app/server/gloss.py`; remove the dev text box from `ui.ts`
- Test: `app/tests/test_gloss.py`, `app/tests/test_stt.py`

**Interfaces:**
- Consumes: `Session.on_transcript` (Task 4), `Vocab` (Task 2), the verified Ink API (Task 9).
- Produces: `MicTranscriber(api_key, on_transcript, is_open: Callable[[], bool])`. It captures the default mic at 16 kHz mono PCM16 via `sounddevice` and streams to Ink only while `is_open()` (Session state `ready`/`listening`). It calls `on_transcript(text, final)` for partials and for the final at Ink's end-of-turn. Captions are sent as partials arrive.
- Produces: `Gloss(client, vocab)` implementing `async plan(english) -> SignPlanMsg` with Claude `claude-haiku-5-5`, timeout 4 s, and the vocabulary table in a cached system prompt.
- Prompt rules:
  - Produce ASL-order gloss using only listed signs.
  - Paraphrase into listed signs where meaning survives.
  - Fingerspell names and anything left as `spell` tokens (uppercase letters only).
  - Set `grammar` from sentence type (yn/wh/neg/statement) and `tone` from how the speaker sounds in words (warm/apologetic/urgent/neutral).
- `validate_plan(raw: dict, vocab) -> SignPlanMsg`: an unknown gloss raises `PlanError`. After one retry, unknown glosses become `spell` tokens of their English word. Spell words are stripped to A–Z, and empty ones are dropped.

- [ ] **Step 1: Write failing tests** (`test_gloss.py`, fake client):
  - `test_valid_plan_passes`.
  - `test_unknown_gloss_retried_then_spelled`: both attempts return `{"kind":"sign","gloss":"ESPRESSO"}` → the final token is `{"kind":"spell","word":"ESPRESSO"}`. *(Review Focus 3)*
  - `test_malformed_json_falls_back_to_spelling_words`: two bad replies → a spell token per word, grammar `statement`, tone `neutral`. *(Review Focus 3)*
  - `test_spell_sanitized`: `"Ana-Marie!"` → `"ANAMARIE"`.
  - `test_english_echoed_for_captions`: `plan.english == input`.
  - `@live test_live_plan_whats_size`: "What size would you like?" → grammar `wh`, and every sign token is in vocab.
- [ ] **Step 2: Write failing tests** (`test_stt.py`, fake socket and fake audio source):
  - `test_audio_not_sent_when_closed`: `is_open` False → 0 bytes sent. *(Review Focus 2)*
  - `test_final_transcript_forwarded_once`.
- [ ] **Step 3:** Run both. Expected: FAIL. Implement. Run. Expected: PASS (live tests with a key: PASS).
- [ ] **Step 4:** Wire both into `main.py`. Manual check: say "Sorry, we're out of oat milk." The caption appears, and the `sign_plan` has grammar `neg`, tone `apologetic`, and `OAT` spelled or paraphrased. Check that `turns.jsonl` `planned - speech_end` ≤ 1500.
- [ ] **Step 5: Commit** `feat(app): Ink transcription and validated English-to-sign plans`.

### Task 14: Sign motion library and avatar playback

**Files:**
- Create: `app/training/build_motion_library.py`, `app/web/src/avatar/vrm.ts`, `app/web/src/avatar/player.ts`, `app/web/public/avatar.vrm`
- Test: `app/web/src/avatar/player.test.ts`

**Interfaces:**
- Consumes: `video_to_frames` (Task 5), `Vocab` (Task 2), `SignPlanMsg` (Task 1).
- Produces: `web/public/signs/{GLOSS}.json` = `{"fps": 30, "frames": [{"pose": [[x,y,z]]×33 world landmarks, "left": [[x,y,z]]×21|null, "right": …}]}`, trimmed to the sign (rest-to-rest motion removed by wrist height) and smoothed (One-Euro or a 5-frame Savitzky–Golay filter).
- Produces: `vrm.ts`: `loadAvatar(canvas): Promise<Avatar>` and `Avatar.applyFrame(f: ClipFrame)` (Kalidokit `Pose.solve` + `Hand.solve` → VRM humanoid bone rotations; face is untouched here).
- Produces: `player.ts`: `buildTimeline(plan: SignPlanMsg, clips: Map<string, Clip>, letters: Map<string, Clip>, vocab: {gloss: string, avatar_ok: boolean}[]) -> TimelineItem[]` and `play(avatar, timeline, onDone)`.
  - Each `TimelineItem` is `{kind: "clip"|"caption-only", clip?, durationMs, gloss}`.
  - Neighbors get a 150 ms cross-fade (slerp of bone rotations), and there's a 300 ms ease from and to rest at the ends.
  - Signs with `avatar_ok: false` or a missing clip become `caption-only` items (a highlighted caption word, avatar holding neutral for 400 ms).
  - Calls `onDone` → client sends `avatar_done`.

- [ ] **Step 1:** Pick a CC0 or permissively licensed VRM (VRoid sample or similar), place it at `public/avatar.vrm`, and note its license in `app/web/public/README.md`.
- [ ] **Step 2:** Implement `build_motion_library.py`. For each active gloss, pick the ASL Citizen video tagged as the reference (or the clearest train video), extract with `video_to_frames`, trim, smooth and write JSON. Run it. Expected: one JSON per active gloss.
- [ ] **Step 3: Write failing vitest tests:**
  - `buildTimeline maps sign tokens to clips in order`.
  - `missing clip becomes caption-only`.
  - `avatar_ok false becomes caption-only`.
  - `spell token expands to one item per letter`.
  - `total duration includes blends and rest easing`.
- [ ] **Step 4:** Run `npx vitest run src/avatar/player.test.ts`. Expected: FAIL. Implement `player.ts` and `vrm.ts`. Run. Expected: PASS.
- [ ] **Step 5:** Wire into `main.ts`: on `sign_plan`, build and play the timeline, and show `english` as captions under the avatar.
- [ ] **Step 6: Avatar sign check (spec constraint).** Add a `?mode=review` view that plays each clip next to its ASL Citizen reference video. For each sign, decide whether it clearly matches; for each one that fails, set `avatar_ok: false` in `vocab.yaml`. Expected: fewer than 10% marked false. If more, hand-correct the worst clips' finger rotations, or re-extract from a different take.
- [ ] **Step 7: Commit** `feat(app): sign motion library and VRM playback`. Commit the clips only if the ASL Citizen license note from Task 5 allows private derived data in a private repo; otherwise gitignore `public/signs/`.

### Task 15: Fingerspelling both ways

**Files:**
- Create: `app/server/fingerspell.py`, `app/training/train_fingerspell.py`, `app/web/public/letters/*.json`; modify `session.py` (route `spell` segments)
- Test: `app/tests/test_fingerspell.py`

**Interfaces:**
- Consumes: `Segment(kind="spell")` (Task 7), the recorder (Task 8), `Clip` format (Task 14).
- Produces: `LetterNet` (an MLP on one hand's 63 wrist-relative coordinates, normalized by hand size, over an 8-frame window to capture J and Z motion). `SpellReader(model_path="models/letters.pt")` exposes `.read(seg) -> list[tuple[str, float]]` (top-3 words). Decoding: per-window letter probabilities → collapse repeats and drop windows with max p < 0.5 → a word. Double letters count only when a short dip in confidence or a small hand bounce separates them. The alternatives are the 2nd-best letter at the least confident position.
- Training data: our recordings via recorder `?mode=record&letters=1` (each person signs A–Z × 5 takes). The builder may add a public fingerspelling landmark dataset if its license allows and it uses the same MediaPipe hand layout; note the choice in `app/data/README.md`.
- Avatar letters: `public/letters/{A..Z}.json` in the Task 14 clip format, extracted from our own recorded letters (the best take per letter) with `build_motion_library.py --letters`.

- [ ] **Step 1: Write failing tests:**
  - `test_collapse_repeats`: per-window argmax `A A A N N A A` → `"ANA"`.
  - `test_low_confidence_windows_dropped`.
  - `test_double_letter_with_dip`: `L L (dip) L L` → `"LL"`.
  - `test_spell_segment_routed_to_spell_reader`: in `test_session.py`, a `spell` segment's word reaches the composer's `spelled` argument, not `signs`.
- [ ] **Step 2:** Run. Expected: FAIL. Implement. Run. Expected: PASS.
- [ ] **Step 3:** Record letters, train with `uv run python -m training.train_fingerspell`, and build the letter clips. Expected: per-letter accuracy on the holdout signer ≥ 85%.
- [ ] **Step 4:** Manual check: fingerspell a 4-letter name and hear it spoken. Say "My name is Priya" and watch the avatar spell P-R-I-Y-A.
- [ ] **Step 5: Commit** `feat(app): fingerspelling recognition and avatar letters`.

### Task 16: Avatar face (grammar and tone)

**Files:**
- Create: `app/web/src/avatar/face.ts`
- Test: `app/web/src/avatar/face.test.ts`

**Interfaces:**
- Consumes: `Avatar` (Task 14), `SignPlanMsg.grammar` and `.tone`.
- Produces: `faceTrack(grammar, tone, durationMs) -> (tMs: number) => FacePose`, where `FacePose = {browUp: number, browDown: number, headYawDeg: number, expressions: Record<string, number>}`, and `Avatar.applyFace(p: FacePose)`. Brows map to VRM expression or blendshape names available on the chosen model (custom `browUp`/`browDown` if the VRM lacks them; document the mapping in `face.ts`).
- Grammar holds across the whole sentence with 200 ms on/off ramps:
  - `yn` → browUp 0.8 and a slight head tilt forward.
  - `wh` → browDown 0.8.
  - `neg` → headshake ±12° at 2.5 Hz through the sentence.
  - `statement` → 0.
- Tone layers on the mouth, eyes and cheeks only, never the brows:
  - warm → `happy` 0.4
  - apologetic → `sad` 0.35 plus slightly lowered gaze
  - urgent → `surprised` 0.25 mouth/eyes only
  - neutral → 0

- [ ] **Step 1: Write failing vitest tests:**
  - `yn holds browUp through middle of sentence`.
  - `neg yaw crosses zero at least 4 times over 2 s`.
  - `tone never sets brow values`: for every tone with grammar `statement`, browUp == browDown == 0 at all t.
  - `ramps start and end at zero`.
- [ ] **Step 2:** Run `npx vitest run src/avatar/face.test.ts`. Expected: FAIL. Implement. Run. Expected: PASS.
- [ ] **Step 3:** Wire into `player.ts` playback (face sampled every frame alongside the body). Manual check: "What size?" shows lowered brows. "Sorry, we're out" shows a headshake with an apologetic face.
- [ ] **Step 4: Commit** `feat(app): avatar facial grammar and tone`.

### Task 17: Top-3 picker and signer UI polish

**Files:**
- Modify: `app/web/src/ui.ts`, `app/web/index.html`, `app/web/src/main.ts`
- Test: `app/web/src/ui.test.ts`

**Interfaces:**
- Consumes: `PickRequestMsg`, `YouSaidMsg`, `CaptionMsg`, `StateMsg`; `ws.send(PickMsg)`.
- Produces: one page with this layout: the avatar large, captions under it, the camera preview small in a corner, a "you said: … (emotion)" strip, a state badge, and a "Calibrate" button. Picker: on `pick_request`, show 3 large buttons (english + gloss), clickable and selectable with keys 1/2/3. Choosing sends `PickMsg` and hides the picker. A pending picker blocks no other display.

- [ ] **Step 1: Write failing vitest tests** (jsdom):
  - `pick_request renders three options`.
  - `key 2 sends pick for option 2`.
  - `you_said replaces previous text and shows emotion`.
  - Buttons have accessible names (`getByRole("button", {name: /coffee/i})`).
- [ ] **Step 2:** Run. Expected: FAIL. Implement. Run. Expected: PASS.
- [ ] **Step 3:** Manual check: sign deliberately sloppily. The picker appears, and pressing 1 speaks within 2 s of the tap.
- [ ] **Step 4: Commit** `feat(app): top-3 picker and signer UI`.

### Task 18: Evaluation, tuning and reliability

**Files:**
- Create: `app/training/eval.py`, `app/README.md` (setup, run, record, train, eval)
- Test: `app/tests/test_eval.py`

**Interfaces:**
- Consumes: everything above; `logs/turns.jsonl`; `data/ours/{holdout}/…`; `data/ours/*/_grid/`.
- Produces: `uv run python -m training.eval signs|grid|spell|latency`, each printing a table plus PASS/FAIL against the spec targets:
  - `signs`: ASL Citizen test top-1 ≥ 0.80; holdout signer top-1 ≥ 0.70 and top-3 ≥ 0.90.
  - `grid`: question-type accuracy ≥ 0.85 over all grid recordings, with a confusion table by emotion.
  - `spell`: holdout names (≤ 8 letters) fully correct ≥ 0.60.
  - `latency`: p50/p90/max of `first_audio - (picked or hands_rest)` and `avatar_start - speech_end`. PASS when max ≤ 2000 ms over the last 5 conversations.

- [ ] **Step 1: Write failing tests:**
  - `test_latency_uses_pick_time_when_present`: a log line with `picked` measures from `picked`.
  - `test_grid_accuracy_counts`: on a fixture of 4 labeled turns with known reader outputs → 0.75.
- [ ] **Step 2:** Run. Expected: FAIL. Implement. Run. Expected: PASS.
- [ ] **Step 3: Tune.** Grid-search `UP_T`, `DOWN_T` and the headshake limits on the grid recordings for ≥ 0.85, and `PAUSE_K`, `PAUSE_FRAMES` and `REST_FRAMES` on the holdout recordings. Set `PICK_THRESHOLD` so wrong top-1 signs below it reach the picker (target: no wrong sign is spoken without a pick on the holdout set). Commit the tuned constants.
- [ ] **Step 4: Listening test.** Generate 12 clips (4 emotions × 3 sentences) with `voice.py`, play them shuffled to 3+ people, and record the labels in `app/data/listening.csv`. Expected: ≥ 60% match. If not, widen the mapping in `voice_controls` and repeat.
- [ ] **Step 5: Reliability.** Run five full conversations as the spec defines them: ≥ 6 turns, both directions, including a yn question, a wh question, a negation, a fingerspelled name and two emotions. Use the spec's failure definition. Any failure resets the count. Run `eval latency` after. Expected: 5 in a row, and latency PASS.
- [ ] **Step 6:** Write `app/README.md`: prerequisites, `.env`, download/extract/train commands, run commands, recorder and review modes, eval commands, and the known limits (no fluent-signer review yet; 150 signs; good lighting needed).
- [ ] **Step 7: Commit** `feat(app): evaluation suite, tuned thresholds, README`.
