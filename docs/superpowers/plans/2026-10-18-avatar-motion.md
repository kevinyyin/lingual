# Avatar Motion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default avatar (Rocketbox `Male_Adult_11`) sign with human-like arms, hands in the right place, smooth transitions, and a head and shoulders that move.

**Architecture:** Measure first: a deterministic in-browser bench mode records motion metrics and avatar renders, plus a round-trip recognizer score, before anything changes. Then replace the arm-solving core in `rocketbox.ts` with two pure, unit-tested modules:
- `armSolver.ts`: a two-bone solve with human joint limits that continues from the previous pose.
- `rotationFilter.ts`: a time-based one-euro filter on the *solved* joint rotations.

Contact and collision corrections move onto the IK targets, before the solve and filter, so their changes reach the screen smoothly. Placement, head and shoulder motion, and arcs in the transitions follow as separate tasks.

**Tech Stack:** TypeScript, three.js 0.186, vitest 5 (jsdom), Vite 8; Python 3.12 with uv, Playwright (`uv run --with playwright`), MediaPipe (`training/landmarks.py`), and the ASL Citizen ST-GCN (`server/citizen_classifier.py`).

**Spec:** `docs/superpowers/specs/2026-10-18-avatar-motion-design.md`

## Global Constraints

- Default body: Rocketbox `Male_Adult_11`. MPFB is reachable as `?model=mpfb`, the other Rocketbox as `?model=Male_Adult_07`, VRoid as `?avatar=vrm`. `vrm.ts` must keep loading but isn't reworked.
- Human-like arm motion is the pass/fail bar. Smoothing must not blur signs, handshapes or holds, or move a sign out of place.
- Speed: motion still starts within one animation frame of `stage.play`, the whole turn stays within 2 s, and the avatar holds 60 fps on this MacBook.
- Idle head motion must never read as grammar. The grammar nod is 8° (pitch) and the headshake 12° at 2.5 Hz (`face.ts`), so idle motion stays ≤ 2° per axis and ≤ 0.3 Hz.
- Grammar on the brows and head (`face.ts`: `browUp` for yes/no, `browDown` for wh, the headshake for negation) must work unchanged.
- Joint limits in the new solver: elbow flexion 0–150° (never past straight), forearm twist within ±90° of the bind pose, and wrist swing ≤ 70° (the existing `WRIST_MAX`).
- Metric definitions, used by every comparison:
  - **Spike:** any arm bone (upper, fore, hand) turning more than 25° in world space between consecutive 60 fps frames.
  - **Backward elbow:** flexion below −5°.
  - **Twist out of range:** |forearm twist| > 90°.
  - **Penetration:** the existing `penetration()` and `armPenetration()`, in metres.
  - **Contact error:** for frames with `touchR`/`touchL` ≥ 0.5, the distance from the nearest hand probe to the avatar's mapped `touchAt` place, in metres.
- "Before" means the current code on `Male_Adult_11` with contact data restored, which is the state on disk.
- Clips are non-commercial research data. Never commit `web/public/signs/`, `data/` or rendered videos. The small JSON/markdown summaries under `app/reports/motion/` are committed.
- Tests: `cd app/web && npx vitest run` and `cd app && uv run pytest -q` stay green after every task.

## Review Focus

1. **Frame-rate changes** (a 120 Hz display, a throttled background tab): the filter is time-based, so 30 fps and 120 fps input give nearly the same pose. Pinned in Task 5.
2. **A fully straight arm**, where the elbow's bend plane is undefined: the solve must not flip the upper arm's roll when the arm passes through straight. Pinned in Task 4.
3. **Unreachable or degenerate targets** (a wrist target beyond reach, at the shoulder itself, or inside the torso): no NaN and no snap, the arm reaches as far as it can. Pinned in Task 4.
4. **The first frame after load, and a hand that appears or leaves mid-plan:** the filter starts at the first solved pose (no sweep from the bind pose), and an arm leaving for rest eases rather than snaps. Pinned in Tasks 5 and 6.
5. **Arms crossing** (two-handed signs where the wrists cross, 8 clips have this for >20% of frames): the arms must not pass through each other. Arm-to-arm distance is measured in Task 1 and gated in Task 6.

---

### Task 1: Bench mode and motion metrics

**Files:**
- Create: `app/web/src/bench.ts`
- Create: `app/web/src/avatar/metrics.ts`, `app/web/src/avatar/metrics.test.ts`
- Modify: `app/web/src/main.ts` (route `?mode=bench`), `app/web/src/avatar/stage.ts` (export `prepare`), `app/web/src/avatar/vrm.ts` (`Avatar` interface), `app/web/src/avatar/rocketbox.ts` (`armState()`, `preserveDrawingBuffer` in bench mode)
- Create: `app/scripts/motion_bench.py`

**Interfaces:**
- Produces:
  - `prepare(avatar: Avatar, plan: SignPlan): Promise<{ timeline: TimelineItem[]; face: (t: number) => FacePose; rest: ClipFrame; durationMs: number }>` in `stage.ts`. `play` is rewritten to use it, with no change in behaviour.
  - `Avatar.armState?(): Record<"L" | "R", ArmState>` where `ArmState = { upper: THREE.Quaternion; fore: THREE.Quaternion; hand: THREE.Quaternion; elbowDeg: number; twistDeg: number; wrist: THREE.Vector3; elbow: THREE.Vector3 }`. Quaternions are world-space. `elbowDeg` is flexion: 0 = straight, positive = bent the human way, negative = backward. `twistDeg` is the forearm's twist about its own axis relative to the bind pose.
  - `Avatar.applyFrame(f: ClipFrame, nowMs?: number): void`. `nowMs` defaults to `performance.now()`.
  - `frameMetrics(prev: Record<"L"|"R", ArmState> | null, cur: Record<"L"|"R", ArmState>): { maxStepDeg: number; backwardElbow: boolean; twistOut: boolean; armGap: number }` in `metrics.ts`, where `armGap` is the minimum distance (m) between the two forearm segments.
  - `window.bench.run(glosses: string[], opts: { fps: number; capture: boolean }): Promise<BenchResult>` in `bench.ts`. It steps the timeline at fixed time steps (`t = i * 1000 / fps`), calling `applyFrame(frame, t)` and `applyFace`, and waits one `requestAnimationFrame` per step so the frame renders. `BenchResult = { frames: number; p95FrameMs: number; spikes: number; maxStepDeg: number; p99StepDeg: number; backwardElbow: number; twistOut: number; maxPenetration: number; maxArmPenetration: number; minArmGap: number; contactErrorMean: number | null; images?: string[] }`. `images` are JPEG data URLs, filled only when `capture`. `p95FrameMs` is the 95th-percentile wall time of `applyFrame` + `applyFace` per step, the 60 fps guard. `contactErrorMean` also covers items given an explicit target: a bench item may be `{ glosses: string[]; at?: { hand: "L" | "R"; place: number[] } }`, where `place` is a point in pose world coordinates, measured instead of the clip's `touchAt`. `contactError?(f: ClipFrame, at?: { hand: "L" | "R"; place: number[] }): number | null` on `Avatar` takes it.
  - `uv run --with playwright python scripts/motion_bench.py --label NAME [--model Male_Adult_11]` writes `app/reports/motion/NAME.json` (one `BenchResult` without `images` per test item, plus totals) and `NAME.md` (a table).

- [ ] **Step 1: Write the failing tests** in `metrics.test.ts`:
  - `frameMetrics reports the largest per-bone turn`: identical states → `maxStepDeg` 0; the `fore` of R turned 30° → `maxStepDeg` ≈ 30 (±0.01).
  - `backward elbow flagged below -5 degrees`: `elbowDeg: -6` → true; `-4` → false.
  - `twist out of range beyond 90 degrees`: `twistDeg: 91` → true; `-89` → false.
  - `armGap is the forearm segment distance`: parallel forearms 0.2 m apart → 0.2 (±1e-6); crossing segments → 0.
- [ ] **Step 2: Run** `cd app/web && npx vitest run src/avatar/metrics.test.ts`. Expected: FAIL (module missing).
- [ ] **Step 3: Implement** `frameMetrics` in `metrics.ts`, using segment–segment closest distance between elbow→wrist pairs for `armGap`.
- [ ] **Step 4: Run** the same command. Expected: PASS.
- [ ] **Step 5: Implement** `prepare` and refactor `play` onto it, then `armState()` and the `nowMs` parameter in `rocketbox.ts`:
  - Flexion is the angle between the upper-arm and forearm directions, signed by whether the forearm lies on the human side of the hinge axis. The hinge is the bind-pose elbow axis carried by the upper arm.
  - Twist is `twistAbout` of the forearm's local rotation (relative to bind) about `arm.fore.axis`.
  - Then `bench.ts` and the `?mode=bench` route.
  - Create the renderer with `preserveDrawingBuffer: true` only when `mode=bench`.
  - Contact error uses the existing `handProbes` and `toAvatar` mapping; expose it from `rocketbox.ts` as `contactError?(f: ClipFrame): number | null` on `Avatar`.
- [ ] **Step 6: Write** `scripts/motion_bench.py`, modelled on `scripts/record_demo.py` (Playwright, headed Chromium, `BASE = "http://localhost:5173"`). Test items, fixed in the script:
  - Single signs: `GRANDMOTHER, MOTHER, FATHER, MY, PLEASE, HELP, MEET, KING, REMEMBER, WHERE, HELLO, THANK-YOU`.
  - Sentences, as gloss lists: `MY GRANDMOTHER TEACH I COOK`, `HELLO NICE MEET YOU`, `YOU HELP I FIND MY KEY`, `TOMORROW I GO STORE BUY APPLE`, `LIVE OR DIE WHICH`, `I SORRY I UNDERSTAND NOT`, `TRAIN LEAVE WHAT TIME`, `YOUR CAR NEW`, `FATHER MOTHER WORK WHERE`, `NOW I NEED SLEEP`.
  - Run at 60 fps, without capture.
  - GRANDMOTHER is also run as `{ glosses: ["GRANDMOTHER"], at: { hand: "R", place: chin } }`, where `chin` is the clip's own nose (`pose[0]`) moved 0.07 m down (+y in MediaPipe world), since ASL-LEX codes no contact for it.
  - `--capture` also saves each item's frames as `data/eval/motion/NAME/<item>.mp4` (30 fps) for the side-by-sides.
- [ ] **Step 7: Run** the app (preview `asl-server` and `asl-web`), then `cd app && uv run --with playwright python scripts/motion_bench.py --label smoke`. Expected: `reports/motion/smoke.md` exists with one row per item and nonzero `frames`.
- [ ] **Step 8: Run** `npx vitest run` and `uv run pytest -q`. Expected: all pass.
- [ ] **Step 9: Commit**

```bash
git add app/web/src app/scripts/motion_bench.py
git commit -m "feat(avatar): bench mode and motion metrics for before/after comparison"
```

### Task 2: Round-trip recognizer harness

**Files:**
- Create: `app/scripts/roundtrip.py`, `app/tests/test_roundtrip.py`

**Interfaces:**
- Consumes: `window.bench.run(glosses, {fps: 30, capture: true})` (Task 1); `training.landmarks.video_to_frames(path)`; `CitizenSignPredictor(vocab, restrict_to_vocab=True).topk(frames, 5)`; `server.vocab.load_vocab()`.
- Produces:
  - `score(rows: list[tuple[str, list[str]]]) -> dict[str, float]`: rows of (true gloss, top-5 predictions), returning `{"top1", "top3", "top5", "n"}`.
  - `uv run --with playwright python scripts/roundtrip.py --label NAME [--model Male_Adult_11] [--n 150]` renders each sign alone, writes an mp4 per sign under `data/eval/roundtrip/NAME/` (cv2 `VideoWriter`, 30 fps), reads it back, and writes `app/reports/motion/roundtrip_NAME.json`.
  - Sign list: the first `n` curated (`vocab.yaml`) signs, sorted by gloss, that have a clip in `web/public/signs/` and `avatar_ok: true`.

- [ ] **Step 1: Write the failing test** `test_score_counts_topk`. With rows `[("A", ["A","B","C","D","E"]), ("B", ["A","C","B","D","E"]), ("C", ["A","B","D","E","F"])]`, `score` returns `top1 == 1/3`, `top3 == 2/3`, `top5 == 2/3` and `n == 3`.
- [ ] **Step 2: Run** `cd app && uv run pytest tests/test_roundtrip.py -q`. Expected: FAIL (import error).
- [ ] **Step 3: Implement** `score` and the script. Decode the data URLs with `base64` and `cv2.imdecode`.
- [ ] **Step 4: Run** the test. Expected: PASS.
- [ ] **Step 5: Smoke run:** `uv run --with playwright python scripts/roundtrip.py --label smoke --n 10`. Expected: the JSON has `n == 10` and the scores are between 0 and 1.
- [ ] **Step 6: Commit**

```bash
git add app/scripts/roundtrip.py app/tests/test_roundtrip.py
git commit -m "feat(eval): round-trip recognizer harness for avatar renders"
```

### Task 3: Default body, setup docs, placement audit, baseline

**Files:**
- Modify: `app/web/src/avatar/rocketbox.ts:15-23` (default character)
- Modify: `app/README.md` (setup and avatar row)
- Create: `app/training/audit_placement.py`, `app/tests/test_audit_placement.py`

**Interfaces:**
- Produces:
  - `CHARACTER` defaults to `"Male_Adult_11"`. `?model=mpfb` selects MPFB; `?model=Male_Adult_07` stays as it is.
  - `aspect_from_pose(pose: list, pose2d: list) -> float`: the image width/height ratio estimated from one frame, `W/H = (|Δy2d| / |Δyw|) / (|Δx2d| / |Δxw|)`. Δx is the shoulder width (points 11→12) and Δy the shoulder-midpoint-to-hip-midpoint height (11,12 → 23,24), in world (w) and image (2d) coordinates. Use the median over the clip.
  - `placement_error(frame, aspect) -> dict[str, float]`: per wrist (15, 16), the across/up offset in shoulder widths between the world-landmark wrist relative to the nose and the aspect-corrected 2D wrist relative to the nose.
  - `uv run python -m training.audit_placement` writes `app/reports/motion/placement_audit.json`. It holds the share of clips whose worst frame with a wrist within 1.3 shoulder widths of the nose (`FACE_FAR` in `rocketbox.ts`) has error > 0.3 shoulder widths, and the 30 worst glosses.

- [ ] **Step 1: Write the failing tests:**
  - `test_aspect_from_pose_recovers_ratio`: build a synthetic pose (world shoulders 0.4 m apart, torso 0.5 m) and project it with W/H = 16/9 to normalized 2D. Expect `aspect_from_pose` ≈ 1.778 (±0.01).
  - `test_placement_error_zero_when_consistent`: the same synthetic frame → both errors < 1e-6.
- [ ] **Step 2: Run** `uv run pytest tests/test_audit_placement.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `audit_placement.py`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Switch the default character.** In README Setup, after `training.clips_from_wlasl`, add `uv run python -m training.touch_from_asllex   # contact data; rerun after every clip rebuild`. In the Models table, make the Avatar row say Rocketbox `Male_Adult_11` is the default and `?model=mpfb` loads MPFB.
- [ ] **Step 6: Record the baselines.** Run `uv run python -m training.audit_placement`, then `motion_bench.py --label before`, then `roundtrip.py --label before --n 150`. Note in `app/reports/motion/README.md`: the share of clips with face-area error > 0.3 shoulder widths (this decides Task 7's path), the `before` spike count, the round-trip top-1/3/5, and the commit hash measured.
- [ ] **Step 7: Run** `npx vitest run` and `uv run pytest -q`. Expected: PASS.
- [ ] **Step 8: Commit**

```bash
git add app/web/src/avatar/rocketbox.ts app/README.md app/training/audit_placement.py app/tests/test_audit_placement.py app/reports/motion
git commit -m "feat(avatar): Rocketbox Male_Adult_11 by default; placement audit and before-baselines"
```

### Task 4: Arm solver with joint limits and continuity

**Files:**
- Create: `app/web/src/avatar/armSolver.ts`, `app/web/src/avatar/armSolver.test.ts`

**Interfaces:**
- Consumes: `joint`, `parentWorld`, `twistAbout`, `Joint` from `rig.ts`.
- Produces:
  - `interface ArmRig { upper: Joint; fore: Joint; hand: Joint; l1: number; l2: number; hinge: THREE.Vector3 }`. `hinge` is the elbow's bend axis in the upper arm's bind-local frame, measured from the bind pose: the cross product of the upper-arm and forearm directions, falling back to the axis whose positive rotation brings the forearm toward the chest.
  - `interface ArmLimits { elbowMaxDeg: number; twistMaxDeg: number; wristMaxDeg: number }` and `DEFAULT_LIMITS = { elbowMaxDeg: 150, twistMaxDeg: 90, wristMaxDeg: 70 }`.
  - `interface ArmTarget { shoulder: THREE.Vector3; elbowHint: THREE.Vector3; wrist: THREE.Vector3; handRot: THREE.Quaternion | null }`. All are world-space. `handRot` is the hand's world rotation from the finger landmarks, or null to keep the forearm's.
  - `interface ArmPose { upper: THREE.Quaternion; fore: THREE.Quaternion; hand: THREE.Quaternion; pole: THREE.Vector3 }`: local rotations, plus the bend-plane pole that is carried to the next frame.
  - `solveArmPose(rig: ArmRig, t: ArmTarget, prev: ArmPose | null, lim?: ArmLimits): ArmPose`. It applies nothing to the bones; the caller copies the quaternions.
- Algorithm (not determined by the tests alone):
  1. The pole is `elbowHint` projected off the shoulder→wrist line. When that projection is under 1 cm, or flips by more than 90° from `prev.pole`, use `prev.pole` (or straight down when there's no `prev`).
  2. Elbow position from `twoBoneElbow` with that pole.
  3. Upper-arm rotation = swing to the elbow direction, *with roll chosen so the hinge axis is perpendicular to the bend plane* (this fixes the roll that `aimLocal` leaves free).
  4. Forearm = rotation about the hinge by the flexion, clamped to [0, `elbowMaxDeg`].
  5. Hand: the swing part relative to the forearm is clamped to `wristMaxDeg`. The twist part goes to the forearm, clamped to ±`twistMaxDeg` and unwrapped toward the previous twist so it never jumps by 360°.
  6. Keep each output quaternion in the same hemisphere as the previous one (`dot < 0` → negate).

- [ ] **Step 1: Write the failing tests** in `armSolver.test.ts`, on the synthetic arm from `rig.test.ts` (copy its `arm()` helper; add a hand child):
  - `reaches a target within reach`: the wrist ends within 1 mm of the target for 50 seeded random targets inside 0.9·(l1+l2).
  - `never bends the elbow backward`: for the same 50 targets with random `elbowHint`s, flexion lies in [0°, 150°].
  - `forearm twist stays within ±90°` for random `handRot`s.
  - `continuous through a straight arm`: sweep the wrist target along a line through full extension in 100 steps, carrying `prev`. Every bone turns less than 5° per step.
  - `no NaN for degenerate targets`: a wrist target equal to the shoulder, one at 3·(l1+l2) away, and an `elbowHint` equal to the shoulder all give finite quaternions.
  - `small target change, small pose change`: nudging the target 1 mm turns no bone more than 1°.
- [ ] **Step 2: Run** `npx vitest run src/avatar/armSolver.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** `armSolver.ts` per the algorithm above.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add app/web/src/avatar/armSolver.ts app/web/src/avatar/armSolver.test.ts
git commit -m "feat(avatar): arm solver with human joint limits and frame-to-frame continuity"
```

### Task 5: Time-based rotation filter

**Files:**
- Create: `app/web/src/avatar/rotationFilter.ts`, `app/web/src/avatar/rotationFilter.test.ts`

**Interfaces:**
- Produces:
  - `class QuatFilter { constructor(minCutoffHz?: number, beta?: number, dCutoffHz?: number); filter(q: THREE.Quaternion, tMs: number): THREE.Quaternion; reset(): void }`.
  - It is a one-euro filter on the rotation: speed is the angular velocity (rad/s) from the previous filtered value, the cutoff is `minCutoff + beta·|speed|`, and the output is a slerp from the previous output by `alpha(cutoff, dt)`.
  - The first call (or the first after `reset`) returns `q` unchanged. A `dt` ≤ 0 returns the previous output.
  - Starting values are `minCutoffHz = 1.5`, `beta = 0.5`, `dCutoffHz = 1.0`. Task 6 tunes them against the bench.

- [ ] **Step 1: Write the failing tests:**
  - `first sample passes through`.
  - `holds a constant input`: 100 frames of the same q → output within 1e-6 of q.
  - `frame-rate independent`: a 60° step input sampled at 30 fps and at 120 fps; at t = 200 ms the two outputs differ by less than 3°.
  - `no overshoot on a step`: the angle to the target decreases monotonically.
  - `fast motion has little lag`: rotate the input at 360°/s for 500 ms at 60 fps; the final lag is under 15°.
  - `jitter is reduced`: ±2° random noise around a fixed q at 60 fps; the output RMS error is less than half the input's.
- [ ] **Step 2: Run** `npx vitest run src/avatar/rotationFilter.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** `QuatFilter`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add app/web/src/avatar/rotationFilter.ts app/web/src/avatar/rotationFilter.test.ts
git commit -m "feat(avatar): time-based one-euro filter for joint rotations"
```

### Task 6: Wire the new core into the Rocketbox avatar (then checkpoint)

**Files:**
- Modify: `app/web/src/avatar/rocketbox.ts` (`makeArm`, `solveArm`, `placeArm`, `applyFrame`; delete the input `SMOOTH` lerp and the `JOINTS` flag paths it replaces)

**Interfaces:**
- Consumes: `solveArmPose`, `ArmRig`, `ArmPose`, `DEFAULT_LIMITS` (Task 4); `QuatFilter` (Task 5); `prepare`, `armState`, `nowMs` (Task 1).
- Produces: unchanged `Avatar` behaviour for callers. `?nocollide` and `?notouch` keep working, and `?smooth=` is removed.
- New per-arm pipeline, per `applyFrame(f, nowMs)`:
  1. Targets: `armTargets` → idle-to-rest ease (keep `IDLE_EASE`, but time-based: `1 - exp(-dt/150ms)`) → `placeShift` and `contactGap` applied to the wrist **target** → the collision push applied to the wrist and elbow **targets** (`body.push(target, MARGIN)`; elbow `MARGIN*2`) → the clavicle lift as today.
  2. `solveArmPose(rig, target, prevPose)`.
  3. Each of upper, fore and hand goes through its own `QuatFilter` keyed by arm and bone, then is copied onto the bone. The fingers keep `aimLocal`, but their rotations are filtered too (one `QuatFilter` per finger joint).
  4. One final safety pass. If `penetration()` > 5 mm after filtering, move the wrist target out by the push and re-solve **once**. Its result also goes through the filters, never straight to the bones.

- Also produces, in `armSolver.ts`: `stepArm(rig: ArmRig, filters: { upper: QuatFilter; fore: QuatFilter; hand: QuatFilter }, t: ArmTarget, prev: ArmPose | null, tMs: number, lim?: ArmLimits): ArmPose`. It is `solveArmPose` followed by the three filters, and returns the filtered pose, which is what is carried as `prev` and copied onto the bones. `rocketbox.ts` calls it once per arm per frame.

- [ ] **Step 1: Write the failing tests** in `armSolver.test.ts`:
  - `stepArm eases a hand appearing mid-plan`: switch the target from a rest wrist to a raised wrist in one frame, stepping at 60 fps. No bone turns more than 25° between consecutive frames, and the wrist is within 1 cm of the raised target by 400 ms.
  - `stepArm first frame is unfiltered`: the first call returns the same pose as `solveArmPose`.
- [ ] **Step 2: Run** `npx vitest run src/avatar/armSolver.test.ts`. Expected: FAIL (`stepArm` not defined).
- [ ] **Step 3: Implement** `stepArm`, then rewire `rocketbox.ts` onto it per the pipeline above.
- [ ] **Step 4: Run** `npx vitest run`. Expected: PASS.
- [ ] **Step 5: Measure:** `motion_bench.py --label core`. Expected, against `before`:
  - `spikes` = 0, `backwardElbow` = 0, `twistOut` = 0.
  - `maxPenetration` and `maxArmPenetration` no worse.
  - `minArmGap` > 0 on the two-handed items.
  - `contactErrorMean` no worse.
  - `p95FrameMs` ≤ 4 ms.
- [ ] **Step 6: Tune** `QuatFilter` (`minCutoffHz`, `beta`) if any item fails Step 5, or if holds visibly smear in `?mode=review` (the clip beside its source video). Record the final values and the reason in a code comment.
- [ ] **Step 7: Run** `roundtrip.py --label core --n 150`. Expected: top-1 and top-3 ≥ `before`.
- [ ] **Step 8: Commit**

```bash
git add app/web/src/avatar app/reports/motion
git commit -m "feat(avatar): new arm core: limits, continuity, filtered rotations, corrections on targets"
```

- [ ] **Step 9: CHECKPOINT with the user.** Record before/after side-by-side videos of the 10 test sentences with `motion_bench.py --capture` (or `record_demo.py`), and show the user. Continue only on their go-ahead; their judgement of "human-like" is the gate.

### Task 7: Hand placement near the face

**Files:**
- Modify: `app/web/src/avatar/rocketbox.ts` (`toAvatar`)
- Test: `app/web/src/avatar/placement.test.ts`; create `app/web/src/avatar/placement.ts`

**Interfaces:**
- Consumes: `app/reports/motion/placement_audit.json` (Task 3).
- Produces: `wristFrom2d(pose: number[][], pose2d: number[][], wrist: 15 | 16, aspect: number): THREE.Vector3`. Across/up come from the aspect-corrected 2D landmarks scaled by world shoulder width over 2D shoulder width; depth comes from the world landmark. `estimateAspect(frames: ClipFrame[]): number` uses the same formula as `training/audit_placement.aspect_from_pose`, median over frames, and is cached per clip.

The path depends on the audit:
- **If ≤ 20% of clips** exceed 0.3 shoulder widths, do the runtime fix: in `toAvatar`, within the face zone (`FACE_NEAR`–`FACE_FAR` weighting as today), use `wristFrom2d` for across/up.
- **If > 20%**, stop and tell the user. Re-extracting the clips is the spec's fallback, and it costs a full rebuild of `clips_from_wlasl`, so it needs their go-ahead.

- [ ] **Step 1: Write the failing tests:**
  - `wristFrom2d matches world when consistent`: build a world pose (shoulders at x ±0.2 m, hips 0.5 m below, nose 0.25 m above the shoulders, right wrist at (−0.05, −0.55, −0.3)), and derive `pose2d` by orthographic projection to a 16:9 image (x' = 0.5 + x/1.6, y' = 0.5 + y/0.9). The result is within 1 mm of the world wrist.
  - `estimateAspect recovers 16:9` within 0.01.
- [ ] **Step 2: Run** `npx vitest run src/avatar/placement.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** `placement.ts` and use it in `toAvatar`.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Measure:** `motion_bench.py --label placement`. Expected: `contactErrorMean` lower than `core`, and the GRANDMOTHER chin item (Task 1) has a minimum error ≤ 0.03 m.
- [ ] **Step 6: Commit**

```bash
git add app/web/src/avatar/placement.ts app/web/src/avatar/placement.test.ts app/web/src/avatar/rocketbox.ts app/reports/motion
git commit -m "fix(avatar): place hands near the face from image landmarks"
```

### Task 8: Head and shoulder motion

**Files:**
- Create: `app/web/src/avatar/headMotion.ts`, `app/web/src/avatar/headMotion.test.ts`
- Modify: `app/web/src/avatar/rocketbox.ts` (`applyFace` adds the head motion; the clavicle protraction in the target step)

**Interfaces:**
- Produces:
  - `headMotion(tMs: number, wristL: THREE.Vector3 | null, wristR: THREE.Vector3 | null, nose: THREE.Vector3): { yawDeg: number; pitchDeg: number; rollDeg: number }`. The head turns toward the more active (higher, further forward) wrist by up to 4° yaw and 3° pitch, eased by its own `QuatFilter`. The idle drift is ≤ 1.5° per axis, made of sines at 0.11, 0.17 and 0.23 Hz.
  - `applyFace` adds this to the grammar head pose. `HEAD_KEEP` stays.
  - Clavicle protraction: forward rotation proportional to how far the wrist is in front of the chest, up to 8°, plus the existing lift.

- [ ] **Step 1: Write the failing tests:**
  - `idle drift stays under 2 degrees and 0.3 Hz`: sample 20 s at 60 fps with both wrists null. Every |angle| is ≤ 2°, and an FFT or zero-crossing count shows no component above 0.3 Hz.
  - `turns toward the working hand`: right wrist raised in front, left at rest → yaw has the sign of the right side and |yaw| ≤ 4°.
  - `never looks like a nod or shake`: the peak-to-peak pitch over any 1 s window is < 4° (the grammar nod is 8°).
- [ ] **Step 2: Run** `npx vitest run src/avatar/headMotion.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** `headMotion.ts` and wire it into `applyFace` and the clavicle step.
- [ ] **Step 4: Run** `npx vitest run` (including `face.test.ts`). Expected: PASS.
- [ ] **Step 5: Measure:** `motion_bench.py --label head`. Expected: no regression against `placement`, and `spikes` still 0.
- [ ] **Step 6: Commit**

```bash
git add app/web/src/avatar
git commit -m "feat(avatar): natural head and shoulder motion that never reads as grammar"
```

### Task 9: Arcs between signs

**Files:**
- Modify: `app/web/src/avatar/player.ts` (`lerpFrame`)
- Test: `app/web/src/avatar/player.test.ts`

**Interfaces:**
- Produces: `lerpFrame(a, b, w, ease?)` bows the wrist and hand points of both arms forward (toward the viewer: −z in MediaPipe world) by `ARC_K · d · sin(π·e)`, where `d` is that wrist's straight-line travel (m), `e` the eased weight, and `ARC_K = 0.25`. The end frames are unchanged (sin 0 = sin π = 0). Export `ARC_K`.

- [ ] **Step 1: Write the failing tests:**
  - `blend endpoints unchanged`: w = 0 and w = 1 return exactly a and b.
  - `blend midpoint bows forward`: two frames whose right wrist moves 0.4 m across. At w = 0.5 its z is lower than the straight-line midpoint by 0.25·0.4 (±1e-6).
  - `no bow when the hand doesn't move`.
- [ ] **Step 2: Run** `npx vitest run src/avatar/player.test.ts`. Expected: FAIL.
- [ ] **Step 3: Implement** the bow in `lerpFrame`.
- [ ] **Step 4: Run** `npx vitest run`. Expected: PASS.
- [ ] **Step 5: Measure:** `motion_bench.py --label arcs`. Expected: `maxPenetration` ≤ `head`, and `spikes` still 0.
- [ ] **Step 6: Commit**

```bash
git add app/web/src/avatar/player.ts app/web/src/avatar/player.test.ts app/reports/motion
git commit -m "feat(avatar): sign-to-sign transitions follow forward arcs"
```

### Task 10: Final comparison and record

**Files:**
- Modify: `app/reports/motion/README.md`, `docs/superpowers/plans/2026-10-16-two-way-asl-mvp-record.md` (a new section, "Avatar motion rebuild (2026-10-18)")

- [ ] **Step 1: Run** `motion_bench.py --label after` and `roundtrip.py --label after --n 150`.
- [ ] **Step 2: Write** the before/after table in `reports/motion/README.md`: spikes, backward elbows, twist out of range, penetrations, minimum arm gap, contact error, round-trip top-1/3/5. Every number comes from the JSON files.
- [ ] **Step 3: Add** the record section: the rulings (filter values, joint limits, the placement path taken and why) with the same numbers.
- [ ] **Step 4: Run** `npx vitest run` and `uv run pytest -q`. Expected: PASS.
- [ ] **Step 5: Commit**

```bash
git add app/reports/motion docs/superpowers/plans/2026-10-16-two-way-asl-mvp-record.md
git commit -m "docs: avatar motion rebuild results"
```

- [ ] **Step 6: Show the user** the final before/after side-by-side of the 10 test sentences. Their judgement of "human-like" is the acceptance gate.
