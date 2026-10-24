# Avatar readability: clean sign edges, steadier motion, a face for tone

Date: 2026-10-21. Status: design approved in conversation, written spec awaiting review. Branch: `avatar-readability`
(from `ios-plushie`, which carries the open review-fixes PR #4).

## Intent

The user reports three things about the toon avatar:

- the arm movement needs smoothing;
- the signs aren't distinct: they blur together in sentences, each sign is mushy, and different signs look alike (the
  user picked all three);
- the face should change with the tone: at least a smile, a frown or neutral.

They watch it on the phone (teaching mode, in the plushie) and on the laptop (the translator) about equally. A demo is
**a day or two away**, so this round favours fast, safe changes; deeper rework waits.

The viewer is a Deaf signer (the 2026-10-18 avatar motion spec), so smooth must not mean mushy: every change here has
to make signs crisper as well as calmer.

## Decided (with the user, 2026-10-21)

- **Approach A**, a quick and safe tune-up, for the demo. Approach B (work out every sign's joint angles offline and
  play them back) and approach C (re-track the source videos or choose clearer signers) are later options.
- **Part 1:** cut each sign clip down to the sign itself and hold its first and last pose.
- **Part 2:** clean each clip once as it loads, with smoothing that looks both ways in time; then keep the lightest
  live smoothing that tests best.
- **Part 3:** tone at full strength in the translator; feeling signs get their face in teaching mode; the tone
  **relaxes after each sentence** rather than being held until the next one (in ASL the face moves with the signs, and a
  held smile would carry into the next exercise in teaching mode).
- **Part 4:** measurements before and after, the user's side-by-side on both screens, and an `?oldmotion` switch.

## What we measured

Toon avatar, current code (`ios-plushie` at `0cba5c5`), stepped at 60 fps in the browser over the motion bench's 12
single signs and 10 sentences, plus 250 random sign clips read directly.

- **Sign edges.** Within 40 ms of a clip boundary, the avatar's wrist moves at a median 11.6 mm per frame against 5.2
  elsewhere, and its frame-to-frame change in velocity (jerkiness) is 2.9 mm against 1.1. The hand should be nearly
  still where a sign starts and ends; this is the stop-and-go hitch, and it also blurs one sign into the next.
- **Why.** `player.ts` trims a fixed 12% from each end of every clip (`COART_TRIM`). But the sign's first hold (the
  first moment the hand slows after it has risen 0.25 m above the hips) sits a median 23% into the clip (p10 14%,
  p90 36%), and its last hold 26% before the end. The fixed trim cuts past those holds in only 2-3% of clips, so about
  0.2 s of the recorded signer's reach stays at each end: the blend eases the hand to a stop at the clip's first frame,
  then the clip moves it on again. (Median clip length 56 frames at 30 fps.)
- **Wobble in the recordings.** The recorded wrist moves a median 1.0 cm per frame, with 0.41 cm of frame-to-frame
  wobble; fingertips wobble 2.9 mm. One-frame glitches (a point that jumps out and back) are rare: none for wrists and
  one for fingertips in 250 clips, so no glitch filter is needed.
- **Snaps.** Frames where a joint turns at the 20°-per-frame speed limit: 2 of 1,711 for single signs, 10 of 5,248 for
  sentences, all fast wrist or forearm turns in signs that flip the hand (BUY, DIE, NEW, WHAT, GRANDMOTHER). Rare, and
  left alone this round.
- **Distinctness.** The round-trip test renders the avatar and reads each sign back with the ASL Citizen recognizer
  (`scripts/roundtrip.py`, 150 signs). At the end of the 2026-10-18 motion work, on the Rocketbox body, it scored 8.7%
  top-1 and 14.0% top-3, and the live filtering cost about 1-2 points. It has not been measured on the toon; the plan
  re-measures the baseline first.
- **Face.** `face.ts` sets tone on the mouth and eyes at 25-40%: warm gives a 0.4 smile and 0.24 cheek lift, apologetic
  a 0.35 frown and 0.15 lowered eyes, urgent 0.125 jaw open and 0.25 wide eyes. The toon's smile lifts the mouth corners
  about 9 mm at full weight, so about 3.6 mm today: hard to see, above all on the phone. Teaching mode always sends tone
  "neutral" (`teach/signer.ts`, `playTokens`).

## Design

### Part 1: clean edges on every sign (`app/web/src/avatar/player.ts`)

- New `signSpan(frames)` returns the frames of the sign itself:
  - The active wrist is the one (15 or 16) whose highest point over the clip rises furthest above the hips (hip
    midpoint y minus wrist y, world landmarks, y down).
  - The hand is up when it has risen more than 0.25 m (where signing hands start; see the note on `IDLE_RISE_M`).
  - Wrist speed per frame, averaged over 3 frames. A frame is a local minimum when its speed is no higher than the
    frames on either side and lower than at least one of them (so a stretch of steady speed doesn't count).
  - Start: the first local minimum at or after the first up frame. End: the last local minimum at or before the last
    up frame.
  - Fall back to today's 12% trim when no frame is up, no minimum is found, the span would be shorter than 8 frames, or
    either end would lose more than 40% of the clip.
- When the sign was found, its first and last frames are repeated so each is held for 100 ms (3 frames at 30 fps). On
  the fallback there are no holds: holding a frame where the hand is still moving would add a stop-and-go.
- Sign clips only. Fingerspelled letters keep today's trim: they are built as held handshapes
  (`training/letters_from_kaggle.py`), not cut from a signer's video.
- `buildTimeline` is otherwise unchanged: its blends (150-450 ms, minimum-jerk, forward arcs) now run from one sign's
  end hold to the next sign's start hold, where the hand is slow at both ends.

### Part 2: steadier, crisper motion

- **Clean at load.** New `smoothClip(clip)` in a new `app/web/src/avatar/clipSmooth.ts`, called in `stage.ts`
  `fetchClip` before `steadyBaseHand` and cached with it. Every point of `pose`, `pose2d`, `left`, `right`, `leftW` and
  `rightW` becomes the 1-2-1 weighted average of itself and the frames on either side (¼ of frame t−1, ½ of frame t,
  ¼ of frame t+1). Frames without both neighbours (a clip's first and last, and the frames beside a stretch where a
  hand is missing) are left as they are, so a steady movement stays exactly as it was everywhere. `touchL/R`,
  `meetL/R`, `touchAtL/R` and `aspect` are left as they are.
  - Why 1-2-1 rather than a wider window: it removes the frame-to-frame shimmer completely (it cancels the fastest
    wobble 30 fps can show, 15 Hz) while keeping 96% of a 2 Hz movement and 83% of a 4 Hz one. A wider 1-4-6-4-1 window
    keeps only 70% at 4 Hz, which would flatten fast repeated movements such as shaking or tapping.
  - It runs once per clip, a fraction of a millisecond, so it adds nothing per frame on the phone.
- **Lighter live smoothing, chosen by measurement** from:
  - (a) today's: arm one-euro filter `FILTER_HZ` 3 (`armSolver.ts`), finger filter `FINGER_HZ` 8 (`rocketbox.ts`);
  - (b) arm 6, fingers 12;
  - (c) the arm one-euro filter switched off (the springs and the speed limit only), fingers 12.

  Each candidate is measured with Part 1 and the load-time cleaning in place. Keep the lightest one that (1) has no
  more frames at the speed limit than today's code, (2) has wrist jerkiness at sign edges and elsewhere no higher than
  candidate (a)'s, and (3) scores a round-trip top-3 no more than 2 points below today's code (run-to-run noise is
  about 2 points). If neither (b) nor (c) passes, keep (a).
- Not changing: the 1,200°/s speed limit (`TURN_MAX_DEG_S`), the springs (`motionSmoother.ts`), the arm solver, the
  body and contact corrections, and hand placement.

### Part 3: a face that shows tone

- **Translator.** The tone strengths in `face.ts` `TONE` go to full. Starting values: warm `MTH_Joy` 1.0, `EYE_Joy` 0.7
  (a smile with the cheeks lifted); apologetic `MTH_Sorrow` 0.9, `EYE_Sorrow` 0.6, head pitch 6° unchanged (a frown,
  eyes lowered); urgent `MTH_Surprised` 0.6, `EYE_Surprised` 0.8 (wide eyes, mouth a little open); neutral nothing. The
  final values are tuned by eye on both screens until they read on the phone without looking cartoonish, and the user
  signs them off. The ramps stay: the tone eases in over 200 ms as signing starts and relaxes as the hands return to
  rest. Tone stays on the mouth and eyes; the brows stay for grammar.
- **Teaching mode.** A feeling-sign map in the teaching code: a smile (tone warm) for HAPPY, EXCITED, GOOD, FINE; a frown
  (tone apologetic) for SAD, ANGRY, TIRED, SICK, SORRY. A pure helper `faceFor(glosses)` returns the plan's tone: the
  shared face when `glosses` has at least one feeling sign and they all map to the same face, otherwise neutral; signs
  not in the map don't count. `playTokens` uses it instead of always sending neutral, so single signs (most teaching plays) show their face,
  and a mixed sequence (the put-in-order exercise) stays neutral.

### Part 4: the switch and the bench

- `?oldmotion` in the page address restores today's motion: the 12% trim, no holds, no load-time cleaning, today's
  filter settings. The face changes are not behind it.
- `bench.ts` also reports, per item, the frames at the speed limit (any arm bone turning more than 19.5° between
  60 fps frames) and the median wrist speed and jerkiness within 40 ms of sign edges and elsewhere, as
  measured above (the higher wrist; speed = half the distance between the frames either side, jerkiness = the length of
  P[t+1] − 2·P[t] + P[t−1], millimetres at 60 fps; an edge is any boundary between timeline items where either item is a
  clip). Before and after runs then come from `scripts/motion_bench.py`.

## How we'll know it's done

1. **Sign edges:** the median wrist speed and jerkiness at sign edges come within 1.5× of their level elsewhere (today
   2.2× and 2.7×).
2. **Snaps:** frames at the speed limit no more than today's (2 of 1,711 single-sign frames, 10 of 5,248 sentence
   frames).
3. **Nothing breaks:** the existing bench shows no new body penetration, backward elbows or over-twisted forearms, and
   the p95 frame cost is no worse.
4. **Distinctness:** the round-trip test re-measured on the toon before and after; top-3 no lower than the before
   baseline minus 2 points. A rise is the hope, not the gate.
5. **The user's eyes,** on the laptop and the phone, before (`?oldmotion`) and after: HELLO, THANK-YOU, GRANDMOTHER,
   HAPPY and SAD, and three sentences including one warm and one apologetic. The user's judgement is the gate.
6. **Tests:** `signSpan` finds the holds on a synthetic clip (rise, hold, sign, hold, drop), falls back and respects the
   caps; the held first and last frames last 100 ms; `smoothClip` removes an alternating wobble, keeps the clip's
   length and a straight-line motion unchanged, and handles a missing hand; the face strengths and `faceFor`. The
   existing `npm test` and `uv run pytest` keep passing.

## Not in this round

- Approaches B and C.
- A face per sign inside a sequence, holding the tone after a sentence, faces reacting to the learner's answers, faces
  for individual signs in the translator, and lowered brows for ANGRY.
- Fingerspelling: the letter clips aren't built on this machine (separate setup with the Kaggle data).
- The phone's camera framing, the speed limit and springs, and the arm solver.

## Risks

- A clip whose first slowdown isn't the sign's start (a sign with a pause inside it) could be trimmed in the wrong
  place. The fallback and the 40% cap limit the damage, and the 150-sign round-trip test and the fixed-set review would
  catch a regression.
- Lighter live smoothing could let tracker noise through. The decision rule keeps today's settings unless a lighter one
  measures no worse.
- Stronger faces could look cartoonish on the laptop yet still be faint on the phone. The user signs off the strengths
  on both screens.
