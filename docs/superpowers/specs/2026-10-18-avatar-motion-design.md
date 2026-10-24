# Avatar body and motion: human-like signing

Date: 2026-10-18. Status: design, awaiting approval. Branch: `eric-testing`.

## Intent

The person watching the avatar is a **Deaf ASL signer**: the person who will most readily notice anything wrong. Today,
watching the avatar sign is uncomfortable. The arms overlap, several limbs suddenly spin quickly instead of moving
smoothly, the elbows and forearms bend and twist like "spaghetti", the head never moves, and the shoulders are too broad
and stay frozen. The aim: the avatar moves like a real person signing: firm, readable and never jarring, with each sign
in the right place.

**The failure line (the user's words):** if the arm movements are not human-like, this is a failure, even if every
sign is technically correct.

This is piece 3 of an agreed sequence. Pieces 1 (translation order) and 2 (sign clips) are done. Piece 4 (emotion from
words and voice, an expressive face) comes after this one and is out of scope here.

## Goals

1. **Body: Rocketbox `Male_Adult_11`** (the plain-shirt Rocketbox character, MIT license) becomes the default avatar,
   replacing the MPFB character. *(The user chose it from a side-by-side of the four characters in the app.)* The others
   stay reachable through the existing URL parameters (`?model=…`, `?avatar=vrm`) for comparison.
2. **Hands where the sign belongs.** Location carries meaning in ASL: GRANDMOTHER starts with the thumb at the chin;
   today every character holds that hand out near the shoulder. Hands reach the chin, forehead, chest and the other
   hand where the signer's did.
3. **Arms that move as solid limbs.** No elbow bending the wrong way, no forearm corkscrewing (a "candy-wrapper"
   twist), no wobble or jitter. The upper arm and forearm move like bones from a shoulder. Arms do not pass through each
   other or the body.
4. **Smooth movement within and between signs.** No frame where one or more limbs flip or spin suddenly. Transitions
   between signs follow natural arcs.
5. **A head and shoulders that move.** The head moves naturally while signing (small nods, tilts, turning a little with
   the body and toward the working hand) instead of staying locked to the viewer. The shoulders rise, fall and turn a
   little with the arms.
6. **Keep what works:** the brow and head grammar (raised brows for yes/no questions, lowered for wh-questions, the
   headshake for negation) must still read clearly, the face shapes must still work, and the hands must stay visible
   against the shirt.

## Non-goals

- Emotion from the voice, and new or stronger facial expressions (that's piece 4).
- New sign data from new sources. Work with the existing WLASL-derived clips in `app/web/public/signs/` (1,982 signs),
  though rebuilding them from the same videos with a better pipeline is allowed.
- Fingerspelling handshapes (they need the Kaggle data, which the user has deferred).
- Continuous-signing grammar (classifiers, directional verbs).
- Changing the sign → speech direction.

## Anti-goals (would make this a failure even if it "works")

- Arm motion that still looks mechanical or rubbery: the failure line above.
- Smoothing so heavy that signs blur into each other, lose their crisp handshapes and holds, or land in the wrong
  place. Smooth must not mean mushy.
- Slower signing: the speech → avatar turn currently starts motion in about 10 ms after the plan arrives, and the
  whole turn must stay within the app's 2-second budget per direction. Per-frame cost must keep 60 fps in the browser
  on this MacBook.
- A head that moves in ways that look like ASL grammar it isn't (a nod reads as "yes", a shake as "not"). Idle head
  motion must be small enough not to be mistaken for grammar.

## Constraints

- Browser, three.js, TypeScript (`app/web/src/avatar/`). The avatar's motion is driven per frame from MediaPipe
  landmark clips (`pose` 33 points, `left`/`right` 21 hand points, world versions `leftW`/`rightW`), built by
  `app/training/clips_from_wlasl.py` + `build_motion_library.py`.
- Licenses: Rocketbox assets are MIT; the WLASL-derived clips are non-commercial research only and are not
  redistributed (git-ignored).
- The existing evaluation tools stay usable: the round-trip test (the avatar's render read back by the ASL Citizen
  recognizer, currently about 0.09 top-1 on Rocketbox) and `?mode=review` (each avatar clip beside its source video).
- No Deaf reviewer is available yet; the project README says one is required before anyone outside the team uses it.

## What the current code does (evidence for the builder)

All in `app/web/src/avatar/rocketbox.ts` and `player.ts` unless noted. These are the likely sources of each symptom,
from reading the code; none are confirmed by measurement yet.

- **Arm solving is per frame, with no memory of the last pose.** `solveArm` builds an elbow target (MediaPipe's elbow,
  half-mixed with a "hanging elbow" guess), solves a two-bone IK (`twoBoneElbow`), then aims each bone at its child
  (`aimLocal`). Aiming sets the bone's direction but not its roll about its own axis, so the upper arm's roll can
  change abruptly when the direction crosses certain angles: a likely cause of the sudden limb spins.
- **Forearm twist** is set from the hand's roll each frame (`FOREARM_TWIST = 0.6` of the wrist roll), with a wrist
  swing limit of 70°. There is no limit on twist itself, and no continuity with the previous frame: a likely
  corkscrew source.
- **Body collision pushes** (up to 3 passes per arm per frame) move the hand and elbow in discrete jumps whenever a
  probe enters the torso; an elbow pushed out one frame and not the next will wobble.
- **Smoothing happens only on the input landmarks** (`SMOOTH = 0.55`, applied per `applyFrame` call, so it depends on
  the frame rate), never on the solved joint rotations. Contact corrections (`placeShift`, `contactGap`) and collision
  pushes come after the smoothing, so their jumps go straight to the screen.
- **Blends between signs** (`player.ts`, `lerpFrame`) linearly interpolate landmark points with a minimum-jerk ease,
  then IK solves each blended frame. Straight-line wrist paths can cut through the body (triggering pushes) and can
  swap which side the elbow bends to mid-blend.
- **Head:** `applyFace` turns the head only for grammar (yes/no nod pitch, negation shake) and undoes half the torso
  turn (`HEAD_KEEP = 0.5`) so it keeps facing the viewer. There is no other head motion. The shoulders: the collarbone
  lifts up to about 26° when an arm rises (`SHRUG`); there is no other shoulder motion. The torso turns up to 8°.
- **Placement:** wrists are mapped from the signer's body to the avatar's relative to the shoulders, and relative to
  the nose when near the face (`toAvatar`). Even so, GRANDMOTHER lands near the shoulder on all four characters, so
  something upstream (the clip, the mapping or a correction) misplaces it. Not yet diagnosed.
- **Contact data was missing, and is now restored.** `training.touch_from_asllex` adds per-frame ASL-LEX contact
  weights (`touchL/R`, `meetL/R`, `touchAtL/R`) that `contactGap` and `placeShift` rely on. Rebuilding clips drops them,
  and the README's setup never ran the script, so no clip had them when the user tested. They were added on 2026-10-18:
  1,370 of 1,982 clips matched ASL-LEX (323 touch the body or head, 468 have the hands meeting, 1,114 have no contact,
  48 have a place the script can't map, and in 29 the hand never reaches the place). The README needs this step.
- **GRANDMOTHER is a mapping problem, not missing data.** ASL-LEX codes no contact for it. In the clip's 2D image
  landmarks the thumb is at chin height beside the nose (as it should be), but in MediaPipe's *world* landmarks the
  right wrist sits straight in front of the right shoulder (x −0.11 m, with the shoulder at −0.12 m and the nose at
  0.00). The avatar is placed from the world landmarks, so it inherits that error. World-landmark x and depth disagree
  with the image; the builder should check how widespread this is before choosing a placement fix.
- **Clip data looks clean on the hand-presence side:** in all 704 clips checked, hands never appear or disappear
  mid-clip, so the problem is not missing-hand gaps. 41 clips have a wrist jump of more than 0.25 shoulder widths in
  one frame, and 8 have the wrists crossed for over 20% of frames.

## Decided

- Rocketbox `Male_Adult_11` is the default body (user).
- The order of work: this piece before emotion (user).
- Human-like arm motion is the pass/fail bar (user).
- Keep the grammar face behaviour (brows, nod, shake) working (agreed in conversation).
- **Rebuild the arm-solving core; keep what surrounds it** (user). The current per-frame solver structurally can't be
  smooth: no memory of the previous pose (bone roll can flip in one frame), smoothing applied before the contact and
  collision corrections (so their jumps reach the screen), and no human joint limits. The new core starts each frame
  from the previous pose, enforces human joint limits (elbow hinge range, forearm twist range, wrist), filters the
  *solved* joint rotations rather than the raw landmarks, and resolves body collision continuously instead of in
  discrete pushes. Kept: body- and face-relative hand placement, contact data, the sign timeline and blends (the
  builder may change the blend paths), the face and grammar, model loading, and the evaluation tools.
- Re-extracting the clips happens **only if measurement shows** the world-landmark placement error (see GRANDMOTHER
  above) is widespread; otherwise placement is corrected at runtime.
- No deadline (user didn't name one). Checkpoint: show the user a side-by-side as soon as the first signs move well,
  before finishing every goal.
- *Designer's call:* "too broad shoulders" was about the MPFB body the user tested. They picked `Male_Adult_11` from a
  side-by-side, at its natural proportions (`?frame` stays 1 for it), so the body's proportions don't change; only
  shoulder *motion* is in scope.
- *Designer's call:* the human-like bar applies to the default body. The shared `rocketbox.ts` path (Rocketbox and
  MPFB) gets the work; the old VRM path (`vrm.ts`, its own solver) only has to keep loading.
- *Designer's call:* "before" for every comparison means the current code **with contact data restored** (the state
  on disk now), so the improvement measured is the motion work's, not the setup fix's.
- *Designer's call:* the round-trip test exists only as numbers in the execution record and comments
  (`training/landmarks.py`, `training/hand_cleanup.py`); no harness is committed. The builder builds a small one
  (render the avatar headless, run MediaPipe, read it back with the ASL Citizen recognizer) and re-measures the
  baseline with it, rather than trusting the quoted 0.09.

## Left to the builder

- How the new core is built inside the decision above: e.g. the IK method, swing-twist decomposition, the filter (one-
  euro or critically damped), the exact joint ranges, arcs instead of straight lines for transitions. Measure the
  current solver first so the before/after numbers exist.
- How much head and shoulder motion is natural, as long as it never reads as grammar (see the anti-goals).
- Whether MPFB-specific code paths stay or are simplified, as long as the other characters still load.

## How we'll know it's done

1. **The user watches a fixed set of test sentences** before and after, side by side, and judges whether the arms look
   human. The builder picks about 10 sentences covering face-location signs (GRANDMOTHER, FATHER, MOTHER), chest
   signs, two-handed signs where the hands meet (HELP, MEET), signs with large arcs, and long multi-sign sentences.
   *(Builder's choice of exact sentences; the user's judgement is the gate.)*
2. **Measured, before vs after, on the same signs:** no frame-to-frame joint rotation spike above a threshold the
   builder sets and reports; no elbow bending backward; forearm twist within human range; arm-body and arm-arm
   penetration (the existing `penetration()` and `armPenetration()` probes) not worse; hand positions at contact
   signs closer to the target places.
3. **The round-trip recognizer score must not drop** from the baseline the builder re-measures (the execution record
   quotes about 0.09 top-1 / 0.15 top-3 on Rocketbox over 150 signs). It is a weak signal for human-likeness, so a rise
   is a bonus, not the goal.
4. The existing unit tests (`npx vitest run`, `uv run pytest`) still pass.
