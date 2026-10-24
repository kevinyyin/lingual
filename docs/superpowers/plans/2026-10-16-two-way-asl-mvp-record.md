# Two-Way ASL MVP — execution record

Decisions made during implementation (from the execution ledger), final-review fixes, deferred minors and pending operator steps. Plan: `2026-10-16-two-way-asl-mvp.md`.

## Rulings

- Ruling: No ANTHROPIC/CARTESIA keys on machine — live tests stay skip-marked; probe_cartesia written against documented shapes but unrun; model IDs unverified — cost if wrong: voice/stt adapters need shape fixes when keys arrive.
- Ruling: Human-only steps (ASL Citizen download/extraction run, recording signers, training to targets, manual camera checks, avatar review, listening test, 5 reliability runs) are left as pending operator steps; code + tests for them are built — cost if wrong: targets unmeasured until a person runs them.
- Ruling: Work on branch asl-mvp in this checkout (user didn't ask for a worktree) — cost if wrong: none, branch is isolated.
- Task 1: Ruling: ws.ts Socket also exposes buffered() — Task 3 needs bufferedAmount for backpressure — cost if wrong: none
- Task 2: Ruling: vocab is 160 signs (upper bound of test range); asl_citizen glosses are best guesses (hyphens removed) pending Task 5 verification against the dataset — cost if wrong: some renames when data arrives
- Task 3: Ruling: frame backpressure (bufferedAmount > 256 KB) lives in main.ts's onFrame (it owns the socket), not camera.ts — cost if wrong: none
- Task 3: Ruling: MediaPipe .task files pinned to version "1" and committed (17 MB) so Python extraction uses identical models; wasm copied by npm postinstall, gitignored. JS and Python mediapipe both 1.1.1 — cost if wrong: repo size
- Task 4: Ruling: Session takes make_segmenter(baseline) instead of a segmenter (segmenter needs calibration first) and vocab (pick options need english) — cost if wrong: none
- Task 4: Ruling: plan said baseline.rest_y > resting wrist y; image y grows downward, so the rest line is REST_MARGIN *above* rest = smaller y; test asserts rest_y < resting y — cost if wrong: turn detection inverted (would show immediately)
- Task 4: Ruling: TurnEnd carries rest_ms (browser-clock time since hands came to rest) so hands_rest stamp counts the rest-detection delay toward the 2 s budget — cost if wrong: latency over-reported slightly
- Task 4: Ruling: avatar_start stamped when sign_plan is sent (browser starts on receipt); speech_end stamped when Ink's final arrives — cost if wrong: excludes Ink endpointing delay from the 2 s budget
- Task 4: Ruling: Session already routes kind="spell" segments to an optional spell_reader (Task 15 plugs it in) — cost if wrong: none
- Task 4: Ruling: dev transcript is POST /dev/transcript to the open session (no protocol change); removed in Task 13 — cost if wrong: none
- Task 5: Ruling: ASL Citizen download (42.8 GB, license = non-commercial research only) NOT started — downloading/accepting terms needs the user's explicit OK; extraction code built and tested (npz roundtrip, CSV matching, video_to_frames runs on a synthetic video) — cost if wrong: Day-1 background download delayed until user approves
- Task 5: Ruling: npz stores float64 (exact roundtrip; landmarks are small) — cost if wrong: ~2x disk
- Task 6: Ruling: training on real data (step 4, target top-1 >= 0.80) pending ASL Citizen download; fit() verified on synthetic separable data (>0.9) — cost if wrong: accuracy unknown until data arrives
- Task 6: Ruling: main.py uses SignPredictor when models/signs.pt exists, else falls back to stub with a warning — cost if wrong: none
- Task 6: Ruling: augmentation hand swap also swaps pose L/R pairs (plan said hand swap + x-mirror only; mirroring without swapping pose pairs corrupts the pose) — cost if wrong: none
- Task 7: Ruling: energy measured in shoulder widths/frame with MIN_ENERGY floor 0.02 (plan's motion_floor*PAUSE_K alone is ~0 for a still signer) — cost if wrong: retune in Task 18
- Task 7: Ruling: adjacent spell segments in a turn merge into one (letter holds would otherwise split a word); hands going down closes the open segment immediately — cost if wrong: two spelled words back-to-back merge into one
- Task 8: Ruling: save logic in server/record.py (testable) instead of inline in main.py; added RecordedMsg ack so the recorder shows saved/failed; letters recorded as gloss "_letter/X" → data/ours/{signer}/_letters/X/ (for Task 15) — cost if wrong: none
- Task 8: Ruling: Space starts and Space saves a take (plan said hands-drop stops); simpler and deterministic for recording — cost if wrong: takes include a little rest motion at the end, which the segmenter trims
- Task 9: Ruling: Composer(client, vocab) — vocab needed for the plain-words fallback; shared server/llm.py ask_json (structured output json_schema, thinking disabled, effort low, 4 s timeout, client max_retries=0) reused by gloss.py — cost if wrong: none
- Task 9: Ruling: use official cartesia SDK 4.2 (cartesia[websockets]) instead of raw WebSocket shapes — avoids guessing protocol; sonic-3.6 + ink-2 per current docs; generation_config.emotion is ONE string (plan assumed list) — cost if wrong: one dependency
- Task 10: Ruling: voice_controls returns {"emotion": str, "speed", "volume"} with emotion omitted for neutral (Sonic takes one emotion string); mild/strong emotion names switch at intensity 0.6 (e.g. sad→disappointed) — cost if wrong: retune in listening test
- Task 10: Ruling: Voice takes injectable client/player (testable); main.py uses StubVoice when CARTESIA_API_KEY is absent — cost if wrong: none
- Task 12: Ruling: FaceReaders lives in server/readers.py (session.py can't import grammar/emotion without a cycle); grammar and emotion run sequentially inside the one worker thread (both take ms) — cost if wrong: none
- Task 12: Ruling: face drives the emotion label; signing speed/size (relative to the first turn) scales intensity rather than voting on the label — prevents "slow signing = sad" with a neutral face — cost if wrong: motion alone never changes the label
- Task 13: Ruling: while the session isn't listening, the mic sends silence (same length) instead of 0 bytes — keeps the Ink socket alive; the voice output still never reaches Ink — cost if wrong: small bandwidth
- Task 13: Ruling: no Ink keyterms — the vocabulary is everyday words, keyterms help only rare terms — cost if wrong: slightly worse transcription of names
- Task 13: Ruling: dev text box kept behind ?dev=1 (plan said remove) so the speech direction is testable without a mic or Cartesia key; mic runs only when CARTESIA_API_KEY is set — cost if wrong: none
- Task 13: Ruling: LLM token schema uses {"kind","value"}; validate_plan strict on attempt 1, lenient (unknown sign → spelled) on attempt 2 — cost if wrong: none
- Task 14: Ruling: avatar = pixiv "VRM1_Constraint_Twist_Sample" (VRM Public License 1.0: everyone, commercial OK, redistribution OK), committed (10 MB) — cost if wrong: swap the file
- Task 14: Ruling: blends interpolate landmarks (smoothstep) before Kalidokit solves, instead of slerping bone rotations — same visual effect, far simpler — cost if wrong: slight elbow shortening mid-blend
- Task 14: Ruling: clip frames carry world pose (for Kalidokit 3D) + image pose2d; video_to_frames gained world_out; _REST.json from first reference frame, DEFAULT_REST built-in until then — cost if wrong: none
- Task 14: Ruling: Kalidokit mirrors the person; signing avatar must copy. Input is mirrored (x flip + L/R landmark swap) so Kalidokit's mirror cancels; VRM1 bones need Euler x,z negated. Verified visually: rest = arms down, signer's right arm → avatar's right arm, fist vs open fingers on the correct hand — cost if wrong: mirrored signing
- Task 14: Ruling: vite proxies /vocab /reference /dev to the backend (was /ws only); review mode at ?mode=review; .claude/launch.json for asl-server + asl-web — cost if wrong: none
- Task 15: Ruling: letter avatar clips built from our recordings with a pseudo-world pose from image landmarks (recordings have no world landmarks; the arm only holds near the shoulder) — cost if wrong: arm position slightly off for letters
- Task 15: Ruling: training data = our own letter recordings only (no public dataset added) — cost if wrong: needs 26 × 5 takes per person
- Task 16: Ruling: the VRM's preset expressions (angry/sad/...) bind whole-face morphs that include brows, so the face drives morph targets directly: brows = BRW_Surprised/BRW_Angry, tone = MTH_*/EYE_* only; FacePose gained headPitchDeg (yn tilt, apologetic lowered gaze) — cost if wrong: morph names are specific to VRoid-made models
- Task 17: Ruling: added @testing-library/dom (dev only) for role-based accessibility assertions; picker hides when the state leaves "picking" (stale-picker test RED→GREEN); camera failure shown in the state badge — cost if wrong: none
- Task 18: Ruling: latency measures from the tap only when the log's result.picks > 0 (session always stamps "picked"); session now logs picks — cost if wrong: none
- Task 18: Ruling: grid eval baseline = the signer's statement_neutral takes (grid takes have no calibration); names recorder mode (&names=1, "_name/X" ≤ 8 letters) added for the spelling eval — cost if wrong: baseline slightly different from live calibration

## Published-model pivot (2026-10-17)

The user asked to replace the in-house models with published ones, widen the vocabulary, make the avatar's hands
human-like and test recognition on online videos of varying quality. Reports: `app/data/eval/` (gitignored).

- Ruling: sign recognition = Microsoft's ASL Citizen ST-GCN (MIT code, weights from GitHub release checkpoints_v1, 2,731 signs, 27 keypoints) — the only published isolated-sign model with released weights trained on many Deaf signers that takes MediaPipe-style keypoints — cost if wrong: swap `server/citizen_classifier.py`
- Ruling: its label order = sorted glosses of ASL Citizen's train.csv, read out of the 42.8 GB zip with HTTP range requests (`training/remote_zip.py`) instead of downloading it — verified: 85% top-1 on ASL Citizen test signers when choosing among the 160 curated signs — cost if wrong: every label shifted (would show as ~0%)
- Ruling: I3D (ASL Citizen's RGB model) rejected: 52 s per sign on this CPU — cost if wrong: none
- Ruling: MediaPipe's handedness is wrong on ~22% of frames; hands are re-keyed by distance to the pose wrists (15 = signer's left, 16 = signer's right) everywhere (recognizers, avatar clips) — cost if wrong: none (pose wrists are reliable)
- Ruling: ensemble = ASL Citizen ST-GCN (restricted to the app vocabulary) + the teammate asl-spike model (143 signs), per-sign weighted average over only the members that know the sign; weight 1:1 (benchmark: 1:1 0.471 top-1, 2:1 0.461, ST-GCN alone 0.463) — cost if wrong: change `CITIZEN_WEIGHT`
- Benchmark (`training/eval_recognition.py`): 772 online videos (ASL Citizen test signers no model trained on + WLASL dictionary videos from many sites) × 7 qualities (clean, 240p, JPEG q12, blur, noise, dark, half fps) = 4,231 landmark runs through the browser's MediaPipe models, 43,220 scored predictions; every model chooses among the app's full ~2,900 signs. Ensemble, live-like span: ASL Citizen signers top-1 0.60 / top-3 0.81 / top-5 0.85 (clean), 0.56–0.60 under 240p, blur, dark, half fps, 0.57 JPEG, 0.31 heavy noise; WLASL videos (different recording setups, often older/lower quality) top-1 0.22–0.27, top-3 ~0.49. Full table: `app/data/eval/report.md`
- Ruling: the browser sends ~15 frames/s (MediaPipe hand+pose+face per frame), measured with a fake webcam; ST-GCN was trained on ~30 fps video and lost 40% of its accuracy at half rate (ASL Citizen top-1 0.51 → 0.32). `citizen_classifier.frames_to_input` now resamples sparse input to 30 fps by timestamp (hands interpolated only between frames that both have them): half-fps top-1 0.32 → 0.53 (= clean) — cost if wrong: none measured
- Ruling: the live segmenter handed the recognizer only the high-motion frames; the published models were trained on whole clips including the hands rising and settling. Segments now carry ±500 ms of context (never past halfway to the next sign; pre-turn frames kept): ASL Citizen top-1 0.49 (±150 ms) → 0.60 (±500 ms) ≈ whole clip 0.61 — cost if wrong: neighbouring signs leak in at fast signing
- Ruling: segmenter thresholds converted from frame counts (set for 30 fps) to milliseconds — at the real ~15 fps the end-of-turn wait had doubled to ~1 s and signs under ~0.45 s were dropped; energy normalized per 30 fps interval. Same values at 30 fps (tests unchanged); recognition now lands 0.5–0.7 s after the hands rest (was ~2 s) — cost if wrong: retune `REST_MS`/`PAUSE_MS`/`MIN_SEG_MS`
- Ruling: Cartesia sockets connect with happy-eyeballs (`voice.WS_OPTIONS`): on this network IPv6 is broken and asyncio tried all 8 of Cartesia's IPv6 addresses before IPv4, so every TTS/STT handshake hit the 10 s timeout (curl was fine) — cost if wrong: none
- Ruling: turn log records each segment's top-3 (also for cancelled turns) so live recognition can be audited from `logs/turns.jsonl`
- Ruling: MIT static-letter fingerspelling model rejected: letter CER 0.951 on Kaggle ASL Fingerspelling (Deaf signers, 210 held-out sequences) — it reads single still letters, not real fingerspelling — cost if wrong: none
- Ruling: fingerspelling = CTC Conv1D+Transformer (~1M params) trained here on Google's Kaggle ASL Fingerspelling landmarks (the competition winners' recipe shrunk to CPU). v1 (5 files, 40 epochs): best held-out CER 0.425 (picked on the same set: optimistic). v2 (20 files, 16,389 sequences, 30 epochs, 99 min): val 0.367, **untouched test 0.363** — kept v2; v1 kept at `models/fingerspell_ctc_v1.pt` — cost if wrong: names spelled with ~1 letter in 3 wrong; the picker/caption shows the reading
- Ruling: vocabulary widened from 160 curated signs to 2,918 (`training/build_vocab.py`): every ASL Citizen and WLASL gloss with English meanings from ASL-LEX 2.0 (LemmaID; DominantTranslation was wrong e.g. PICKLE → "pineapple"); curated entries win on conflict; generated file gitignored (license) — cost if wrong: odd meanings for rare signs
- Ruling: Claude is offered only signs the avatar can show (curated, or with a built clip) so it fingerspells the rest — cost if wrong: none
- Ruling: gloss prompt rewritten with ASL grammar (time-topic-comment, wh-sign last, yn on the face only, negation + headshake, keep possessives and NEED/WANT/CAN, drop articles/be/do) and the input as JSON `{"say": ...}` (a bare question was answered instead of translated). `training/eval_gloss.py`, 182 live sentences from 20 templates: v1 67.6% → v2 83.5% → v3 93.4% → final **100%** pass every check, p50 992 ms, p90 1,231 ms
- Ruling: avatar kept (VRM, the only free rig with full finger bones that runs in the browser); motion rebuilt instead: clips from WLASL dictionary videos with MediaPipe world hand landmarks, arms by direction alignment, every finger bone by a chain direction solver against the rig's own rest directions; minimum-jerk easing, blend time grows with wrist travel, coarticulation trim between signs, idle breathing and blinks — cost if wrong: retune `player.ts` constants
- Ruling: avatar clips for 1,993 signs WLASL has direct-file videos for (`clips_from_wlasl --workers 8`); spot-checked YES, HELLO, NAME, COFFEE, WATER, HAPPY, PLEASE, COMPUTER beside their reference videos
- Ruling: screen recordings via Playwright (`scripts/record_demo.py`): sign-to-speech uses a Chromium fake webcam fed a y4m built from one ASL Citizen test signer's videos (P35; stitching different signers made every cut register as motion), so the whole live path runs on real signing. A script stands in for the signer at the picker and picks only the sign actually made (else "none of these"). Run 2026-10-17 09:35: 9 signs → 6 turns, 4 spoken ("I need help.", "I need to use the bathroom." from TOILET, "Hello.", "Thank you."; 3 via the picker), 2 cancelled (true sign not in the top 3, or a doubled segment); signs made while the picker is open are not read (by design). First audio 2.3–2.9 s after the hands rest, over the 2 s target: compose ~1 s + TTS ~0.5 s
- Pending (operator): live check with a real signer on the laptop camera; the end-of-turn and picker thresholds still untuned for real users

## Realistic avatar (2026-10-17)

The user asked for a non-anime avatar whose hands no longer pass into the shirt, and for every sign to stay accurate.

- Ruling: avatar = Microsoft Rocketbox `Male_Adult_11` (MIT; 81-bone 3ds Max Biped rig, 3 bones per finger, 52 ARKit face blendshapes), picked from 40 adults for a plain dark short-sleeved shirt (skin-coloured hands against a dark solid top, as interpreters dress); textures converted TGA -> 2048 JPEG (4.5 MB total) — cost if wrong: swap the files in `web/public/avatar/rocketbox/`
- Ruling: skeleton-agnostic retargeting (`web/src/avatar/rig.ts`): every bone's rest direction read from the rig's bind pose; the hand oriented from the palm frame with 60% of its roll moved into the forearm (no twist bones); fingers aimed bone by bone
- Ruling: arm *directions* copied onto a body with other proportions put hands too low and wide (HELLO above the head, MY/COFFEE at the belly). Wrists are now placed where the signer's were *relative to the body* (origin between the shoulders, shoulder widths), and relative to the nose when within 0.6-1.3 shoulder widths of it (MediaPipe exaggerates depth: the nose came out 0.5 shoulder widths in front of the shoulders), elbows by two-bone IK in the signer's bend plane. Checked: THANK-YOU/MOTHER at the chin, EAT at the mouth, FATHER/THINK at the forehead/temple, MY/PLEASE on the chest
- Ruling: body collision (`web/src/avatar/body.ts`): the torso measured from the avatar's own mesh as 2 cm ellipse slices, the head as an ellipsoid; wrist, knuckles, fingertips and forearm pushed out to 1.2 cm off the surface and the arm re-solved (3 passes). All 1,984 sign clips + 26 letters (92,265 frames): frames with a hand point > 5 mm inside the body 615 (0.67%) without -> **5 (0.01%)** with; residual only LAZY, COUGH (hand tapping the shoulder)
- Ruling: round-trip legibility test: 156 curated signs rendered on each avatar and read back by the ASL Citizen recognizer (app vocabulary, ~2,900 signs), same MediaPipe pipeline. Top-1/top-3/top-5: source WLASL videos 0.280/0.453/0.513, **Rocketbox 0.093/0.153/0.173**, old VRoid 0.000/0.007/0.007 (MediaPipe barely tracks the stylised body). Side by side the signs match visually (WATER, GOOD, WANT); the remaining gap is not hand detection (Rocketbox hands are found *more* often than the source signers': 77% vs 58% of frames) — open: likely movement dynamics and synthetic appearance. Input smoothing kept (off: 0.125 vs 0.175 top-1 on a 40-sign subset)
- Ruling: fingerspelling handshapes: the medoid of the reader's surest half of samples per letter; Kaggle image depth recovered from finger bone lengths; knuckle bend over 35 degrees amplified (x2.6, max 100) because MediaPipe under-reads curled knuckles (Kaggle A measured 56-81 degrees where a fist is ~90; the posed avatar reproduced the data exactly, so the input was the cause) — A and S now read as fists — cost if wrong: retune `KNUCKLE_*` in `training/letters_from_kaggle.py`
- Ruling: rest pose wrists moved a little below and outside the hip joints (at hip height the body push turned rest into hands-on-hips)

## Hand motion accuracy (2026-10-17)

The user asked for any way to make the hand animations more accurate. Approved: clean the sign clips the way the letters
were cleaned, hold one handshape per phase, pick the best take per sign. Each change was measured with the round-trip
test above (150 signs, avatar render read back by the recognizer, paired against the current clips).

| Clips (avatar, 150 signs) | top-1 | top-3 | top-5 |
|---|---|---|---|
| Current: first WLASL take that downloads | 0.093 | 0.153 | 0.173 |
| Best take + heavy pose model + hand clean-up | 0.027 | 0.073 | 0.093 |
| Best take + hand clean-up (lite pose) | 0.087 | 0.187 | 0.220 |
| Best take + hand clean-up without knuckle step | 0.080 | 0.193 | 0.240 |
| **Best take only** | **0.093** | **0.213** | **0.280** |

- Ruling: **best take** (`training/clips_from_wlasl.best_take`): up to 4 takes per sign; among those whose leading hand
  was tracked >= 80% as well as the best, right-handed first, the one closest to the others (wrist path relative to the
  shoulders + handshape, 16 steps), else the least noisy. Against the current clips top-5 +23/-7 signs (one-sided sign
  test p = 0.003); top-1 unchanged (+7/-7). The chosen takes also read better as videos (source 0.28 -> 0.31 top-1)
- Ruling: MediaPipe's **heavy pose model rejected**: its world landmarks put the wrists ~0.5 shoulder widths further back
  than lite; the body push then shoved hands sideways (REMEMBER never reached the forehead), top-1 0.093 -> 0.027
- Ruling: **hand clean-up kept opt-in** (`training/hand_cleanup.py`, `--clean`): depth and knuckle correction, gaps
  interpolated instead of frozen (a quarter of some clips' frames were frozen copies), handshape held per segment
  (piecewise-constant split, at most 6 segments; frame noise is as fast as a real handshape change, so speed thresholds
  failed). Visibly steadier (handshape jitter roughly halved; S, A and H hands exact) but read *worse* than best take
  alone (top-5 0.220 vs 0.280, p = 0.047): the recognizer learned signers through MediaPipe's own noise and under-curl.
  It also over-curled some open hands (PLEASE, SEE) — a fluent signer's judgement should decide, not this metric alone
- Rolled out (2026-10-18): all 1,984 WLASL signs rebuilt with best take (722 compared 4 takes, 595 three, 512 two,
  155 one; 1,337 changed video); the previous clips are in `data/signs_backup_2026-10-17/`. Body collision after the
  swap: 5 of 115,402 frames > 5 mm inside the body (0.00%); worst KING 51 mm (hand crosses the chest), COUGH, KNEEL
- Fixed on the way: MediaPipe 1.1.1 on Windows takes ~42 s to close each detector (3 per video, ~25x the detection
  time): `training/landmarks.video_to_frames` now keeps them open per process (identical landmarks); 634 WLASL
  "videos" were HTML error pages (now rejected; dead links remembered as `.dead` and downloaded 32 at a time)

## Final review fixes

- Final review: reviewer on opus (fable unavailable: no usage credits). Verdict: ready to merge with fixes; 0 Critical, 5 Important, 11 Minor.
- Final: fixed overlapping turns (Important 1) — test_second_turn_end_while_busy_is_ignored, test_second_final_transcript_while_planning_is_dropped RED→GREEN
- Final: fixed picker with no way out (Important 2) — test_none_of_these_cancels_turn, test_pick_not_offered_is_ignored, test_recalibrate_during_picking_ends_turn, ui "none of these sends an empty pick" RED→GREEN
- Final: fixed Ink drop sticking in listening (Important 3) — test_connection_drop_returns_session_to_ready RED→GREEN
- Final: fixed stalled Sonic stream hanging speaking (Important 4) — test_stalled_stream_times_out RED→GREEN
- Final: fixed /ws and /dev/transcript open to any website (Important 5) — test_origin (local allowed, others rejected) RED→GREEN; live check: evil origin ws refused, dev 403
- Final: fixed (re-graded Minor→Important: a hang) signer leaving frame mid-turn — test_signer_leaving_frame_mid_turn_times_out RED→GREEN
- Final: suite 111 passed, 2 deselected in 5.64s; web       Tests  20 passed (20)

## Avatar motion rebuild (2026-10-18)

Spec `docs/superpowers/specs/2026-10-18-avatar-motion-design.md`. Numbers from `app/reports/motion/*.json` (table in its README).

- Result: spikes 129 -> 0 of 7624 frames; max step 179.7 -> 20.3 deg; twist out of range 316 -> 0; backward elbows 0 -> 0; penetration 0; min arm gap 0.065 -> 0.071 m; mean contact error 0.149 -> 0.139 m; p95 frame 1.2 -> 1.6 ms (re-measured after the final-review fixes); CONTINUOUS (10 sentences, no reset): 0 spikes, 0 twist out, 0 backward elbows, max penetration 0.0023 m on one frame
- Ruling: default body is Rocketbox Male_Adult_11 (the user's choice); contact data restored (`touch_from_asllex` had never been run on this machine; README step added) — evidence: before baseline measured on Male_Adult_11 with contact data — cost if wrong: before numbers are for another body
- Ruling: new arm core in `armSolver.ts`: two-bone solver with joint limits (elbow 0-150 deg, forearm twist +-90 deg, wrist <= 70 deg); pole rule (hint >= 5 cm off the shoulder-wrist line accepted, 1-5 cm turns the carried pole <= 10 deg per call, < 1 cm keeps previous); twist demand stored unclamped and unwrapped against itself, so the twist stays at the clamp instead of jumping ~170 deg — evidence: spikes 129 -> 0, twist-out 316 -> 0 — cost if wrong: a fast real elbow flip takes a few frames to follow
- Ruling: one-euro filters on the solved rotations: arm QuatFilter(3 Hz, beta 0.5), fingers (8, 0.5), clavicle (1.5, 0.5, 1); a 1200 deg/s turn cap (20 deg per 60 fps frame, under the 25 deg spike gate) kept as a backstop, saturating 143 of 7444 frames (1.9%) at the core stage (FATHER end, GRANDMOTHER 41-45, COOK, NOT); corrections moved onto the targets before solving, plus a continuous safety pass — cost if wrong: robotic sweeps on palm flips; retune
- Ruling: user checkpoint (side-by-side, Task 6): "The arms on the right look human enough." — cost if wrong: n/a, the user's eye is the acceptance gate
- Ruling: placement took the runtime path — audit found 6.8% of clips (134 of 1976) over 0.3 shoulder widths (<= 20% rule); ramped 2D-wrist correction (0.15-0.35 sw disagreement) is neutral on the bench but covers those clips; contact data extended: image-detected face contact only for signs whose ASL-LEX MajorLocation is Head or that are missing from ASL-LEX (74 clips; spot check about 2 of 10 doubtful) — evidence: non-head image contact gave about 4 of 10 false touches — cost if wrong: a few real non-head face touches stay uncorrected
- Ruling: GRANDMOTHER@chin gate (<= 0.03 m) unmet and parked: mean 0.513 m, min 0.205 m — the source video's hand never comes within 0.12 sw of the estimated chin, loosening the detector to 0.22-0.25 sw flags 19-22% of clips (false touches) and still misses, and the chin target (nose -0.07 m) is itself an assumption — cost if wrong: GRANDMOTHER lands off the chin
- Ruling: head and shoulders: idle drift <= 1.5 deg per axis at 0.11-0.23 Hz, turn toward the working hand <= 4 deg yaw and <= 3 deg pitch (head QuatFilter(0.8 Hz, beta 0, dcutoff 1)), clavicle protraction <= 8 deg, both filtered; designed never to read as grammar (no nod/shake/shrug-sized motion) — cost if wrong: too subtle to see (the controller could not see the 4 deg turn in screenshots)
- Ruling: arcs: ARC_K = 0.25 forward bow (z only, scaled by transition length) on blends between signs only, not inside clips — contact error +0.0005 m — cost if wrong: retune one constant
- Ruling: round-trip reported plainly: after .087 / .140 / .227 (top-1/3/5) vs before .087 / .153-.173 / .233-.240 — top-1 equal, top-3 1-3 points below, top-5 within noise; core .073-.080 / .133-.147 / .227-.233; placement .080 / .140 / .200. The spec calls it a weak signal, and run-to-run drift is about 2 points — cost if wrong: legibility slightly worse than before
- Deferred minors: contact error maps raw `f.pose` not smoothed; `at` targets ungated and TOTAL double-counts GRANDMOTHER; bench ignores fingerspelling fallbacks; soft reach lands extended signs 1-4% short (SOFT knob); about 8 solveArmPose calls per arm per frame (allocation debt); palm-flip cap saturation to track upstream; forward-push fallback may alternate on wedged-hand frames (NOT); clavicle lift filtered (Task 8) and TORSO_EASE time-based (final review); non-empty ASL-LEX entry with empty MajorLocation gets no image contact; maxPenetration 6.6e-06 on FATHER MOTHER WORK WHERE at the placement stage (0 in the final run); YOU HELP I FIND MY KEY contact .13 -> .18 and HELLO NICE MEET YOU .047 -> .063 after false contacts removed; head turn picks the wrist by height only, no hysteresis; arc tests cover the right arm only; hinge sign assumes the avatar faces +z (unrun on Male_Adult_07)

## Deferred minors

- second tab (recorder) takes over _current and opens a second Ink connection
- 4 s LLM timeout means a retry always misses the 2 s target (plan value)
- gloss double-failure fallback spells in-vocabulary words and drops digits
- compose fallback drops the question mark
- avatar_start stamped before the browser fetches clips (cold-cache time not logged)
- turn task not cancelled on socket close; ws.ts never reconnects after a backend restart
- hidden tab pauses requestAnimationFrame so the session stays in avatar
- train_signs doesn't check holdout signer isn't in --train-signers
- thresholds untuned — README/eval should flag it before reliability runs

## Pending operator steps

- Task 3: Step 6 manual webcam check pending (operator) — no camera in this session
- Task 4: Manual browser check simulated with a WebSocket script (calibrate, sign turn, transcript) — real webcam check pending (operator)
- Task 7: Manual check done via WebSocket simulation (two signs → two segments); real webcam check pending (operator)
- Task 8: Step 3 (recording Kevin + 1–2 teammates, choosing the holdout signer, retraining) pending (operator)
- Task 10: Step 3 live check (hear it ≤ 2 s) pending — no CARTESIA_API_KEY; speaker output verified with silence
- Task 13: Step 4 live check (spoken sentence → plan ≤ 1.5 s) pending — no keys; verified the fallback path end to end via WebSocket
- Task 14: Pending (operator, needs ASL Citizen): build clips, run the review pass (<10% captions-only). Wrist orientation looked off with a crude synthetic hand — confirm on real clips in review mode.
- Task 15: Steps 3–4 (record letters, train to ≥ 85% per letter, build clips, live spelling check) pending (operator)
- Task 17: Visual check done (picker + you-said); live "pick then speak within 2 s" pending (operator)
- Task 18: Steps 3–5 (threshold tuning on real recordings, listening test, 5 reliability runs) pending (operator); thresholds left at plan defaults (not tuned)

## Open avatar issues after the motion rebuild (2026-10-19, handed to the avatar owner)

The user's verdict on the final build: the elbows still look wrong and the elbow mesh distorts, most visibly in the first
two seconds of HELLO NICE MEET YOU and TOMORROW I GO STORE BUY APPLE; and the realistic Rocketbox character is "too uncanny".
Decision: keep Rocketbox `Male_Adult_11` for now; a teammate owns the avatar from here. What was found (none of it is in the code):

- **Root cause of the "crazy" elbow in HELLO (confirmed):** the contact/place correction in `rocketbox.ts` `solveArm`
  pulls the wrist *target* ~38 cm back to the forehead (MediaPipe exaggerates the signer's hand depth: target 45 cm in
  front of the shoulder before correction), but the elbow hint stays where MediaPipe's too-deep arm put it (19-26 cm in
  front). The arm then folds ~135 deg with the elbow pointing at the camera, hiding and collapsing the elbow. Moving the
  elbow hint by 0.5 x the wrist's correction gives a natural salute (elbow out to the side) — verified visually.
  Open problem: the depth (z) part of that elbow shift pushes the hand 18-33 mm into the chest on chest-contact signs
  (I SORRY I UNDERSTAND NOT, PLEASE); the across/up part alone doesn't fix HELLO. Not a matter of the elbow going behind
  the shoulder (clamping that changed nothing). Try: follow only for head contacts, or only z when the touch place is on
  the head (`touchAt` near the nose).
- **Elbow "distortion" / wrung skin:** Rocketbox has no twist bones (only UpperArm, Forearm, Hand), and the new solver
  puts 100% of forearm twist on the Forearm bone, so the skin at the elbow seam twists. Forearm twist is large: median
  50 deg from bind, 16% of frames pinned at the +-89.5 deg limit. Moving part of it to the hand bone smoothed the elbow
  only slightly. A rig with forearm twist bones (or re-weighting) is the real fix.
- **Mirrored hands:** 10-16% of MediaPipe hand frames have the wrong handedness (palm normal flipped); they demand a
  ~180 deg palm flip and account for about a third of the twist-at-limit frames. Rejecting those frames' hand rotation
  (hold the last good one) cut pinned frames from 20% to 14.5% (left) and 11.6% to 7.5% (right).
- **Right forearm starts pinned at +89 deg at rest**, so the palm faces up while the hand rises (HELLO, first ~0.3 s),
  then unwinds ~150 deg in 0.25 s. Check the rest frame's right-hand rotation (`_REST.json` / `DEFAULT_REST`).
- **Bench mode with extra URL parameters renders the arms as stumps** — a non-finite value reaches the bones on a path
  that skips `stepArm`'s guards (the safety re-solve); guard it.
- Stylizing the realistic MPFB mesh with toon shading does not give a cartoon look (bald mannequin); a cartoon avatar
  needs a cartoon-shaped mesh with a finger rig and ARKit face shapes. No free one was found; Ready Player Me shut down
  on 2026-01-31.

## Anatomical hand (2026-10-20)

The user asked for a joint-by-joint model of the avatar's hands that keeps every joint where a human hand can go and stops
fingers clipping. Code: `app/web/src/avatar/handModel.ts` (+ tests), wired into `rocketbox.ts` `placeFingers`; bench columns
`handOutFrames … heldHands` in `scripts/motion_bench.py` (`--no-hand-limits` = before, `--all-signs` = every clip on its own,
`--channel msedge` where Windows blocks Playwright's own Chromium). Reports: `app/reports/motion/hands_*`.

- Evidence (source data): across the 156 curated clips' world hand landmarks, middle finger joints were asked to bend backward
  past 45 deg in 6.2% of hand frames and hinge joints sideways 30-60 deg (p5/p95); the non-dominant (left) hand is the worst
  (a finger bent back > 20 deg in 18-20% of left-hand frames vs ~5% right). The bone-by-bone copy put those poses on the avatar.
- Ruling: each finger joint has a frame (bend toward the palm about a hinge carried down the finger, sideways) and a range:
  knuckle -30..95 / +-25 sideways, middle -5..110 / +-12, end -15..90 / +-12, thumb metacarpal in a cone (-15..75 across,
  -25..60 out of the palm plane), thumb knuckle -10..65 / +-20, thumb end -20..80 / +-15 (published ranges, widened where
  real signers in the clips went past them; +-12 sideways on hinges = MediaPipe depth noise, measured ~10 deg median) —
  cost if wrong: retune `FINGER`/`THUMB` in `handModel.ts`
- Ruling: fingers are copied *relative to the palm* (signer palm frame -> avatar palm frame), not as world directions: a wrist
  the arm solver limits no longer leaves the fingers pointing where the signer's did
- Ruling: clamp again after the finger filter (it blends between in-range poses and could leave the range); a hand with two
  or more middle joints bent back > 45 deg is treated as mirrored/garbage and the last good hand is held (3,399 hand-frames)
- Ruling: capsule separation (fingers 8.5 mm, thumb 9.5 mm radius): fingers of one hand are turned apart through their limits
  (thumb first, else the finger nearer the little finger; the other one if that cannot clear it); the hand posed second is
  pushed out of the first hand's fingers on the shown pose unless that moves it into the body; every finger joint is a body probe
- Result, every curated clip on its own (156 clips, 25,569 frames, before -> after): finger joint-frames out of range
  308,771 -> 693 (frames with any: 99.6% -> 2.2%; worst excess 175 -> 19 deg); clips with fingers through each other > 5 mm
  153 -> 63 (median per-clip worst 13.8 -> 4.2 mm); clips with a finger in the body > 5 mm 6 -> 1 (worst 60 -> 20 mm, MILK);
  clips with the hands through each other > 5 mm 30 -> 25 (worst 15 -> 14 mm); unchanged: 0 spikes, 0 twist out, 0 backward
  elbows, contact error 0.113-0.114 m; p95 frame cost 1.1 -> 2.4 ms. CONTINUOUS (10 sentences): out-of-range joint-frames
  55,313 -> 34, finger in body 25 -> 0 mm. Visual check (raw vs anatomical, Edge): fingers curl toward the palm on both hands,
  no new backward bends
- Open: fists (YES, COFFEE, BUS) still put the thumb up to 15 mm into the curled fingers (both fingers at their limits);
  two-hand contact signs (WEEK, SCHOOL, LEARN, HOW, HARD) overlap up to 14 mm (palms are not modelled, only fingers); MILK's
  fingers 20 mm into the body on one stretch. Not reviewed by a fluent signer.

## Elbow on forehead signs (2026-10-20, investigated, not changed)

The user reported the elbows "not working". Same before/after the anatomical hand (it does not move the elbow).
- Finding: on head-contact signs the avatar's upper arm sits above the signer's (all 167 clips, raised-arm frames: +7 deg
  mean, 8.6% over 20 deg, 2.8% over 30 deg; worst THINK, KNOW, FATHER, REMEMBER, HAPPY, EXCITED). FATHER: signer ~90 deg
  (elbow at shoulder height), avatar 123-128 deg. Cause is geometry: the contact correction puts the wrist ~28 cm above and
  only ~9 cm in front of the shoulder, so the arm must fold, and the point of the elbow's circle nearest a sideways hint is
  above the shoulder (lowest reachable elbow ~6 cm above it).
- Tried, measured, shown to the user side by side (`app/data/eval/elbow_options.jpg`): (B) cap the elbow at the signer's
  mapped elbow height + 3 cm by turning the pole about the shoulder-wrist axis — over-30-deg frames 2.8% -> 0.3%, max body
  penetration 20 -> 4 mm, contact error unchanged, but the elbow swings low and forward with the forearm upright (FATHER's
  forearm in front of the face), even with the turn capped at 45 deg; (C) half the head-contact wrist move added to the elbow
  hint (the 2026-10-19 notes' idea) — FATHER becomes a level salute, numbers unchanged. Also found: the collision pass
  replaces the elbow hint with the trial's elbow whenever any hand point touches (`el.copy(e.add(pe))`); limiting that to
  elbow pushes alone changed nothing measurable.
- Ruling: the user chose to keep the elbow as it was (A); all of the above reverted — cost if wrong: re-apply B or C
  (the investigation notes above have the numbers)

## Avatar on the phone (2026-10-21)

The user tested the toon avatar on an iPhone: the hand that isn't signing pointed oddly, HELLO's lift looked like part of
the sign, and the arms looked pixelated. Causes and fixes (bench: `app/reports/motion/toon_plushie_before|fixes`):
- **Idle hand:** the recorded signer's resting hand often sits at the bottom edge of the video; MediaPipe still reports it
  with an arbitrary orientation, and the avatar copied it (THANK-YOU's left hand pointed across the hip). Across the clips,
  idle hands rise at most 0.2 m above the hips and signing hands at least 0.25 m. `player.ts` `restIdleHands` gives a hand
  under `IDLE_RISE_M` = 0.2 m the rest arm (hung from the clip's shoulder) and a relaxed hand, unless the hands touch
  (ASL-LEX meet > 0: COOK, CHEESE, HOLD … keep a low base hand; without this guard MY GRANDMOTHER TEACH I COOK went 4.5 cm
  into the body).
- **Lead-in:** clips keep the signer's travel up to and back from the sign (`trim_to_sign` only drops frames at rest), and
  the player trimmed it only between two signs; a sign alone played the blend from rest and then the recorded lift,
  palm out, which reads as a wave before HELLO's salute. Now every clip loses `COART_TRIM` (12%, the median lead-in across
  the clips) at both ends; the blends are the travel. Left: a ~0.1-0.2 s palm-up flick as the hand leaves rest (the
  forearm twist at rest noted on 2026-10-19).
- **Pixelated arms:** 16-sided arm tubes under the toon ramp shade every facet as a stripe. `build_toon_avatar.py`: arms
  48 sides, wrist ball 40x24, palm 48, fingers 24, sleeves 48 (glb 4.2 -> 5.1 MB). Extra rings along the arm made the
  elbow balloon under the distance-based skinning, so the arm shafts keep their two end rings.
- Bench (same items, before -> after): spikes 0 -> 0, max body penetration 0.027 -> 0.007 m, CONTINUOUS 0.018 -> 0.014 m,
  min arm gap 0.083 -> 0.083 m; contact error min unchanged, mean up on HELLO (0.138 -> 0.177) and THANK-YOU
  (0.085 -> 0.105). Not yet checked on the phone.
