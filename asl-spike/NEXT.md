# NEXT: what the 36-hour build should reuse (as knowledge, not code)

This is a pre-event research spike, not product code. Nothing here is copied into the build. These are
lessons, in priority order. Numbers are from `REPORT-SPIKE.md` (round 1) and `REPORT-SPIKE-2.md` (round 2).

1. **Recognize isolated signs from landmarks, keep the vocabulary small, and set up the camera like ASL Citizen.**
   - A ~0.8M-parameter landmark model (Conv1d + BiGRU, CPU, minutes to train) gets 84% top-1 / 93% top-3 on
     143 signs for unseen ASL Citizen signers and **72–73% top-1 / 87% top-3 on new WLASL signers with no
     retraining**. The 34% on PopSign was its recording style (phone held in one hand, close-up selfie: both
     hands visible in <1% of frames), not "new people".
   - So: **phone on a stand, landscape, both hands free, head to waist in frame.**
   - **Correct for the aspect ratio:** MediaPipe's x is divided by width and y by height. Use x·(w/h) before
     any geometry (shoulder normalization, speeds, angles).
   - Normalize by the shoulders, mirror the dominant hand into one slot, resample to 32 frames, merge variants
     (WANT1/WANT2) into one class. Show top-3 with tap-to-confirm when confidence is low.
   - MediaPipe Holistic on CPU: about 40 ms per frame.

2. **Calibrate the demo signer on every demo sign: 3 examples each.**
   - On stand-in signers (the hardest case, PopSign selfie style), fine-tuning on 3 examples per sign took
     top-1 from 35% to 67% (top-3 82%) in ~24 s of CPU. With an ASL Citizen-style camera the starting point
     is ~73%, so expect better, but this was not measured.
   - Recording ≈ 2 min for ~50 signs (plus retakes). Fine-tune 10 epochs, LR 1e-3, mixed 1:1 with replayed
     base training clips.
   - **Calibrating only some signs makes the others worse.** Calibrate all of them, or none.
   - Embedding prototypes are faster but weaker (58% at k = 3).

3. **Phrase mode works as *script matching*, not free recognition.**
   - Train a CTC model (same encoder, initialized from the isolated model) on synthetic sentences made by
     stitching trimmed citation clips with 80–250 ms minimum-jerk transitions, plus OOV signs as `<unk>`.
     About 55 min of CPU.
   - Decode by scoring every scripted café line with CTC forced alignment and take the top 3: right line in
     the top 3 **97%** of the time on unseen signers, top-1 89%. A Claude reranker (haiku 5.5, structured
     JSON, low effort, with the barista's previous line) lifts top-1 to **97%** for ~1.3 s and ~$0.0001 per call.
   - Free recognition of arbitrary sentences is not good enough: 63% recall on synthetic, 5–26% on real
     RWTH sentences. Real continuous signing is the open risk. Keep sign-by-sign with short holds as the fallback.
   - Delay is fine: 300 ms hands-at-rest hold + ~30 ms compute.

4. **Trim every clip to its active segment.** Wrist raised 0.15 shoulder widths above the clip's resting
   height and moving > 0.3 sw/s, 0.15 s hysteresis, static signs keep 0.3 s, pad 4 frames. Removes ~47% of
   each citation clip with < 0.5 point accuracy loss. Use it for the avatar, the synthetic sentences and the
   recognizer's segmenter.

5. **Core vocabulary ≈ 50 signs, chosen for robustness, and edit the script around weak signs.**
   - `results2/vocab_core.json`. Weak on new signers: CARD, WANT, WHAT, BLUEBERRY, YOU, CAN, THANK-YOU
     (confused with GOOD). Robust: COFFEE, NAME, KIND, WHICH, BYE, SORRY, NONE, MILK.
   - Proposed edits (all need Deaf advisor review): HELLO COFFEE PLEASE; MUFFIN KIND WHICH; PAY FS-CARD;
     THANK-YOU BYE (never GOOD next to THANK-YOU); fingerspell BLUEBERRY/OAT.

6. **Facial grammar: brows tell question *type*, not question *vs statement*.**
   - Once an utterance is a question, sustained lowered brows separate wh- from yes/no questions (AUC ≈ 0.92–0.95).
   - Raised brows also mark topics and conditionals, so a whole-utterance "brows up → ?" rule mislabels
     topic-comment statements. Combine the wh-sign in the gloss, *where* the raise is, and the LLM.
   - Headshake → negation works (AUC ≈ 0.93). Calibrate per signer on the first seconds.
   - Face size is not the bottleneck: brow readings stay stable down to a 40 px-wide face (smallest tested).
     Aim for ≥ 60 px, i.e. within ~3 m of a 1080p landscape phone camera; head-to-waist framing is far bigger.

7. **The pitch slide is real.** An off-the-shelf expression classifier called 10 of 13 wh-questions angry.
   Separate grammar from affect; affect must not use the brows.

8. **Avatar: trim, direct transitions, 1.3×, snapped handshapes, and a model with visible brows.**
   - Round 1's 9–14 s for a 6-sign sentence becomes ≈ 4.8 s at 1.3× with trimmed clips and direct 160 ms
     minimum-jerk transitions (rest only at the end). Internal-hold compression adds almost nothing.
   - Retarget arms by aligning bone directions to shoulder→elbow→wrist (kalidokit's arm solver is wrong for VRM 1.0).
   - Fingers: set them from a canonical handshape table keyed by ASL-LEX handshape (finger-angle error
     26° → 17° vs raw landmarks). Image landmarks underestimate curvature: C, O, flat-O, curved-5 need hand-tuned
     values. Thumb: the metacarpal's −y axis folds it across the palm; use ASL-LEX ThumbPosition.
   - The sample VRM's bangs hide the brows; cutting them leaves a ragged fringe. Make a VRoid model with short bangs.
   - **MediaPipe cannot see an avatar's hands** (≈ 3–7% of frames), so an automatic recognizer round trip
     cannot measure avatar legibility. Only Deaf reviewers can.
   - First motion ≈ 10 ms after the gloss JSON. Fallback: captions plus pre-rendered clips for the core phrases.

9. **Voice: the words carry the emotion; the tag barely does.**
   - Cartesia emotion settings produced no measurable change on neutral café sentences. Express affect
     through speed and volume; send the tag only when the English itself is emotional.
   - Never let the LLM add emotional words the signer did not sign.
   - First audio ≈ 135 ms on a warm WebSocket. The listening test is still not run.

10. **LLM glue: pass the previous turn, use structured output, validate and retry.**
    claude-haiku-5-5, JSON schema, low effort, thinking off: 1.1–1.5 s per call. Validate English → gloss
    against the vocabulary and send invalid glosses back once (9/30 → 0/30 rejections).

11. **Speech → text is solved; use eager end.** Cartesia Ink 2 `turn.eager_end` ≈ 0.35 s after speech ends.

12. **Licensing decides the demo assets.** ASL Citizen: non-commercial research, no redistribution of data or
    derivatives (avatar poses included). WLASL: research use (C-UDA). ASL-LEX: CC BY-NC 4.0. PopSign: CC BY 4.0.
    Plan for the Deaf advisor to perform the core phrases for the avatar lexicon, with consent.

13. **Windows and environment gotchas.**
    - `uv run python -m …` (Application Control blocks `.exe` shims).
    - The MediaPipe Tasks Holistic Landmarker replaces `mp.solutions`.
    - Hidden or embedded browser panes render WebGL in software: render the avatar in real Edge through
      Playwright (`--use-angle=d3d11`).
    - Many WLASL sources block scripts (signingsavvy 403, aslpro Flash); about half the videos download.
