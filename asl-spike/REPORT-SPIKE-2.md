# REPORT-SPIKE-2: early research spike, round 2; not product code

Early research spike, round 2, run 2026-10-16 on the same Windows 11 laptop (Intel Core Ultra 7 258V, 32 GB, CPU only; Intel Arc 140V GPU for WebGL through Microsoft Edge). Every number comes from `results2/`. Splits are signer-disjoint; CIs are 95% bootstrap over signers. Gloss, handshape and facial-grammar choices are by hearing non-signers and need Deaf advisor review.

## What changes in the build plan

1. **Camera: phone on a stand, landscape, both hands free, signer framed head to waist.** On new signers with ordinary two-handed video (WLASL) the round-1 model already gets 72.9% top-1; the 34% on PopSign came from its one-handed, phone-in-hand selfie style, not from new people. Correct x for the camera aspect ratio (x·w/h) before any geometry.
2. **Calibrate every sign the demo uses, with the demo signer, before the demo: 3 examples per sign.** On stand-in signers this took top-1 from 35.4% to 67.3% (top-3 81.5%) in ~24 s of CPU; signing time ≈ 1.9 min for 53 signs (more with pauses and retakes). Calibrating only some signs makes the others worse — calibrate all of them. Keep tap-to-confirm on the top 3.
3. **Phrase mode: yes, but only as script matching with a reranker.** A CTC model trained on synthetic sentences picks the right café line among the top 3 97.0% of the time on unseen signers, and a Claude reranker lifts top-1 from 87.0% to 97.0%. Free recognition of arbitrary sentences is not good enough (vocabulary recall 62.8% on synthetic, 20.8% on real RWTH sentences). Fallback: sign-by-sign with short holds.
4. **Delay is not the problem:** the phrase model's final hypothesis arrives 332 ms (p90 362) after the last sign, including the 300 ms hands-at-rest hold; use the bidirectional model.
5. **Core vocabulary of ~50 signs** (`results2/vocab_core.json`) with script edits that avoid weak or confusable signs (THANK-YOU↔GOOD, WHAT↔COST, CARD); all need Deaf advisor review.
6. **Avatar: trim clips to the active segment, direct 160 ms minimum-jerk transitions, play at 1.3×** — a typical 6-sign sentence drops from 9–14 s (round 1) to ≈4.77 s. Snap fingers to ASL-LEX handshapes (finger-angle error 26.2° → 16.7°), but curved shapes (C, O, flat-O) need a hand-tuned table, and the sample model's bangs must be replaced by a model with visible brows.
7. **Face size is not a constraint at normal distances:** brow readings held up down to the smallest face tested (40 px wide; the native ASL Citizen median is 128 px). Aim for ≥ 60 px for margin — on a 1080p phone in landscape that is within ≈ 2.9–3.4 m of the camera, far beyond a head-to-waist framing.

## Decision gates

| Gate | Pass if | Measured | Verdict |
|---|---|---|---|
| R1: PopSign gap explained | Two-handed signs account for most of the drop | one-handed 39.7% [34.5–44.8] vs two-handed 19.2% [17.0–21.5] top-1; of a gap of 51.1 points, one-handed signing explains 6.8 points, variants 2.0 points, aspect ratio 2.8 points; 42.3 points remain (PopSign's selfie style) | **FAIL** (informational) |
| R2: new signers, two-handed, no retraining (WLASL) | ≥ 70% top-1 or ≥ 90% top-3 | 72.9% [70.2–75.6] top-1, 86.8% top-3 (n = 1146 clips, 48 signers, 136 classes) | **PASS** |
| R3: calibration (base model) | ≤ 3 examples/sign: ≥ 90% top-1, ≥ 97% top-3, ≤ 2 min CPU | k = 3: 67.3% [57.4–75.7] top-1, 81.5% [72.4–88.6] top-3, 24 s CPU (6 PopSign stand-in signers × 3 draws) | **FAIL** |
| R4: robust training | WLASL held-out top-1 +10 points, ASL Citizen ≤ 2 points drop | best 'baseline': WLASL 71.7% vs 71.7%, ASL Citizen 84.4% vs 84.4% (3 seeds) | **FAIL** (informational) |
| P: phrase mode, synthetic free sentences | recall ≥ 80% and WER ≤ 30% | recall 62.8% [57.1–68.5], WER 44.1% [38.9–49.5] (2,000 sentences, 11 unseen signers) | **FAIL** |
| P: phrase mode, script lines | top-3 ≥ 95% and top-1 ≥ 85% | top-3 97.0% [94.2–99.4], top-1 83.1% [76.2–89.4] with the out-of-script fallback (89.0% without it; 97.0% with the Claude reranker on 100 items) | **PASS** (with reranker or without fallback) |
| P: real sentences (RWTH) | recall ≥ 40% at precision ≥ 40% (round 1: 18.6% / 22.8%) | synthetic only 5.5% / 20.8%; + 2 RWTH signers 20.8% / 70.4%; + same-signer calibration 26.0% / 67.5% | **FAIL** |
| P: delay | final hypothesis ≤ 1.0 s after the last sign (incl. 300 ms hold) | bidirectional 332 ms median, 362 ms p90; streaming 300 ms | **PASS** |
| V: avatar duration | 6-sign sentence ≤ 5 s at ≤ 1.3× | typical 6-sign sentence 4.77 s at 1.3× (6.11 s at 1.0×); the 3 round-1 test sentences 5.8, 7.1, 5.1 s at 1.2× | **PASS** (typical) / long signs exceed |
| V: avatar legibility | (c) beats (a) on round-trip top-3; path ≥ 90% | round-trip top-3 (c) 2.1% vs (a) 2.1% — MediaPipe finds the avatar's hands in only 7.1% of frames (median), so the proxy is unusable; fallback: finger-angle error 26.2° (a) → 16.7° (c); path ratio median 100%, 2 signs < 90% | **INCONCLUSIVE** |
| F1 (optional) | report minimum face width | brows stable down to 40 px (smallest tested) | measured |

## Human steps needed

| Step | Why | Status |
|---|---|---|
| WLASL missing-videos request form (https://github.com/dxli94/WLASL, links arrive within ~7 days) | only 1146/2202 (52%) instances downloaded: signingsavvy/handspeak/aslsignbank/aslsearch refuse scripts (403/404), aslpro is Flash (.swf) | needed (below 60%) |
| ASLLRP DAI account (https://dai.cs.rutgers.edu/dai/s/dai) | real continuous signing with time-aligned glosses and yes/no questions (P5, F2) | optional; not set (F2 not run) |
| An avatar with visible brows (make one in VRoid Studio with short bangs; put it in `avatar/models/` and load with `?model=models/<name>.vrm`) | removing the sample model's bangs leaves a ragged fringe | optional |
| Round-1 Cartesia listening test (`listening/index.html`, 3–5 people) | still not run | open |
| Deaf advisor review (checklist below) | handshapes, avatar, brows, core vocabulary, script edits | open |
| OSF (https://osf.io/zpha4/) was down for maintenance; ASL-LEX 2.0 was taken from asl-lex.org's own data file | re-check against the OSF CSV when it is back | optional |

## Reuse audit

| Item | Found at | Status |
|---|---|---|
| ASL Citizen landmark cache (5,958 clips, 143 classes; hands, pose, pose_world, 24 face points, blendshapes) | `cache/s2/citizen/*.npz` | reused |
| Round-1 feature pipeline | `spike/features.py` (`from_holistic`, `normalize`), `spike/extract.py` | reused unchanged |
| Round-1 models (ASL Citizen only; + PopSign) | `cache/s2/model.pt`, `cache/s2/model_with_kaggle.pt` | reused (base for R1–R3, P1, V1, V4) |
| Gloss alias table / class map | `vocab.json` (`asl_citizen_class_videos`, `kaggle_asl_signs`, `wlasl`) | reused |
| Kaggle asl-signs (PopSign) features | `cache/s2/kaggle_feats.npz` (32-frame features only) | reused; raw landmarks re-read from the zip for R1 hand statistics; aspect-corrected features regenerated (`cache/s2r2/kaggle_feats_aspect.npz`) |
| RWTH-BOSTON-104 + round-1 harness (75 sentences with ≥ 2 vocabulary signs) | `data/rwth-boston-104/`, `cache/s3/rwth/*.npz`, `results/s3_rwth_ncslgr_match.json`, `spike/s2_continuous.py` | reused (same 75 sentences) |
| NCSLGR SignStream 2 decoder | `spike/ss2.py` | reused (via round-1 match file) |
| Avatar page, P11 lexicon, 3 sentences | `avatar/index.html`, `avatar/signs.json` (16 signs), `avatar/sentences.json` | kept; round 2 builds `avatar/index2.html` + `avatar/signs2.json` (all 143 signs) |
| S5 outputs (30 barista lines → gloss + NMM) | `results/s5_*.jsonl` | reused for V1/V4 durations |
| Missing: hand world landmarks | not cached in round 1 | V2 uses aspect-corrected image landmarks instead (approximation) |
| Missing: OOV sign landmarks for P2 | only the 143 classes were extracted | regenerated for 60 seeded other ASL Citizen glosses (train split fully, val/test 4 clips per gloss) |
| Missing: WLASL, ASL-LEX | new this round | downloaded |

## R0 · ASL-LEX linking

Source: ASL-LEX 2.0 sign properties (2,723 signs; CC BY-NC 4.0) from `asl-lex.org/visualization/data/sign_props.json` (OSF was down). Linked by ASL Citizen's `ASL-LEX Code` column: 143/143 classes. Sign types: {'AsymmetricalSameHandshape': 11, 'OneHanded': 71, 'AsymmetricalDifferentHandshape': 25, 'SymmetricalOrAlternating': 32, 'SymmetryViolation': 1, 'DominanceViolation': 3}. Dominant handshapes needed: 36; signs with a flexion change: 23; signs without path movement (static holds): 20. WELCOME1 has no code and no ASL-LEX entry (WELCOME2 → welcome_1 is used).

Classes whose merged variants disagree on sign type (the primary = most ASL Citizen videos is used): ALL (all=AsymmetricalSameHandshape, whole=SymmetricalOrAlternating); BANANA (banana_3=AsymmetricalSameHandshape, banana=AsymmetricalDifferentHandshape, banana_2=AsymmetricalDifferentHandshape, banana_4=AsymmetricalDifferentHandshape); CAN (can=SymmetricalOrAlternating, can_2=AsymmetricalDifferentHandshape); CUP (glass_3=OneHanded, glass_2=OneHanded, glass_1=AsymmetricalDifferentHandshape, cup=AsymmetricalDifferentHandshape); HOW (how=SymmetricalOrAlternating, how_2=AsymmetricalSameHandshape); INSIDE (in=AsymmetricalDifferentHandshape, inside=AsymmetricalSameHandshape); MEDIUM (middle=AsymmetricalDifferentHandshape, average=AsymmetricalSameHandshape); MILK (milk_2=SymmetricalOrAlternating, milk=OneHanded); MONEY (money_2=OneHanded, money=AsymmetricalDifferentHandshape); MY (mine=AsymmetricalDifferentHandshape, my=OneHanded); NONE (nothing_2=SymmetricalOrAlternating, none=AsymmetricalDifferentHandshape, nothing=SymmetricalOrAlternating); NUTS (nut=OneHanded, nut_2=AsymmetricalDifferentHandshape); RIGHT (right_2=OneHanded, right_3=AsymmetricalSameHandshape, right_1=SymmetricalOrAlternating); SAME (same_2=SymmetricalOrAlternating, same_1=OneHanded); SANDWICH (sandwich_4=SymmetricalOrAlternating, sandwich_3=SymmetricalOrAlternating, sandwich=AsymmetricalDifferentHandshape, sandwich_2=SymmetricalOrAlternating); SIT (sit=AsymmetricalDifferentHandshape, chair=DominanceViolation); SODA (pop_2=SymmetricalOrAlternating, pop_3=SymmetricalOrAlternating, soda=AsymmetricalDifferentHandshape, pop_4=SymmetricalOrAlternating, pop=AsymmetricalDifferentHandshape); WHAT (what_2=AsymmetricalDifferentHandshape, what_1=SymmetricalOrAlternating, W.H.A.T=OneHanded).

## R1 · Why PopSign scored 34%

| Slice (round-1 model, no retraining) | Classes | Top-1 [signer CI] | Top-3 |
|---|---|---|---|
| All overlapping signs | 53 | 33.7% [29.7–37.9] | 49.8% [44.8–54.7] |
| AsymmetricalDifferentHandshape | 3 | 24.5% [17.9–31.7] | 46.9% [39.1–54.9] |
| AsymmetricalSameHandshape | 3 | 7.7% [5.1–10.5] | 12.7% [8.5–17.2] |
| OneHanded | 37 | 39.7% [34.5–44.8] | 56.7% [50.4–62.4] |
| SymmetricalOrAlternating | 10 | 21.1% [19.2–23.1] | 35.5% [32.1–38.6] |

n = 20,358 sequences, 21 signers.

**Both hands detected** (fraction of hand frames with both hands):

| ASL-LEX sign type | Classes | PopSign | ASL Citizen |
|---|---|---|---|
| AsymmetricalDifferentHandshape | 3 | 0.3% | 64.6% |
| AsymmetricalSameHandshape | 3 | 0.5% | 69.9% |
| OneHanded | 37 | 0.4% | 8.3% |
| SymmetricalOrAlternating | 10 | 0.3% | 69.5% |

PopSign almost never shows two hands — for any sign type — consistent with signing while holding the phone. Two-handed classes signed one-handed: ALL, CAN, CHOCOLATE, FAST, FINISH, GO, HAVE, HOW-MANY, MILK, MY, NICE, NOW, SAME, SIT, WAIT, WINDOW. Likely different variants (under 15% top-1 and a different dominant-hand location or flexion): MORNING, TIME, WHY.

**Decomposition** (class-level counterfactuals): ASL Citizen test top-1 on the same classes 84.8% vs PopSign 33.7% — a gap of 51.1 points. One-handed signing explains 6.8 points, variants 2.0 points; even the unaffected classes only reach 42.5%, leaving 42.3 points.

**Framing and aspect ratio.** PopSign faces are bigger in frame (median face width 28% vs 19% of image width) but the shoulders are in frame and the shoulder/face ratio is similar (1.966 vs 2.1). The hand looked 32% smaller relative to the face, which pointed to aspect ratio: MediaPipe's normalized x is divided by width and y by height, and PopSign is portrait (estimated w/h 0.73) while ASL Citizen is 4:3. Correcting x (×0.55) raises PopSign top-1 from 33.7% to 36.5% [32.6–40.4] — real but small.

**Conclusion.** Two-handed signs signed one-handed explain only part of the drop (~7 points), variants ~2 points and aspect ratio ~3 points. Most of the gap is PopSign's recording style itself (phone held in one hand, close-up selfie, casual fast signing). R2 confirms it: new signers with ordinary two-handed video do much better.

## R2 · New signers, two-handed, no retraining (WLASL)

Downloaded 1146/2202 instances of glosses mapping to our classes (asllex excluded: 106). Success by source: asl5200 132/132, aslbrick 59/59, asldeafined 178/178, aslpro 0/171, aslsearch 0/168, aslsignbank 0/108, aslu 224/227, elementalasl 6/6, handspeak 1/218, lillybauer 0/62, nabboud 0/59, northtexas 47/56, scott 18/18, signingsavvy 0/259, signschool 206/206, spreadthesign 132/132, startasl 103/103, valencia-asl 40/40. Failure reasons: HTTPError: HTTP Error 403: Forbidden (259); HTTPError: HTTP Error 404: Not Found (217); aslpro Flash .swf (unsupported video format) (171); RuntimeError: tiny response (114 bytes) (168); URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED] ce (108); RuntimeError: ERROR: unable to download video data: HTTP Err (9); RuntimeError: ERROR: The downloaded file is empty (2); RuntimeError: ERROR: [youtube] HZMTQJdBZ9g: This video is un (1); RuntimeError: ERROR: [youtube] Lk2Vws8homw: This video is un (1); RuntimeError: ERROR: [youtube] 6gU3Ln98sWA: This video is un (1); RuntimeError: ERROR: [youtube] Ddd2ZZNFhVI: This video is un (1); RuntimeError: ERROR: [youtube] j2hvnGiBrRI: This video is un (1). Clips cut to frame_start/frame_end on WLASL's 25 fps timeline; landmarks with the identical round-1 pipeline.

| Features | Top-1 [signer CI] | Top-3 | Top-5 |
|---|---|---|---|
| round1_pipeline | 72.3% [69.9–74.9] | 87.0% [85.2–89.0] | 90.8% [89.1–92.5] |
| aspect_fix | 72.9% [70.2–75.6] | 86.8% [84.7–89.2] | 91.4% [89.8–93.0] |

n = 1146 clips, 48 signers, 136 classes.

| Breakdown (aspect fix) | Group | Top-1 [CI] | Top-3 |
|---|---|---|---|
| sign_type | AsymmetricalDifferentHandshape | 65.1% [58.6–71.5] | 83.4% |
| sign_type | AsymmetricalSameHandshape | 74.0% [64.1–82.1] | 86.5% |
| sign_type | DominanceViolation | 70.8% [58.3–83.3] | 75.0% |
| sign_type | OneHanded | 74.5% [71.4–78.3] | 89.0% |
| sign_type | SymmetricalOrAlternating | 74.5% [69.5–79.7] | 85.2% |
| sign_type | SymmetryViolation | 62.5% [25.0–87.5] | 100.0% |
| variation | 0 | 74.8% [72.2–77.6] | 88.6% |
| variation | >=1 | 36.8% [23.3–49.1] | 52.6% |
| youtube | False | 73.1% [68.9–76.8] | 87.6% |
| youtube | True | 72.5% [70.3–77.6] | 85.7% |
| source | asl5200 | 69.7% [69.7–69.7] | 81.8% |
| source | aslbrick | 78.0% [68.4–83.9] | 93.2% |
| source | asldeafined | 69.7% [54.0–79.5] | 87.1% |
| source | aslu | 71.9% [63.6–72.3] | 84.8% |
| source | elementalasl | 50.0% [0.0–75.0] | 83.3% |
| source | handspeak | 0.0% [0.0–0.0] | 0.0% |
| source | northtexas | 74.5% [74.5–74.5] | 93.6% |
| source | scott | 77.8% [66.7–83.3] | 94.4% |
| source | signschool | 70.9% [64.2–77.4] | 85.9% |
| source | spreadthesign | 73.5% [66.0–79.3] | 84.1% |
| source | startasl | 82.5% [80.4–85.1] | 94.2% |
| source | valencia-asl | 80.0% [60.0–83.3] | 90.0% |

Weakest signs (top-1, instances/signers): CARD 0% (3/3), PREFER 0% (6/4), GIVE 10% (10/6), SMALL 11% (9/7), THANK-YOU 17% (6/6), WANT 20% (10/9), WHAT 22% (9/9), BLUEBERRY 33% (3/3), MEDIUM 33% (9/6), MONEY 33% (6/5), SIGN 35% (20/14), YOU 36% (11/8), CAKE 40% (5/5), LATER 40% (10/7), MINUTE 40% (5/5).

Signer-disjoint split for R4 saved to `results2/wlasl_split.json`: {'test_signers': 14, 'val_signers': 5, 'train_signers': 29, 'test_class_coverage': 1.0}.

## R3 · Calibration curve

Stand-in demo signers: the 6 held-out PopSign signers, 53 signs, aspect-corrected features (conservative: one-handed selfie style). Tuned on validation signer 55372: fine-tune 10 epochs at LR 0.001 (calibration clips mixed 1:1 with replayed ASL Citizen training clips); prototype mixing α = 1.0. 3 draws per signer.

**round-1 base model**

| k | Fine-tune top-1 | top-3 | CPU s | Prototypes top-1 | top-3 | Recording (min) |
|---|---|---|---|---|---|---|
| 0 | 35.4% [25.9–43.6] | 51.0% [38.8–60.6] | 0.0 | 35.4% [25.9–43.6] | 51.0% [38.8–60.6] | 0.0 |
| 1 | 55.2% [45.1–64.1] | 70.8% [60.5–79.1] | 8.4 | 46.1% [32.1–58.3] | 61.0% [46.3–73.4] | 0.6 |
| 2 | 62.0% [52.0–70.5] | 78.0% [68.9–85.2] | 16.54 | 53.5% [40.2–65.0] | 70.4% [56.5–81.4] | 1.2 |
| 3 | 67.3% [57.4–75.7] | 81.5% [72.4–88.6] | 23.89 | 57.9% [44.5–68.9] | 73.9% [60.6–84.2] | 1.9 |
| 5 | 71.2% [61.7–78.9] | 84.7% [76.8–90.8] | 40.28 | 61.5% [48.4–72.0] | 77.1% [64.2–86.8] | 3.1 |

Partial calibration (k = 3 on some signs, test on the others): fine-tune on 10 signs → 23.4% [15.7–30.4], 25 signs → 27.1% [19.8–33.6], vs uncalibrated 35.4% [25.6–43.6]. Calibrating some signs does **not** fix the style for the others; fine-tuning makes the uncalibrated signs worse (forgetting), and prototypes with α = 1 always pick a calibrated sign.

Recording time = k × signs × median PopSign clip (0.7 s): signing time only — pauses between takes, setup and retakes will multiply it several times.

## P1 · Active-segment trimmer

Method (to re-implement in the main build): x·(w/h); dominant hand = more wrist motion; height h = (shoulder-midline y − wrist y)/shoulder width (missing hand = at rest); rest = 10th percentile of h in the clip; speed = wrist speed in shoulder widths/s (3-frame mean). Active = raised (h > rest + **0.15**) and fast (> **0.3** sw/s) from the first to the last such frame; hysteresis: extend up to **0.15 s** while h > rest + RISE/2; static-hold signs (ASL-LEX movement None) keep ≥ **0.3 s** after the last fast frame; pad **4 frames**. Thresholds tuned on ASL Citizen validation signers (grid in `results2/p1_trimmer.json`, padding in `results2/p1_pad.json`).

| | Value |
|---|---|
| Removed per clip (test, pad 2) | median 53%, p90 69% |
| Removed per clip (final, pad 4) | median 47%; clip 2.47 s → 1.23 s |
| Round-1 classifier, ASL Citizen test, untrimmed → trimmed (pad 4) | 84.4% → 84.0% top-1 (93.0% → 92.6% top-3); n = 2376 clips, 11 signers |
| Validation drop with pad 2 / pad 4 | 1.5 points / 0.3 points |

20 contact sheets (lead-in, onset, middle, offset, lead-out): `results2/p1_contact/` (local; ASL Citizen frames).

## V1 · Avatar: trim, direct transitions, speed

Lexicon: one ASL Citizen training clip per class (143 signs; signers {'P11': 117, 'P40': 11, 'P31': 9, 'P51': 2, 'P43': 2, 'P37': 2}). Per sign: P1 active segment; internal holds (< 0.3 sw/s for > 0.15 s) compressed to 0.1 s except static signs — the hold sweep showed citation signs have almost no internal holds (higher thresholds cut real movement: `results2/v1_hold_sweep.json`). Direct 160 ms minimum-jerk transitions; rest only at the end (300 ms).

| | Value |
|---|---|
| Median sign: whole clip → active segment | 1.67 s → 0.84 s |
| Dominant-wrist path length kept | median 100%, min 78%; below 90%: BREAKFAST, CUP |
| Round-1 classifier on lexicon landmarks, full → processed | 100.0% → 99.3% top-1 (training clips: optimistic) |
| Typical 6-sign sentence at 1.0 / 1.2 / 1.3 / 1.4× | 6.11 / 5.14 / 4.77 / 4.45 s |

| Sentence | Signs played | Round 1 (s) | 1.0× | 1.2× | 1.4× | Signs/s at 1.2× |
|---|---|---|---|---|---|---|
| round1:wh: And what size: small, medium or large? | 6 | 11.0 | 6.8 | 5.8 | 5.0 | 1.04 |
| round1:yn: Why don't you try the chocolate cake? It's really | 6 | 13.6 | 8.5 | 7.1 | 6.1 | 0.84 |
| round1:neg: Sorry, we're out of oat milk today. | 5 | 9.3 | 6.1 | 5.1 | 4.5 | 0.97 |
| s5:Hi there, welcome in! What can I get started for you? | 6 | 11.9 | 7.5 | 6.3 | 5.5 | 0.95 |
| s5:Sure thing. Hot or iced? | 4 | 7.5 | 3.9 | 3.3 | 2.9 | 1.22 |
| s5:And what size: small, medium or large? | 6 | 11.0 | 6.8 | 5.8 | 5.0 | 1.04 |
| s5:Do you want any milk in it? | 3 | 4.9 | 3.3 | 2.8 | 2.4 | 1.08 |
| s5:We do. We have oat, almond, soy, and regular. | 2 | 3.2 | 2.6 | 2.2 | 1.9 | 0.9 |
| s5:Got it. Anything to eat? The muffins are fresh this morni | 4 | 7.2 | 3.8 | 3.3 | 2.8 | 1.23 |
| s5:Blueberry, chocolate chip, and banana nut. | 4 | 10.2 | 5.8 | 4.9 | 4.2 | 0.82 |
| s5:Of course. Is that for here or to go? | 2 | 4.1 | 2.4 | 2.0 | 1.8 | 0.99 |
| s5:Can I get a name for the order? | 2 | 3.3 | 1.9 | 1.7 | 1.5 | 1.21 |
| s5:Thanks, Maya. That comes to seven fifty. | 3 | 5.3 | 2.9 | 2.5 | 2.1 | 1.22 |
| s5:Seven dollars and fifty cents. | 4 | 8.2 | 5.6 | 4.7 | 4.1 | 0.85 |
| s5:Yep, just tap right here. Do you want a receipt? | 3 | 5.2 | 2.8 | 2.4 | 2.1 | 1.26 |
| s5:Your drink will be ready at the end of the counter. | 2 | 3.5 | 2.1 | 1.8 | 1.6 | 1.11 |
| s5:Right over there, by the window. | 3 | 6.6 | 4.2 | 3.6 | 3.1 | 0.84 |
| s5:Sorry, we're out of oat milk today. | 5 | 9.3 | 6.1 | 5.1 | 4.5 | 0.97 |
| s5:Would you like whipped cream on that? | 3 | 6.4 | 3.7 | 3.1 | 2.7 | 0.96 |
| s5:We don't take cash, only card. | 2 | 5.0 | 4.6 | 3.9 | 3.3 | 0.52 |
| s5:Is that everything? | 2 | 4.1 | 2.7 | 2.3 | 2.0 | 0.88 |
| s5:Where would you like to sit? | 4 | 8.7 | 4.4 | 3.8 | 3.3 | 1.06 |
| s5:It'll be about five minutes. | 2 | 3.5 | 2.2 | 1.9 | 1.7 | 1.06 |
| s5:Do you want it decaf? | 2 | 3.3 | 1.7 | 1.5 | 1.3 | 1.37 |
| s5:Who is the order for? | 2 | 3.1 | 1.8 | 1.6 | 1.4 | 1.27 |
| s5:Why don't you try the chocolate cake? It's really good. | 6 | 13.6 | 8.5 | 7.1 | 6.1 | 0.84 |
| s5:The bathroom is in the back, on the left. | 3 | 6.9 | 4.5 | 3.8 | 3.3 | 0.8 |
| s5:Do you need a straw? | 3 | 6.0 | 3.1 | 2.6 | 2.3 | 1.16 |
| s5:Sorry, I didn't understand. Can you show me again? | 5 | 8.9 | 4.6 | 3.9 | 3.4 | 1.29 |
| s5:How many sugars? | 2 | 2.9 | 1.7 | 1.4 | 1.3 | 1.4 |
| s5:We close at nine. | 1 | 1.6 | 1.0 | 0.9 | 0.8 | 1.1 |
| s5:That's not on the menu, sorry. | 2 | 3.7 | 1.9 | 1.7 | 1.5 | 1.21 |
| s5:Have a great day! | 5 | 8.3 | 5.0 | 4.2 | 3.7 | 1.18 |

Median 1.06 signs/s at 1.2× vs ~2 signs/s in fluent signing: the citation clips are still slow because each ASL Citizen clip is a careful, isolated production; per-sign speed-up is the remaining lever.

## P2 · Synthetic continuous sentences

Generator (`spike2/p2_gen.py`): clips converted to body coordinates first (x·w/h relative to 4:3, shoulder-centered/scaled, dominant hand in the right slot) so clips from different videos join; P1 active segment with an extra random 0–20% trim at each end; per-sign speed 0.8–1.5×; 80–250 ms minimum-jerk transitions of all landmarks; optional lead-in/lead-out rest from the clip's own trimmed-off segments; out-of-vocabulary ASL Citizen signs (60 other glosses) inserted at ~15% of positions as <unk>; jitter (sd 0.01 shoulder widths) and 3% dropped hand frames. Content: 50% café lines (30 gloss lines; the first 14 are the scenario's customer turns), 35% bigram samples, 15% uniform; 2–7 signs. **Synthetic transitions are not real coarticulation; the RWTH tests are the check.**

| Set | Split | Sentences | Mean signs | <unk> rate | Mean frames | Signs needing a second signer |
|---|---|---|---|---|---|---|
| val | val | 500 | 4.57 | 18.5% | 175.4 | 0.0% |
| test | test | 2000 | 4.57 | 18.1% | 179.5 | 0.0% |
| val_script | val | 140 | 3.36 | 10.6% | 133.7 | 24.3% |
| test_script | test | 462 | 3.36 | 10.6% | 133.1 | 0.9% |

The sentence signer is the one covering most of its signs (ties broken at random); every free sentence came from a single signer.

## P3 · Phrase-mode model

Encoder as round 1 (Linear 228→192, 3 residual Conv1d k = 5, GRU) initialized from the round-1 isolated model where shapes match; CTC head over 143 classes + <unk> + blank. Bidirectional: 794,065 parameters; streaming (causal convolutions, unidirectional GRU 192): 849,361. Batches of 16 generated on the fly (train-split signers); AdamW, gradient clip 1.0. 1 seed per variant (time).

| Model | Training | Steps | CPU minutes |
|---|---|---|---|
| (a) bidirectional, synthetic | from round-1 weights, LR 1e-3 then 5e-4 | 1423 + 782 | 55 |
| (a) streaming, synthetic | from round-1 weights, LR 1e-3 | 1313 | 20 |
| (b) + RWTH, 2 of 3 signers (3 folds) | from (a), half the batches real RWTH sentences, LR 5e-4 | 627, 745, 767 | 8 each |
| (c) + half of the held-out signer's sentences | from (b), LR 3e-4 | 456, 456, 431 | 5 each |

## P4–P5 · Decoding and phrase-mode evaluation

Decoders: greedy; CTC prefix beam (width 10) with a gloss bigram LM from the café lines (LM weight 0.5, insertion bonus 0.5); script-constrained (CTC forced-alignment log-likelihood of each of the 30 café gloss lines, top 3) with an out-of-script fallback (free decode wins if it beats the best line by > τ nats; τ = 12 chosen on validation as the smallest τ keeping script top-1 within 3 points of no fallback).

| Set (unseen ASL Citizen signers unless noted) | Model / decoder | Result |
|---|---|---|
| Free synthetic sentences (2,000; 11 signers) | bidirectional (55 min), greedy | WER 44.1% [38.9–49.5], recall 62.8% [57.1–68.5], precision 73.0% |
| Free synthetic sentences (2,000; 11 signers) | bidirectional (55 min), beam | WER 44.2% [37.1–48.0], recall 58.5% [53.6–66.9], precision 83.6% |
| Free synthetic sentences (2,000; 11 signers) | bidirectional (25 min), greedy | WER 49.6% [44.2–55.2], recall 53.0% [48.0–59.6], precision 74.5% |
| Free synthetic sentences (2,000; 11 signers) | bidirectional (25 min), beam | WER 53.5% [46.3–57.0], recall 45.7% [41.9–54.4], precision 86.6% |
| Free synthetic sentences (2,000; 11 signers) | streaming, greedy | WER 54.5% [49.5–58.7], recall 52.2% [48.3–59.4], precision 64.0% |
| Free synthetic sentences (2,000; 11 signers) | streaming, beam | WER 54.7% [47.9–56.6], recall 45.3% [43.0–54.5], precision 80.2% |
| Café script lines (462 = 14 lines × 33 renders, 11 signers) | bidirectional (55 min), script decoding | top-1 83.1% [76.2–89.4], top-3 97.0% [94.2–99.4]; fallback fired 6.3%; without fallback top-1 89.0% |
| Café script lines (462 = 14 lines × 33 renders, 11 signers) | streaming, script decoding | top-1 84.0% [76.8–90.9], top-3 93.3% [88.1–98.0]; fallback fired 2.2%; without fallback top-1 84.9% |
| Café script lines, 100 seeded items | + Claude reranker (claude-haiku-5-5) | top-1 87.0% → 97.0%, top-3 98.0%; latency 1348 ms median, 1702 p90; cost $0.01 |
| RWTH-BOSTON-104 (75 sentences, 3 signers, leave-one-signer-out) | synthetic only, greedy | recall 5.5% [1.5–7.2], precision 20.8% (10/183 vocabulary signs) |
| RWTH-BOSTON-104 (75 sentences, 3 signers, leave-one-signer-out) | + 2 RWTH signers, greedy | recall 20.8% [11.6–25.6], precision 70.4% (38/183 vocabulary signs) |
| RWTH-BOSTON-104 (44 sentences, 3 signers, leave-one-signer-out) | + same-signer calibration (other half), greedy | recall 26.0% [13.9–34.6], precision 67.5% (27/104 vocabulary signs) |
| RWTH-BOSTON-104, round 1 | sliding window | recall 18.6%, precision 22.8% |

**Delay** (end of the last sign known exactly in synthetic data; includes the 300 ms hands-at-rest hold): bidirectional final hypothesis 332 ms median / 362 p90 / 432 max (compute 32 ms per utterance on CPU); streaming: the model emits the last sign 0 ms (p90 67) after it ends, so the hold dominates (300 ms). Partial hypotheses: with the streaming model each sign appears as soon as it ends, but it is less accurate.

**Error analysis** (bidirectional, beam): deletions dominate (the model under-emits). Most common — substitutions: <unk>→INSIDE (23), CAN→<unk> (17), <unk>→TIME (16), LARGE→FAST (13), IX-1P→SORRY (12), THANK-YOU→GOOD (11), <unk>→MY (11), SAME→<unk> (10), MINUTE→<unk> (9), <unk>→MILK (9), GOOD→THANK-YOU (8), HOW-MUCH→COST (8), MILK→<unk> (7), CAKE→<unk> (7), MEDIUM→<unk> (7); deletions: <unk> (629), HOW-MUCH (70), NUTS (63), NOT (60), IX-1P (51), MILK (51), PAY (47), COFFEE (46), COOKIE (39), CARD (37), THANK-YOU (37), CAN (36), CAKE (33), WHAT (32), WANT (30); insertions: <unk> (15), COFFEE (11), HOT (7), SMALL (6), MILK (6), YES (6), HOW-MANY (5), WRONG (5), ONE (4), BLUEBERRY (4), WANT (3), COLD (3), HERE (3), UNDERSTAND (2), SANDWICH (2).

Limitations: 1 seed per phrase model; ~55 min of CPU training is short for CTC; synthetic transitions are interpolations, not coarticulation; RWTH is 2007 black-and-white video of 3 signers with mostly non-café vocabulary; DAI (ASLLRP) not available, so no boundary error.

## V2 · Handshape library and snapping

Canonical handshapes: frames from the middle 40% of P1 active segments of ASL Citizen training clips whose ASL-LEX entry has no flexion change, dominant hand mirrored to right-hand geometry, aspect-corrected image landmarks (hand world landmarks were not cached). Joint angles in a hand-local frame; robust median per handshape. The vocabulary needs 36 dominant handshapes; the library covers 37 labels (non-dominant labels B, C, 8, stacked_5 aliased to open_b, c, open_8, 5).

| Handshape | Frames | Clips | Signers | Index MCP/PIP | Middle MCP/PIP | Pinky MCP/PIP | Thumb MCP | Thumb abd |
|---|---|---|---|---|---|---|---|---|
| 1 | 5278 | 371 | 29 | 25/7 | 57/81 | 51/80 | 23 | 14 |
| open_b | 4080 | 276 | 30 | 27/6 | 16/5 | 21/9 | 10 | 10 |
| c | 2542 | 174 | 23 | 30/22 | 23/25 | 29/21 | 11 | 12 |
| o | 2239 | 147 | 22 | 40/25 | 34/35 | 37/25 | 19 | 12 |
| bent_1 | 1731 | 122 | 22 | 31/42 | 64/76 | 64/83 | 20 | 11 |
| 5 | 1720 | 122 | 21 | 17/7 | 12/7 | 16/10 | 10 | 9 |
| a | 1709 | 115 | 24 | 62/92 | 60/104 | 63/105 | 11 | 8 |
| f | 1487 | 89 | 23 | 42/38 | 20/19 | 19/15 | 18 | 11 |
| curved_5 | 1435 | 92 | 22 | 28/27 | 24/29 | 29/22 | 14 | 13 |
| s | 1323 | 87 | 21 | 70/84 | 76/92 | 73/93 | 21 | 15 |
| flat_b | 1104 | 74 | 22 | 55/19 | 55/19 | 47/18 | 10 | 11 |
| flat_o | 1048 | 74 | 18 | 45/12 | 43/14 | 41/17 | 22 | 13 |
| baby_o | 1013 | 58 | 19 | 33/50 | 48/77 | 64/73 | 18 | 9 |
| y | 955 | 67 | 20 | 48/105 | 42/109 | 14/11 | 7 | 12 |
| closed_b | 733 | 50 | 20 | 23/8 | 14/7 | 16/11 | 15 | 8 |
| w | 664 | 45 | 17 | 12/6 | 14/10 | 51/57 | 22 | 9 |
| i | 643 | 45 | 20 | 44/94 | 43/102 | 20/13 | 21 | 14 |
| h | 636 | 43 | 17 | 33/7 | 32/9 | 43/47 | 30 | 14 |
| curved_v | 501 | 30 | 20 | 25/44 | 23/48 | 43/90 | 27 | 14 |
| open_8 | 431 | 28 | 14 | 31/8 | 34/13 | 18/13 | 8 | 9 |
| curved_h | 430 | 28 | 16 | 32/6 | 28/9 | 52/30 | 28 | 11 |
| p | 418 | 27 | 15 | 21/4 | 39/9 | 38/110 | 28 | 15 |
| flat_1 | 406 | 30 | 19 | 34/6 | 74/91 | 67/95 | 31 | 14 |
| flatspread_5 | 397 | 31 | 15 | 35/24 | 31/23 | 30/18 | 16 | 10 |
| l | 387 | 27 | 17 | 20/6 | 76/74 | 84/73 | 12 | 14 |
| r | 334 | 28 | 16 | 24/7 | 12/11 | 40/100 | 31 | 10 |
| g | 292 | 14 | 13 | 34/28 | 79/66 | 96/65 | 13 | 12 |
| flat_h | 255 | 16 | 16 | 37/13 | 34/16 | 71/62 | 16 | 9 |
| spread_open_e | 252 | 15 | 14 | 30/46 | 34/48 | 34/37 | 16 | 10 |
| curved_1 | 249 | 16 | 14 | 20/25 | 30/130 | 34/114 | 38 | 10 |
| t | 213 | 14 | 14 | 35/72 | 43/112 | 63/109 | 28 | 4 |
| open_e | 213 | 14 | 14 | 41/14 | 28/19 | 20/19 | 9 | 8 |
| 4 | 207 | 15 | 15 | 10/5 | 6/4 | 8/5 | 40 | 13 |
| 7 | 198 | 14 | 13 | 8/5 | 10/5 | 11/10 | 35 | 9 |
| bent_l | 198 | 15 | 15 | 29/24 | 54/100 | 56/106 | 10 | 14 |
| 3 | 187 | 14 | 14 | 14/4 | 16/7 | 26/141 | 7 | 16 |
| v | 172 | 14 | 14 | 14/4 | 18/7 | 34/133 | 37 | 16 |

(All angles in degrees; full table with spread, DIP and thumb CMC/IP/opposition in `results2/handshapes.json`.)

Avatar mapping (`avatar/index2.html`): finger flexion = (angle − flat-hand baseline) × 1.3 (image landmarks underestimate flexion), about the bone's z axis (sign mirrored for the left hand); spread about y; the thumb's two base degrees of freedom were found by a render sweep (metacarpal about −y folds it across the palm) and its fold is set from ASL-LEX ThumbPosition (Closed → folded). Non-dominant hand: one-handed → relaxed; symmetrical/alternating or asymmetrical-same → the dominant shape; asymmetrical-different → the coded non-dominant shape. Flexion-change signs interpolate to a data-derived end shape over the active segment. Wrist rotation from the palm frame (wrist, index MCP, pinky MCP); One-Euro smoothing (min cutoff 1.2 Hz, β 0.4) of arm and hand landmarks; round-1 vector alignment for shoulder→elbow→wrist.

**Honest result:** straight and fist shapes read correctly (5, open_b, 1, l, y, s); curved shapes (c, o, flat_o, curved_5, f) come out nearly flat because image landmarks register little curvature; the thumb is rule-based, not data-driven. Contact sheets (avatar vs reference crop): `results2/v2_sheets/` (local; ASL Citizen frames).

## V3 · Brows and face

The sample model's hair is one mesh (bangs included). Change: at load, hair triangles whose bind-pose centroid lies in front of the forehead (within 7 cm of the face's front, between 58% and 72% of the face-box height, within 75% of its half-width) are removed — about 1,100 triangles. Clipping planes did not work (MToon materials ignored them). Result: brows visible, but the fringe looks ragged; a model with short bangs is the real fix (human step).

Non-manual markers: yes/no = Fcl_BRW_Surprised 1.0 + Fcl_EYE_Surprised 0.55 + head pitch forward 0.10 rad; wh = Fcl_BRW_Angry 1.0 + Fcl_EYE_Angry 0.35 + pitch 0.05 + roll 0.07 rad; negation = head yaw ±13.6° at 1.21 Hz, matched to round-1 S3 negations (n = 9; yaw-proxy sd 0.0831). Screenshots for advisor review: rendered with `spike2/v_shots.py face` (neutral / raise / lower); the raise/lower difference is visible but subtle.

## V4 · Avatar evaluation

| Configuration | Hand detection (median per sign) | Round-trip top-1 | Top-3 | Finger-angle error vs canonical (where detected) |
|---|---|---|---|---|
| (a) round-1 avatar | 3.1% | 0.7% | 2.1% | 26.2° (1324 frames) |
| (b) + V1 trim/transitions | 5.6% | 0.7% | 2.1% | 25.1° (991 frames) |
| (c) + V2 snapping, wrist, smoothing | 7.1% | 0.0% | 2.1% | 16.7° (961 frames) |
| (d) = (c) at 1.3× | 7.1% | 1.4% | 2.8% | 19.8° (775 frames) |
| Upper bound: human source landmarks | — | 100.0% | 100.0% | — |

Renders: Edge + Playwright on the Intel Arc GPU, 1280×720 at 30 fps, frontal head-to-waist framing, plain background. **MediaPipe finds the avatar's hands in only a few percent of frames** (it is trained on real hands), so the automatic round trip cannot measure legibility; the body pose is found (81.4% of frames). The fallback finger-angle comparison favors snapping. Sentence-level round trip (3 round-1 sentences + 5 barista lines through the phrase model) decoded essentially nothing for the same reason (`results2/v4_sentences.json`). Contact sheets per sign (human vs avatar key frames): `results2/v4_contact/`; capture of the 3 sentences: `avatar/capture2.mp4` (both local). **A classifier recognizing (or not recognizing) the avatar says nothing about whether Deaf people find it natural or correct; only advisor review can.**

## R4 · Robust training ablation

| Variant | ASL Citizen test top-1 (mean, range) | WLASL held-out top-1 | PopSign held-out top-1 | Train min/seed |
|---|---|---|---|---|
| baseline | 84.4% (84.3%–84.6%) | 71.7% (69.8%–73.5%) | 34.6% (34.0%–35.2%) | 13 |

Test sets: ASL Citizen 2376 clips (11 signers), WLASL 351 clips (14 signers), PopSign 5761 sequences (6 signers). 3 seeds each. Best variant by WLASL held-out top-1 (ASL Citizen within 2 points): **baseline**. PopSign training data is one-handed selfie style.

## R5 · Core vocabulary proposal (needs Deaf advisor review)

Robustness score = min(ASL Citizen test top-1, WLASL top-1 without retraining, post-calibration top-1 at k = 3 on PopSign stand-ins — only for the signs PopSign has) − 0.15 × share of errors going to the single most-confused sign. WLASL per-sign numbers rest on few instances (3–15), so they are noisy.

| Sign | Score | ASL Citizen | WLASL | Calibrated k=3 | Most confused with | Script needs it |
|---|---|---|---|---|---|---|
| CARD | -0.05 | 57% | 0% | — | HELP | yes |
| THANK-YOU | 0.042 | 50% | 17% | 55% | GOOD | yes |
| WANT | 0.17 | 79% | 20% | — | HERE | yes |
| WHAT | 0.178 | 54% | 22% | — | COST | yes |
| BLUEBERRY | 0.183 | 83% | 33% | — | BANANA | yes |
| YOU | 0.314 | 75% | 36% | — | SAME | yes |
| CAN | 0.327 | 83% | 88% | 40% | CHEAP | yes |
| NOT | 0.369 | 83% | 44% | 66% | CHANGE | yes |
| ONE | 0.4 | 100% | 40% | — | — | yes |
| HOW-MUCH | 0.405 | 46% | 56% | — | WHAT | yes |
| NEED | 0.406 | 80% | 78% | 53% | WHERE | yes |
| GOOD | 0.444 | 67% | 56% | — | THANK-YOU | yes |
| HOT | 0.471 | 73% | 57% | 72% | BAD | yes |
| YES | 0.489 | 100% | 77% | 49% | — | yes |
| AGAIN | 0.523 | 58% | 71% | — | SLOW | yes |
| MY | 0.527 | 85% | 70% | 60% | PLEASE | yes |
| PLEASE | 0.54 | 85% | 100% | 62% | MY | yes |
| IX-1P | 0.55 | 85% | 62% | — | LUNCH | yes |
| PAY | 0.564 | 92% | 71% | — | CAKE | yes |
| HAVE | 0.575 | 92% | 100% | 72% | PREFER | yes |
| WHERE | 0.581 | 85% | 100% | 66% | RIGHT | yes |
| LARGE | 0.6 | 75% | 79% | — | SIGN | yes |
| UNDERSTAND | 0.6 | 93% | 75% | — | CENT | yes |
| NO | 0.609 | 100% | 75% | 61% | — |  |
| HELLO | 0.618 | 93% | 100% | 77% | KNOW | yes |
| FINE | 0.639 | 94% | 71% | 76% | CHEESE | yes |
| HERE | 0.675 | 86% | 75% | — | SALAD | yes |
| SWEET | 0.677 | 73% | 83% | — | YOUR | yes |
| WARM | 0.7 | 75% | 100% | — | DELICIOUS | yes |
| DAY | 0.725 | 80% | 100% | — | TABLE | yes |
| COFFEE | 0.739 | 91% | 89% | — | SODA | yes |
| MILK | 0.773 | 91% | 100% | 85% | OUTSIDE | yes |
| NAME | 0.779 | 93% | 100% | — | COOKIE | yes |
| DRINK | 0.796 | 88% | 92% | 85% | FINE |  |
| STRAW | 0.8 | 100% | 80% | — | — |  |
| BREAKFAST | 0.802 | 84% | 100% | — | FINE |  |
| KIND | 0.807 | 96% | 100% | — | MONEY | yes |
| JUICE | 0.808 | 96% | — | — | LUNCH |  |
| PHONE | 0.818 | 100% | 82% | 84% | — |  |
| BANANA | 0.82 | 92% | 86% | — | CHANGE |  |
| STRAWBERRY | 0.825 | 88% | 100% | — | LUNCH |  |
| BREAD | 0.833 | 100% | 83% | — | — |  |
| SORRY | 0.857 | 100% | 86% | — | — | yes |
| NONE | 0.859 | 91% | 91% | — | AGAIN | yes |
| DIFFERENT | 0.9 | 100% | 90% | — | — |  |
| EIGHT | 1.0 | 100% | 100% | — | — |  |
| NINE | 1.0 | 100% | — | — | — |  |
| SEVEN | 1.0 | 100% | 100% | — | — |  |
| THREE | 1.0 | 100% | 100% | — | — |  |
| WITH | 1.0 | 100% | 100% | — | — |  |

**Proposed script edits** (all need Deaf advisor review; English stays the same, the signed form changes):

- `HELLO IX-1P WANT COFFEE PLEASE` → `HELLO COFFEE PLEASE` — WANT is weak on new signers (WLASL 20%, n = 5) and IX-1P/WANT add little in an order
- `YES YOU HAVE OAT MILK` → `OAT MILK HAVE? (raised brows)` — YOU is weak (WLASL 36%, confused with SAME); OAT is not in ASL Citizen (fingerspell)
- `MUFFIN WHAT KIND` → `MUFFIN KIND WHICH` — WHAT is weak and confused with COST; KIND (0.81) and WHICH (0.78) are robust
- `BLUEBERRY ONE WARM CAN` → `FS-BLUEBERRY ONE WARM PLEASE` — BLUEBERRY is confused with BANANA; CAN is weak after calibration (40%)
- `MY NAME FS-MAYA` → `NAME IX-1P FS-MAYA` — MY confused with PLEASE
- `SORRY HOW-MUCH AGAIN PLEASE` → `SORRY HOW-MUCH` — shorter; AGAIN is mid-strength
- `CARD PAY CAN` → `PAY FS-CARD` — CARD scored 0% on WLASL (n = 5); fingerspell or tap-to-confirm
- `FINE THANK-YOU GOOD DAY` → `THANK-YOU BYE` — GOOD and THANK-YOU are each other's top confusion; BYE is robust (0.67)

Signs the dialogue truly needs stay even when weak and are flagged: THANK-YOU (confused with GOOD), WANT, WHAT, CARD, NOT. Mitigations: avoid GOOD in the same exchange, rely on headshake for negation, tap-to-confirm.

## F1 · How big must the face be? (optional)

ASL Citizen test clips: 60 wh-sign clips, 60 non-wh; native face width median 128.3 px.

| Face width | Detection | corr(browDown) with native | corr(browInnerUp) | AUC wh vs non-wh (sustained browDown) |
|---|---|---|---|---|
| native | 100.0% | 1.0 | 1.0 | 0.708 |
| 40 px | 100.0% | 0.837 | 0.86 | 0.706 |
| 60 px | 100.0% | 0.927 | 0.938 | 0.712 |
| 80 px | 100.0% | 0.962 | 0.973 | 0.717 |
| 120 px | 100.0% | 0.99 | 0.992 | 0.708 |
| 160 px | 100.0% | 0.995 | 0.995 | 0.708 |

Criterion: detection >= 90%, browDown correlation with native >= 0.8, wh AUC within 0.05 of native. Every tested size passed, so the limit is **at or below 40 px** (the smallest size tested); at 40 px the browDown correlation (0.837) is close to the 0.8 bar, so use ≥ 60 px for margin. Maximum distance for 40 px: landscape 4.29–5.14 m, portrait 2.41–2.89 m (face width 0.15 m; 1080p camera; HFOV 70-80 deg (landscape) - portrait uses the 1080-px width); for 60 px multiply by 0.67. The wh-vs-non-wh AUC is only ≈ 0.71 even at native size: isolated citation signs do not reliably carry grammatical facial expressions, so this measures signal stability, not a gate.

## F2 · Facial grammar with timing rules (optional)

Not run: DAI_ACCESS is not set (no ASLLRP utterances with face close-ups).

## Numbers to carry into the build

| What | Value |
|---|---|
| Aspect correction | x_px = x · (w/h) before any geometry |
| Trimmer | rise 0.15 sw above the clip's 10th-percentile wrist height; speed 0.3 sw/s; hysteresis 0.15 s at rise/2; static holds ≥ 0.3 s; pad 4 frames |
| Avatar transitions | direct minimum-jerk, 160 ms; rest only at sentence end (300 ms) |
| Avatar playback speed | 1.3× (≈4.77 s for a typical 6-sign sentence) |
| Calibration | k = 3 per sign for every demo sign; fine-tune 10 epochs, LR 0.001, 1:1 replay of base training data; ~24 s CPU; ≈1.9 min recording for 53 signs |
| Phrase model | Conv1d + BiGRU, 794,065 params, CTC over classes + <unk> + blank; ~55 min CPU from round-1 weights |
| Decoder | script-constrained (CTC forced alignment over the café lines), top 3; out-of-script τ = 12 nats; Claude reranker (haiku 5.5) with the barista's last line |
| Hands-at-rest hold | 300 ms; bidirectional model compute ≈ 32 ms |
| Handshape table | `results2/handshapes.json` (data); curved shapes need hand-tuned values |
| Headshake | ±13.6° at 1.21 Hz (round-1 S3 negations) |
| Minimum face width | ≤ 40 px works; aim for ≥ 60 px |

## Needs Deaf advisor review

- Handshape contact sheets: `results2/v2_sheets/` (avatar canonical handshape vs a real signer's hand).
- Avatar contact sheets per sign: `results2/v4_contact/`; capture of the 3 sentences: `avatar/capture2.mp4`.
- Brow and headshake renders (neutral / raise / lower; headshake strip): render with `uv run python spike2/v_shots.py face <out.jpg>`.
- Core vocabulary (`results2/vocab_core.json`) and the script edits above; whether pointing (IX-1P/YOU) should be recognized at all.
- The non-dominant-hand rules, the thumb rules, and the 1.3× playback speed.

## Estimated API spend

Claude reranker only: 100 calls, 63,928 input / 7,249 output tokens ≈ **$0.01** (claude-haiku-5-5 at $0.10 / $0.50 per MTok). No other paid APIs were used this round.
