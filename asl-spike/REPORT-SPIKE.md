# REPORT-SPIKE: two-way ASL MVP feasibility (early research spike; not product code)

Early research spike, run 2026-10-16 on one Windows 11 laptop (Intel Core Ultra 7 258V, 32 GB, no CUDA; Intel Arc 140V for WebGL). Every number below comes from `results/`. n (clips, signers) is given beside each result. Glosses and facial-grammar choices are by hearing non-signers and need Deaf advisor review.

## Decision gates

| Gate | Pass if | Measured | Verdict |
|---|---|---|---|
| Recognition, unseen ASL Citizen signers | ≥ 80% top-1 on the ~150-sign vocabulary | 84.4% top-1, 93.0% top-3, 95.0% top-5 (143 signs; n = 2376 videos, 11 test signers); released ST-GCN: 81.1% top-1 | **PASS** |
| Recognition, different dataset (PopSign Deaf signers, phone cameras) | ≥ 70% top-1 (or ≥ 90% top-3) on overlapping signs | 33.7% top-1, 49.8% top-3 over 53 overlapping signs (n = 20,358 sequences, 21 signers; argmax over all 143 classes). With the other PopSign signers added to training: 76.0% top-1, 87.3% top-3 on 6 held-out PopSign signers (n = 5,761), ASL Citizen test unchanged at 83.9% | **FAIL as specified (no retraining); PASS with PopSign in training** |
| Continuous signing (RWTH-BOSTON-104 fallback; no DAI access) | ≥ 70% recall of vocabulary signs, ≤ 1.5 s after last sign | recall 18.6%, precision 22.8% at confidence 0.5 (n = 75 utterances, 3 signers); delay ≈ 300 ms rest hold + 208 ms compute | **FAIL** |
| Facial grammar (RWTH-BOSTON-104 + NCSLGR labels, leave-one-signer-out) | ≥ 85% sentence type on held-out signers | 81.2% accuracy but 57.4% balanced (statement vs wh, n = 85, 3 signers; no yes/no questions in this corpus); wh vs raised-brow questions AUC 0.923 | **FAIL (inconclusive data)** |
| Naive baseline contrast | Baseline mislabels most yes/no questions as surprise | wh-questions labelled anger/disgust: 10/13; raised-brow questions labelled surprise: 0/10 (rhetorical; no yes/no in corpus); statements labelled anger: 23/72 | **PARTLY (wh → anger confirmed; yes/no untested)** |
| Cartesia emotion | Intended emotion detected ≥ 60% by the classifier | tag only: 22.2% (chance 25%; no tag: 25.0%); best condition 'e': 47.9% (SpeechBrain; HuBERT 39.6%); the same emotional wording *without* the tag: 49.0%, so the gain is from the words, not the tag | **FAIL** |
| Avatar | 5-sign sentence with brow grammar, first motion ≤ 2 s | first motion 7 ms median (47 ms cold), 5–6-sign sentences with brow/headshake NMM; handshapes poor, brows hidden by the model's hair | **PASS (latency) / quality poor** |
| Glasses | ≥ 15 fps, ≤ 500 ms added delay via mock device | not run: deferred (S7) | **NOT RUN** |

## Human steps needed

| Step | Why | Status |
|---|---|---|
| Accept the Kaggle rules for asl-signs and asl-fingerspelling (browser, signed in) | Kaggle download | done |
| RWTH-BOSTON-104 download form | S2 continuous / S3 fallback data | done (archive placed in the repo) |
| ASLLRP DAI account (https://dai.cs.rutgers.edu/dai/s/dai), hand-approved | real ASLLRP/NCSLGR videos with face close-up and yes/no questions: the missing piece for the S3 gate | not requested; fallbacks used |
| Download How2Sign frontal clips in a browser (Google Drive refuses scripted downloads: 'too many users have viewed or downloaded this file'). Files: val/test 'Green Screen RGB clips (frontal)' from http://how2sign.github.io/ into `asl-spike/data/how2sign/` | S3 fallback (b), weak labels | blocked; only the translation CSVs downloaded |
| NCSLGR XML: the public XML link on the BU page is missing and the Rutgers mirror (secrets.rutgers.edu) does not respond | we decoded the public SignStream 2 database files instead (validated against the 3 XML files in the BU parser package) | worked around |
| Ask Microsoft (ASL_Citizen@microsoft.com) whether a public demo and its derived avatar poses/model weights are permitted | ASL Citizen license: non-commercial research, no distribution of data or modified data | open |
| Run the S4 listening sheet with 3–5 people (`listening/index.html`) | the classifiers disagree with each other; human judgement needed | ready, not run |
| Deaf advisor review of every gloss, NMM rule and S5 output (checklist at the end) | we are hearing beginners | open |
| S7 camera paths: phone on local Wi-Fi; Meta Wearables Device Access Toolkit (developer account, Android Studio + JDK, Android phone with USB debugging; Mock Device Kit; iOS needs a Mac) | glasses gate | deferred |

## S1 · Scenario and vocabulary

`scenario.md`: 28-turn café dialogue (yes/no ×3, wh ×3, negation ×3, fingerspelled name). `vocab.json`: 171 candidate glosses (dialogue + café neighbours + numbers 1–20 + function signs).

| Coverage | Glosses |
|---|---|
| In ASL Citizen (any alias) | 146 of 171 |
| Distinct recognition classes (one ASL Citizen sign trains one class) | 143 |
| Dialogue glosses in ASL Citizen | 35 of 38 |
| In WLASL | 149 |
| In Kaggle asl-signs (PopSign 250) and a model class | 53 |

Not in ASL Citizen, with the proposed substitute: ONLY → FS-ONLY; OAT → FS-OAT; ALMOND → FS-ALMOND; SOY → FS-SOY; LATTE → FS-LATTE; ESPRESSO → FS-ESPRESSO; DECAF → FS-DECAF; MUFFIN → FS-MUFFIN; BAGEL → FS-BAGEL; ALLERGY → FS-ALLERGY; SIZE → FS-SIZE; CASH → MONEY; RECEIPT → FS-RECEIPT; TIP → FS-TIP; FIVE → FS-FIVE; TEN → FS-TEN; TWELVE → FS-TWELVE; THIRTEEN → FS-THIRTEEN; FOURTEEN → FS-FOURTEEN; FIFTEEN → FS-FIFTEEN; SIXTEEN → FS-SIXTEEN; SEVENTEEN → FS-SEVENTEEN; EIGHTEEN → FS-EIGHTEEN; NINETEEN → FS-NINETEEN; TWENTY → FS-TWENTY.
Numbers 12–20 (and FIVE, TEN) are missing from ASL Citizen, so a number/fingerspelling handshape path is needed for prices.

Shared signs (recognized as another class): ALSO → SAME; SUGAR → SWEET; EXTRA → MORE.

## S2 · Isolated sign recognition

Landmarks: MediaPipe Holistic (Tasks API) on CPU. Features: both hands (21×2), 8 upper-body points, 6 face points; centered and scaled by the shoulders. The dominant hand is mirrored into one slot, and clips are resampled to 32 frames. Model: Conv1d + BiGRU + attention, 793,872 parameters, trained on CPU in 4 min. Official signer-independent splits: train 2856 videos / 34 signers, val 726 / 5, test 2376 / 11.

| Model | Test top-1 | top-3 | top-5 |
|---|---|---|---|
| Ours, ASL Citizen only | 84.4% [83–86%] | 93.0% | 95.0% |
| Ours, ASL Citizen + PopSign/Kaggle signers | 83.9% | 93.5% | 95.7% |
| ASL Citizen released ST-GCN, restricted to our 143 classes (no fine-tuning) | 81.1% | 92.0% | 94.7% |
| ASL Citizen released ST-GCN, full 2,731-way (paper: 63% top-1) | 53.5% | — | 77.6% |

Weakest signs (test accuracy, n): COST 25% (12), HOW-MUCH 46% (11), BUTTER 46% (13), SEE 46% (13), CHEAP 50% (12), THANK-YOU 50% (12), WHAT 54% (37), CARD 57% (14), COOKIE 57% (14), AGAIN 58% (12), SOUP 62% (13), SOME 64% (14), WHO 64% (14), GOOD 67% (12), MINUTE 67% (12).

15 most-confused pairs (count): GOOD↔THANK-YOU (8), COST↔WHAT (8), CREAM↔CUP (7), CHEAP↔SMALL (4), NEED↔WHERE (4), COST↔HOW-MUCH (4), SEE↔TWO (4), CREAM↔SOME (3), SLOW↔TIME (3), SANDWICH↔SOME (3), FRESH↔RIGHT (3), WAIT↔WANT (3), BUY↔MONEY (3), LARGE↔SIGN (3), CHANGE↔MINUTE (3).

**Cross-dataset (PopSign Deaf signers on phone selfie cameras, Kaggle landmark files, no retraining):** 53 vocabulary signs overlap, 20,358 sequences from 21 signers.

| Scoring | top-1 | top-3 | top-5 |
|---|---|---|---|
| argmax over all vocabulary classes | 33.7% [33–34%] | 49.8% | 57.1% |
| restricted to the 53 overlapping classes | 40.7% | 58.5% | 65.8% |
| held-out PopSign signers (6), ASL-Citizen-only model | 33.6% | 49.5% | 57.0% |
| same held-out signers, model trained with the other PopSign signers | 76.0% | 87.3% | 90.1% |

Per sign (top-1 / top-3): GO 0%/0%, FAST 1%/4%, TIME 1%/21%, GIVE 2%/5%, WINDOW 2%/6%, HOW-MANY 5%/16%, NICE 6%/11%, CAN 6%/20%, MORNING 8%/30%, WHY 9%/25%, SEE 10%/26%, HAVE 10%/52%, SIT 10%/31%, FINISH 12%/25%, ALL 15%/20%, WAIT 16%/35%, NOT 17%/37%, TABLE 17%/29%, CHOCOLATE 18%/43%, HUNGRY 19%/35%, THIRSTY 23%/51%, LATER 27%/38%, FOR 28%/52%, HOT 29%/54%, NOW 30%/58%, NEED 31%/52%, CENT 33%/54%, WHERE 34%/54%, BAD 34%/54%, WHO 36%/64%, HELLO 36%/50%, YES 42%/71%, MY 45%/66%, PLEASE 46%/70%, NUTS 50%/74%, HAPPY 50%/65%, THANK-YOU 51%/70%, FINE 51%/66%, HEARING 51%/70%, BYE 51%/69%, TASTE 52%/66%, SAME 55%/64%, BATHROOM 56%/68%, LIKE 56%/72%, OUTSIDE 60%/69%, NO 61%/75%, PHONE 62%/77%, WATER 70%/77%, DRINK 70%/80%, APPLE 71%/85%, ORANGE 73%/78%, EAT 74%/82%, MILK 76%/82%.

**Continuous signing (fallback: RWTH-BOSTON-104, because DAI access was not requested).** 75 sentences containing at least two vocabulary signs, 3 signers, a full-body black-and-white camera at 25 fps, 336×312. The recognizer uses sliding windows of 12–36 frames with the S2 model, a confidence threshold, temporal non-maximum suppression, and merging of consecutive duplicates. Recall and precision use an order-preserving alignment against the gloss sequence. Boundary error cannot be measured because RWTH-104 annotations have no per-sign times.

| Confidence threshold | Recall | Precision | False positives per OOV sign | Compute per utterance (median / p90 ms) |
|---|---|---|---|---|
| 0.3 | 23.5% (43/183) | 15.3% | 1.29 (238/185) | 177 / 295 |
| 0.5 | 18.6% (34/183) | 22.8% | 0.62 (115/185) | 208 / 317 |
| 0.7 | 13.1% (24/183) | 33.3% | 0.26 (48/185) | 209 / 330 |
| 0.85 | 2.2% (4/183) | 23.5% | 0.07 (13/185) | 193 / 314 |

**Not done:**
- Fingerspelling (optional). Time went to the core risks. The Kaggle asl-fingerspelling rules are accepted, and the data can be downloaded later.
- Gemini zero-shot baseline. The ASL Citizen license forbids transferring the materials to third parties, and only PopSign *landmarks* (no video) were downloaded. Skipped: license.
- SignGemma. Not publicly available as of 2026-10-16: no Hugging Face repository, and nothing in the Gemma release notes.

## S3 · Facial grammar vs emotion

**Data (source a):** RWTH-BOSTON-104 videos, camera 0. This is a full-body view in which the face is about 40 px wide; it is upscaled 2× before MediaPipe. Labels come from the NCSLGR annotations: the public SignStream 2 database files, decoded by `spike/ss2.py`. The decoder was validated on 45 of 45 aligned utterances against the XML files shipped in the BU parser package. RWTH sentences were matched to NCSLGR utterances by normalized gloss sequence, using translations to break ties. Primary set: 95 exact matches. Classes: {'statement': 72, 'rhq': 10, 'wh': 13}. By signer: {'man-1': {'statement': 29, 'wh': 5, 'rhq': 2}, 'woman-1': {'statement': 19, 'rhq': 7, 'wh': 8}, 'woman-2': {'statement': 24, 'rhq': 1}}. Negation: 9. **This corpus has no yes/no questions**, so yes/no detection is untested.

Signals are MediaPipe blendshapes (browInnerUp/OuterUp, browDown) plus geometric brow height. The sentence-level feature is a 0.5 s rolling mean, which captures sustained raise or lower. Head pose is from mesh geometry, and a headshake is yaw oscillation at 1.5–6 Hz. Signals are z-scored per signer without labels; this is transductive calibration, and a product would calibrate on the first seconds of a conversation.

| Test (primary exact set, leave-one-signer-out) | Result |
|---|---|
| Statement vs wh, logistic regression on brow features | accuracy 81.2% [72–88%], balanced 57.4%, n = 85; confusion {'statement': {'statement': 66, 'wh': 6}, 'wh': {'statement': 10, 'wh': 3}} |
| Statement vs wh, single brow-down threshold | accuracy 75.3% [65–83%], balanced 69.7%, n = 85; confusion {'statement': {'statement': 56, 'wh': 16}, 'wh': {'statement': 5, 'wh': 8}} |
| AUC wh vs statement brow down sus max | 0.676 [0.49–0.853] (n = 13 vs 72) |
| AUC wh vs statement brow contrast | 0.764 [0.615–0.896] (n = 13 vs 72) |
| AUC rhq vs statement brow up sus max | 0.74 [0.617–0.856] (n = 10 vs 72) |
| AUC raised vs lowered questions brow contrast | 0.923 [0.777–1.0] (n = 13 vs 10) |
| AUC negation vs rest hs amp rel | 0.944 [0.826–1.0] (n = 9 vs 86) |
| Negation, headshake amplitude threshold | precision 100.0%, recall 77.8% (7 TP, 0 FP, 2 FN; n = 95) |
| Negation, logistic regression on headshake features | precision 80.0%, recall 88.9% (8 TP, 2 FP, 1 FN; n = 95) |
| 3-way brow rule (neutral / lowered / raised) | accuracy 57.9%, balanced 51.7% |

**Key finding: raised brows are not only questions.** Mean per-signer z-scores of sustained brow raise and lower:

| Group | n | brow raise | brow lower |
|---|---|---|---|
| rhq | 10 | +1.77 | +0.37 |
| statement plain | 54 | +0.17 | +0.81 |
| statement+topic/cond | 18 | +2.81 | +0.27 |
| wh | 13 | -0.23 | +1.56 |

Statements with topic or conditional marking raise the brows *more* than rhetorical questions do, because topics and conditionals are marked with raised brows in ASL. Whole-utterance brow raise therefore cannot mean "yes/no question". The grammar layer needs *where* the raise is: over the whole clause or ending at the sentence end means a question, while a raise on the first constituents followed by a drop means a topic or conditional. Gloss context also matters: a wh-sign means a wh-question. Once an utterance is known to be a question, the brows separate wh from raised-brow questions well (AUC 0.923).

**Naive baseline (HSEmotion enet_b0_8, frame by frame on face crops, majority label per utterance):**

| Sentence type | n | Majority labels |
|---|---|---|
| rhq | 10 | {'neutral': 5, 'fear': 3, 'anger': 2} |
| statement | 72 | {'anger': 23, 'sadness': 4, 'disgust': 10, 'surprise': 7, 'neutral': 24, 'fear': 4} |
| wh | 13 | {'anger': 10, 'fear': 1, 'neutral': 1, 'sadness': 1} |

Headline: the naive classifier calls **10/13 wh-questions angry or disgusted**. Our grammar layer labels them {'statement': 10, 'wh': 3}. The naive classifier also calls 23 of 72 statements angry, because the low-resolution faces and lowered brows read as anger. Raised-brow (rhetorical) questions were labelled surprise in 0/10; yes/no questions are untested. Association with sentence type (Cramér's V): naive 0.35 vs our affect layer 0.16.

**Affect layer** (mouth, cheeks, eye squint and nose sneer, plus hand speed, amplitude and tempo; brows excluded): label distribution by type: {'rhq': {'neutral': 7, 'happy': 1, 'sad': 2}, 'statement': {'neutral': 58, 'sad': 4, 'happy': 4, 'angry': 6}, 'wh': {'neutral': 10, 'angry': 1, 'sad': 2}}. Its association with sentence type is low, so it does not simply track question type. The corpora have no emotion labels, so emotion *accuracy* is untested and needs labeled recordings later.

`to_voice(grammar, affect, kinematics)` is in `spike/voice_map.py`:
- **Text:** questions get "?".
- **Emotion:** one of Cartesia's well-supported emotions plus "happy". It is sent only at intensity ≥ 0.35, and dropped when it contradicts the LLM's reading of the words.
- **Speed:** signing tempo ratio, clamped to 0.8–1.3.
- **Volume:** movement amplitude ratio, clamped to 0.8–1.5.
Examples are in `results/s3_rwth_eval.json` → `to_voice_examples`.

Weak-label How2Sign fallback (b): not run. Google Drive blocked scripted downloads; see Human steps.

## S4 · Cartesia expressive voice

Sonic 3.6 over one WebSocket, sequential, one fresh context per request. Twelve scenario sentences × 4 emotions × 3 voices (Leo, Maya, Gavin), across several conditions:
- **(a)** no emotion;
- **(b)** `generation_config.emotion`;
- **(c)** emotion plus speed and volume from `to_voice`;
- **(d)** inline `<emotion value=…/>` tag;
- **(e/f)** an LLM rewrite of each line so the words carry the emotion, with (e) and without (f) the tag; Leo and Maya only.

Two independent 4-class speech-emotion classifiers scored every clip: SpeechBrain wav2vec2-IEMOCAP and HuBERT-large SUPERB-ER. Chance is 25%.

Latency (n = 696): first audio byte median 122 ms, p90 153, max 1,228; total synthesis median 317 ms, p90 460.

| Condition | SpeechBrain hit rate | HuBERT hit rate | n | per emotion (SpeechBrain) |
|---|---|---|---|---|
| a · no emotion (scored vs each intent) | 25.0% [19–33%] | 25.0% | 144 | neutral 58%, happy 0%, sad 0%, angry 42% |
| b · emotion tag | 22.2% [16–30%] | 25.0% | 144 | neutral 58%, happy 0%, sad 0%, angry 31% |
| c · tag + speed/volume | 25.7% [19–33%] | 25.7% | 144 | neutral 64%, happy 3%, sad 0%, angry 36% |
| d · inline tag | 25.0% [19–33%] | 22.2% | 144 | neutral 61%, happy 3%, sad 0%, angry 36% |
| e · emotional wording + tag | 47.9% [38–58%] | 39.6% | 96 | neutral 75%, happy 71%, sad 0%, angry 46% |
| f · emotional wording, no tag | 49.0% [39–59%] | 38.5% | 96 | neutral 79%, happy 67%, sad 0%, angry 50% |

**Words, not tags:** emotional wording with the tag scored 47.9% and the same wording without the tag 49.0%. All tag-only conditions sit at chance. Neither classifier detected 'sad' in any condition, which may be a classifier limit on synthetic speech as much as a Sonic limit; the listening test should settle it.

**Mismatch test** (6 texts × 3 voices; emotion contradicting the words). With the contradicting tag, the intended emotion was detected in 3/18 clips (SpeechBrain), and the *text's* emotion in 8/18. With the matching tag: 9/18 (HuBERT 66.7%). This confirms the docs: the words win. `to_voice` drops the emotion when it conflicts with the LLM's reading of the sentence.

An acoustic check (duration, RMS, median F0) shows the tags barely change these neutral café sentences: about 1.85–1.90 s, RMS 0.073–0.077 and F0 197–203 Hz across tags. The two classifiers also disagree with each other: SpeechBrain says neutral or angry, HuBERT mostly happy. That is why the listening sheet matters.

**Recommendation.** The default voice is provisional until the listening test. Gavin read as neutral most consistently (83% of untagged clips, vs 67% for Maya and 25% for Leo), and Leo's anger was the most detectable (75% with the tag). Rules:
1. Convey affect mainly through speed and volume, which are measurable and controllable.
2. Send an emotion tag only when the LLM's English already carries that emotion.
3. Never let the LLM add emotional words the signer did not sign. The e/f rewrites invented content such as "Get it right this time" and "It's been a long day". That misrepresents a Deaf user and needs Deaf advisor review.
4. Keep the set of emotions small: neutral, happy, sad, angry.

Listening sheet: `listening/index.html` (60 blind clips, key in `listening/key.json`, score with `spike/s4_listening.py score`). Not run.

## S5 · LLM glue (claude-haiku-5-5, structured JSON, effort low, thinking off)

| Direction | Variant | Acceptable meaning (A / B / W, my judgement) | Latency |
|---|---|---|---|
| Gloss → English, 30 clean | v1: glosses only | 24 A / 1 B / 5 W | median 1,257 ms, p90 1,895, max 4,447 (n = 40) |
| Gloss → English, 10 with one wrong/missing gloss | v1 | 5 A / 2 B / 3 W | |
| Gloss → English, 30 clean | v2: + barista's previous line as context | 28 A / 2 B / 0 W | median 1,239 ms, p90 2,013, max 11,338 (n = 40) |
| Gloss → English, 10 with errors | v2 | 6 A / 3 B / 1 W | |
| English → gloss + NMM, 30 barista lines | v1: vocabulary in the prompt | rejected (gloss outside vocabulary) 7/30 | median 1,397 ms, p90 2,121, max 2,536 (n = 30) |
| English → gloss + NMM | v2: + worked examples, one repair retry | first pass invalid 8/30, after retry **0/30 rejected**; meaning 23 A / 6 B / 1 W; NMM errors: 2 | median 1,485 ms, p90 2,970, max 4,084 (n = 30) (incl. retries) |

A = meaning preserved; B = understandable but lossy or changed; W = wrong meaning. Judged on meaning only, not on ASL quality. Every output is in `results/s5_*.jsonl` and the prompts are in `results/s5_config.json`; all of them need Deaf advisor review. Lessons:
1. Pass the last barista utterance as context. It fixed most v1 errors, such as "It's hot." turning into "Hot, please."
2. A misrecognized off-topic gloss (FISH, DOG) still leaks through about half the time. The UI should show low-confidence words.
3. A missing negation sign with the headshake flag set is recovered in v2.
4. Validate-and-retry makes English → gloss output always valid, at the cost of a second call on 30% of lines.

## S6 · Avatar for speech → sign

| Option | Assessment |
|---|---|
| (a) sign.mt spoken-to-signed-translation | MIT code. Lexicons are Swiss (sgg/ssr/slf); ASL is a rule-based 'lexical plan' with no ASL lexicon listed. Outputs .pose/video. Non-manuals are explicitly 'downstream work'. The app is CC BY-NC-SA 4.0. Not usable for ASL as is. |
| (b) AWS GenASL | MIT-0 code. Stitched 2D avatar videos rendered from ASLLVD citation clips with RTMPose. Needs Transcribe, Bedrock, Step Functions, Lambda, DynamoDB, S3 and Amplify. No facial grammar. ASLLVD videos come from BU, so their terms must be checked. Too heavy for the build's time budget. |
| (c) ours: per-gloss MediaPipe poses → VRM avatar in three.js | Built (`avatar/index.html`). three.js 0.169, @pixiv/three-vrm 3.1, kalidokit 1.1.5 for fingers. VRM sample model 'VRM1_Constraint_Twist_Sample' (VRM Public License 1.0: commercial use and redistribution allowed). |

Prototype: 3 sentences from S5 v2 outputs (wh: 6 signs, yes/no: 6, negation: 5). Glosses missing from ASL Citizen were dropped (SIZE, OAT). The page applies 200 ms slerp blends between signs, brow-only morph targets (Fcl_BRW_Surprised for raise, Fcl_BRW_Angry for lower) over each NMM span, and a 3 Hz head yaw for negation. Measured in Microsoft Edge (Playwright) on ANGLE (Intel, Intel(R) Arc(TM) 140V GPU (16GB) (0x000064A0) Direct3D11 vs_5_0 ps_5_0, D3D11):

| Metric | Value |
|---|---|
| Gloss JSON → first avatar motion (warm, lexicon loaded) | median 7.2 ms, p90 9.7, n = 15 |
| Same, first run (fetching a 5 MB lexicon) | 47 ms |
| Paint cost per frame | 0.7 ms |
| Page ready (model load) | 1.18 s |

Screen capture: `avatar/capture.webm` (local, git-ignored). Key frames: `avatar/shots/`.

**Visual problems, honestly:**
1. kalidokit's arm solver puts the arms behind the body on a VRM 1.0 model under every flip/mirror setting. We replaced it with direct vector alignment of the shoulder→elbow→wrist directions, which puts the hands in signing space. Arm twist and wrist roll are still missing.
2. Handshapes from kalidokit's finger solver are unreliable: an index point (YOU) does not read clearly.
3. The sample model's bangs cover the eyebrows, so brow grammar is nearly invisible. Pick an avatar with visible, thick brows and add eye aperture and head tilt.
4. Citation clips are slow: 7–11 s for a 5–6-sign sentence. Trim holds and play at about 1.3–1.5×.
5. Transitions between different signers' clips would jump; the prototype uses one signer (P11).
6. The hidden in-app browser pane renders WebGL in software and cannot even compile the model's shaders in 45 s. Test in a real browser with a GPU.

**License blocker for a public demo:** the per-gloss poses come from ASL Citizen, whose license forbids distributing the data or modified data. A public demo should use a lexicon we may distribute (PopSign is CC BY 4.0; check the Kaggle data terms) or one recorded by the team's Deaf advisor with consent.

## S7 · Camera input paths

Deferred; not run. Instructions to do it later:
- **Phone browser path:** the page must be served over HTTPS (a self-signed certificate is accepted on the phone) for getUserMedia on a non-localhost origin. Allow the Windows Firewall prompt for the local server, then measure fps and frame age at the laptop.
- **Meta glasses path:**
  1. Create a Meta Wearables developer account and enroll in Device Access Toolkit access.
  2. Install Android Studio and a JDK, and connect an Android phone with USB debugging.
  3. Build the DAT sample app, using a GitHub token with `read:packages` if the SDK is hosted on GitHub Packages.
  4. Use the Mock Device Kit to inject a licensed signing video, such as a PopSign clip (CC BY 4.0).
  5. Stream frames to the laptop service.
  6. Check Bluetooth audio output to the glasses' speakers from the phone app.

## S8 · End-to-end latency budget (from measured stages)

**Sign → speech (last sign ends → first audio)**

| Stage | Median ms | p90 ms | n | Source |
|---|---|---|---|---|
| landmarks for the last frame (MediaPipe Holistic, CPU) | 143 | 194 | 400 | S2 extraction, ms per frame over 400 ASL Citizen videos while 7 extraction workers shared the CPU; a single unloaded process measured ~40 ms/frame on RWTH |
| hands-at-rest detected (hold time) | 300 | 300 | — | design parameter (300 ms of low hand speed); not measured on real end-of-turn behaviour |
| gloss sequence final (windowed classifier over the utterance) | 208 | 317 | 75 | S2 continuous test (RWTH, CPU) |
| LLM gloss -> English (claude-haiku-5-5) **(biggest)** | 1,239 | 2,245 | 40 | S5 v2, non-streaming request, sequential |
| Cartesia Sonic 3.6 first audio byte | 122 | 153 | 696 | S4, WebSocket, warm connection |
| **Total** | **2,010** | **3,209** (sum of p90s, conservative) | | |

**Speech → sign (speech ends → first avatar motion)**

| Stage | Median ms | p90 ms | n | Source |
|---|---|---|---|---|
| Ink 2 turn.end (final transcript) | 721 | 923 | 30 | S8 Ink test, 30 barista lines, real-time streaming |
| LLM English -> gloss + NMM (incl. one repair retry when invalid) **(biggest)** | 1,484 | 3,006 | 30 | S5 v2 |
| first avatar motion after gloss JSON | 7 | 10 | 15 | S6, Edge + Intel Arc GPU; lexicon preloaded (cold first run: 47 ms) |
| **Total** | **2,213** | **3,938** (sum of p90s, conservative) | | |

Ink's `turn.eager_end` arrives at a median 334 ms (p90 742). Starting the LLM on eager end, and discarding the result on `turn.resume`, saves roughly the difference. Ink transcripts were essentially perfect on these clean TTS lines; examples are in `results/s8_ink.jsonl`. Not included: camera capture and transport (S7), audio buffering, and the time the avatar takes to sign (7–11 s per sentence at citation speed).

## Dataset and asset licenses

| Item | License / terms | Used for | Demo impact |
|---|---|---|---|
| ASL Citizen (Microsoft Research) | Microsoft Research License Terms: non-commercial research only; no sharing or distribution of the data or modified data; no transfer to third parties; personal data kept confidential | S2 training and testing; S6 lexicon | Training locally is OK for research. Public demo assets derived from it are likely not allowed; ask Microsoft |
| ASL Citizen code + ST-GCN weights | MIT (code repository) | S2 baseline | OK |
| Kaggle asl-signs (from PopSign) | competition rules accepted; PopSign itself is CC BY 4.0 | S2 cross-dataset test | Check the competition data terms before using it in a demo |
| RWTH-BOSTON-104 | CC BY-NC-SA 4.0 (form on download) | S2 continuous, S3 | Research only |
| NCSLGR annotations (BU ASLLRP) | public download page; cite ASLLRP and C. Neidle; videos via DAI only | S3 labels | Research only |
| How2Sign | CC BY-NC 4.0 | not used (download blocked) | — |
| VRM1_Constraint_Twist_Sample (pixiv) | VRM Public License 1.0: commercial use and redistribution allowed | S6 avatar | OK |
| SpeechBrain / HuBERT SUPERB-ER / HSEmotion models | open model licenses (Apache 2.0 / MIT-style) | S4, S3 evaluation only | OK for evaluation |

## Go/no-go recommendation

**Go, with a narrowed hero demo.** The parts that must be fast are fast. Sign → speech is about 2.0 s median (3.2 s p90), and speech → sign is about 2.2 s median (3.9 s p90). The LLM call is the largest stage in both directions. Isolated-sign recognition clears the gate on unseen ASL Citizen signers: 84.4% top-1 and 93.0% top-3 over 143 signs. It does **not** transfer to other cameras and signers without retraining: PopSign phone signers get 33.7% top-1, and continuous signing gets 18.6% recall. Adding the other PopSign signers to training lifts held-out PopSign signers from 33.6% to 76.0% top-1 (87.3% top-3) while ASL Citizen test stays at 83.9%.

So the demo should be isolated signs with brief holds, a top-3 tap-to-confirm, and a vocabulary of about 50 core café signs. Ideally the demo signer is the team's Deaf advisor, with a few minutes of their own examples added to training before the demo. Use facial grammar for wh vs yes/no (once a question is detected), not for question vs statement. Use the headshake for negation. Convey voice affect with speed and volume rather than Cartesia emotion tags. For speech → sign, use captions plus the avatar, and keep pre-rendered clips of the core phrases as the fallback. The avatar's handshape quality is not demo-ready, and its ASL Citizen-derived poses cannot be shown publicly.

## Five biggest risks for the main build

1. **Recognition collapses on a new signer or camera.** Measured: 33.7% top-1 on PopSign phone signers vs 84.4% on ASL Citizen. Mitigation: about 50 signs, tap-to-confirm top-3, the demo signer's own samples added before the demo, and PopSign data in training.
2. **Continuous signing is not recognized** (18.6% recall on RWTH sentences). Mitigation: sign-by-sign with holds, and segmentation by hands at rest. Do not promise sentence-level recognition.
3. **Wrong grammar or affect in the voice misrepresents the Deaf user.**
   - Raised brows also mark topics.
   - Emotion classifiers read grammar as anger.
   - LLM rewrites invent words.

   Mitigation: show the English on screen before or while speaking, keep tags conservative, and get Deaf advisor sign-off.
4. **The avatar's signing is not legible:** handshapes are wrong, the brows are hidden, and citation clips are slow. Mitigation: captions first, the avatar as an extra, the advisor's recorded clips for the core phrases, and a different avatar model.
5. **Licensing and data:**
   - ASL Citizen forbids redistribution, so public demo assets must come from CC BY data or the team.
   - Glasses camera access (S7) was not tested.

   Mitigation: phone camera as the default input, and ask Microsoft and Meta early.

## Needs Deaf advisor review

Every item below was chosen by hearing non-signers.

- **Glosses:** HELLO, BYE, THANK-YOU, PLEASE, SORRY, EXCUSE-ME, WELCOME, NICE, GOOD, BAD, FINE, DAY, MORNING, NAME, YES, NO, IX-1P, YOU, MY, YOUR, WANT, NEED, HAVE, CAN, NOT, NONE, FINISH, AGAIN, MORE, WITH, WITHOUT, ALSO, ONLY, WAIT, HELP, UNDERSTAND, KNOW, LIKE, PREFER, CHANGE, GIVE, BUY, EAT, DRINK, GET, KIND, SAME, DIFFERENT, FOR, OR, WHAT, WHERE, WHEN, WHO, WHY, HOW, HOW-MUCH, HOW-MANY, WHICH, COFFEE, TEA, HOT, COLD, ICE, WATER, MILK, OAT, ALMOND, SOY, CREAM, SUGAR, SWEET, CHOCOLATE, LATTE, ESPRESSO, DECAF, JUICE, ORANGE, APPLE, SODA, CUP, STRAW, MUFFIN, COOKIE, CAKE, BREAD, SANDWICH, BAGEL, CHEESE, EGG, BUTTER, BLUEBERRY, BANANA, STRAWBERRY, SALAD, SOUP, BREAKFAST, LUNCH, HUNGRY, THIRSTY, NUTS, ALLERGY, WARM, FRESH, DELICIOUS, TASTE, SIZE, SMALL, MEDIUM, LARGE, EXTRA, HALF, ALL, SOME, MONEY, DOLLAR, FIVE-DOLLARS, CENT, CARD, CASH, PAY, COST, RECEIPT, TIP, CHEAP, EXPENSIVE, HERE, GO, TABLE, SIT, WINDOW, BATHROOM, OUTSIDE, INSIDE, READY, NOW, LATER, MINUTE, TIME, FAST, SLOW, DEAF, HEARING, SIGN, WRITE, PHONE, SEE, HAPPY, LOVE, WRONG, RIGHT, ONE, TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, ELEVEN, TWELVE, THIRTEEN, FOURTEEN, FIFTEEN, SIXTEEN, SEVENTEEN, EIGHTEEN, NINETEEN, TWENTY.
- **Substitutions:** ALSO → SAME; ONLY → FS-ONLY; OAT → FS-OAT; ALMOND → FS-ALMOND; SOY → FS-SOY; SUGAR → SWEET; LATTE → FS-LATTE; ESPRESSO → FS-ESPRESSO; DECAF → FS-DECAF; MUFFIN → FS-MUFFIN; BAGEL → FS-BAGEL; ALLERGY → FS-ALLERGY; SIZE → FS-SIZE; EXTRA → MORE; CASH → MONEY; RECEIPT → FS-RECEIPT; TIP → FS-TIP; FIVE → FS-FIVE; TEN → FS-TEN; TWELVE → FS-TWELVE; THIRTEEN → FS-THIRTEEN; FOURTEEN → FS-FOURTEEN; FIFTEEN → FS-FIFTEEN; SIXTEEN → FS-SIXTEEN; SEVENTEEN → FS-SEVENTEEN; EIGHTEEN → FS-EIGHTEEN; NINETEEN → FS-NINETEEN; TWENTY → FS-TWENTY.
- **Alias matches to confirm** (same English label, possibly different sign): ALSO ↔ SAME1, SAME2; KIND ↔ TYPE1, TYPE2; HOW-MUCH ↔ COST, PRICE; HOW-MANY ↔ MANY; SUGAR ↔ SWEET; BLUEBERRY ↔ BERRY; NUTS ↔ NUT1, NUT2; FRESH ↔ NEW; MEDIUM ↔ MIDDLE, AVERAGE; LARGE ↔ BIG; EXTRA ↔ MORE; DOLLAR ↔ 1DOLLAR; CENT ↔ PENNY; CARD ↔ CREDITCARD.
- **Facial grammar rules:**
  - raised brows over the clause mark a yes/no question;
  - lowered brows mark a wh-question;
  - a headshake marks negation;
  - raised brows on initial constituents mark a topic or conditional, not a question;
  - the avatar's 3 Hz head yaw is our stand-in for a negation headshake.
- **S5 outputs:** all 40 gloss → English and 30 English → gloss outputs (`results/s5_*.jsonl`), including the NMM spans.
- **Voice affect:**
  - whether conveying affect by speed and volume is acceptable to Deaf users;
  - whether a voice should ever add emotional wording.
- **Avatar:** handshape accuracy, movement speed, the whole avatar approach (vs. captions or recorded clips), and the choice of avatar appearance.

## Estimated API spend

| Service | Usage | Estimated cost |
|---|---|---|
| Anthropic claude-haiku-5-5 | about 200 calls (S5 v1/v2, retries, 48 S4 rewrites, smoke tests): about 170k input / 15k output tokens | ≈ $0.03 ($0.10 / $0.50 per MTok) |
| Cartesia Sonic 3.6 TTS | 696 S4 clips + 30 S8 barista lines + smoke tests: about 28.5k characters | ≈ 28.5k credits |
| Cartesia Ink 2 STT | 30 streamed utterances: about 1.8 min of audio | small (per-minute STT pricing) |
| OpenAI, Gemini | not used | $0 |

Please confirm against the Anthropic and Cartesia dashboards.