# Teaching videos: are their labels right, how does the app read them, and why it fails (2026-10-24)

How this was measured: 1,029 clips of single signs, each signed slowly in citation form, one sign per clip:

- 912 from the ASL Citizen **test** split: 11 Deaf signers the shipped model never trained on, up to 6 clips per sign.
- 117 from ASL Signbank: fluent dictionary signers, studio video, an outside source.

Together they cover 152 of the model's 155 signs. Landmarks came from the browser's own MediaPipe models. Each clip was replayed at 15 frames/s through the app as it runs (`signs.pt`, the real `Segmenter`, recut, the picker at 0.5). A scripted signer picks the right sign when the picker offers it and otherwise presses "None of these".

**Right** means the turn says exactly that sign, straight away or after picking. Code: `app/training/teach_eval/`. Data, sheets and per-clip results stay local in `app/data/teaching/`.

Sources left out: YouTube (its terms forbid downloading), sites whose terms forbid reuse (Handspeak, Signing Savvy, Spread The Sign, SignASL), and sites that need permission (Lifeprint, ASL Core, the WLASL host sites). The local WLASL clips were not used for that reason.

## 1. Are the videos' labels right?

**Automatic checks on every clip:**
- framing, hands seen, blur, light and caption band;
- the number of turns the app cuts the clip into;
- agreement with ASL-LEX on hands, location, fingers and movement.

**Agent review of 400 clips:**
- All 137 clips whose checks raised a doubt about the label, plus a sample stratified by source × sign.
- Each clip was judged by two independent agents. Each agent saw 6 frames of the clip above 6 frames of an ASL Citizen training video of the labelled sign.
- A third agent settled the 47 items where the two disagreed or either was unsure.
- 40 planted controls (a clip shown under a wrong sign's label) measured how often a wrong label gets caught.

| | ASL Citizen test | ASL Signbank |
|---|---|---|
| clips reviewed | 302 | 98 |
| label right (verified) | 91.7% (95% CI 88.1–94.3) | 90.8% (83.5–95.1) |
| label wrong (rejected) | 13 | 5 |
| unsure | 12 | 4 |

Review quality:
- The two reviewers agreed with κ = 0.73.
- They caught 85% of the planted wrong labels (34 of 40); 5 controls ended unsure and 1 was missed.
- On real items, only 4% of first-round votes said "no".

The automatic checks did **not** predict wrong labels: 4 of 137 flagged clips were rejected, against 14 of 263 unflagged clips. Caption band, dark and blur describe the camera, not the label.

**What the rejected clips were:** another sign under the label (e.g. HAPPY showing a thumb at the chin), clips with no hands, and homonyms:
- **RIGHT:** some clips show the direction "right", not "correct".
- **CAN:** some clips show the modal "can", and the ASL Citizen reference looks like the container "can". **Worth checking which CAN the course teaches.**

Rejected and unsure clips (34) were dropped. That leaves **995 trusted clips**: 366 verified plus 629 unreviewed clips that passed every check.

## 2. How does the app read them?

On the 995 trusted clips:

| | value | 95% CI (signer bootstrap) |
|---|---|---|
| translator right (straight away + via picker) | **87.0%** | 83.8–90.0 |
| — straight away | 69.3% | |
| — via the picker | 17.7% | |
| wrong word spoken | 5.4% | |
| lost at "None of these" | 4.8% | |
| sign cut into pieces / no turn | 2.7% | |
| single-sign read, top-1 / top-3 | 84.0% / 95.1% | |
| lesson pass (51 course signs, 326 clips) | **96.6%** | 94.2–98.8 |

By source:
- **ASL Signbank (outside source):** 83.3% right (108 clips).
- **ASL Citizen test:** 87.5% right (887 clips).
- **Left-handed signers:** 85.1%, against 88.3% for right-handed.
- **Per signer:** from 74% to 95%.

Weakest signs (right in at most half of 6–8 attempts): GOOD, MINUTE, AGAIN, NIGHT, MY, NOT, WHAT, YES. Next come WHO and YOU (60%).

Calibration is reasonable:
- top score ≥ 0.9: right 98% of the time;
- top score 0.5–0.6: right 67%;
- top score < 0.3: right 48% or less.

**Read with care.** The model was trained on ASL Citizen. The test signers are new, but the camera setup is the same (a webcam at home). These numbers are therefore an upper bound, above the 64% measured on WLASL dictionary clips (`sign_reliability.md`). The ASL Signbank figure (83%) is the fairer outside estimate.

## 3. Why it fails: patterns in the 129 failures

**Method:**
- Two analyst agents tagged every failure. Each failure came with a sheet showing the clip above a reference of the sign the app read, plus the clip's measurements.
- Shares below count a tag only when **both** analysts gave it; the share with either analyst's tag is in brackets.
- **Measured** means the statistics on all 995 clips support the claim. They are:
  - a within-sign comparison: failing vs passing clips of the same sign, with a permutation test;
  - quartile failure rates;
  - models cross-validated by signer.
- **Hypothesis** means only the analysts saw it.

1. **Look-alike signs: 43% (57%) of failures; κ 0.72.** **Measured.**
   - The true and read signs differ in few ASL-LEX parameters: 17% differ in one, 16% in two.
   - Handshape (72%) and movement (66%) are the parameters that differ most often.
   - Repeat pairs:
     - MINUTE→HOUR ×4
     - GOOD→THANK-YOU ×4, and THANK-YOU→GOOD ×2
     - MY→PLEASE ×3
     - WE→I, STOP→MONEY and YES→BATHROOM, 2 each
   - GOOD signed one-handed *is* THANK-YOU's form; only context tells them apart.
   - *Fix to try:* in lessons, accept the known pair when it's the top guess with a close second. In the translator, offer both signs in the picker for these pairs.
2. **Turn cutting: 22% (22%); κ 0.98.** **Measured.**
   - In 20 failures the sign was read right but the turn was split (`split_right`). In 6 more, no turn opened at all.
   - Causes:
     - long holds and repeated taps (MORE, WRITE, SOON);
     - a hand resting on the chin before signing (WHEN);
     - clips under 0.3 s.
   - The longest quarter of clips (over 1.27 s) fails 18% of the time, against 8–13% for the rest. Duration also ranks first in the boosting model's importance.
   - *Fix to try:* in `server/segmenter.py`, don't split on `PAUSE_MS` (300 ms) while the hands stay up near the last position. A hold is not the end of a sign.
3. **Head movement: 17% (26%); κ 0.73.** **Measured.**
   - Within the same sign, failing clips turn the head more: yaw range +0.49 SD (p = 0.002), pitch range +0.34 SD (p = 0.008), shoulder shrug +0.26 SD (p = 0.01).
   - The top quarter of head-yaw range fails 19% of the time, against about 10% for the rest.
   - Clips the analysts tagged with head motion have much higher peak head speed (Cliff's δ 0.70).
   - *Fix to try:* hand positions are measured relative to the body. Normalise them to the shoulders rather than the nose or head, so head turns don't move the hands' coordinates, and add head-turn augmentation in training.
4. **Hands overlapping: 19% (25%); κ 0.84.** **Measured.**
   - Clips whose hands overlap in more than a third of frames fail 18% of the time, against 11%.
   - When the hands stack (LOVE, NAME, CUP, CHEESE), MediaPipe keeps one hand, so the sign looks one-handed: LOVE read as I, CUP read as MY.
   - *Fix to try:* when a hand vanishes while the two wrists are close, carry the last seen hand forward instead of zeroing it.
5. **Blur and fast motion: 27% (44%); κ 0.63.** **Hypothesis.**
   - The analysts saw it on the frames, but measured hand speed does not back it up (δ 0.02). Sharper clips do fail less, though: the sharpest quarter fails 8%, against about 15%.
   - It is mostly dark or blurred home webcams, where fingers are lost.
   - *Fix to try:* camera guidance on the page: light in front, and the camera at chest height.
6. **Signer small in frame: 14% (15%); κ 0.97.** **Measured, for that source only.**
   - All the tagged clips are ASL Signbank's full-body studio shots, with the face about 50–60 px wide.
   - Clips with the face under 60 px are right 82% of the time.
   - *Fix to try:* in the app, ask the user to sit closer when the face is under about 80 px.
7. **Different form of the sign: 16% (29%); κ 0.63.** **Hypothesis.**
   - Examples: fingerspelled W-H-A-T and P-I-Z-Z-A, compound DEAF and 8-HOUR, an index-finger DOCTOR, a one-handed MILK.
   - *Fix to try:* add these variants to training, or list the taught form in the lesson.
8. **Left-handed signers and fingers close together.** **Measured, small effects.**
   - Within the same sign, failing clips are more often left-handed (+0.30 SD, p = 0.008), and their fingers sit closer together (closest fingertip gap −0.22 SD, p = 0.01).
   - The model's features don't re-key MediaPipe's left/right labels.
   - *Fix to try:* mirror left-handed signing to the right before reading, and re-key the hands to the pose wrists in `server/features.py`.

Hands near the face on their own did **not** go with failure: no gap between failing and passing clips (δ −0.01), and failure rate is flat across its quartiles. It shows up only together with occlusion or head movement.

The boosting model separates failing from passing clips by measurements alone only weakly (AUC 0.65, cross-validated by signer). Most failures come from which sign it is (the look-alike pairs) rather than from how it was filmed.

## Fixes tried (2026-10-25)

How each fix was judged:
- Each was prototyped as an in-memory patch and replayed on all 995 trusted clips with `training/teach_eval/quick_eval.py`.
- A separate skeptic agent then reviewed it.
- Only changes that broke no currently-right clip and added no model calls on the normal path were kept.

| fix | verdict | 995 clips: fixed / broke |
|---|---|---|
| merge recut pieces with the same top sign inside one segment (≤ 2.5 s); keep a 2-frame lone piece (`server/recut.py`, `server/segmenter.py`) | **shipped** | 15 / 0 |
| look-alike guard: same handshape + location (ASL-LEX, `vocab/lookalikes.json`) close behind → picker offers both (`server/session.py`) | **shipped** | 3 / 0 |
| camera tips "too dark" / "sit closer" (`web/src/capture.ts`) | **shipped** (not measurable on clips) | — |
| bridge a hand lost while the hands overlap | rejected | 2 / 2 on the subset; needs a retrain |
| re-key hands to pose wrists / mirror test-time augmentation | rejected | net −3 to −4 |
| head-turn normalisation | needs a retrain | — |

**With both shipped:**
- translator right went from 866 to **884 of 995 (88.8%)**: 18 fixed, 0 broke;
- `split_right` went from 20 to 8;
- lesson pass went from 315 to 317 of 326;
- picker taps rose by about 1 per 1,000 turns that were already right.

Sentences repeating the same sign with a rest in between stay as two words (tested).

**Retrain later:**
- key hands the same way in training and at runtime;
- add head-yaw augmentation;
- bridge overlap gaps in both training and inference.

## Caveats

- **The agents are not fluent signers.** They judged from 6 frames per clip and are not a Deaf advisor.
  - Weakest agreement: handedness_or_mirrored κ 0.35, other κ 0.36.
  - A Deaf advisor should check first: the 18 rejected clips, the CAN and RIGHT homonyms, and the GOOD/THANK-YOU pair.
- **Sample sizes are small.** There are 129 failures, with 6–8 clips per sign, so per-sign rates move a lot with one miss.
- **One reference sign was missing.** The reference sign I has no ASL Citizen training video, so its sheets had no reference strip.
- **Controls could be spotted from duplicates.** A planted control sometimes landed in the same batch as the real clip, and reviewers noticed the duplicate frames. That makes the 85% catch rate somewhat optimistic.
- **The lead-analyst pass was skipped for time.** Tag shares use both-analysts agreement (conservative), with the either-analyst share in brackets.

## Attribution

- ASL Signbank (2026), aslsignbank.com, Haskins Laboratories / University of Connecticut. Videos used under CC BY-NC-SA 4.0 for non-commercial evaluation; not redistributed.
- ASL Citizen (Desai et al. 2023, Microsoft Research), test split, under its research licence.
- ASL-LEX 2.0 (Caselli et al. 2017; Sehyr et al. 2021) for sign properties.
