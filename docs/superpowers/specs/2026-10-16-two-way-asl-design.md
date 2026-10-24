# Two-Way ASL: MVP Design

Oct 16, 2026 · builds on "Two-Way ASL MVP: Product Spec" (@Zack, Oct 16)

## Intent

A Deaf ASL signer and a hearing person who doesn't sign hold a real conversation face to face, with no interpreter booked. The signer signs and the hearing person hears natural English in a voice that carries the signer's emotion. The hearing person talks and a 3D avatar signs it back with correct facial grammar and the speaker's tone, beside English captions.

What sets it apart: it treats ASL as a language with grammar on the face. Raised brows on a yes/no question are read as a question, not as "surprised", and emotion is read from the rest of the face and the signing itself. Both directions carry emotion.

The product spec targeted Cal Hacks 13.0. This build is **not tied to the hackathon**, so the "no code before the event" rule doesn't apply. No prior ASL code exists: the repo holds only `latency-bench/`, which stays untouched, so everything is built from zero. **This document overrides the product spec** wherever they differ. The spec is background only, and its demo script, P1 features and tracks are out of scope here.

## Goals (what "done" means)

The MVP is done when both directions work end to end on one laptop and these all hold:

1. **Sign to speech works.** At a natural pace with short pauses, the signer signs a sentence from the vocabulary and drops their hands. Within 2 s, the laptop speaks a natural English sentence that means what they signed.
2. **Speech to sign works.** The hearing person speaks any everyday English sentence. Within 2 s of them finishing, the avatar starts signing it with vocabulary signs, paraphrasing and fingerspelling where needed, with English captions.
3. **Facial grammar works.** Yes/no questions (raised brows), wh-questions (lowered brows) and negation (headshake) are read across the whole sentence and change the English: question intonation and punctuation, negation in the sentence. The avatar shows the same grammar on its face.
4. **Emotion works in both directions**, proven three ways:
   - **Grid test:** the same question signed happy, sad and angry keeps the correct question type every time while the detected emotion changes.
   - **Listening test:** the spoken voice audibly matches the detected emotion (Sonic's emotion, speed and volume controls).
   - **Avatar tone:** the avatar's face reflects the hearing person's tone (warm, apologetic, urgent, neutral) on top of the grammar.
5. **Fingerspelling works in both directions.** The signer can spell a name and have it spoken, and the avatar fingerspells names and words it can't paraphrase.
6. **Reliability:** five full conversations in a row (at least 6 turns each, both directions, including a yes/no question, a wh-question, a negation, a fingerspelled name and two different emotions) with no failure. The signer's sentences are planned in advance from the vocabulary; the hearing person speaks freely. *(My call:)* A **failure** is any of these: the wrong meaning is spoken or signed, a crash or hang, a needed restart, or a turn over the 2 s target. A top-3 picker tap, a sign the avatar shows as captions-only, or fingerspelling in place of a paraphrase is **not** a failure.

### When time runs short

The deadline holds: a working product as soon as possible, 5 days. If the schedule slips, cut in this order and keep everything else:

1. **Vocabulary from 150 to ~75 signs.** That means higher accuracy and fewer avatar clips to check.
2. **Natural pace to a brief hold per sign.** That makes segmentation easier.

Nothing else is cut. Fingerspelling both ways, the grammar and emotion readers, and avatar tone all stay.

### Numeric targets

| Measure | Target |
|---|---|
| Sign recognition, held-out ASL Citizen signers, 150 signs | ≥ 80% top-1 |
| Sign recognition, our own signing | ≥ 70% top-1, ≥ 90% top-3 |
| Question type across the emotion grid | ≥ 85% correct |
| Voice emotion recognized by listeners | ≥ 60% |
| Fingerspelled names (≤ 8 letters), our signing | ≥ 60% fully correct *(my call; revise after first measurement)* |
| Sign to speech: hands at rest → first audio (if the picker appears, the clock restarts at the tap; *my call*) | ≤ 2 s |
| Speech to sign: speaker stops → avatar starts | ≤ 2 s |

## Non-goals (not in this MVP)

- Open-vocabulary or fluent ASL translation. The vocabulary is ~150 signs plus fingerspelling.
- Ray-Ban Meta glasses mode (later).
- Voice picker, live lag meter, "ours vs naive" overlay (later). The signer gets one good default expressive voice.
- Scenario-specific vocabularies (café, pharmacy packs), own-voice cloning.
- Accounts, storage of recordings, security hardening, more than two people, mobile or app-store builds.
- Any sign language other than ASL; any medical or legal use.

## Anti-goals (failure even if it technically works)

- **A robotic voice:** speech that sounds the same whatever the emotion.
- **Questions read as emotion:** raised brows treated as surprise, or emotion overriding question type.
- **A wrong sign spoken confidently:** the system voices something the signer didn't mean, without giving them a chance to correct it.
- **An avatar that misrepresents ASL:** garbled handshapes or wrong facial grammar that a signer would find wrong or insulting. When unsure, show fewer signs and lean on captions rather than signing nonsense.

## Constraints

- **Team and time:** Kevin plus Claude build the MVP in **5 days**. The four-person team joins afterward for polish and P1.
- **Hardware:** one Apple Silicon Mac, used for training (PyTorch MPS) and for running. No cloud GPU is assumed.
- **Physical setup:** one laptop and one browser page. The webcam faces the signer, the laptop mic hears the hearing person, and the laptop speaker plays the voice. Steady front lighting (ring light) and a plain background.
- **Privacy:** only landmarks leave the browser, never video. Nothing is stored except the turn log, which holds no video or audio.
- **No fluent signer is available yet.** "Works" is measured on our own beginner signing plus held-out ASL Citizen signers. **Our own signing** comes from Kevin plus 1–2 teammates. One person is always held out of training, and the "our own signing" targets are measured on that person.
- **Avatar sign check:** each vocabulary clip is compared side by side with its ASL Citizen reference video. A clip that doesn't clearly match is shown as captions-only, which still counts toward goal 2. Fewer than ~10% of signs may be captions-only. A fluent (ideally Deaf) signer review comes after the MVP and is required before anyone outside the team uses it.
- **Data license:** ASL Citizen use must follow its license. Derived avatar motion stays private (not published) until the license is confirmed for that use.
- **Keys:** Cartesia (Sonic and Ink) and Anthropic. They live in a local `.env` and never reach the browser.

## Decisions already made

| Area | Decision |
|---|---|
| Vocabulary | ~150 **general everyday** signs (greetings, people, wants and needs, questions, numbers, time, places, food and drink, feelings, yes/no/not), plus the fingerspelling alphabet. The café is one demo situation, not the scope |
| Out-of-vocabulary English (hearing side) | Claude paraphrases into known signs first, then fingerspells names and leftovers. Captions always show the full English |
| Signing style | Natural pace with short pauses between signs; automatic segmentation. Hands back to rest ends the turn |
| Low-confidence signs | Top-3 picker: when any sign is uncertain, the signer taps the right guess before anything is spoken |
| Grammar vs emotion | Separate readers running in parallel. Grammar uses brows and headshake; emotion **never uses the brows** |
| Emotion proof | Grid test, listening test, avatar tone (see Goals) |
| Avatar | three.js VRM avatar driven by pose clips extracted from reference videos, with programmatic brow, headshake and expression overrides |
| Training data | ASL Citizen (the ~150 chosen signs) plus our own recordings |
| Stack | Browser page (MediaPipe Tasks JS, three.js plus a VRM, pose retargeting such as Kalidokit). Python FastAPI plus WebSockets backend, PyTorch on MPS. A fast Claude model for both language steps; Cartesia Sonic (TTS, streaming) and Ink (STT) |
| Latency | ≤ 2 s per direction |
| Extras | Only an internal **turn log** (timestamped per stage) for tuning and latency checks *(my call)* |

## How it works

### Sign to speech

1. **Camera client (browser):** MediaPipe Tasks gives hand, pose and face landmarks plus face blendshapes per frame, streamed to the backend over a WebSocket. Every session starts with a short **calibration**: a few seconds of neutral face and hands at rest, giving a per-user brow and face baseline and the rest position.
2. **Segmenter (backend):** motion-energy rules on the hand landmarks find each sign's start and end and spot the end of the turn (hands at rest past a threshold). It must also separate fingerspelling stretches from signs (e.g., one dominant hand held up near the shoulder with fast finger changes and little arm movement).
3. **Sign classifier:** a landmark-sequence model (PyTorch, trained on MPS) over the ~150 signs. It returns top-3 with confidences per segment.
4. **Fingerspelling reader:** a per-frame handshape classifier on hand landmarks, with letters collapsed over time into a word. Returns the word plus alternatives.
5. **Facial grammar reader:** sentence-level yes/no question, wh-question, negation or statement, from brow blendshapes relative to the calibration baseline plus head yaw oscillation for headshake. Rules first, with a small classifier added only if rules miss the target.
6. **Emotion reader:** emotion (at least happy, sad, angry, neutral; adding surprised or disappointed is the builder's call) and an intensity, from mouth, eye and cheek blendshapes plus signing speed and size. Never uses the brows.
7. **Picker:** if any segment's top-1 confidence is below threshold, the signer sees the top 3 for that segment and taps one. If all are confident, this step is skipped.
8. **English composer (Claude, JSON out):** signs, grammar and emotion go in. One natural English sentence plus voice controls (emotion, speed, volume) come out.
9. **Voice (Cartesia Sonic, streaming):** speaks the sentence. The signer's screen shows "you said: …".

### Speech to sign

1. **Transcription (Cartesia Ink):** the laptop mic streams to Ink, with turn detection marking the end of the speaker's turn.
2. **English to gloss (Claude, JSON out plus validator):** returns an ordered list of tokens, each a vocabulary sign or a fingerspelled word, plus a sentence-level grammar tag and a tone tag. The validator rejects any sign not in the vocabulary and retries once, then falls back to fingerspelling the unknown token.
3. **Avatar renderer (browser):** plays each sign's motion clip with smooth blends, fingerspells letters with per-letter handshapes, and overrides brows and headshake for grammar and expression for tone. English captions appear underneath.

### Shared

- **Orchestrator (FastAPI):** one session; turn-taking (one direction at a time, and the mic is ignored while the voice plays); routing; and the **turn log** with a timestamp at every stage boundary, for the latency targets.
- **Sign motion library:** one cleaned clip per vocabulary sign, extracted from ASL Citizen reference videos with MediaPipe and retargeted to the VRM, plus 26 fingerspelling handshapes.
- **UI (one page):** camera view, avatar with captions, a "you said" strip, the picker, and a start or calibrate button.

## Testing and evaluation

- **Unit tests** for the logic that branches: segmenter, grammar rules, gloss validator, fingerspelling collapse, orchestrator turn-taking.
- **Offline eval scripts:**
  - classifier top-1 and top-3 on held-out ASL Citizen signers and on our recordings
  - grammar accuracy on the emotion grid (3 question types × 3+ emotions × several takes, recorded by us)
  - fingerspelling on a set of spelled names
- **Listening test:** a few people label the emotion of generated voice clips without seeing the source.
- **Latency:** read from the turn log over the five-run reliability check.

## Left to the builder

- Choosing exactly which ~150 signs, guided by everyday frequency and by what ASL Citizen covers. Ask the fluent signer to review later.
- The classifier architecture and features, the segmentation thresholds, and the confidence threshold for the picker.
- The fingerspelling training data source (ASL Citizen has no alphabet). For example, a public fingerspelling landmark dataset or our own recordings; check its license.
- How the 26 avatar handshapes are made (extracted or hand-posed).
- Which VRM model, and the retargeting approach.
- Exact Cartesia and Claude model IDs and API shapes; verify against current docs.
- Repo layout. Suggested: a new top-level `app/` (backend, web, training) beside `latency-bench/`, which stays untouched.
- How to order the 5 days. Suggested: a working stub skeleton in both directions on day 1, then real components replace the stubs.
