# ASL Dragon (lingual)

A two-way American Sign Language translator and an ASL course that share one signing avatar. It runs in a laptop
browser, or on an iPad or iPhone inside a plush toy.

## The problem

A Deaf ASL signer and a hearing person who doesn't sign have no shared language when they meet face to face, and an
interpreter is not always there. Someone learning ASL at home has a different gap: nobody watches their signing and
tells them whether it was right.

## What it does

- **Translator.** Signing becomes spoken English in a voice that carries the signer's emotion. Speech becomes a 3D
  avatar signing it, with the speaker's tone and ASL facial grammar, beside English captions.
- **Teaching mode.** A Duolingo-style course: units of short lessons, an avatar that shows each sign, exercises that
  check your own signing through the camera, spoken encouragement, and a "show me a sign" box for any word.

Both halves share one avatar, one camera pipeline and one backend. In the plush toy (the plushie), the phone's screen
shows the avatar, its front camera watches the kid, its speaker gives the plushie a voice, and an Arduino flaps the
plushie's arms and shows a check or an X on a small display in its neck. Kids pick their signing buddy: six avatars,
each with its own voice.

## Status

This is a working prototype. A fluent Deaf signer has not reviewed it yet, and that review is required before anyone
outside the team uses it. It reads single signs, short sentences of everyday signs and fingerspelling, not fluent
continuous ASL. It is meant to complement interpreters and teachers, never to replace them.

## Accuracy and limitations

Measured accuracy:

- Single signs: right 89% of the time on 995 held-out teaching videos (signers the model never saw), counting the
  "which sign did you mean?" picker; 83% on an outside dictionary (ASL Signbank). Details:
  [`app/reports/teaching_videos.md`](app/reports/teaching_videos.md).
- Short sentences: 2-4 signs come out exactly right about 53% of the time, 59% with a brief pause between signs. This
  is from 300 sentences spliced from single-sign clips the app already reads right on their own, replayed through the
  translator (`app/training/teach_eval/sentence_eval.py`). Most errors come from cutting the turn into signs: it gets
  the wrong number of pieces 34-40% of the time, usually two signs read as one. When a piece is cut right, its top
  guess is right 90% of the time.
- Live webcams do worse than these videos. Dim light, a signer far from the camera, or a learner's slower signing all
  lower confidence, and the picker then appears more often.

### Demo sentences (off by default)

The translator has an optional demo mode that snaps short sentences to a fixed list. It exists because open
translation gets only about half of short sentences exactly right. The code history calls it Plan C (design note:
[`docs/plan-C-scripted-demo-sentences.md`](docs/plan-C-scripted-demo-sentences.md)). It is off unless the page is
opened with `?demo=1` or the server runs with `LINGUAL_DEMO_SENTENCES=1`. When it is on:

1. After a signed turn is cut into pieces, each piece is read again for its 10 most likely signs.
2. The pieces are aligned against each of the 14 sentences in
   [`app/vocab/demo_sentences.yaml`](app/vocab/demo_sentences.yaml), such as HELLO NICE MEET and WANT COFFEE PLEASE.
   The alignment allows one piece to stand for two signs, or one sign to be cut into two pieces.
3. The best sentence wins only if its score per sign clears a floor (which still allows roughly one sign in three to
   be missing from the guesses) and it leads the runner-up by a margin (`FLOOR` and `MARGIN` in
   `app/server/demo_sentences.py`). The app then shows that sentence's signs and speaks its fixed English, for example
   "Hello, nice to meet you!". It skips the picker and Claude.
4. A turn with no clear match is translated the normal way.

Lessons and sign-by-sign mode never snap. The avatar mirror page (`?mode=mirror&demo=1`) snaps too, but only shows
the sentence. The 53% and 59% above are for normal translation, without snapping. Snapping has not been measured
live. Before it was built, an offline estimate put correct snaps at 96-98%, with the caveat that the test sentences
overlapped less than the real list does (the design note expects 85-95%). A recording made with `?demo=1` shows the
scripted sentences, not open translation.

### Other limitations

- Vocabulary: the camera reads 155 everyday signs plus fingerspelling.
- Sign by sign (in the translator's More menu, or `?bysign=1`) avoids the cutting problem: you lower your hands after
  each sign, the words build up on screen, and Say it (or a 3 s pause) speaks them. It has not been measured live.
- Fingerspelling uses a Core ML model, so it reads letters only on a Mac. Elsewhere spelled words go to the sign model.
- The phone needs the backend running on a laptop, reached through a Cloudflare tunnel; it does not run on its own yet.
- One page per backend: opening a second tab takes the session away from the first.
- Licences: the sign model and sign data come from research datasets (ASL Citizen, WLASL, ASL-LEX) that allow
  non-commercial research use only. See [Data and licenses](#data-and-licenses).

---

## Contents

1. [What you can do with it](#what-you-can-do-with-it)
2. [How it works](#how-it-works)
3. [Setup](#setup)
4. [Run it on a laptop](#run-it-on-a-laptop)
5. [Run it on an iPad or iPhone (the plushie)](#run-it-on-an-ipad-or-iphone-the-plushie)
6. [Troubleshooting](#troubleshooting)
7. [The avatar](#the-avatar)
8. [Recognition, grammar, voice](#recognition-grammar-voice)
9. [Testing and measuring](#testing-and-measuring)
10. [Repository map](#repository-map)
11. [Tech stack](#tech-stack)
12. [Data and licenses](#data-and-licenses)
13. [Team](#team)

---

## What you can do with it

### Translator (`/`)

| Direction | What happens |
|---|---|
| **Sign to speech** | The camera reads the signer's hands, body and face. The backend finds the turn, reads fingerspelled words with the fingerspelling reader, and cuts the rest into signs where the sign model is surest (so a sentence signed without pauses still splits) and recognizes each. It reads facial grammar (yes/no question, wh-question, negation) separately from emotion, and Claude writes a natural English sentence. Cartesia speaks it in the buddy's voice, with emotion, speed and volume following the signer's. If the model is under 50% sure of a sign, a **picker** asks which of three signs they meant. Then a **mini buddy** pops up under the sentence and signs back what the camera read ("I saw: HELLO NAME"); tap it to see it again. |
| **Speech to sign** | The hearing person holds Space (or the Hold to talk button) and talks; nothing said otherwise is translated. Cartesia transcribes them, Claude turns the English into an ASL-order sign plan (paraphrasing into signs the avatar knows, fingerspelling names and the rest), and the avatar signs it with the matching brows and head movement. The caption and a row of sign chips show what is being signed; the current sign lights up. |

Turn-taking is one direction at a time: the **Voice / Signing** switch at the top picks it (voice mode ignores the
camera, signing mode ignores the mic). The status pill says whose turn it is, and on the phone the whole stage's edge
takes the status colour: **blue** listening or reading signs, **green** speaking or signing, **yellow** getting into
position, calibrating or picking, **red** an error.

### Teaching mode (`/teach.html`)

- **Course:** five units (Greetings, People, Questions, Food & drink, Feelings), each a path of short lessons ending in a
  review. Each lesson shows every sign, has you sign it to the camera (**"Your turn"**), then asks what each sign means
  (pick-the-sign, match and order are off for now). Redo any answer, go back a step, and a lesson you leave picks up
  where you were.
- **"Your turn" grading:** the camera points go to the recognizer, which chooses among the course's signs; you pass
  when the target sign is in its top three guesses. It reads the whole attempt and its last 1.5 s, so a wrong sign
  corrected without resting counts the correction, and raising your hands into position and holding before you sign
  doesn't spoil the read. Replayed correct signing passes 94% of the time; a wrong sign passes about 5%. A miss gets
  a second try with "That looked like *X*". It tells you which sign it saw, not what to fix in your handshape.
- **The plushie talks** in the buddy's voice: a greeting, "Your turn! Sign hello.", praise, "Almost! Let's try
  again.", "You did it!". Only one line ever waits behind the one playing, so speech never lags the screen. The
  **Sound effects** switch in Profile turns the voice off too. On a right answer the plushie flaps its arms (if its
  Arduino is plugged in, see [The plushie's arms and neck display](#the-plushies-arms-and-neck-display)).
- **Signing buddy** (Profile tab): Leo, Benji, Zeke, Lexi, Avery or Daisy, each with their own look and Cartesia voice.
  Tapping one swaps the avatar everywhere (lessons and translator) and says hello in that voice; the choice is saved
  on the device.
- **Show me a sign** (Practice tab): type any word; the avatar signs it, or fingerspells it if there's no sign.
- **Practice tab:** review weak signs, or sign words you know to the camera. **Challenges** and **Profile** track XP
  and streaks; progress is stored in the browser.

### Debug and tool pages

| URL | What |
|---|---|
| `/?dev=1` | Adds a box to type what the hearing person says (no mic needed), and shows sign IDs in the picker |
| `/?camera=0` | Speech to sign only |
| `/?mode=review` | Each avatar clip beside the video it came from |
| `/?mode=record&signer=NAME` | Record your own signing as landmarks (see [`app/README.md`](app/README.md)) |
| `/?mode=bench` | The motion benchmark harness (driven by `scripts/motion_bench.py`) |
| `?model=Male_Adult_11`, `?model=mpfb` | Other avatar bodies |
| `?phone=1` / `?phone=0` | Turn phone-audio mode on or off for that browser (the iOS app sets it for you) |
| `/?mode=mirror` | The avatar mirror: the avatar copies the signer live and labels each sign as it is read, then shows the English sentence (never spoken) |
| `?bysign=1` / `?bysign=0` | Sign by sign on or off (also in the More menu); remembered per browser |
| `?demo=1` | Demo sentences: signed turns that closely match a scripted sentence snap to it (see [Demo sentences](#demo-sentences-off-by-default)) |

---

## How it works

### On a laptop

Everything runs on one machine. The browser does the camera, tracking and avatar; the Python backend does
recognition, Claude, Cartesia and the laptop's own mic and speaker.

```
browser (localhost:5173, Vite dev server)           backend (localhost:8000, FastAPI)
┌────────────────────────────────────┐              ┌───────────────────────────────────────────────┐
│ camera → MediaPipe hand/pose/face  │── points ───▶│ segmenter → re-cut → signs / fingerspell      │
│   (only points leave, never video) │   WebSocket  │           → facial grammar + emotion          │
│ 3D avatar (three.js) + captions    │◀── plans ────│ → Claude (English) → Cartesia Sonic → speaker │
│ lessons, show-me, picker, mini     │              │ mic → Cartesia Ink → Claude (sign plan)       │
└────────────────────────────────────┘              └───────────────────────────────────────────────┘
```

### On the phone (the plushie)

```
iPad/iPhone (plushie)                            Laptop
┌────────────────────────────┐                   ┌────────────────────────────────────┐
│ iOS app (Capacitor)        │                   │ FastAPI :8000                      │
│  start screen → paste URL  │  HTTPS tunnel     │  ├ serves the built page (web/dist)│
│  full-screen web view      │◀────────────────▶│  │   and clips/models (web/public)  │
│   ├ camera + MediaPipe     │  (Cloudflare)     │  ├ sign recognition (PyTorch)      │
│   ├ avatar, lessons        │                   │  ├ Claude                          │
│   ├ mic  ─ 16 kHz audio ──▶│── one WebSocket ─▶│  ├ Cartesia speech-to-text         │
│   └ speaker ◀─ 24 kHz ─────│◀─────────────────│  └ Cartesia voice                  │
└────────────────────────────┘                   └────────────────────────────────────┘
```

The design choices behind this, and why:

- **The app is a thin shell.** [Capacitor](https://capacitorjs.com) wraps a full-screen iOS web view. It bundles one
  tiny start screen (`app/web/shell/`); you paste the laptop's address and it loads the real page from the laptop. So
  the 900+ MB of sign clips stay on the laptop, and a web change reaches the phone by reopening the app, with no
  reinstall. The only native code is one line in `AppDelegate.swift` that keeps the screen awake, plus the camera and
  mic permission texts in `Info.plist`.
- **HTTPS through a tunnel.** iOS lets a page use the camera and mic only over HTTPS. A Cloudflare quick tunnel gives the
  laptop a free public HTTPS address that works on any Wi-Fi or a phone hotspot. Its address changes every run.
- **What runs where.** The phone does what must happen live and close to you: camera, tracking, avatar, lesson screens.
  The laptop does what is heavy or secret: the PyTorch recognizer, Claude, Cartesia and the API keys. Only small data
  crosses the tunnel: hand points (never video), text, and short audio chunks.
- **The phone brings its own mic and speaker.** The app opens pages with `?phone=1`, so the page connects with
  `/ws?audio=client`. The server then reads mic audio from that socket (`server/phone_audio.py` `ClientMic`) and sends
  the voice back over it (`WsPlayer`) instead of using the laptop's `sounddevice`. A laptop browser never sets this, so
  laptop mode is unchanged.
- **The plushie never hears itself.** While it speaks, the server feeds silence to speech-to-text. The voice is sent at
  the pace it plays and the server waits half a second after it ends, so the mic only reopens once the phone is quiet.
  Each speech-to-text reconnect starts with fresh audio, so a network blip can't replay old sound.
- **A fast page on the phone.** For the phone, the backend serves the production build (one bundled, minified page with
  its styles in the head) instead of the Vite dev server's dozens of modules. The 30 MB of hand/face/body trackers start
  downloading in the background as soon as teaching mode opens, so the first "Your turn" doesn't wait.
- **One session at a time.** The server keeps one live page; opening another takes over and the old page shows "Open
  elsewhere" with a **Use it here** button.

---

## Setup

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/) (Python 3.12), Node 20+, a webcam, and API keys for
[Anthropic](https://console.anthropic.com) and [Cartesia](https://play.cartesia.ai). For the phone: Xcode, an Apple
ID, and `brew install cloudflared`.

```bash
git clone https://github.com/kevinyyin/lingual.git && cd lingual/app
uv sync                         # Python backend
cp .env.example .env            # add ANTHROPIC_API_KEY and CARTESIA_API_KEY
npm --prefix web install        # also fetches MediaPipe wasm + models into web/public
```

The repo has no API keys. Keys live only in `app/.env`, which is git-ignored; `.env.example` lists the names with
empty values.

The sign model (`app/models/signs.pt`) is in the repo. The fingerspelling reader (18 MB, Macs only) is downloaded
once from Hugging Face, and the vocabulary and the avatar's sign clips are built from public datasets. The commands
(and how to rebuild the sign model from ASL Citizen videos) are in [`app/README.md`](app/README.md#setup). Without
keys the app still runs: no Cartesia key means no voice or mic (use `?dev=1` to type), no Anthropic key means English
is the plain sign words and plans are fingerspelled. Without the fingerspelling reader, spelled words go to the sign
model; without any sign model a scripted demo recognizer stands in.

---

## Run it on a laptop

From `app/`, in two terminals:

```bash
uv run asl-server                # backend on :8000
```

```bash
npm --prefix web run dev         # page on :5173
```

Open <http://localhost:5173> for the translator or <http://localhost:5173/teach.html> for teaching mode in Chrome or
Edge, and allow the camera and mic. Keep one tab open: the backend serves one page at a time.

- Signing to speech: switch to **Signing**, sit head-to-waist in frame, hold still for 2 s while it calibrates, sign,
  then lower your hands.
- Speech to signing: switch to **Voice** and hold **Space** while you talk.
- No mic: add `?dev=1` to type what the hearing person says instead.

On Windows, if Application Control blocks the `asl-server.exe` launcher, run
`uv run python -c "from server.main import run; run()"` instead.

Signing tips: sit head-to-waist in frame with steady light from the front. The camera tells you where to sit (closer,
back, to the middle, hands in your lap) and calibrates by itself once you're there; if you move, it calibrates again,
and **Recalibrate** in the More menu forces it. In the translator, sign a short sentence at your own pace (pauses
between signs are fine but not needed) and lower your hands for about half a second to finish a turn. Fingerspell
with one hand, the wrist fairly still.

---

## Run it on an iPad or iPhone (the plushie)

### Every time

From `app/`, in three terminals:

```bash
npm --prefix web run build                        # after any web change; the phone gets web/dist
```

```bash
cloudflared tunnel --url http://localhost:8000    # prints https://….trycloudflare.com
```

```bash
ALLOWED_ORIGINS=https://….trycloudflare.com uv run asl-server   # paste the URL the tunnel printed
```

Then on the phone: open the app, clear the address field, paste the tunnel URL and tap **Open**. Teaching mode opens;
the translator is the icon in its top-right corner.

`ALLOWED_ORIGINS` is the server's guard against other websites reading the session: it only accepts pages from
localhost and the exact addresses you list (comma-separated, a trailing `/` is fine).

### First time (installing the app)

```bash
cd app/web && npx cap sync ios && npx cap open ios
```

In Xcode: plug in the iPad or iPhone, pick it as the run target, set **App → Signing & Capabilities → Team** to your
Apple ID, and press **Run**. If the phone asks, turn on Developer Mode (Settings → Privacy & Security) and trust the developer
(Settings → General → VPN & Device Management). Allow the camera (and, for the translator, the mic) when asked.

Re-run `npx cap sync ios` and press Run again only after changing `app/web/shell/` (the start screen) or `app/web/ios/`.
Everything else reaches the phone by rebuilding and reopening the app.

Keep your signing Team out of commits: each teammate signs with their own Apple ID, and Xcode writes it into
`project.pbxproj`.

### Live edits instead of a build

`npm --prefix web run dev:tunnel` with the tunnel pointed at `:5173` serves the dev server to the phone, so edits show
on reload without a build. It loads much more slowly on the phone, so use the build otherwise.

### The plushie's arms and neck display

An Arduino Uno R4 WiFi, plugged into the laptop by USB, drives two 9 g servos (arm A on pin 6, arm B on pin 9), a
MAX7219 8x8 dot matrix in the plushie's neck and two LEDs. Upload `app/arduino/plush_arms/plush_arms.ino` with the
Arduino IDE. The server finds the board by itself (`server/plush.py`) and tells it each lesson answer: a right answer
shows a check, lights the right LED and flaps the arms three times; a wrong one shows an X; a missed first try shows a
!. With no board plugged in, nothing happens. Wiring is in [`app/README.md`](app/README.md).

To set the angles for your plush, stop the server, open the Serial Monitor at 115200 baud, and type `a90` or `b90` to
turn arm A or B to 90°. Find where each arm hangs down and where it's raised without straining, then put those in
`A_DOWN`/`A_UP` and `B_DOWN`/`B_UP` at the top of the sketch.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Phone shows "Cloudflare Tunnel error" | The app is still opening an old address (it pre-fills the last one), or the tunnel stopped. Clear the field and paste the current URL; if the tunnel died, restart it and the server with the new `ALLOWED_ORIGINS`. |
| Nothing happens after tapping Open | The tunnel address no longer exists (quick tunnels can drop). Start a new one. |
| "Connection lost" / "Backend disconnected" | The server stopped or restarted, or `ALLOWED_ORIGINS` doesn't match the tunnel URL. Check the server is running with the right URL, then tap **Reconnect**. |
| "Open elsewhere" | Another tab or device took the session. Tap **Use it here**, or close the other one. |
| Teaching page slow or unstyled on the phone | The tunnel points at the Vite dev server (`:5173`). Point it at `:8000` and run `npm --prefix web run build`. |
| Camera slow to start | First load downloads ~30 MB of trackers; it starts in the background when teaching mode opens. Later loads are cached. |
| "Camera blocked" | Allow the camera for the app in the device's Settings. |
| Stuck on "Rest your hands in your lap…" | It can't see you calibrate: step back so head to waist is in frame, light from the front, hands still. |
| No sound from the plushie | Check **Sound effects** is on in teaching mode's Profile; turn the volume up; reopen the app (iOS unlocks audio on the first tap). |
| No voice at all, server log shows HTTP 402 | The Cartesia account is out of credit. Top it up and restart the server. |
| Every line is in Leo's voice, or the phone's avatar doesn't load | The server or the phone's page is older than the buddies. Restart the server and run `npm --prefix web run build`. |
| Fingerspelled names come out as random signs | The fingerspelling reader isn't installed: download it into `app/models/fingerspell_coreml/` (Macs only, see `app/README.md`) and restart the server. |
| Arms don't flap | The Arduino isn't plugged in, or the server started before it was: plug it in and get a sign right again (the server looks for it on every cheer). |
| Laptop browser stuck in phone mode | Open the page once with `?phone=0`. |
| Xcode: the app expired | Free Apple ID installs last 7 days. Press Run again. |

---

## The avatar

The body is a stylized "toon" character built by `scripts/build_toon_avatar.py` (rounded shapes, flat colours, big eyes,
20 ARKit face shapes on the MPFB rig; no Blender needed), one file per signing buddy in `app/web/public/avatar/toon/`
(`leo.glb` … `daisy.glb`, with a headshot each for the Profile picker). All six share one body and rig, so every sign
plays the same on each; they differ in hair (side part, curls, quiff, puffs, ponytail, bob), skin and clothes. Hair
beyond Leo's short cut sits in a mesh the hands don't collide with, so the face the hands touch is the same for every
buddy. The mini buddy that signs back what the camera read is a second, small copy of the same avatar
(`app/web/src/avatar/mini.ts`). Motion comes from real signers:

- **Clips:** each sign is extracted with MediaPipe from WLASL dictionary videos (the most typical of up to four takes),
  and fingerspelled letters from the Kaggle fingerspelling data. Wrists go where the signer's were relative to their
  body (relative to the nose near the face); contact data from ASL-LEX pins hands to the chin, forehead or other hand.
- **Arms:** a two-bone solver with human joint limits, frame-to-frame continuity, filtered rotations and a body-collision
  push. **Hands:** every finger joint is copied relative to the palm and kept within anatomical ranges.
- **Between signs:** blends follow minimum-jerk arcs whose length grows with the distance travelled. Every clip drops the
  recorded signer's own lift from and drop back to rest (12%, the median across the clips), because the blends already
  supply that travel; kept, it read as a wave before HELLO.
- **The hand that isn't signing:** if a hand never rises 0.2 m above the hips in a clip, it is the recorded signer's
  resting hand (often half out of the video, with garbage orientation), so the avatar uses its own relaxed rest arm
  instead, unless the hands touch (a low base hand, as in COOK).
- **Head and shoulders** move naturally but stay small enough never to read as grammar; grammar (raised brows for yes/no
  questions, furrowed for wh-questions, headshake for negation) is driven by the sign plan.

`scripts/motion_bench.py` measures the motion (spikes, backward elbows, twist, penetration, contact error, finger
limits) before and after any change; reports are in `app/reports/motion/`.

Known limits: a brief palm-up flick as the hand leaves rest on some signs; fists can put the thumb into the curled
fingers; two-hand contact signs can overlap slightly.

---

## Recognition, grammar, voice

| Part | How |
|---|---|
| Sign recognition | Our own model for the camera's signs (`app/models/signs.pt`: a bidirectional GRU over shoulder-normalized hand and arm landmarks), trained on ~5,600 ASL Citizen videos of 155 of the 160 hand-checked signs, each also shown inside made-up sentences between other signs, cut the way the live translator cuts. On ASL Citizen's unseen signers: 85% top-1 alone, 84% inside sentences. On other people's WLASL dictionary videos, through the live path: 69% of single signs right (the published ASL Citizen ST-GCN it replaced: 58%), 53% of signs inside sentences (48%). The ST-GCN is still the fallback when `signs.pt` is missing. |
| Cutting sentences | The segmenter finds the turn and cuts at 300 ms pauses; then each run of signs is cut again where the sign model reads the pieces most surely (`server/recut.py`, every dip in wrist speed a candidate, ~50 ms a sentence). Replayed sentences: signs wrong 75% → 60%, the right number of signs 42% → 67%, exactly right 5% → 9%. |
| Fingerspelling | A pretrained CTC reader on MediaPipe Holistic landmarks (Hugging Face `Skunk0/openglasses-fingerspelling-ctc`, Apache-2.0, trained on Google's Kaggle ASL Fingerspelling corpus), run by Core ML on the Mac (`server/fingerspell_coreml.py`). It needs the whole face mesh, which the browser sends while a hand is up. Kaggle names fed the way the app sends frames: 17% of letters wrong (37% without the face). A spelled piece is recognized by one hand, a still wrist and busier fingers (95% of real spelling caught); a weak letter read goes back to the sign model. |
| Facial grammar and emotion | Rules over MediaPipe's face blendshapes and head angle: brows and headshake for grammar, mouth, eyes and signing speed and size for emotion, kept separate so a raised-brow question isn't read as "surprised". |
| Signs to English | Claude (`claude-haiku-5-5`) composes a sentence from the recognized signs, grammar and emotion. |
| English to signs | Claude turns English into a validated sign plan (ASL order, only signs the avatar can show, fingerspelling the rest) with grammar and tone; 182/182 plan checks pass. |
| Demo sentences | Off by default (`?demo=1`): a signed turn that closely matches a scripted sentence is said as its fixed English (`server/demo_sentences.py`, see [Demo sentences](#demo-sentences-off-by-default)). |

### Voice and speech-to-text (Cartesia)

| Feature | Where | What it does here |
|---|---|---|
| Sonic 3.6 TTS, streaming over one kept-open WebSocket | `server/voice.py` | Says the signer's sentence out loud. It starts speaking as audio arrives (raw PCM, 24 kHz) and reuses one socket across turns, so there is no handshake per line. |
| `generation_config`: emotion, speed, volume | `server/voice.py` `voice_controls` | The signer's face and signing energy drive the voice. Happy → `happy`/`excited`, sad → `sad`/`disappointed`, angry → `frustrated`/`angry`. Speed and volume scale with intensity, so a Deaf signer's tone is heard, not just their words. |
| Six voices | `server/voice.py` `BUDDY_VOICES` | Each signing buddy (Leo, Benji, Zeke, Lexi, Avery, Daisy) speaks in its own Sonic voice; the page sends the buddy's name and the server picks the voice. |
| Context cancel | `server/voice.py` | Cuts a line off when the conversation moves on (in teaching mode, the newest line replaces a stale one), keeping the socket for the next line. |
| Ink 2 STT with auto-finalize | `server/stt.py` | The hearing person's push-to-talk speech becomes text with Cartesia's turn detection, then Claude plans it as ASL for the avatar. |
| Audio over the page's socket | `server/phone_audio.py` | On the phone, its mic feeds Ink 2 and Sonic's audio plays on its speaker: the plushie's voice. |

Vocabulary: 160 hand-checked signs (`app/vocab/vocab.yaml`) plus every ASL Citizen and WLASL sign (~2,900) in the
generated `vocab_extended.yaml`, with English meanings from ASL-LEX 2.0. The camera reads only the hand-checked signs:
155 of the 160, because HOW-MANY, FIVE, TEN, FOOD and FORGET have no ASL Citizen videos to learn from. The avatar shows
any sign it has a clip for; anything else the hearing person says is fingerspelled.

Laptop latency (simulated input, before the sentence re-cutting added ~50 ms): sign to speech p50 1.42 s from hands at
rest to first audio; speech to sign 1.4 s from the speaker stopping to the avatar starting.

---

## Testing and measuring

```bash
cd app
uv run pytest                    # backend: session, protocol, phone audio, origin check, voice, recognition, …
npm --prefix web test            # web: avatar timeline and solver, lessons, picker, phone audio, show-me, …
```

Measurement scripts (details in [`app/README.md`](app/README.md#evaluate)):

| Script | Measures |
|---|---|
| `training.train_signs` | Trains the sign model and prints ASL Citizen test accuracy, alone and inside sentences |
| `training.eval_recognition` | Recognition on real online videos across 7 quality conditions |
| `training.eval_gloss` | ~180 live English-to-sign plans checked for grammar, signs and order |
| `training.eval_fingerspell` | The letter model on Kaggle signers |
| `training.eval latency` | Turn delays from `logs/turns.jsonl` |
| `scripts/motion_bench.py` | Avatar motion quality (Playwright, `?mode=bench`) |
| `scripts/roundtrip.py` | Renders avatar signs and feeds them back to the recognizer |
| `scripts/record_demo.py` | Screen recordings of both directions |

---

## Repository map

| Path | What's there |
|---|---|
| `app/server/` | FastAPI backend: `main.py` (WebSocket, origin check, serving the built page), `session.py` (turn-taking, practice mode, lesson lines), `phone_audio.py` (phone mic and speaker over the socket), `stt.py` / `voice.py` (Cartesia, the buddies' voices), `compose.py` / `gloss.py` (Claude), `demo_sentences.py` (demo-sentence snapping), `classifier.py` (the sign model), `segmenter.py` and `recut.py` (cutting turns into signs), `fingerspell_coreml.py` (fingerspelling), `plush.py` (the plushie's Arduino), grammar and emotion readers. |
| `app/models/signs.pt` | The sign model (the other models are downloaded, see `app/README.md`). |
| `app/web/src/` | Translator (`main.ts`, `ui.ts`, `style.css`), camera and tracking (`camera.ts`), socket (`ws.ts`), phone audio (`phone.ts`), the buddies (`buddies.ts`), the avatar and mini buddy (`avatar/`), and teaching mode (`teach/`). |
| `app/arduino/plush_arms/` | The plushie's Arduino sketch: arms, neck display and LEDs. |
| `app/web/shell/` | The iOS app's start screen (the only page bundled in the app). |
| `app/web/ios/` | The Xcode project generated by Capacitor (`AppDelegate.swift`, `Info.plist`). |
| `app/web/public/` | Sign clips (`signs/`), letters, avatar bodies, MediaPipe wasm and models; served in place, never copied into the build. |
| `app/training/` | Building the vocabulary, clips and contact data; training and evaluation. |
| `app/scripts/` | Avatar builders, the motion bench, round-trip and demo recording. |
| `app/README.md` | Full setup, training, evaluation and own-data recording steps. |
| `app/vocab/` | The hand-checked vocabulary (`vocab.yaml`) and the demo sentences (`demo_sentences.yaml`). |
| `app/reports/` | Measurement write-ups: teaching-video accuracy, sign reliability, avatar motion. |
| `docs/plan-C-scripted-demo-sentences.md` | The design note for demo-sentence snapping. |
| `asl-spike/` | The team's earlier recognition spike and its models. |
| `latency-bench/` | A separate harness that measured live speech-to-speech latency across providers. |

---

## Tech stack

**Phone and browser** (`app/web/`, Vite + TypeScript): [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe) (hand,
pose and face landmarkers, on-device), [three.js](https://threejs.org) for the avatar, Web Audio for the phone's mic and
voice, [Capacitor](https://capacitorjs.com) for the iOS app, Vitest + Testing Library.

**Backend** (`app/server/`, Python 3.12, [uv](https://docs.astral.sh/uv/)): [FastAPI](https://fastapi.tiangolo.com) +
WebSockets, [PyTorch](https://pytorch.org) (the sign model, and the ST-GCN fallback),
[Core ML](https://github.com/apple/coremltools) via coremltools (the fingerspelling reader, Macs only), NumPy for
segmenting, grammar and emotion, the Anthropic SDK ([Claude](https://docs.anthropic.com)), the
[Cartesia](https://docs.cartesia.ai) SDK (Sonic and Ink), sounddevice for the laptop's mic and speaker, pyserial for
the plushie's Arduino, pytest.

**Getting to the phone:** [Cloudflare quick tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
(`cloudflared`) and Xcode.

---

## Data and licenses

Code is MIT ([LICENSE](LICENSE)). The sign model, clips and letters derive from ASL Citizen, WLASL, ASL-LEX and the
Kaggle ASL competitions, which allow **non-commercial research use only**; anything derived from them inherits that
limit (fine for a research prototype, not for an App Store release). `app/models/signs.pt` is trained on ASL Citizen,
whose licence also forbids redistributing modified data; it is published here at the repo owner's decision and is
for non-commercial research use only. The fingerspelling reader is Apache-2.0, trained on Google's ASL Fingerspelling
corpus (CC-BY 4.0), and is downloaded, not committed. The six buddy avatars are generated here (MIT) on the MPFB rig
(CC0); the Rocketbox bodies are MIT. Recordings of our own signing are stored as landmarks only, never video, and are
not committed.

---

## Team

Kevin Yin, Eric Wang (the avatar mirror), z57333 (sign-by-sign mode) and Nathan Tien.
