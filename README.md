# ASL Dragon (lingual): Two-Way ASL

A conversation and learning tool for American Sign Language. It has two halves that share one avatar, one camera
pipeline and one backend:

- **Translator.** A Deaf ASL signer and a hearing person who doesn't sign talk face to face. Signing becomes spoken
  English in a voice that carries the signer's emotion. Speech becomes a 3D avatar signing it, with the speaker's tone
  and ASL facial grammar, beside English captions.
- **Teaching mode.** A Duolingo-style ASL course: units of short lessons, an avatar that shows each sign, exercises that
  check your own signing through the camera, spoken encouragement, and a "show me a sign" box for any word.

It runs on a **laptop** in the browser, and on an **iPad inside a plush toy** (the plushie): the phone's screen shows
the avatar, its front camera watches the kid, and its speaker gives the plushie a voice.

> **Status: a working MVP for a pitch and demo video.** It has not been reviewed by a fluent Deaf signer, which is
> required before anyone outside the team uses it. It recognizes isolated signs and fingerspelling, not continuous
> ASL. It is meant to complement interpreters and teachers, never to replace them.

---

## Contents

1. [What you can do with it](#what-you-can-do-with-it)
2. [How it works](#how-it-works)
3. [Setup](#setup)
4. [Run it on a laptop](#run-it-on-a-laptop)
5. [Run it on an iPhone (the plushie)](#run-it-on-an-iphone-the-plushie)
6. [Demo-day checklist](#demo-day-checklist)
7. [Troubleshooting](#troubleshooting)
8. [The avatar](#the-avatar)
9. [Recognition, grammar, voice](#recognition-grammar-voice)
10. [Testing and measuring](#testing-and-measuring)
11. [Repository map](#repository-map)
12. [Tech stack](#tech-stack)
13. [Data and licenses](#data-and-licenses)

---

## What you can do with it

### Translator (`/`)

| Direction | What happens |
|---|---|
| **Sign to speech** | The camera reads the signer's hands, body and face. The backend splits the movement into signs, recognizes each one (or reads fingerspelling), reads facial grammar (yes/no question, wh-question, negation) separately from emotion, and Claude writes a natural English sentence. Cartesia speaks it in a voice whose emotion, speed and volume follow the signer's. If the recognizer isn't sure, a **picker** asks the signer which of three signs they meant. |
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
- **"Your turn" grading:** the camera points go to the recognizer; you pass when the target sign is in its top three
  guesses (it picks exactly the right sign about half the time, so top-1 alone would fail good signing). A miss gets a
  second try with "That looked like *X*". It tells you which sign it saw, not what to fix in your handshape.
- **The plushie talks:** a greeting, "Your turn! Sign hello.", praise, "Almost! Let's try again.", "You did it!". Only
  one line ever waits behind the one playing, so speech never lags the screen. The **Sound effects** switch in Profile
  turns the voice off too.
- **Show me a sign** (Practice tab): type any word; the avatar signs it, or fingerspells it if there's no sign.
- **Practice tab:** review weak signs, or sign words you know to the camera. **Challenges** and **Profile** track XP,
  streaks and hearts; progress is stored in the browser.

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

---

## How it works

### On a laptop

Everything runs on one machine. The browser does the camera, tracking and avatar; the Python backend does
recognition, Claude, Cartesia and the laptop's own mic and speaker.

```
browser (localhost:5173, Vite dev server)            backend (localhost:8000, FastAPI)
┌──────────────────────────────────┐               ┌───────────────────────────────────────────┐
│ camera → MediaPipe hand/pose/face │── points ───▶│ segmenter → sign recognizer / fingerspell  │
│   (only points leave, never video)│   WebSocket   │           → facial grammar + emotion       │
│ 3D avatar (three.js) + captions   │◀── plans ────│ → Claude (English) → Cartesia Sonic ──▶ 🔈  │
│ lessons, show-me, picker          │               │ 🎙 → Cartesia Ink → Claude (sign plan)      │
└──────────────────────────────────┘               └───────────────────────────────────────────┘
```

### On the iPhone (the plushie)

```
iPhone (in the plushie)                          Laptop
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

Requirements: macOS or Linux, [uv](https://docs.astral.sh/uv/) (Python 3.12), Node 20+, and API keys for
[Anthropic](https://console.anthropic.com) and [Cartesia](https://play.cartesia.ai). For the iPhone: Xcode, an Apple
ID, and `brew install cloudflared`.

```bash
cd app
uv sync                         # Python backend
cp .env.example .env            # add ANTHROPIC_API_KEY and CARTESIA_API_KEY
cd web && npm install && cd ..  # also fetches MediaPipe wasm + models into web/public
```

The sign models, the vocabulary and the avatar's sign clips are built from public datasets. The full download and
build steps (ASL Citizen weights, `build_vocab`, `clips_from_wlasl`, `touch_from_asllex`, the fingerspelling reader)
are in [`app/README.md`](app/README.md#setup). Without keys the app still runs: no Cartesia key means no voice or mic
(use `?dev=1` to type), no Anthropic key means English is the plain sign words and plans are fingerspelled. Without
any sign model a scripted demo recognizer stands in.

---

## Run it on a laptop

From `app/`, in two terminals:

```bash
uv run asl-server                # backend on :8000
```

```bash
npm --prefix web run dev         # page on :5173
```

Open <http://localhost:5173> for the translator or <http://localhost:5173/teach.html> for teaching mode.

**Signing tips:** sit head-to-waist in frame with steady light from the front. The camera tells you where to sit (closer,
back, to the middle, hands in your lap) and calibrates by itself once you're there; if you move, it calibrates again.
Sign one sign per movement, pause briefly between signs, and lower your hands for about half a second to finish a turn.

---

## Run it on an iPhone (the plushie)

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

In Xcode: plug in the iPhone, pick it as the run target, set **App → Signing & Capabilities → Team** to your Apple ID,
and press **Run**. If the phone asks, turn on Developer Mode (Settings → Privacy & Security) and trust the developer
(Settings → General → VPN & Device Management). Allow the camera (and, for the translator, the mic) when asked.

Re-run `npx cap sync ios` and press Run again only after changing `app/web/shell/` (the start screen) or `app/web/ios/`.
Everything else reaches the phone by rebuilding and reopening the app.

Keep your signing Team out of commits: each teammate signs with their own Apple ID, and Xcode writes it into
`project.pbxproj`.

### Live edits instead of a build

`npm --prefix web run dev:tunnel` with the tunnel pointed at `:5173` serves the dev server to the phone, so edits show
on reload without a build. It loads much more slowly on the phone, so use the build for demos.

---

## Demo-day checklist

1. Charge the phone and keep a cable in the plush; the screen never auto-locks while the app is open.
2. Laptop plugged in, sleep off. Build, start the tunnel, start the server with the new `ALLOWED_ORIGINS`.
3. Open the app, paste the new URL, and run one lesson through: the plushie talks, "Your turn" grades a sign, the
   avatar looks right.
4. Translator: sign THANK-YOU (the phone speaks it), say "nice to meet you" (the avatar signs it).
5. Flip the ringer switch to silent and check the voice is still audible.
6. Signer about 1 m from the upright phone, steady front light, head to waist in frame.
7. Close every other tab or device that has the app open (one session at a time).
8. Free Apple ID installs expire after 7 days: reinstall from Xcode the day before.
9. After the demo, stop the tunnel (Ctrl-C): anyone with the address can use the app while it runs.

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
| "Camera blocked" | Allow the camera for the app in iPhone Settings. |
| Stuck on "Rest your hands in your lap…" | It can't see you calibrate: step back so head to waist is in frame, light from the front, hands still. |
| No sound from the plushie | Check **Sound effects** is on in teaching mode's Profile; turn the volume up; reopen the app (iOS unlocks audio on the first tap). |
| Laptop browser stuck in phone mode | Open the page once with `?phone=0`. |
| Xcode: the app expired | Free Apple ID installs last 7 days. Press Run again. |

---

## The avatar

The default body is a stylized "toon" character built by `scripts/build_toon_avatar.py` (rounded shapes, flat colours,
big eyes, 20 ARKit face shapes on the MPFB rig; no Blender needed), as one of six signing buddies picked on the Profile
tab (Leo, Benji, Zeke, Lexi, Avery, Daisy): same body, own hair, skin, clothes and Cartesia voice. Motion comes from real signers:

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
| Sign recognition | The published ASL Citizen ST-GCN (Microsoft Research, 2,731 signs from 27 pose and hand keypoints), ensembled with the team's asl-spike model (143 signs). On signers neither model trained on, choosing among ~2,900 signs: top-1 0.60, top-3 0.81, holding up under 240p, blur, low light and half frame rate. The camera now reads only the 160 hand-checked signs (a lesson only its course's): replayed one sign per turn through the live path, 61% of everyday signs come out right in the translator (18% after the picker) and 14% as a wrong word, and 87% of correct signing passes a lesson. |
| Fingerspelling | A CTC reader trained here on Google's Kaggle ASL Fingerspelling landmarks from Deaf signers; 0.36 letter error rate on held-out signers. |
| Facial grammar and emotion | Rules over MediaPipe's face blendshapes and head angle: brows and headshake for grammar, mouth, eyes and signing speed and size for emotion, kept separate so a raised-brow question isn't read as "surprised". |
| Signs to English | Claude (`claude-haiku-5-5`) composes a sentence from the recognized signs, grammar and emotion. |
| English to signs | Claude turns English into a validated sign plan (ASL order, only signs the avatar can show, fingerspelling the rest) with grammar and tone; 182/182 plan checks pass. |
| Voice and speech-to-text | Cartesia Sonic 3.6 (emotion, speed and volume controls over a kept-open socket) and Ink 2 (turn-detecting transcription). |

Vocabulary: 160 hand-checked signs (`app/vocab/vocab.yaml`) plus every ASL Citizen and WLASL sign (~2,900) in the
generated `vocab_extended.yaml`, with English meanings from ASL-LEX 2.0. The camera reads only the 160; the avatar
shows any sign it has a clip for.

Laptop latency (simulated input): sign to speech p50 1.42 s from hands at rest to first audio; speech to sign 1.4 s
from the speaker stopping to the avatar starting.

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
| `app/server/` | FastAPI backend: `main.py` (WebSocket, origin check, serving the built page), `session.py` (turn-taking, practice mode, lesson lines), `phone_audio.py` (phone mic and speaker over the socket), `stt.py` / `voice.py` (Cartesia), `compose.py` / `gloss.py` (Claude), recognizers, segmenter, grammar and emotion readers. |
| `app/web/src/` | Translator (`main.ts`, `ui.ts`, `style.css`), camera and tracking (`camera.ts`), socket (`ws.ts`), phone audio (`phone.ts`), the avatar (`avatar/`), and teaching mode (`teach/`). |
| `app/web/shell/` | The iOS app's start screen (the only page bundled in the app). |
| `app/web/ios/` | The Xcode project generated by Capacitor (`AppDelegate.swift`, `Info.plist`). |
| `app/web/public/` | Sign clips (`signs/`), letters, avatar bodies, MediaPipe wasm and models; served in place, never copied into the build. |
| `app/training/` | Building the vocabulary, clips and contact data; training and evaluation. |
| `app/scripts/` | Avatar builders, the motion bench, round-trip and demo recording. |
| `app/README.md` | Full setup, training, evaluation and own-data recording steps. |
| `docs/superpowers/specs/` | Design documents (the original two-way design, avatar motion, the iOS plushie). |
| `docs/superpowers/plans/` | Implementation plans and the execution record: every decision made while building, with its evidence. |
| `asl-spike/` | The team's earlier recognition spike and its models. |
| `latency-bench/` | A separate harness that measured live speech-to-speech latency across providers. |

---

## Tech stack

**Phone and browser** (`app/web/`, Vite + TypeScript): [MediaPipe Tasks](https://ai.google.dev/edge/mediapipe) (hand,
pose and face landmarkers, on-device), [three.js](https://threejs.org) for the avatar, Web Audio for the phone's mic and
voice, [Capacitor](https://capacitorjs.com) for the iOS app, Vitest + Testing Library.

**Backend** (`app/server/`, Python 3.12, [uv](https://docs.astral.sh/uv/)): [FastAPI](https://fastapi.tiangolo.com) +
WebSockets, [PyTorch](https://pytorch.org) (ST-GCN and the fingerspelling reader), NumPy rules for segmenting,
grammar and emotion, the Anthropic SDK ([Claude](https://docs.anthropic.com)), the
[Cartesia](https://docs.cartesia.ai) SDK (Sonic and Ink), sounddevice for the laptop's mic and speaker, pytest.

**Getting to the phone:** [Cloudflare quick tunnels](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
(`cloudflared`) and Xcode.

---

## Data and licenses

Code is MIT ([LICENSE](LICENSE)). The sign models, clips and letters derive from ASL Citizen, WLASL, ASL-LEX and the
Kaggle ASL competitions, which allow **non-commercial research use only**; anything derived from them inherits that
limit and must not be redistributed (fine for a pitch and demo, not for an App Store release). The Rocketbox bodies
are MIT; the MPFB/MakeHuman body is CC0. Recordings of our own
signing are stored as landmarks only, never video, and are not committed.
