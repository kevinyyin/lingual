# iOS plushie: the app on a phone inside a plush toy

Date: 2026-10-21. Status: design, awaiting approval. Branch: `eric-testing`.

## Intent

A kid and their parents sit in front of a plush toy (on a desk, a table, a bed). An iPhone stands upright inside it: the
screen shows the signing avatar, the front camera watches the kid. The plushie runs the teaching mode (the lesson path,
and "show me how to sign ___"), talks with a Cartesia voice, and can also translate both ways (speech to sign, sign to
speech). In the story it is the family's own phone; in practice it is a team member's iPhone.

**What this is for now:** the pitch and a demo video. Not real families, not the App Store.

**Must work on camera (the user's words):** speech to sign, sign to speech, the lesson, and the interaction with the
child and parents.

**Keep what exists:** the laptop web app (translator and teaching mode, `npm run dev`) keeps working unchanged.

## Goals

1. **An iOS app** that installs from Xcode onto a team iPhone, opens full screen, asks for camera and microphone the
   normal iOS way, and keeps the screen awake.
2. **Teaching mode on the phone**, portrait, readable at about 1 m: lesson path, exercises, grading of the kid's signs.
3. **The plushie talks in lessons** with a Cartesia voice through the phone speaker: a greeting, "now sign X",
   praise, "try again".
4. **"Show me a sign"**: a parent taps or types a word on the phone, the avatar signs it.
5. **Translator on the phone**: the phone mic hears the hearing person (speech to sign); the kid's or parent's signing
   is spoken through the phone speaker (sign to speech).
6. **Laptop mode unchanged**: opened in a laptop browser, the app still uses the laptop's mic and speaker as today.
   *My call:* the new lesson lines and "show me a sign" appear in the laptop teaching mode too (voice from the laptop
   speaker); "unchanged" means nothing existing breaks or behaves differently, not that the laptop gets no new features.

## Non-goals

- The kid talking to the plushie to ask for signs (parent taps or types for now; revisit after the iOS basics work).
- App Store, offline use, logins, accounts, multiple families or devices at once.
- Running recognition, Claude or Cartesia on the phone.
- Android.
- A real plush enclosure: for the demo the phone is held upright in the plush, jury-rigged.

## Anti-goals (a failure even if it technically works)

- Laggy or broken in the video: the avatar stutters, a sign takes seconds to be recognized, the voice cuts out.
- The plushie's own voice is transcribed as speech and triggers a reply loop.
- The laptop app breaks or changes behaviour.
- API keys (Anthropic, Cartesia) inside the app bundle.

## How it works

### Shape: a native shell that loads the page from the laptop

*Decided (my call, approved in chat):* wrap the existing `web/` app with **Capacitor**, no Swift rewrite.
*My call:* the app does **not** bundle the web files. It opens the page served by the laptop, through Capacitor's
`server.url`. Reasons:

- `web/public/signs` is 911 MB of avatar clips; bundling it makes every install slow.
- iOS allows the camera and mic only in a secure context; a page from `https://…` is one.
- Same origin as today: `/ws`, `/vocab`, `/reference`, `/dev` and the clip files keep their relative URLs; no URL
  config in the web code.
- A web change shows up on the phone on reload, no reinstall.

The native part is small: full screen, portrait, `Info.plist` texts for camera and microphone, keep-screen-awake (a
Capacitor plugin, e.g. `@capacitor-community/keep-awake`), and the start URL (`teach.html`).

### Where things run

| Phone (in the web view) | Laptop (server, as today) |
|---|---|
| Camera, MediaPipe tracking, the 3D avatar, lesson screens and progress (local storage), mic capture, playing voice audio | Sign recognition (PyTorch), Claude (compose, gloss), Cartesia STT and TTS, API keys, the Vite dev server |

Frames sent to the server are landmarks (JSON), not video, so the link carries little.

### Getting HTTPS to the phone

*My call:* a **Cloudflare quick tunnel** (`cloudflared tunnel --url http://localhost:5173`) gives a trusted
`https://….trycloudflare.com` address to the laptop's Vite server. It works on any network (pitch-room Wi-Fi, a phone
hotspot), needs no certificates on the phone, and Vite already proxies `/ws` and the API routes to `:8000`. The
tunnel URL changes every run, so *my call:* the app bundles one tiny start screen with a single URL field (remembering
the last value) that navigates to the pasted address; a tunnel restart on pitch day then needs no Xcode reinstall.
A named Cloudflare tunnel (fixed URL) is an alternative if the team has a domain. Vite must accept the tunnel's host
name (`server.allowedHosts`).

Server change this needs: `origin_ok` (`server/main.py`) only accepts `localhost` today; it must also accept the
tunnel origin, from an env var (e.g. `ALLOWED_ORIGINS`). Keep the check: it is what stops other websites reading the
session.

Risk: anyone who learns the random tunnel URL can open the page and use the session (and the API keys' quota) while
the tunnel runs. Acceptable for a demo; stop the tunnel after. Left to the builder: a fallback for a room with no
internet is a laptop hotspot plus `mkcert`, but Claude and Cartesia need internet anyway.

### Phone audio (translator and the plushie's voice)

The server already takes its audio ends as parameters: `MicTranscriber(mic=…)` (`server/stt.py`) and
`Voice(player_factory=…)` (`server/voice.py`). Today both default to the laptop's devices via `sounddevice`.

- **Mic:** the page captures the phone mic (`getUserMedia` audio, 16 kHz mono PCM16, ~100 ms chunks) and sends it over
  the existing WebSocket. On the server a per-connection queue becomes the `mic` iterator.
- **Voice:** a `player_factory` that sends each PCM chunk to the page over the WebSocket; the page plays it with Web
  Audio (24 kHz PCM16). The `"first_audio"` / `"done"` timing the session relies on stays as is.
- **Which ends:** the page says at connect time whether it brings its own audio. *My call:* the app does (it runs in
  Capacitor); a laptop browser does not, so the laptop keeps `sounddevice` exactly as today.
- **Echo:** the existing gate stays: while the session is `speaking` or `avatar`, the mic sends silence
  (`server/main.py`, `is_open`). Builder checks that phone speaker to phone mic does not leak into STT after playback
  ends; if it does, extend the gate briefly after `done`.
- `VOICE` in `server/main.py` is one object shared by all connections; the phone player has to be per connection
  (how is left to the builder).
- Left to the builder: binary WebSocket frames vs base64 in JSON, and how the client message is named.

### The plushie talks in lessons

Teaching mode today is silent except a chime, and practice mode tells the server to ignore speech. *My call:* the page
asks the server to say a short line (new client message, text only); the server speaks it through `Voice` and the
audio plays on the phone as above. Lines are short and fixed (greeting, "Now sign MOTHER", "Nice!", "Try again"),
written in the teaching code. Left to the builder: exact wording, when each plays, and whether to cache repeated lines.
The kid's sign attempts still go through the existing practice flow.

### "Show me a sign"

A box in teaching mode: a parent taps a suggestion or types a word; the avatar signs it. *My call:* look the word up in
`/vocab` (English meaning to gloss) and play that clip, fingerspelling when there is no match; no Claude call for a
single word. Left to the builder: suggestions to show, where the box sits.

### Translator on the phone

The translator page (`index.html`) runs in the app too, reached from the teaching home (the link already exists the
other way). It needs a portrait layout for the phone. Speech to sign and sign to speech use the phone audio above;
otherwise the translator logic is unchanged.

### Portrait and the camera

The phone stands upright, front camera at the top. The recognizer needs head to waist in frame: expect the kid about
1 m back. `camera.ts` asks for 640x480; in portrait the builder confirms what the camera delivers and that tracking and
recognition still hold. Check this early, on the real phone, in the real plush.

## Order of work

1. Capacitor shell + tunnel + origin change: teaching mode loads on the phone, camera works, a lesson can be graded.
   Confirm recognition holds in portrait at ~1 m, and the avatar runs smoothly on the phone. **Go/no-go gate** for the
   rest.
2. Phone audio: voice playback, then mic. Sign to speech and speech to sign working on the phone.
3. The plushie's lesson lines.
4. "Show me a sign".
5. Portrait layout pass on both pages; a full run-through of the demo script.

## Demo facts (the user's answers)

- **Phone: iPhone 17 Pro.** If step 1 shows stutter, look at the code before anything else; a newer phone is not an
  option.
- **The signer on camera is an adult or older kid acting the part**, not a young child. No child-specific recognition
  work.
- **The demo script is not fixed and will change.** *My call:* step 1's check uses Unit 1 (Greetings) signs plus a
  few common words for show-me and speech to sign; anything that must be on camera has to be a sign the recognizer
  knows and that has an avatar clip (`avatar_ok`). Nothing in the build hard-codes a script.

## Constraints

- Team iPhones, installed from Xcode (free Apple ID: apps expire after 7 days, reinstall before the pitch; a paid
  account avoids it).
- One session at a time: the server closes the older WebSocket when a new one opens (`server/main.py`). Only one
  phone or tab can be live.
- Models are for non-commercial research use (`app/README.md`): fine for a pitch, not for distribution.
- The laptop must be on, with the server, Vite and the tunnel running, during the demo.

## Testing

- Existing tests stay green (`uv run pytest`, `npm test`).
- Server: unit tests for the phone mic queue and the WebSocket player (fakes, like the existing session tests), and
  for `origin_ok` with an allowed extra origin.
- Web: a test for the audio-chunk encoder; the "show me a sign" lookup.
- On the phone: a scripted run of the demo (lesson, show-me, speech to sign, sign to speech) before recording.
