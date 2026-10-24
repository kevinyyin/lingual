# iOS Plushie Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the existing web app (teaching mode and translator) on an iPhone 17 Pro inside a plushie, with the
plushie's Cartesia voice and the phone's mic, while the laptop app keeps working as today.

**Architecture:** A Capacitor iOS shell whose bundled start screen opens the laptop's Vite dev server through a
Cloudflare quick tunnel (HTTPS, so iOS allows camera and mic). Recognition, Claude and Cartesia stay on the laptop
server. A page opened in the app adds `?audio=client` to its WebSocket; the server then reads mic audio from, and
sends voice audio to, that socket as binary frames instead of using the laptop's `sounddevice`.

**Tech Stack:** Capacitor (iOS), Vite + TypeScript + vitest, FastAPI + pytest, Cartesia Ink/Sonic, Web Audio,
cloudflared.

**Spec:** `docs/superpowers/specs/2026-10-21-ios-plushie-design.md`

## Global Constraints

- Laptop mode unchanged: a page without `?phone=1` behaves exactly as today (laptop mic and speaker via `sounddevice`).
- No API keys in the app or the web bundle; Claude and Cartesia are called only by the server.
- Keep `origin_ok`: only localhost plus the exact origins in `ALLOWED_ORIGINS` may open `/ws` or POST `/dev/transcript`.
- Phone: iPhone 17 Pro, portrait only, installed from Xcode; screen never auto-locks while the app is open.
- Wire audio: mic is PCM16 mono 16 kHz (`server/stt.py` `SAMPLE_RATE`); voice is PCM16 mono 24 kHz
  (`server/voice.py` `SAMPLE_RATE`). Both travel as binary WebSocket frames on the existing `/ws`.
- One live session at a time (existing behaviour in `server/main.py`).
- Run from `app/`: `uv run pytest`, `npm --prefix web test`. Both stay green after every task.

## Review Focus

1. **The plushie hears itself:** the phone plays voice while the session is still `speaking`, so the echo gate covers
   it only if the server stops counting as speaking no earlier than the phone stops playing. Pinned by
   `test_player_paces_to_real_time` (Task 3).
2. **The phone drops mid-sentence:** sending voice to a closed socket must end the utterance as a `VoiceError`, and the
   session must return to `ready`, not hang. Pinned by `test_player_write_raises_when_send_fails` (Task 3).
3. **Tunnel restarted on pitch day:** the start screen must show the last URL, editable, and the server must accept the
   new origin from `ALLOWED_ORIGINS` (comma list, spaces tolerated). Pinned by `test_allowed_origins_env` (Task 1).
4. **iOS blocks sound until a tap:** the first voice line after load is silent unless audio is unlocked on the first
   touch. `unlockAudio()` on the first `pointerdown` (Task 4); checked on the phone in Task 4 step 6.
5. **Untrusted text into Cartesia:** `say` text over 200 characters, or `say` in translator mode, must be refused.
   Pinned by `test_say_rejected_outside_practice` and `test_say_text_capped` (Task 5).

---

### Task 1: Server and Vite accept the tunnel

**Files:**
- Modify: `app/server/main.py` (`origin_ok`)
- Modify: `app/web/vite.config.ts`, `app/web/package.json`
- Test: `app/tests/test_origin.py`

**Interfaces:**
- Produces: env var `ALLOWED_ORIGINS` (comma-separated exact origins, e.g. `https://abc.trycloudflare.com`);
  npm script `dev:tunnel`.

- [ ] **Step 1: Write the failing test** in `tests/test_origin.py`

```python
def test_allowed_origins_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://abc.trycloudflare.com, https://x.example")
    assert origin_ok("https://abc.trycloudflare.com")
    assert origin_ok("https://x.example")
    assert not origin_ok("https://other.trycloudflare.com")
    assert not origin_ok("http://abc.trycloudflare.com")  # exact origin, scheme included
    monkeypatch.delenv("ALLOWED_ORIGINS")
    assert not origin_ok("https://abc.trycloudflare.com")
```

- [ ] **Step 2: Run it, expect FAIL** — `uv run pytest tests/test_origin.py -v`
- [ ] **Step 3: Implement** in `origin_ok`: read `os.environ.get("ALLOWED_ORIGINS", "")` on every call, split on
  commas, strip; return True on an exact match, else the existing localhost rule. Update the docstring in one line.
- [ ] **Step 4: Run it, expect PASS** (all of `tests/test_origin.py`).
- [ ] **Step 5: Vite.** In `vite.config.ts` add `server.allowedHosts: [".trycloudflare.com"]`, and
  `server.hmr: process.env.TUNNEL ? false : undefined` (Vite's HMR socket cannot reach the phone through the tunnel,
  and its reconnect logic reloads the page). In `package.json` add `"dev:tunnel": "TUNNEL=1 vite"`.
- [ ] **Step 6: Verify by hand:** `brew install cloudflared`; `npm --prefix web run dev:tunnel`;
  `cloudflared tunnel --url http://localhost:5173` prints `https://….trycloudflare.com`;
  `ALLOWED_ORIGINS=<that url> uv run asl-server`. Opening `<url>/teach.html` in desktop Safari loads the course
  (vocab fetched, no "Can't reach the backend").
- [ ] **Step 7: Commit** — `git commit -m "feat(app): accept a tunnel origin for the phone"`

### Task 2: Capacitor iOS shell (GO/NO-GO gate)

**Files:**
- Create: `app/web/capacitor.config.ts`, `app/web/shell/index.html`, `app/web/ios/` (generated by Capacitor)
- Create: `app/web/src/phone.ts`, `app/web/src/phone.test.ts`
- Modify: `app/web/package.json` (devDeps), `app/web/ios/App/App/AppDelegate.swift`, `app/web/ios/App/App/Info.plist`
- Modify: `app/web/src/teach/main.ts`, `app/web/src/main.ts` (call `rememberPhone()` at boot)
- Modify: `app/README.md` (new "Run on iPhone" section)

**Interfaces:**
- Produces: `isPhone(): boolean` and `rememberPhone(): void` in `web/src/phone.ts`. `rememberPhone` stores
  `localStorage["lingual.phone"] = "1"` when the URL has `?phone=1`; `isPhone` returns whether that key is `"1"`.
  Storage is per origin, so a laptop browser on `localhost` is never a phone.

- [ ] **Step 1: Write the failing test** `src/phone.test.ts`

```ts
test("phone flag is remembered from ?phone=1", () => {
  localStorage.clear();
  history.replaceState(null, "", "/teach.html");
  rememberPhone(); expect(isPhone()).toBe(false);
  history.replaceState(null, "", "/teach.html?phone=1");
  rememberPhone(); history.replaceState(null, "", "/"); expect(isPhone()).toBe(true);
});
```

- [ ] **Step 2: Run, expect FAIL** — `npm --prefix web test -- phone`
- [ ] **Step 3: Implement** `src/phone.ts` (wrap `localStorage` in try/catch, false on error); call `rememberPhone()`
  first thing in `teach/main.ts` and `src/main.ts`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Capacitor.** In `app/web`: install `@capacitor/core`, `@capacitor/cli`, `@capacitor/ios` (same
  current major) as devDependencies; `npx cap init lingual com.lingual.plushie --web-dir shell`; `npx cap add ios`.
  `capacitor.config.ts`: `webDir: "shell"`, `server.allowNavigation: ["*.trycloudflare.com"]` (otherwise navigation
  opens Safari). Commit `ios/` (Capacitor's own `.gitignore` there excludes build output).
- [ ] **Step 6: Start screen** `shell/index.html`: one text field (prefilled from `localStorage["lingual.url"]`), an
  Open button. On Open: trim, require `https://`, strip a trailing `/`, save, then
  `location.href = url + "/teach.html?phone=1"`. Large touch targets; inline CSS and script; no other files.
- [ ] **Step 7: Native.** `Info.plist`: `NSCameraUsageDescription` = "The plushie watches your signing to check it.",
  `NSMicrophoneUsageDescription` = "The plushie listens so it can sign what you say.",
  `UISupportedInterfaceOrientations` = portrait only. `AppDelegate.swift` `didFinishLaunching`:
  `UIApplication.shared.isIdleTimerDisabled = true` (keeps the screen awake; no plugin).
- [ ] **Step 8: Run on the phone:** `npx cap sync ios && npx cap open ios`, select the iPhone 17 Pro, Run. Paste the
  Task 1 tunnel URL, Open.
- [ ] **Step 9: GO/NO-GO, on the phone standing upright, signer about 1 m back:**
  1. Teaching mode loads; camera prompt appears with the text above; screen does not lock after 2 minutes idle.
  2. Avatar plays a Unit 1 sign smoothly (no visible stutter).
  3. A perform exercise grades HELLO, THANK-YOU and NICE correctly at least 2 of 3 tries each.
  Record results in `docs/superpowers/plans/2026-10-16-two-way-asl-mvp-record.md`. **If any check fails, stop and
  report before Task 3.**
- [ ] **Step 10: README** "Run on iPhone": the three laptop commands from Task 1 step 6, then Xcode Run, paste URL.
  Note: free Apple ID installs expire after 7 days; stop the tunnel after the demo.
- [ ] **Step 11: Commit** — `git commit -m "feat(ios): Capacitor shell that opens the laptop app through a tunnel"`

### Task 3: Server sends voice to, and hears from, the phone

**Files:**
- Create: `app/server/phone_audio.py`
- Modify: `app/server/main.py` (`ws_endpoint`, `build_session`)
- Test: `app/tests/test_phone_audio.py`

**Interfaces:**
- Consumes: `MicTranscriber(mic=…)`, `Voice(client=…, voice_id=…, player_factory=…)`, `voice.SAMPLE_RATE`,
  `stt.SAMPLE_RATE`, `stt.CHUNK`.
- Produces:
  - `class ClientMic` — `push(b: bytes) -> None`; calling the instance returns an async iterator of `bytes`
    (usable as `MicTranscriber(mic=client_mic)`). Yields pushed chunks in order; when nothing arrives for 0.1 s
    yields `bytes(CHUNK * 2)` of silence so the Ink socket stays alive.
  - `class WsPlayer(send: Callable[[bytes], Awaitable[None]], loop: asyncio.AbstractEventLoop, ahead_s: float = 0.3,
    clock=time.monotonic, sleep=time.sleep)` — `write(b: bytes) -> None` (called from a worker thread, as `Voice`
    does), `close() -> None`. `write` sends through `asyncio.run_coroutine_threadsafe(send(b), loop).result()`, then
    sleeps while audio sent so far is more than `ahead_s` ahead of real time (24 kHz × 2 bytes per second).
  - `/ws?audio=client`: binary frames in → `ClientMic.push`; voice out as binary frames.

- [ ] **Step 1: Write the failing tests** `tests/test_phone_audio.py`

```python
async def test_client_mic_yields_pushed_then_silence():
    m = ClientMic(); m.push(b"\x01\x00" * 10)
    it = m().__aiter__()
    assert await it.__anext__() == b"\x01\x00" * 10
    assert await it.__anext__() == bytes(3200)

def test_player_paces_to_real_time():
    t = [0.0]; slept = []
    def sleep(s): slept.append(s); t[0] += s
    sent = []
    loop = asyncio.new_event_loop(); th = threading.Thread(target=loop.run_forever, daemon=True); th.start()
    async def send(b): sent.append(b)
    p = WsPlayer(send, loop, ahead_s=0.3, clock=lambda: t[0], sleep=sleep)
    for _ in range(3): p.write(bytes(48000))  # 1 s of 24 kHz PCM16 each
    assert len(sent) == 3
    assert t[0] >= 3.0 - 0.3 - 1e-6  # never more than ahead_s ahead of playback
    loop.call_soon_threadsafe(loop.stop)

def test_player_write_raises_when_send_fails():
    loop = asyncio.new_event_loop(); th = threading.Thread(target=loop.run_forever, daemon=True); th.start()
    async def send(b): raise ConnectionError("closed")
    p = WsPlayer(send, loop, sleep=lambda s: None)
    with pytest.raises(ConnectionError): p.write(bytes(4800))
    loop.call_soon_threadsafe(loop.stop)
```

(`Voice.speak` already turns any player exception into `VoiceError`, and `Session` already returns to `ready` on it;
no new session code.)

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_phone_audio.py -v`
- [ ] **Step 3: Implement** `server/phone_audio.py` (an `asyncio.Queue` and `asyncio.wait_for(q.get(), 0.1)` for the
  mic; the pacing loop above for the player).
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Wire `ws_endpoint`.** `client_audio = ws.query_params.get("audio") == "client"`. If so:
  `mic = ClientMic()`; voice = a per-connection `Voice(client=VOICE.client, voice_id=VOICE.voice_id,
  player_factory=lambda: WsPlayer(send_audio, loop))` when `VOICE` is a real `Voice` (else keep `VOICE`, the stub),
  started with `asyncio.create_task(voice.warm())`; `send_audio` holds the same `lock` as `send` and calls
  `ws.send_bytes`. Pass `mic=mic` to `MicTranscriber`. `build_session(send, voice=VOICE)` gains the `voice` parameter.
  Replace `ws.receive_text()` with `ws.receive()`: `"websocket.disconnect"` → leave the loop; `bytes` → `mic.push`
  when `client_audio` (ignored otherwise); `text` → the existing parse path. Without `?audio=client` nothing changes.
- [ ] **Step 6: Run the whole suite** — `uv run pytest` — expect PASS.
- [ ] **Step 7: Commit** — `git commit -m "feat(app): voice and mic over the socket for the phone"`

### Task 4: The phone plays voice and sends its mic

**Files:**
- Modify: `app/web/src/phone.ts`, `app/web/src/phone.test.ts`, `app/web/src/ws.ts`, `app/web/src/main.ts`

**Interfaces:**
- Consumes: `isPhone()` (Task 2); server `?audio=client` binary protocol (Task 3).
- Produces in `phone.ts`:
  - `toPcm16(f: Float32Array): Int16Array` — clamp to [-1, 1], scale by 32767, round.
  - `playPcm(buf: ArrayBuffer): void` — 24 kHz PCM16 mono; schedules each chunk right after the previous one on one
    shared `AudioContext` (start at `max(ctx.currentTime, next)`).
  - `startMic(send: (b: ArrayBuffer) => void): Promise<void>` — `getUserMedia({ audio: { echoCancellation: true,
    noiseSuppression: true } })` into an `AudioContext({ sampleRate: 16000 })`, a `ScriptProcessorNode` (buffer
    2048), each buffer → `toPcm16` → `send`.
  - `unlockAudio(): void` — resumes both contexts; registered once on the first `pointerdown`.
- `connect(onMsg, onClose?, opts?: { mic?: boolean })` in `ws.ts`: when `isPhone()`, opens `/ws?audio=client`, sets
  `binaryType = "arraybuffer"`, sends binary messages to `playPcm`, and when `opts.mic` calls `startMic` with a sender
  that writes binary only while the socket is open. Text messages unchanged.

- [ ] **Step 1: Write the failing test** in `phone.test.ts`

```ts
test("toPcm16 clamps and scales", () => {
  expect(Array.from(toPcm16(new Float32Array([0, 1, -1, 2, -2, 0.5])))).toEqual([0, 32767, -32767, 32767, -32767, 16384]);
});
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** the four functions and the `connect` change. `src/main.ts` (translator) calls
  `connect(…, …, { mic: true })`; `teach/signer.ts` keeps `connect` without `mic` (practice ignores speech).
- [ ] **Step 4: Run** `npm --prefix web test` — expect PASS.
- [ ] **Step 5: Laptop unchanged:** `localhost:5173` translator still hears the laptop mic and speaks on the laptop
  speaker (the socket URL has no `audio=client`).
- [ ] **Step 6: On the phone** (Task 2 setup): open the Translator from the teaching header. Sign THANK-YOU → the
  phone speaks the English; say "hello, nice to meet you" → the avatar signs it. After the phone finishes speaking,
  its own voice never shows up as a caption (Review Focus 1). The first spoken line after a fresh load is audible
  (Review Focus 4).
- [ ] **Step 7: Commit** — `git commit -m "feat(web): phone mic and voice playback"`

### Task 5: The plushie talks during lessons

**Files:**
- Modify: `app/server/protocol.py`, `app/server/session.py`, `app/web/src/ws.ts` (`ClientMsg`),
  `app/web/src/teach/signer.ts`, `app/web/src/teach/lessonView.ts`, `app/web/src/teach/main.ts`
- Test: `app/tests/test_session.py`, `app/tests/test_protocol.py`

**Interfaces:**
- Produces:
  - `class SayMsg(BaseModel): type: Literal["say"] = "say"; text: str = Field(min_length=1, max_length=200)`, added
    to `ClientMsg`; TS `{ type: "say"; text: string }`.
  - `Session` on `SayMsg` while `self.practice`: starts a task that, under an `asyncio.Lock` (lines queue, never
    overlap), runs `async for _ in self.voice.speak(text, SAY_EMOTION)`; errors are logged, never raised. Does not
    change `self.state`. Ignored when not in practice. `SAY_EMOTION = {"label": "happy", "intensity": 0.4}`.
  - `practice.say(text: string): void` in `signer.ts` — opens the socket if needed (split `open()` so the socket can
    open without the camera) and sends `say`.
  - `LessonHost.say(text: string): void`; `teach/main.ts` passes `(t) => settings.sound && practice.say(t)`.

- [ ] **Step 1: Write the failing tests**

```python
async def test_say_speaks_in_practice(tmp_path):
    voice = RecordingVoice(); s = make_session(tmp_path, voice=voice)
    await s.on_client(PracticeMsg(on=True)); before = s.state
    await s.on_client(SayMsg(text="Nice!"))
    await asyncio.sleep(0.01)  # let the task run
    assert voice.said == [("Nice!", {"label": "happy", "intensity": 0.4})] and s.state == before

async def test_say_rejected_outside_practice(tmp_path):
    voice = RecordingVoice(); s = make_session(tmp_path, voice=voice)
    await s.on_client(SayMsg(text="hi")); await asyncio.sleep(0)
    assert voice.said == []

def test_say_text_capped():
    with pytest.raises(ValueError): parse_client(json.dumps({"type": "say", "text": "a" * 201}))
```

(`RecordingVoice.speak` appends `(text, emotion)` and yields `"done"`.)

- [ ] **Step 2: Run, expect FAIL** — `uv run pytest tests/test_session.py tests/test_protocol.py -k say -v`
- [ ] **Step 3: Implement** server side as in Interfaces.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: Lines in `lessonView.ts`**, one `LINES` constant, sent through `host.say`:
  lesson start "Hi! Let's learn some signs together."; a perform exercise mounts `Your turn! Sign ${english}.`;
  correct → the same `PRAISE` text shown on the card; wrong → "Almost! Let's try again."; lesson complete
  "You did it! Great job." The wording may be tuned; keep each line under 60 characters.
- [ ] **Step 6: Check** on the laptop (lines from the laptop speaker) and on the phone (from the phone speaker) through
  one Unit 1 lesson. `npm --prefix web test` and `uv run pytest` PASS.
- [ ] **Step 7: Commit** — `git commit -m "feat(teach): the plushie speaks during lessons"`

### Task 6: "Show me a sign"

**Files:**
- Create: `app/web/src/teach/showMe.ts`, `app/web/src/teach/showMe.test.ts`
- Modify: `app/web/src/teach/signer.ts` (`playTokens`), `app/web/src/teach/home.ts` (practice tab card,
  `HomeCtx.showMe`), `app/web/src/teach/main.ts`

**Interfaces:**
- Consumes: `VocabSign`, `fetchVocab()` (`signer.ts`); `SignToken` (`ws.ts`).
- Produces:
  - `lookupSign(word: string, vocab: VocabSign[]): SignToken | null` — trims and lower-cases; `null` for empty; a vocab
    entry whose `english` or `gloss` (case-insensitive) equals the word → `{ kind: "sign", gloss }`, preferring
    `avatar_ok`; otherwise `{ kind: "spell", word: <letters A-Z only, upper-cased> }`, `null` if no letters remain.
  - `playTokens(tokens: SignToken[], opts?)` in `signer.ts`; `playSigns` becomes a one-line call to it.
  - `HomeCtx.showMe(word: string): void`.

- [ ] **Step 1: Write the failing test** `showMe.test.ts`

```ts
const V = [{ gloss: "THANK-YOU", english: "thank you", avatar_ok: true }, { gloss: "DOG", english: "dog", avatar_ok: true }];
test("matches english or gloss", () => {
  expect(lookupSign(" Thank You ", V)).toEqual({ kind: "sign", gloss: "THANK-YOU" });
  expect(lookupSign("dog", V)).toEqual({ kind: "sign", gloss: "DOG" });
});
test("unknown words are fingerspelled, junk is nothing", () => {
  expect(lookupSign("Kev1n!", V)).toEqual({ kind: "spell", word: "KEVN" });
  expect(lookupSign("   ", V)).toBeNull();
  expect(lookupSign("123", V)).toBeNull();
});
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `lookupSign`, `playTokens`.
- [ ] **Step 4: Run, expect PASS.**
- [ ] **Step 5: UI.** A "Show me a sign" card at the top of the practice tab: a text input with a `<datalist>` of the
  vocab's English words, a Show button (Enter also submits), and an `.avatar-slot` holding `avatarCanvas`. `main.ts`
  keeps the full avatar-ok `VocabSign[]` from `boot()` and implements `showMe` with `lookupSign` + `playTokens`.
  When it fingerspells, `practice.say` says "Here's how to spell it." (lesson lines rule: only when sound is on).
- [ ] **Step 6: Check** in the browser and on the phone: "dog", "thank you", and an unknown name.
- [ ] **Step 7: Commit** — `git commit -m "feat(teach): show me a sign"`

### Task 7: Portrait pass and demo run-through

**Files:**
- Modify: `app/web/src/style.css` (translator), `app/web/src/teach/teach.css` as needed

- [ ] **Step 1:** In the browser pane at 402×874 (iPhone 17 Pro portrait), screenshot the teaching home, a perform
  exercise, the show-me card, and the translator. Fix anything that overflows horizontally, is cut off, or has touch
  targets under 44 px, using media queries only (desktop layout unchanged).
- [ ] **Step 2:** `npm --prefix web test`, `uv run pytest` PASS; desktop screenshots of the same screens unchanged.
- [ ] **Step 3:** Full run on the phone in the plush: a Unit 1 lesson with spoken lines, show-me, sign to speech,
  speech to sign. Note any failure in the MVP record.
- [ ] **Step 4: Commit** — `git commit -m "feat(web): portrait layout for the phone"`
