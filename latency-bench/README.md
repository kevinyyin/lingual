# latency-bench

An isolated measurement instrument for live speech-to-speech translation. Two people who
speak different languages talk naturally, overlapping, through earbuds; this harness answers
one question: **how far behind the speaker does each system's translated audio arrive, and
where does that time go?** It streams recorded speech to each provider at exactly 1× real
time, timestamps every chunk sent and every event received on one monotonic clock, and
computes every number in the report from those timestamps. Nothing is hard-coded or
vendor-claimed. It shares no code with anything outside this folder.

## Setup

```bash
cd latency-bench
uv sync                      # Python 3.11+, everything installs into .venv
cp .env.example .env         # then fill in the keys you need
```

| Provider (`--providers`) | Keys in `.env` | Notes |
|---|---|---|
| `mock` | none | fixed-delay echo, used by `selftest` |
| `openai` | `OPENAI_API_KEY` | `gpt-realtime-translate` over WebSocket |
| `gemini` | `GEMINI_API_KEY` | `gemini-3.5-live-translate-preview`, Live API WebSocket |
| `cascade` | `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, plus the translate LLM's key (`OPENAI_API_KEY` by default) | Deepgram Flux → LLM → Cartesia Sonic |
| `hibiki` | none | local, fr→en only; needs CUDA or Apple Metal, otherwise skipped cleanly |

Analysis extras: EVS word alignment and the optional LLM judge use the `aligner` / `judge`
LLM roles in `config/bench.yaml` (OpenAI by default). `bench prepare` synthesizes missing
clips with Cartesia (default), OpenAI TTS (`--tts openai`), or local OS voices (`--tts os`).
`faster-whisper` and `silero-vad` download their models on first use.

Only `latency-bench/.env` is read; keys never leave this folder and are never written to
results (provider configs in run headers are filtered for anything named `*key*`).

## Commands, in order

```bash
uv run bench selftest                       # 1. prove the harness measures a known 300 ms delay
uv run bench prepare                        # 2. synthesize missing clips, build derived clips, pad, VAD
uv run python scripts/probe_openai.py       # 3. (recommended) confirm live event shapes before trusting adapters
uv run python scripts/probe_gemini.py
uv run bench run --providers openai,gemini,cascade --pairs es-en --categories all --runs 5   # 4.
uv run bench analyze                        # 5. per-run metrics -> results/analysis/runs.jsonl, results/summary.csv
uv run bench report                         # 6. results/report.html (self-contained)
uv run pytest                               # unit + protocol tests (offline)
```

Useful options: `bench run --pairs all`, `--categories short,medium`, `--clips id1,id2`,
`--real` / `--synthetic`, `--set cascade.translate_mode=incremental` (override any provider
config key for one batch; recorded in each run header and shown as `cascade[incremental]`); `bench prepare --pairs es-en --tts openai`;
`bench analyze --no-evs` (no LLM calls), `--whisper-model tiny`, `--batch <id>`;
`bench report --batch <id>`.

**Run real benchmarks on AC power with a high-performance power plan and heavy apps
closed.** Disconnect VPNs unless you mean to measure through them. Power source, power
plan, Wi-Fi vs wired and VPN adapters are recorded in every run and shown in the report.

## What a run does

1. The prepared clip (speech padded with **1.0 s** of room tone before and **6.0 s** after —
   low-level pink noise, never digital zero) is resampled to the provider's input rate and
   cut into **20 ms** PCM16 chunks.
2. `t_connect_start` → `provider.connect()` → `t_session_ready`. Setup time is reported on
   its own, never mixed into translation lag.
3. Chunks are sent at exactly 1× real time on an absolute schedule (chunk *i* due at
   *t₀ + i·20 ms*), so scheduling error never accumulates. The actual send time of every
   chunk is recorded.
4. After the last chunk the provider is closed gracefully and drained (`session.close` →
   `session.closed`; `audioStreamEnd` + idle drain; `ForceEndTurn`/`CloseStream` + TTS drain).
5. `results/raw/<run_id>.jsonl` gets the header (clip, provider config, machine, network,
   VAD timing), the connect record, every send time, every event with its `perf_counter_ns`
   receive time, and a footer (`ok` / `failed` / `unsupported`). Output audio is written to
   `results/audio/<run_id>.wav` in arrival order with per-event offsets, so the stream can
   be reconstructed exactly as it arrived.

Order of (provider, clip, repetition) is shuffled with a fixed seed. Before each batch the
harness measures TCP-connect and TLS-handshake time to every provider host. Concurrency
defaults to 1. A failed run is recorded and shown as a failure, never dropped.

## Metric definitions (plain language)

* **Speech onset / offset**: where silero-vad says the speaker starts and stops, mapped to
  the moment the chunk containing that sample was sent.
* **Audible output**: output samples louder than `output_gate_dbfs` (−45 dBFS). Silent
  frames some providers stream do not count as "speaking".
* **Session setup**: from opening the connection until the provider says it is ready.
* **Start lag**: first audible translated audio arrives − speaker starts.
* **End lag (arrival)**: last audible translated audio arrives − speaker stops.
* **End lag (playback)**: imagine a speaker playing each chunk as soon as it arrives, but
  never two at once (late chunks queue). When the last translated word finishes playing −
  speaker stops. The best proxy for "how long after they stop talking do I finish hearing it".
* **First-byte lag**: first output audio of any kind − speaker starts (for reference).
* **EVS (ear-voice span), approximate**: for each source content word, how much later its
  translation is heard. Word times come from faster-whisper on both sides (source mapped to
  send times, output mapped to simulated playback times); an LLM pairs source words with
  target words (cached in `results/cache/align/`). Reported as mean and median per run.
* **Cascade translate modes**: `on_turn_end` translates each finished turn (started
  speculatively on Flux's EagerEndOfTurn and released at EndOfTurn if the transcript is
  unchanged). `incremental` is simultaneous-interpretation style: once a transcript prefix
  is stable across `stable_updates` partial updates, it commits fragments of at least
  `min_segment_words` (holding back the newest `holdback_words`) while the speaker is still
  talking; the remaining tail is translated at end of turn, also speculatively from
  EagerEndOfTurn. Each fragment's translation sees the previous source and translation.
* **Cascade stage breakdown** (for the segment that carries the last audible output):
  STT finalization (speech end → Flux EndOfTurn), translate wait (EndOfTurn → first text
  sent to TTS), TTS first byte, TTS remaining stream. These four add up to end lag
  (arrival). Also LLM first-token and completion time, TTS first-byte and completion time.
* **Behavior flags** for `same_language`, `silence`, `overlap`: did it produce output, how
  much, and what it said (provider transcript and Whisper transcript), with audio links.
* **Quality sanity**: chrF of the Whisper transcript of the output audio against the
  reference translation, plus an optional 1–5 LLM adequacy judge. It exists only to catch
  "fast but wrong" and is never blended with latency.

Aggregates per provider × pair × category × audio source (real / synthetic): median, p90,
min, max, count for every latency metric; failure rate; median chrF → `results/summary.csv`.

## Corpus

`corpus/manifest.yaml` lists 46 base clips across es→en, en→es, fr→en, de→en, ja→en and
zh→en, plus 42 derived clips. Categories: `short` (1–3 s), `medium` (5–8 s), `long`
(20–30 s), `verb_final` (German subordinate clauses, Japanese SOV), `noisy` (medium clips
under café/street noise at 10 dB SNR), `overlap` (two speakers in different languages on
one mic with 1–2 s overlap), `same_language` (target-language speech into the session),
`silence` (10 s of room tone). `verb_final` has no Spanish/French/Chinese clips: those
languages are not verb-final.

Real human recordings are preferred; see `corpus/audio/README.md` to drop them in. Any clip
`prepare` synthesizes is flagged `synthetic` and broken out separately in every table. Put
real café/street ambience in `corpus/noise/` (see its README); without it, noisy clips use
six-talker babble built from the corpus. Reference translations were drafted by an LLM;
have a fluent speaker review them before relying on chrF.

## Self-test

`bench selftest` streams short and medium clips (corpus clips if prepared, otherwise offline
OS-voice clips) through the mock provider at 300 ms with 0 jitter, then ±20 ms jitter. It
fails (exit 1) unless:

* start lag and end lag (arrival) are within ±10 ms of 300 ms (zero jitter; ± jitter bound otherwise);
* end lag (playback) equals an independent re-simulation of the playback rule, and with zero
  jitter equals 300 + 20 ms (one chunk of playback) within ±10 ms plus any observed mock
  emission lateness (a late chunk stays queued in playback, exactly as on a real speaker);
* pacing shows < 2 ms systematic drift and p99 send lateness ≤ 10 ms. A single chunk sent
  more than 20 ms late is printed as a warning, not a failure: every lag is measured from
  the recorded actual send time, so it cannot corrupt numbers.

On the dev laptop (Windows 11, on battery) the zero-jitter start lag measured 300.0–301.7 ms.
The mock "speaks" a tone during the VAD speech region in selftest so the output onset is
unambiguous; its default `echo` mode returns the input audio.

### Timing on Windows (why the clock code looks the way it does)

Measured while building this: asyncio's loop timers on CPython 3.11/Windows follow the
15.6 ms system tick (`time.monotonic` is `GetTickCount64`), and Windows may ignore
`timeBeginPeriod(1)` for background processes, so plain `asyncio.sleep` pacing was up to
43 ms late. The harness therefore (a) runs on an event loop whose clock is
`perf_counter`, (b) does coarse waits with `time.sleep` in a timer thread (a high-resolution
waitable timer on Windows), finishing with a 0.3 ms yield-spin, (c) shortens the GIL switch
interval to 0.5 ms, (d) pauses the garbage collector while a stream is live, and (e) raises
process priority to above-normal. Result: p99 send lateness ≈ 1–3 ms, drift ≈ 0.

## Optional hardware test (earbud delay)

```bash
uv sync --extra hw
uv run python scripts/hw_bt_latency.py --list
uv run python scripts/hw_bt_latency.py --out <out id> --in <mic id> --label "wired reference"
uv run python scripts/hw_bt_latency.py --out <out id> --in <mic id> --label "Bluetooth: <model>"
```

It plays a sharp click, records it through a mic held against the earbud, cross-correlates
to find the delay, repeats 20 times and reports median / p90 (appended to
`results/hw_latency.jsonl`). The number includes OS output and input buffering, so measure
a **wired** reference first (wired earbuds, same mic, same settings), then the **Bluetooth**
earbuds; the difference is what Bluetooth adds. Add it to a provider's end lag (playback) for
a realistic end-to-end figure.

## Where the docs differ from the original brief (docs were followed)

Docs read 2026-10-13. Adapters log every unknown event type as a `stage_mark`, so API
changes show up in raw data rather than being silently dropped.

* **OpenAI**: the engine consumes **200 ms** frames; 20 ms appends are buffered
  server-side (we keep 20 ms to mirror a live mic). Source transcripts
  (`session.input_transcript.delta`) only arrive when `audio.input.transcription.model` is
  set (`gpt-realtime-whisper` in config). The docs require an `OpenAI-Safety-Identifier`
  header (we send a hash). `session.created` arrives first; we treat `session.updated` as
  ready.
* **Gemini**: the model id `gemini-3.5-live-translate-preview` is confirmed. Input is 16 kHz
  PCM16, output 24 kHz. The docs recommend 100 ms chunks (we send 20 ms). There is no
  source-language field (auto-detected). `echoTargetLanguage` defaults to `false` (stays
  silent on target-language speech). Chinese is `zh-Hans`. Translation has no single
  terminal event, so after `audioStreamEnd` we drain until 3 s without output.
* **Deepgram Flux**: needs `/v2/listen`; `flux-general-multi` supports 10 languages
  (en, es, fr, de, hi, ru, pt, ja, it, nl), **not Chinese**, so the cascade records zh→en as
  `unsupported`. Flux recommends 80 ms chunks. `CloseStream` does not finalize a turn, so
  `finish()` sends `ForceEndTurn` first if a turn is open. Both pair languages are sent as
  `language_hint`, as a shared-mic product would.
* **Cartesia**: WebSocket needs `cartesia_version` (`2026-08-14`). Managed buffering
  defaults to 3000 ms; config uses 1000 ms (`max_buffer_delay_ms`). Each translated segment
  gets its own context so audio is attributable; a client-side sequencer keeps playback order.
* **Hibiki**: the official code is file-based (`moshi.run_inference`); a streaming adapter
  would drive moshi's frame generator. It was not written because this machine has no CUDA
  or Metal and the moshi streaming API could not be verified; the provider skips cleanly
  and the report says why.

## Known limitations

* **Synthetic audio bias**: TTS speech is clean, evenly paced and crisply bounded; it
  can flatter every provider. Results are split real vs synthetic; prefer real recordings.
* **Approximate EVS**: depends on Whisper word boundaries (weak for ja/zh, which are not
  space-delimited) and on an LLM alignment.
* **Onset/offset asymmetry**: source boundaries come from VAD, output boundaries from an
  energy gate. VAD offsets include quiet trailing speech that a gate would not, which can
  shift end lags by a few tens of ms in either direction, equally for all providers.
* **Mic capture not included**: lags start when the chunk containing the event was sent.
  A live 20 ms mic buffer adds up to 20 ms; earbud output delay is measured separately.
* **Network variance**: results reflect the machine, network, region and time of day they
  were run on. TCP/TLS probes and machine metadata are in every report so runs can be compared.
* **Probes not yet run**: the adapters follow current docs and pass offline protocol tests
  against local fake servers (`tests/test_provider_protocols.py`), but the live probes need
  API keys. Run `scripts/probe_*.py` before trusting the first real batch.

## TODO: anticipation mode for the cascade

Add a fourth `translate_mode: anticipation`: an LLM predicts the rest of the sentence from
the stable partial transcript and commits translated speech early, correcting when the
prediction diverges. The plug-in point is the `TranslateStrategy` interface in
`bench/providers/cascade.py` (`on_update`, `on_eager_eot`, `on_turn_resumed`,
`on_end_of_turn`, using `host.start_segment()` / `cancel_segment()`); the `Anticipation`
class there is a stub that raises `NotImplementedError`. No runner changes are needed.

## Layout

```
config/bench.yaml        providers, pairs, padding, LLM roles, analysis settings
corpus/manifest.yaml     clips, derived clips, reference translations
bench/clock.py           perf_counter loop, timer-thread sleeps, paced sender
bench/runner.py          randomized batches, connect/send/drain, raw JSONL + wav
bench/providers/         mock, openai_translate, gemini_translate, cascade, hibiki_local
bench/analysis/          metrics (all latency math), align (Whisper + LLM aligner), quality, pipeline
bench/report/build.py    results/report.html
scripts/                 probe_openai.py, probe_gemini.py, hw_bt_latency.py
tests/                   clock, metrics, mock round trip, provider protocols (fake servers)
```
