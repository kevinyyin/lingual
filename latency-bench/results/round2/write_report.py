"""Assemble results/round2/REPORT-ROUND2.md (run with: uv run python results/round2/write_report.py)."""
from bench.round2.report import assemble

INTRO = """# Ear-to-Ear Lag, Round 2: Six Speed-Up Experiments (Spanish → English)

*Lingual team · benchmark harness and analysis prepared with Claude · 13–15 October 2026 · internal, preliminary*

## Abstract

Round 1 found OpenAI's realtime translator ahead (median 2.74 s from end of speech to end of hearing, sentences)
of a phrase-by-phrase cascade (3.77 s) and a turn-end cascade (5.08 s), plus three problems: OpenAI stalls, cascade
output that runs longer than the source, and cascade commit-policy waits. Round 2 ran six experiments
(1,435 new runs) testing speed-ups. Catch-up playback (playing queued audio at 1.25–1.4×) is the largest win for the
cascade (sentences 3.77 → 2.98 / 2.64 s, monologue 11.6 → 6.5 / 4.3 s) and costs OpenAI almost nothing. OpenAI
stalls (13 of 419 stall-study runs) were not removed by append size or trailing audio; faint background chatter
made OpenAI invent speech. Flushing OpenAI on each pause cut its worst case (26 s → 8–16 s) but not its median, and
reached 2.3 s only together with catch-up. Gemini Live Translate had no stalls in 65 speech runs and higher chrF, but
started speaking later (3.0 vs 1.7 s). A single synthesis context with per-fragment flush and a concise prompt was the
best cascade (sentences 3.50 s, monologue 9.1 s, output/source 1.06); pipelining fragments changed nothing. No
configuration met the decision-gate targets at normal playback speed.

# Introduction

Round 2 asked which concrete changes close the gap to a conversational lag of about 2.2–2.3 s. Six experiments:

1. **Catch-up playback** (offline re-score of round 1): play queued translated audio faster when it backs up.
2. **OpenAI stall study:** how often OpenAI holds a translation, and whether append size or trailing audio matters.
3. **Flush on pause / watchdog:** force OpenAI to emit by closing the session at each pause (session rotation).
4. **Gemini Live Translate** baseline against an OpenAI control in the same batch.
5. **Cascade output pace:** one synthesis context per turn, buffer and flush settings, a concise prompt, adaptive speed.
6. **Translation host and pipelined fragments** (Groq was blocked: no key).

# Methods (what changed since round 1)

- **Same harness, same rules:** 1× real-time pacing on an absolute schedule, one clock, every number from recorded
  timestamps, randomized order with a fixed seed, TCP/TLS probes per batch, failures recorded not dropped. Latency
  medians come only from concurrency-1 batches with 1 s between runs; the stall-count phases 2a and 2b ran at
  concurrency 4 and are labelled. Round-1 raw data, audio and analysis were not modified (checksums verified; the
  prepared-clip index only gained new entries).
- **New clips (synthetic, Cartesia Spanish voices):** `two_utterances` (1.5 s gap), `hesitation` (a 500 ms pause
  inserted after "Barcelona", found with Whisper word timings), `three_turns` (four short sentences with 0.8, 1.2 and
  2.0 s pauses: the brief listed three pause lengths for "three turns", read as three turn boundaries). Each records
  its true sentence spans.
- **Per-utterance scoring:** source speech segmented with silero-vad (min silence 300 ms); each audible output chunk
  is assigned to the utterance whose offset most recently precedes its arrival.
- **Stall:** start lag over 5 s, or no audible output before the harness began closing the session. Same-language
  clips are excluded (silence is the desired behaviour there).
- **Playback with parallel sessions (experiment 3):** during a rotation two OpenAI sessions stream at once, each with
  continuous silent filler. Each session is simulated on its own playback track and the tracks are mixed; the
  literal arrival-order concatenation is kept in the data but overstates lag by 10 s or more.
- **Statistics:** medians with 95% bootstrap CIs (5,000 resamples of runs; clips are few, so CIs understate
  uncertainty; independent-clip counts are given beside every interval), Wilson intervals for stall rates, per-clip
  paired differences against round 1 or the same-batch control.
- **Environment:** one Windows laptop, home Wi-Fi; Philadelphia for the first half of 2a, Pittsburgh for everything
  after (recorded per run); on battery until 15 Oct, then AC. Windows Modern Standby killed 4 runs once (tossed and
  redone); the harness now asks Windows to stay awake.
"""

DISCUSSION = """# Discussion

**Principal findings.** (1) How audio is *played* matters as much as how fast it is *produced*: every cascade
variant delivers its last audio 1–1.5 s after the speaker stops but makes the listener wait 3.5 s or more, because
synthesis arrives in bursts; 1.25–1.4× catch-up recovers most of that at a small intelligibility cost. (2) OpenAI's
tail is a property of the model, not of what we send: neither 200 ms appends nor any trailing audio removed stalls,
and forcing flushes trades a shorter worst case for slightly longer typical lag unless the player catches up.
(3) Gemini is slower to start but steadier: no stalls, tighter p90, higher chrF; OpenAI is faster when it works.
(4) The cascade's remaining gap is its commit policy and speech length, not the LLM: fragment translations take about
0.45 s to first token and pipelining them changed nothing.

**What we would decide now.** Build catch-up playback into any player we ship. If choosing a single provider today,
the choice is between OpenAI's faster median with a ~3% stall and hallucination risk, and Gemini's ~0.5 s slower but
stall-free output; a watchdog or pause flush on OpenAI caps the damage but needs catch-up to pay off. The cascade is
not competitive on latency yet.

**Limitations.** All speech synthetic; one language pair; few independent clips (3 per sentence category, 1 monologue,
1 of each multi-utterance clip); two locations and power states; chrF against LLM-drafted references (and Whisper
hallucinates on near-silence, e.g. "Thank you for watching"); catch-up evaluated offline (listening check by ASR, not by
people); Groq not tested; phase 2a/2b at concurrency 4.

**Next steps.** (a) Real recordings and a second language pair before any external claim. (b) Ship-ready catch-up
player with a human listening test at 1.25× and 1.4×. (c) OpenAI with watchdog + catch-up vs Gemini in one batch,
20 repetitions, to settle the provider choice on tails. (d) Cascade: experiments 7–8 (commit rules with `stable_ms`,
anticipation) are prepared and unit-tested. (e) Groq once a key is available.
"""

if __name__ == "__main__":
    p = assemble(INTRO, DISCUSSION)
    print("wrote", p)
