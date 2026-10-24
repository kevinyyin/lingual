# Ear-to-Ear Lag, Round 2: Six Speed-Up Experiments (Spanish → English)

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

# Results

## Experiment 1 · Catch-up playback (offline re-score of round 1)

Every successful round-1 run (209) was replayed through the playback simulator under each policy. Recorded arrivals are unchanged; only the simulated listener speeds up when queued audio exceeds the threshold (and returns to 1x below half of it). `catchup_end` allows speed-up only after a source utterance offset. Values: median end lag at playback (ms); in parentheses the median per-run change vs `none`. n = runs / independent clips.

| System · category | n | none | catchup@1.15x/300ms | catchup@1.15x/500ms | catchup@1.25x/300ms | catchup@1.25x/500ms | catchup@1.4x/300ms | catchup@1.4x/500ms | catchup_end@1.25x/500ms |
|---|---|---|---|---|---|---|---|---|---|
| cascade[incremental] · short | 15/3 | 2,821 | 2,588 (-194) | 2,609 (-184) | 2,443 (-298) | 2,475 (-282) | 2,268 (-426) | 2,309 (-402) | 2,475 (-282) |
| cascade[incremental] · medium | 15/3 | 3,771 | 3,224 (-550) | 3,239 (-537) | 2,937 (-816) | 2,977 (-805) | 2,605 (-1,040) | 2,635 (-1,040) | 3,069 (-698) |
| cascade[incremental] · noisy | 15/3 | 3,606 | 3,152 (-532) | 3,165 (-507) | 2,852 (-795) | 2,856 (-755) | 2,515 (-1,017) | 2,560 (-984) | 2,964 (-657) |
| cascade[incremental] · long | 5/1 | 11,641 | 8,210 (-3,512) | 8,249 (-3,499) | 6,419 (-5,385) | 6,459 (-5,341) | 4,252 (-7,389) | 4,318 (-7,324) | 9,253 (-2,388) |
| cascade[incremental] · overlap | 5/1 | 9,747 | 8,036 (-1,718) | 8,059 (-1,680) | 7,140 (-2,594) | 7,176 (-2,558) | 6,069 (-3,665) | 6,155 (-3,592) | 7,838 (-1,920) |
| cascade[on_turn_end] · short | 15/3 | 2,455 | 2,250 (-184) | 2,264 (-179) | 2,141 (-282) | 2,161 (-275) | 2,007 (-403) | 2,035 (-392) | 2,161 (-275) |
| cascade[on_turn_end] · medium | 15/3 | 5,084 | 4,584 (-482) | 4,597 (-472) | 4,315 (-739) | 4,334 (-723) | 3,988 (-1,056) | 4,012 (-1,033) | 4,334 (-723) |
| cascade[on_turn_end] · noisy | 15/3 | 4,822 | 4,347 (-484) | 4,359 (-469) | 4,093 (-742) | 4,112 (-720) | 3,781 (-1,061) | 3,808 (-1,028) | 4,112 (-720) |
| cascade[on_turn_end] · long | 5/1 | 20,533 | 18,055 (-2,446) | 18,068 (-2,433) | 16,733 (-3,750) | 16,753 (-3,730) | 15,104 (-5,358) | 15,133 (-5,329) | 16,753 (-3,730) |
| cascade[on_turn_end] · overlap | 5/1 | 7,561 | 6,732 (-530) | 6,745 (-514) | 6,290 (-813) | 6,310 (-789) | 5,745 (-1,161) | 5,774 (-1,127) | 6,310 (-789) |
| openai · short | 14/3 | 2,268 | 2,038 (-174) | 2,128 (-94) | 2,003 (-195) | 2,117 (-107) | 1,972 (-221) | 2,109 (-96) | 2,115 (-112) |
| openai · medium | 15/3 | 2,738 | 2,564 (-203) | 2,677 (-110) | 2,529 (-218) | 2,658 (-110) | 2,507 (-237) | 2,634 (-109) | 2,658 (-84) |
| openai · noisy | 15/3 | 2,606 | 2,428 (-166) | 2,530 (-76) | 2,426 (-181) | 2,546 (-93) | 2,424 (-200) | 2,520 (-91) | 2,546 (-64) |
| openai · long | 5/1 | 2,753 | 2,569 (-198) | 2,672 (-95) | 2,569 (-228) | 2,672 (-104) | 2,568 (-255) | 2,672 (-103) | 2,672 (-95) |
| openai · overlap | 5/1 | 722 | 548 (-177) | 680 (-83) | 532 (-197) | 679 (-124) | 512 (-210) | 679 (-97) | 674 (-63) |
| openai · same_language | 2/1 | 4,675 | 4,249 (-426) | 4,323 (-352) | 4,062 (-613) | 4,135 (-540) | 3,832 (-843) | 3,913 (-762) | 4,137 (-538) |

95% bootstrap CIs of the per-run change for `catchup@1.25x/500ms` (runs resampled; clips are few, so these understate uncertainty):

- cascade[incremental] · short: -282 ms [-375–-265] (n = 15 runs, 3 clips)
- cascade[incremental] · medium: -805 ms [-1,050–-671] (n = 15 runs, 3 clips)
- cascade[incremental] · noisy: -755 ms [-929–-643] (n = 15 runs, 3 clips)
- cascade[incremental] · long: -5,341 ms [-5,643–-5,182] (n = 5 runs, 1 clips)
- cascade[incremental] · overlap: -2,558 ms [-2,655–-2,164] (n = 5 runs, 1 clips)
- cascade[on_turn_end] · short: -275 ms [-291–-259] (n = 15 runs, 3 clips)
- cascade[on_turn_end] · medium: -723 ms [-755–-659] (n = 15 runs, 3 clips)
- cascade[on_turn_end] · noisy: -720 ms [-752–-646] (n = 15 runs, 3 clips)
- cascade[on_turn_end] · long: -3,730 ms [-3,781–-3,686] (n = 5 runs, 1 clips)
- cascade[on_turn_end] · overlap: -789 ms [-1,251–-758] (n = 5 runs, 1 clips)
- openai · short: -107 ms [-171–-56] (n = 14 runs, 3 clips)
- openai · medium: -110 ms [-203–-64] (n = 15 runs, 3 clips)
- openai · noisy: -93 ms [-145–-60] (n = 15 runs, 3 clips)
- openai · long: -104 ms [-178–0] (n = 5 runs, 1 clips)
- openai · overlap: -124 ms [-304–0] (n = 5 runs, 1 clips)
- openai · same_language: -540 ms — (n = 2 runs, 1 clips)

**Listening check** (pitch-preserving WSOLA, `audiotsm` 0.1.2): chrF of the Whisper transcript of the stretched rendering against the transcript of the unstretched output. 100 = identical words.

| System | Run | Policy | Content sped up (s) | chrF vs unstretched |
|---|---|---|---|---|
| cascade[incremental] | cascade_es_en_medium_1_r0 | catchup@1.25x/500ms | 5.35 | 100.0 |
| cascade[incremental] | cascade_es_en_medium_1_r0 | catchup@1.4x/300ms | 5.36 | 95.38 |
| cascade[incremental] | cascade_es_en_medium_2_r0 | catchup@1.25x/500ms | 3.94 | 100.0 |
| cascade[incremental] | cascade_es_en_medium_2_r0 | catchup@1.4x/300ms | 4.14 | 100.0 |
| cascade[incremental] | cascade_es_en_medium_3_r0 | catchup@1.25x/500ms | 4.11 | 100.0 |
| cascade[incremental] | cascade_es_en_medium_3_r0 | catchup@1.4x/300ms | 4.31 | 100.0 |
| cascade[on_turn_end] | cascade_es_en_medium_1_r0 | catchup@1.25x/500ms | 3.79 | 94.71 |
| cascade[on_turn_end] | cascade_es_en_medium_1_r0 | catchup@1.4x/300ms | 3.89 | 94.71 |
| cascade[on_turn_end] | cascade_es_en_medium_2_r0 | catchup@1.25x/500ms | 3.04 | 100.0 |
| cascade[on_turn_end] | cascade_es_en_medium_2_r0 | catchup@1.4x/300ms | 3.14 | 100.0 |
| cascade[on_turn_end] | cascade_es_en_medium_3_r0 | catchup@1.25x/500ms | 3.63 | 100.0 |
| cascade[on_turn_end] | cascade_es_en_medium_3_r0 | catchup@1.4x/300ms | 3.72 | 100.0 |
| openai | openai_es_en_medium_1_r0 | catchup@1.25x/500ms | 16.8 | 93.82 |
| openai | openai_es_en_medium_1_r0 | catchup@1.4x/300ms | 23.7 | 88.15 |
| openai | openai_es_en_medium_2_r0 | catchup@1.25x/500ms | 21.58 | 95.62 |
| openai | openai_es_en_medium_2_r0 | catchup@1.4x/300ms | 26.64 | 95.62 |
| openai | openai_es_en_medium_3_r0 | catchup@1.25x/500ms | 17.96 | 90.45 |
| openai | openai_es_en_medium_3_r0 | catchup@1.4x/300ms | 25.14 | 86.9 |

Rendered files: `results/round2/catchup_audio/`.

## Experiment 2 · OpenAI stall study

A **stall** is a run whose start lag exceeds 5 s, or that produced no audible output before the harness began closing the session (`session.close`, sent 30 s after speech ended). **Held until flush** = audible output exists, but only after `session.close`.

Both phases stream short and medium clips (6 clips) with a 30 s trailing pad. They run at concurrency 4 because they count events; their milliseconds are not used as latency medians.

### 2a · Append size (20 ms vs 200 ms), trailing room tone at −60 dBFS, 20 repetitions

| Condition | runs ok | failed | stalls | stall rate [Wilson 95%] | held until flush | start lag p10 / p50 / p90 / max (ms) |
|---|---|---|---|---|---|---|
| openai/a20-rt60 | 120 | 0 | 4 | 3.3% [1.3–8.3%] | 1 | 1,132 / 1,633 / 2,510 / 37,991 |
| openai/a20-rt60 · medium | 60 | 0 | 3 | 5.0% [1.7–13.7%] | 1 | 1,125 / 1,633 / 2,914 / 37,991 |
| openai/a20-rt60 · short | 60 | 0 | 1 | 1.7% [0.3–8.9%] | 0 | 1,154 / 1,645 / 2,380 / 9,811 |
| openai/a200-rt60 | 119 | 1 | 5 | 4.2% [1.8–9.5%] | 0 | 1,164 / 1,900 / 3,227 / 28,757 |
| openai/a200-rt60 · medium | 60 | 0 | 4 | 6.7% [2.6–15.9%] | 0 | 1,259 / 2,095 / 3,755 / 28,757 |
| openai/a200-rt60 · short | 59 | 1 | 1 | 1.7% [0.3–9.0%] | 0 | 1,131 / 1,651 / 2,343 / 26,561 |

The batch was interrupted and resumed after a move, so its runs come from two networks. Stall counts by location (the pooled table above is the primary result):

| Condition | runs ok | failed | stalls | stall rate [Wilson 95%] | held until flush | start lag p10 / p50 / p90 / max (ms) |
|---|---|---|---|---|---|---|
| openai/a20-rt60 @ Philadelphia (inferred: before location was recorded) | 65 | 0 | 1 | 1.5% [0.3–8.2%] | 0 | 1,142 / 1,682 / 2,395 / 5,582 |
| openai/a20-rt60 @ Pittsburgh, home Wi-Fi | 55 | 0 | 3 | 5.5% [1.9–14.9%] | 1 | 1,144 / 1,618 / 2,713 / 37,991 |
| openai/a200-rt60 @ Philadelphia (inferred: before location was recorded) | 54 | 1 | 1 | 1.9% [0.3–9.8%] | 0 | 1,192 / 1,989 / 3,586 / 20,709 |
| openai/a200-rt60 @ Pittsburgh, home Wi-Fi | 65 | 0 | 4 | 6.2% [2.4–14.8%] | 0 | 1,139 / 1,666 / 2,864 / 28,757 |

Lower stall rate: **openai/a20-rt60** (ties broken by median start lag).

Stalled runs:

| Condition | Clip | rep | location | start lag (ms) | held until flush | close sent (s after offset) | end lag arrival (ms) | kind | output transcript |
|---|---|---|---|---|---|---|---|---|---|
| openai/a20-rt60 | es_en_medium_1 | 17 | Pittsburgh,  | 37,991 | True | 30.0 | 34,745 | held until flush | Tomorrow morning, I have to take my daughter to the doctor before goin |
| openai/a20-rt60 | es_en_medium_1 | 18 | Pittsburgh,  | — | False | 30.0 | — | text without audible speech | Tomorrow morning I have to take my daughter to the doctor before going |
| openai/a20-rt60 | es_en_medium_3 | 16 | Philadelphia | 5,582 | False | 30.1 | 2,461 | late start | Restaurant Recomendados was full, so we had dinner at the hotel. |
| openai/a20-rt60 | es_en_short_1 | 7 | Pittsburgh,  | 9,811 | False | 30.0 | 8,874 | late start | Where is the train station? |
| openai/a200-rt60 | es_en_medium_2 | 9 | Pittsburgh,  | 28,757 | False | 30.1 | 28,574 | late start | Could you tell me if there's a pharmacy open near here at this hour? |
| openai/a200-rt60 | es_en_medium_2 | 19 | Pittsburgh,  | 5,341 | False | 30.1 | 2,640 | late start | We could say if there's a pharmacy open nearbyat this hour? |
| openai/a200-rt60 | es_en_medium_3 | 1 | Pittsburgh,  | 20,724 | False | 30.1 | 16,264 | late start | The restaurant you recommended was full, so we had dinner at the hotel |
| openai/a200-rt60 | es_en_medium_3 | 19 | Philadelphia | 20,709 | False | 30.1 | 16,301 | late start | The restaurant you recommended was full, so we had dinner at the hotel |
| openai/a200-rt60 | es_en_short_3 | 19 | Pittsburgh,  | 26,561 | False | 30.1 | 29,947 | late start | Many thanks for your help, |

### 2b · Trailing audio (room tone −60 dBFS, digital silence, noise bed −50 dBFS), 10 repetitions

| Condition | runs ok | failed | stalls | stall rate [Wilson 95%] | held until flush | start lag p10 / p50 / p90 / max (ms) |
|---|---|---|---|---|---|---|
| openai/a20-nb50 | 60 | 0 | 0 | 0.0% [0.0–6.0%] | 0 | 1,159 / 1,630 / 2,394 / 3,101 |
| openai/a20-nb50 · medium | 30 | 0 | 0 | 0.0% [0.0–11.4%] | 0 | 1,241 / 1,704 / 2,481 / 3,101 |
| openai/a20-nb50 · short | 30 | 0 | 0 | 0.0% [0.0–11.4%] | 0 | 1,156 / 1,581 / 2,019 / 2,804 |
| openai/a20-rt60 | 60 | 0 | 3 | 5.0% [1.7–13.7%] | 0 | 1,250 / 1,702 / 2,507 / 24,574 |
| openai/a20-rt60 · medium | 30 | 0 | 2 | 6.7% [1.8–21.3%] | 0 | 1,296 / 1,984 / 3,408 / 24,574 |
| openai/a20-rt60 · short | 30 | 0 | 1 | 3.3% [0.6–16.7%] | 0 | 1,210 / 1,640 / 2,437 / 2,509 |
| openai/a20-zero | 60 | 0 | 1 | 1.7% [0.3–8.9%] | 0 | 1,242 / 1,633 / 2,500 / 8,995 |
| openai/a20-zero · medium | 30 | 0 | 0 | 0.0% [0.0–11.4%] | 0 | 1,258 / 1,690 / 2,789 / 3,650 |
| openai/a20-zero · short | 30 | 0 | 1 | 3.3% [0.6–16.7%] | 0 | 1,214 / 1,631 / 2,094 / 8,995 |

Lowest stall rate: **openai/a20-nb50**.

Stalled runs:

| Condition | Clip | rep | location | start lag (ms) | held until flush | close sent (s after offset) | end lag arrival (ms) | kind | output transcript |
|---|---|---|---|---|---|---|---|---|---|
| openai/a20-rt60 | es_en_medium_1 | 9 | Pittsburgh,  | 8,141 | False | 30.0 | 4,959 | late start | Tomorrow morning, I have to take my daughter to the doctor before goin |
| openai/a20-rt60 | es_en_medium_3 | 0 | Pittsburgh,  | 24,574 | False | 30.1 | 25,420 | late start | The restaurant you recommended was full. So we had dinner at the hotel |
| openai/a20-rt60 | es_en_short_2 | 1 | Pittsburgh,  | — | False | 30.0 | — | late start |  |
| openai/a20-zero | es_en_short_1 | 7 | Pittsburgh,  | 8,995 | False | 30.0 | 7,525 | late start | Where is the train station? |

### 2c · Best stall condition at concurrency 1, all speech categories, 5 repetitions

| Condition · category | runs/clips | start lag med | end lag arrival med | end lag playback med [95% CI] | p90 | max | stalls (rate, Wilson 95%) | output/source | chrF |
|---|---|---|---|---|---|---|---|---|---|
| openai/best-a20-noise_bed_m50 · long | 5/1 | 2,081 | 2,646 | 2,751 [2,593–7,097] | 6,705 | 7,097 | 0/5 (0%, 0–43%) | 0.77 | 79 |
| openai/best-a20-noise_bed_m50 · medium | 15/3 | 2,011 | 2,558 | 2,875 [2,473–8,937] | 16,837 | 58,628 | 3/15 (20%, 7–45%) | 0.82 | 82 |
| openai/best-a20-noise_bed_m50 · noisy | 15/3 | 1,880 | 2,097 | 2,429 [1,896–2,815] | 19,803 | 61,164 | 0/15 (0%, 0–20%) | 0.89 | 82 |
| openai/best-a20-noise_bed_m50 · sentences | 30/6 | 1,937 | 2,213 | 2,576 [2,437–2,952] | 22,275 | 61,164 | 3/30 (10%, 3–26%) | 0.86 | 82 |
| openai/best-a20-noise_bed_m50 · short | 15/3 | 1,902 | 2,523 | 2,962 [2,351–29,649] | 32,432 | 33,017 | 2/15 (13%, 4–38%) | 1.18 | 74 |

Per-clip paired difference in end lag at playback vs round-1 OpenAI (median of condition − median of round 1, same clip; negative = faster):

- openai/best-a20-noise_bed_m50: median +56 ms over 10 clips; faster on 5/10. Per clip: long_1 -2, medium_1 +4,786, medium_2 -974, medium_3 +350, noisy_1 -161, noisy_2 -379, noisy_3 -332, short_1 +650, short_2 +29,280, short_3 +113

Stalled runs:

| Condition | Clip | rep | location | start lag (ms) | held until flush | close sent (s after offset) | end lag arrival (ms) | kind | output transcript |
|---|---|---|---|---|---|---|---|---|---|
| openai/best-a20-noise_bed_m50 | es_en_medium_1 | 0 | Pittsburgh,  | 8,779 | False | 30.0 | 5,623 | late start | Tomorrow morning, I have to take my daughter to the doctor before goin |
| openai/best-a20-noise_bed_m50 | es_en_medium_1 | 3 | Pittsburgh,  | 6,466 | False | 30.0 | 2,873 | late start | Tomorrow morningI have to take my daughter to the doctor before going  |
| openai/best-a20-noise_bed_m50 | es_en_medium_2 | 0 | Pittsburgh,  | 20,984 | False | 30.0 | 17,618 | late start | Could it be—tell me, if there's a pharmacy open nearby this late? |
| openai/best-a20-noise_bed_m50 | es_en_short_1 | 2 | Pittsburgh,  | 6,008 | False | 30.0 | 20,093 | late start | Where is the trainstation? |
| openai/best-a20-noise_bed_m50 | es_en_short_2 | 3 | Pittsburgh,  | 8,328 | False | 30.0 | 30,551 | late start | I’m going to ask you, tell me, how much this is, but if you were asked |

## Experiment 3 · Flush on pause and watchdog flush (OpenAI, session rotation)

Translation sessions accept no commit event, so a flush is a `session.close` on the active session (its remaining translation keeps draining into the output) while new source audio goes to a pre-opened, pre-configured standby session. A causal silero VAD on the sent stream (no look-ahead) triggers `on_pause` flushes; `watchdog` flushes only if no audible translation arrives within 1,200 ms of the VAD speech end (min silence 400 ms). A probe showed an idle configured session survives 90 s without audio and then translates normally, so standby sessions get no keepalive audio. Multi-utterance clips are scored per utterance (VAD, min silence 300 ms). During a rotation two sessions stream at once; each session is played on its own track and the tracks are mixed (concatenating two real-time streams in arrival order, as first specified, would double the audio per second and inflate lags by 10 s or more; that variant is kept in the data as `end_lag_playback_concat_ms`).

| Condition · category | runs/clips | start lag med | end lag arrival med | end lag playback med [95% CI] | p90 | max | stalls (rate, Wilson 95%) | output/source | chrF |
|---|---|---|---|---|---|---|---|---|---|
| openai/flush-none-a20 · hesitation | 5/1 | 1,669 | 2,721 | 2,942 [2,348–3,000] | 2,991 | 3,000 | 0/5 (0%, 0–43%) | 0.91 | 73 |
| openai/flush-none-a20 · long | 5/1 | 1,615 | 2,647 | 3,136 [2,483–3,964] | 3,660 | 3,964 | 0/5 (0%, 0–43%) | 0.73 | 79 |
| openai/flush-none-a20 · medium | 15/3 | 1,988 | 2,297 | 2,632 [2,394–2,904] | 3,397 | 4,107 | 0/15 (0%, 0–20%) | 0.87 | 89 |
| openai/flush-none-a20 · noisy | 15/3 | 2,035 | 2,177 | 2,533 [2,154–2,831] | 3,184 | 26,026 | 0/15 (0%, 0–20%) | 0.83 | 85 |
| openai/flush-none-a20 · sentences | 30/6 | 2,003 | 2,255 | 2,614 [2,380–2,823] | 3,229 | 26,026 | 0/30 (0%, 0–11%) | 0.84 | 87 |
| openai/flush-none-a20 · three_turns | 5/1 | 2,054 | 1,707 | 2,401 [2,000–2,657] | 2,618 | 2,657 | 0/5 (0%, 0–43%) | 0.53 | 59 |
| openai/flush-none-a20 · two_utterances | 5/1 | 1,786 | 1,424 | 2,932 [2,749–3,726] | 3,451 | 3,726 | 0/5 (0%, 0–43%) | 0.75 | 91 |
| openai/flush-pause400-a20 · hesitation | 5/1 | 1,992 | 1,120 | 2,763 [2,658–3,797] | 3,488 | 3,797 | 0/5 (0%, 0–43%) | 0.99 | 78 |
| openai/flush-pause400-a20 · long | 5/1 | 1,992 | 1,264 | 3,190 [2,089–5,893] | 4,906 | 5,893 | 0/5 (0%, 0–43%) | 0.76 | 81 |
| openai/flush-pause400-a20 · medium | 15/3 | 1,793 | 1,126 | 2,874 [2,365–3,301] | 4,334 | 13,598 | 0/15 (0%, 0–20%) | 0.85 | 84 |
| openai/flush-pause400-a20 · noisy | 15/3 | 1,693 | 1,064 | 2,578 [2,437–2,649] | 2,729 | 3,345 | 0/15 (0%, 0–20%) | 0.86 | 90 |
| openai/flush-pause400-a20 · sentences | 30/6 | 1,754 | 1,064 | 2,642 [2,475–2,804] | 3,741 | 13,598 | 0/30 (0%, 0–11%) | 0.86 | 86 |
| openai/flush-pause400-a20 · three_turns | 5/1 | 1,855 | 1,079 | 2,596 [2,387–3,213] | 3,049 | 3,213 | 0/5 (0%, 0–43%) | 0.60 | 55 |
| openai/flush-pause400-a20 · two_utterances | 5/1 | 1,446 | 930 | 3,106 [2,815–3,457] | 3,421 | 3,457 | 0/5 (0%, 0–43%) | 0.84 | 90 |
| openai/flush-pause600-a20 · hesitation | 5/1 | 1,607 | 1,185 | 2,714 [2,351–2,764] | 2,751 | 2,764 | 0/5 (0%, 0–43%) | 0.89 | 85 |
| openai/flush-pause600-a20 · long | 5/1 | 1,604 | 1,422 | 3,316 [2,560–4,353] | 4,126 | 4,353 | 0/5 (0%, 0–43%) | 0.75 | 77 |
| openai/flush-pause600-a20 · medium | 15/3 | 1,666 | 1,278 | 2,630 [2,122–3,110] | 10,262 | 15,555 | 0/15 (0%, 0–20%) | 0.87 | 83 |
| openai/flush-pause600-a20 · noisy | 15/3 | 1,923 | 1,181 | 2,470 [2,174–2,612] | 3,040 | 13,749 | 0/15 (0%, 0–20%) | 0.86 | 91 |
| openai/flush-pause600-a20 · sentences | 30/6 | 1,899 | 1,233 | 2,481 [2,221–2,740] | 4,660 | 15,555 | 0/30 (0%, 0–11%) | 0.87 | 85 |
| openai/flush-pause600-a20 · three_turns | 5/1 | 1,670 | 1,102 | 2,371 [2,300–2,617] | 2,597 | 2,617 | 0/5 (0%, 0–43%) | 0.58 | 55 |
| openai/flush-pause600-a20 · two_utterances | 5/1 | 1,833 | 1,262 | 3,414 [2,740–3,744] | 3,712 | 3,744 | 0/5 (0%, 0–43%) | 0.85 | 91 |
| openai/flush-pause800-a20 · hesitation | 5/1 | 1,986 | 1,298 | 2,555 [2,322–2,665] | 2,635 | 2,665 | 1/5 (20%, 4–62%) | 0.85 | 93 |
| openai/flush-pause800-a20 · long | 5/1 | 1,603 | 1,574 | 2,993 [2,503–3,811] | 3,645 | 3,811 | 0/5 (0%, 0–43%) | 0.76 | 78 |
| openai/flush-pause800-a20 · medium | 15/3 | 1,672 | 1,435 | 2,758 [2,591–2,881] | 3,591 | 7,909 | 0/15 (0%, 0–20%) | 0.88 | 83 |
| openai/flush-pause800-a20 · noisy | 15/3 | 1,676 | 1,389 | 2,640 [2,416–2,739] | 2,933 | 3,557 | 0/15 (0%, 0–20%) | 0.84 | 89 |
| openai/flush-pause800-a20 · sentences | 30/6 | 1,674 | 1,412 | 2,681 [2,596–2,771] | 3,382 | 7,909 | 0/30 (0%, 0–11%) | 0.87 | 84 |
| openai/flush-pause800-a20 · three_turns | 5/1 | 1,928 | 1,336 | 2,722 [2,613–3,148] | 3,098 | 3,148 | 0/5 (0%, 0–43%) | 0.58 | 64 |
| openai/flush-pause800-a20 · two_utterances | 5/1 | 1,838 | 1,430 | 3,439 [2,418–3,750] | 3,643 | 3,750 | 0/5 (0%, 0–43%) | 0.83 | 87 |
| openai/flush-watchdog1200-a20 · hesitation | 5/1 | 1,592 | 2,796 | 3,469 [2,327–14,312] | 10,210 | 14,312 | 0/5 (0%, 0–43%) | 1.00 | 73 |
| openai/flush-watchdog1200-a20 · long | 5/1 | 1,628 | 2,582 | 2,732 [2,266–3,339] | 3,300 | 3,339 | 0/5 (0%, 0–43%) | 0.76 | 79 |
| openai/flush-watchdog1200-a20 · medium | 15/3 | 1,923 | 2,217 | 2,560 [2,355–3,153] | 3,353 | 3,485 | 0/15 (0%, 0–20%) | 0.86 | 90 |
| openai/flush-watchdog1200-a20 · noisy | 15/3 | 1,734 | 1,726 | 2,310 [2,030–2,897] | 3,068 | 3,332 | 0/15 (0%, 0–20%) | 0.86 | 84 |
| openai/flush-watchdog1200-a20 · sentences | 30/6 | 1,882 | 2,125 | 2,512 [2,304–2,705] | 3,179 | 3,485 | 0/30 (0%, 0–11%) | 0.86 | 90 |
| openai/flush-watchdog1200-a20 · three_turns | 5/1 | 1,976 | 1,988 | 2,402 [2,305–3,359] | 3,177 | 3,359 | 0/5 (0%, 0–43%) | 0.55 | 64 |
| openai/flush-watchdog1200-a20 · two_utterances | 5/1 | 1,463 | 1,785 | 2,988 [2,765–3,638] | 3,422 | 3,638 | 0/5 (0%, 0–43%) | 0.79 | 86 |

**Pass criterion:** median end lag at playback ≤ 2,300 ms, no value over 4,000 ms, and chrF no more than 3 points below the no-flush control (values pooled over all clips; per utterance on multi-utterance clips).

| Condition | runs/clips | values | median end lag | max | over 4 s | with 1.25x catch-up: median / over 4 s | stalls | chrF | rotations/run | premature flushes (hesitation) | drain med (ms) | sessions/run | text-only sessions | audio min | pass (1x) | pass (catch-up) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| openai/flush-none-a20 | 50/10 | 75 | 2,630 | 26,026 | 5 | 2,517 / 2 | 0 | 80 | 0.0 | 0/5 runs | — | 1.0 | 0 | 11.2 | fail | fail |
| openai/flush-pause400-a20 | 50/10 | 74 | 2,733 | 13,598 | 7 | 2,297 / 5 | 0 | 81 | 1.0 | 5/5 runs | 4,724 | 3.0 | 1 | 11.2 | fail | fail |
| openai/flush-pause600-a20 | 50/10 | 75 | 2,688 | 15,555 | 9 | 2,294 / 7 | 0 | 83 | 1.0 | 0/5 runs | 4,754 | 3.0 | 1 | 11.2 | fail | fail |
| openai/flush-pause800-a20 | 50/10 | 73 | 2,749 | 7,909 | 4 | 2,436 / 3 | 1 | 83 | 1.0 | 0/5 runs | 4,589 | 3.0 | 1 | 11.2 | fail | fail |
| openai/flush-watchdog1200-a20 | 50/10 | 75 | 2,592 | 25,744 | 6 | 2,560 / 2 | 0 | 84 | 0.0 | 2/5 runs | 5,752 | 2.0 | 0 | 11.2 | fail | fail |

Premature flushes (a flush triggered inside a known sentence span) and accuracy:

- openai/flush-pause400-a20: 5 premature of 70 checked flushes; chrF median with a premature flush 77.84 (n = 5) vs without 84.865 (n = 40).
- openai/flush-pause600-a20: 0 premature of 65 checked flushes; chrF median with a premature flush — (n = 0) vs without 85.29 (n = 45).
- openai/flush-pause800-a20: 0 premature of 60 checked flushes; chrF median with a premature flush — (n = 0) vs without 85.85 (n = 45).
- openai/flush-watchdog1200-a20: 3 premature of 4 checked flushes; chrF median with a premature flush 71.29 (n = 3) vs without 56.89 (n = 1).

Per-clip paired difference vs round-1 OpenAI (same clips; medium, noisy, long):

- openai/flush-none-a20: median -131 ms over 7 clips (faster on 4).
- openai/flush-pause400-a20: median -6 ms over 7 clips (faster on 4).
- openai/flush-pause600-a20: median -315 ms over 7 clips (faster on 5).
- openai/flush-pause800-a20: median +44 ms over 7 clips (faster on 2).
- openai/flush-watchdog1200-a20: median -21 ms over 7 clips (faster on 4).

## Experiment 4 · Gemini Live Translate baseline

Gemini and an OpenAI control (round-1 configuration) ran in one randomized batch so network conditions match.

| Condition · category | runs/clips | start lag med | end lag arrival med | end lag playback med [95% CI] | p90 | max | stalls (rate, Wilson 95%) | output/source | chrF |
|---|---|---|---|---|---|---|---|---|---|
| gemini · hesitation | 5/1 | 2,998 | 2,906 | 3,246 [3,241–3,431] | 3,368 | 3,431 | 0/5 (0%, 0–43%) | 1.03 | 91 |
| gemini · long | 5/1 | 3,029 | 2,463 | 2,869 [2,613–3,032] | 3,021 | 3,032 | 0/5 (0%, 0–43%) | 0.90 | 83 |
| gemini · medium | 15/3 | 3,037 | 2,577 | 3,100 [2,789–3,168] | 3,228 | 3,331 | 0/15 (0%, 0–20%) | 0.92 | 90 |
| gemini · noisy | 15/3 | 2,974 | 2,397 | 2,835 [2,718–3,039] | 3,126 | 3,260 | 0/15 (0%, 0–20%) | 0.90 | 93 |
| gemini · same_language | 10/2 | 2,934 | 1,657 | 2,064 [1,563–2,385] | 2,413 | 2,478 | 0/10 (0%, 0–28%) | — | — |
| gemini · sentences | 30/6 | 3,007 | 2,435 | 2,870 [2,795–3,074] | 3,186 | 3,331 | 0/30 (0%, 0–11%) | 0.91 | 90 |
| gemini · short | 15/3 | 2,988 | 3,159 | 3,576 [3,405–3,611] | 3,736 | 3,772 | 0/15 (0%, 0–20%) | 1.10 | 82 |
| gemini · silence | 5/1 | — | — | — — | — | — | 0/5 (0%, 0–43%) | — | — |
| gemini · three_turns | 5/1 | 2,921 | 2,417 | 2,887 [2,746–3,056] | 3,021 | 3,056 | 0/5 (0%, 0–43%) | 0.69 | 66 |
| gemini · two_utterances | 5/1 | 3,028 | 2,245 | 3,411 [3,296–3,690] | 3,590 | 3,690 | 0/5 (0%, 0–43%) | 0.77 | 91 |
| openai/none-control · hesitation | 5/1 | 2,005 | 2,442 | 2,701 [2,673–2,888] | 2,827 | 2,888 | 0/5 (0%, 0–43%) | 0.87 | 81 |
| openai/none-control · long | 5/1 | 1,895 | 2,557 | 3,042 [2,800–19,298] | 12,889 | 19,298 | 1/5 (20%, 4–62%) | 0.76 | 76 |
| openai/none-control · medium | 15/3 | 1,709 | 2,135 | 2,340 [2,034–2,809] | 8,376 | 25,469 | 1/15 (7%, 1–30%) | 0.88 | 90 |
| openai/none-control · noisy | 15/3 | 1,679 | 2,237 | 2,619 [2,169–2,968] | 3,545 | 27,056 | 0/15 (0%, 0–20%) | 0.89 | 85 |
| openai/none-control · same_language | 10/2 | — | — | — — | — | — | 0/10 (0%, 0–28%) | — | — |
| openai/none-control · sentences | 30/6 | 1,694 | 2,159 | 2,488 [2,252–2,708] | 4,427 | 27,056 | 1/30 (3%, 1–17%) | 0.89 | 88 |
| openai/none-control · short | 15/3 | 1,347 | 1,706 | 2,225 [1,931–2,621] | 3,101 | 3,827 | 0/15 (0%, 0–20%) | 1.25 | 75 |
| openai/none-control · silence | 5/1 | — | — | — — | — | — | 0/5 (0%, 0–43%) | — | — |
| openai/none-control · three_turns | 5/1 | 1,743 | 1,886 | 2,542 [2,362–2,790] | 2,706 | 2,790 | 0/5 (0%, 0–43%) | 0.51 | 60 |
| openai/none-control · two_utterances | 5/1 | 1,792 | 1,361 | 2,962 [2,682–3,245] | 3,174 | 3,245 | 0/5 (0%, 0–43%) | 0.73 | 85 |

Per-clip paired difference, Gemini − OpenAI (end lag at playback, ms): hesitation_1 +443, long_1 -173, medium_1 +801, medium_2 +953, medium_3 -19, noisy_1 -211, noisy_2 +531, noisy_3 -4, short_1 +382, short_2 +1,832, short_3 +1,291, three_turns_1 +690, two_utterances_1 +726

**same_language:**
- gemini: output in 8/10 runs. ; ; ; ; ; ; ; 
- openai/none-control: output in 0/10 runs. 

**silence:**
- gemini: output in 0/5 runs. 
- openai/none-control: output in 0/5 runs.

## Experiment 5 · Cascade output pace and single speech context

Phrase-by-phrase cascade. *Baseline* = round-1 configuration (one Cartesia context per fragment, 1,000 ms buffer). *single* = one context per turn: fragments are sent with `continue: true`, the tail with `continue: false`; `mbd` = `max_buffer_delay_ms`; *flush* = Cartesia's default 3,000 ms buffer with `flush: true` after each fragment. *concise* = interpreter prompt asking for natural, concise spoken English (~10% fewer syllables, nothing dropped). *adaptive* = `generation_config.speed` 1.2 when more than 500 ms of audio is queued (1.3 above 1,000 ms), back to 1.0 under 250 ms.

### 5a

| Condition · category | runs/clips | start lag med | end lag arrival med | end lag playback med [95% CI] | p90 | max | stalls (rate, Wilson 95%) | output/source | chrF |
|---|---|---|---|---|---|---|---|---|---|
| cascade/baseline · hesitation | 5/1 | 3,276 | 1,763 | 3,705 [3,491–3,950] | 3,854 | 3,950 | 0/5 (0%, 0–43%) | 1.01 | 91 |
| cascade/baseline · long | 5/1 | 2,538 | 1,483 | 11,349 [10,919–11,962] | 11,805 | 11,962 | 0/5 (0%, 0–43%) | 1.21 | 80 |
| cascade/baseline · medium | 15/3 | 2,607 | 973 | 3,743 [3,338–3,899] | 4,031 | 4,091 | 0/15 (0%, 0–20%) | 1.02 | 81 |
| cascade/baseline · noisy | 15/3 | 2,592 | 978 | 3,762 [3,426–3,903] | 4,062 | 4,244 | 0/15 (0%, 0–20%) | 1.05 | 81 |
| cascade/baseline · sentences | 30/6 | 2,601 | 975 | 3,753 [3,440–3,857] | 4,038 | 4,244 | 0/30 (0%, 0–11%) | 1.04 | 81 |
| cascade/baseline · short | 15/3 | 2,168 | 1,212 | 2,810 [2,233–2,889] | 2,960 | 3,075 | 0/15 (0%, 0–20%) | 1.15 | 100 |
| cascade/baseline · three_turns | 5/1 | 3,629 | 1,394 | 4,416 [3,917–4,759] | 4,645 | 4,759 | 0/5 (0%, 0–43%) | 0.74 | 67 |
| cascade/baseline · two_utterances | 5/1 | 2,180 | 997 | 3,117 [3,045–3,180] | 3,165 | 3,180 | 0/5 (0%, 0–43%) | 0.85 | 90 |
| cascade/single-flush · hesitation | 5/1 | 3,212 | 1,888 | 3,459 [3,385–3,625] | 3,572 | 3,625 | 0/5 (0%, 0–43%) | 0.94 | 91 |
| cascade/single-flush · long | 5/1 | 2,566 | 1,416 | 9,644 [9,226–10,783] | 10,450 | 10,783 | 0/5 (0%, 0–43%) | 1.09 | 82 |
| cascade/single-flush · medium | 15/3 | 2,650 | 1,054 | 3,634 [3,468–4,215] | 4,269 | 4,338 | 0/15 (0%, 0–20%) | 0.98 | 81 |
| cascade/single-flush · noisy | 15/3 | 2,650 | 975 | 3,674 [3,341–4,044] | 4,803 | 5,800 | 0/15 (0%, 0–20%) | 1.00 | 81 |
| cascade/single-flush · sentences | 30/6 | 2,650 | 1,007 | 3,654 [3,450–4,022] | 4,279 | 5,800 | 0/30 (0%, 0–11%) | 0.99 | 81 |
| cascade/single-flush · short | 15/3 | 2,125 | 1,098 | 2,716 [2,269–2,882] | 2,923 | 3,146 | 0/15 (0%, 0–20%) | 1.07 | 100 |
| cascade/single-flush · three_turns | 5/1 | 3,614 | 1,382 | 4,250 [3,526–4,461] | 4,394 | 4,461 | 0/5 (0%, 0–43%) | 0.67 | 67 |
| cascade/single-flush · two_utterances | 5/1 | 2,166 | 943 | 3,238 [2,925–3,609] | 3,480 | 3,609 | 0/5 (0%, 0–43%) | 0.84 | 90 |
| cascade/single-mbd0 · hesitation | 5/1 | 3,200 | 2,057 | 7,311 [7,047–7,788] | 7,665 | 7,788 | 0/5 (0%, 0–43%) | 1.74 | 91 |
| cascade/single-mbd0 · long | 5/1 | 2,539 | 1,612 | 42,203 [39,423–49,706] | 47,724 | 49,706 | 0/5 (0%, 0–43%) | 1.94 | 74 |
| cascade/single-mbd0 · medium | 15/3 | 2,629 | 1,232 | 10,044 [9,613–11,255] | 12,435 | 12,596 | 0/15 (0%, 0–20%) | 1.74 | 81 |
| cascade/single-mbd0 · noisy | 15/3 | 2,678 | 1,177 | 9,752 [9,371–11,897] | 12,102 | 12,325 | 0/15 (0%, 0–20%) | 1.75 | 81 |
| cascade/single-mbd0 · sentences | 30/6 | 2,632 | 1,220 | 9,986 [9,518–11,058] | 12,331 | 12,596 | 0/30 (0%, 0–11%) | 1.74 | 81 |
| cascade/single-mbd0 · short | 15/3 | 2,084 | 1,148 | 4,954 [4,605–5,164] | 6,673 | 6,809 | 0/15 (0%, 0–20%) | 2.22 | 100 |
| cascade/single-mbd0 · three_turns | 5/1 | 3,587 | 1,740 | 8,132 [7,770–9,779] | 9,578 | 9,779 | 0/5 (0%, 0–43%) | 1.12 | 49 |
| cascade/single-mbd0 · two_utterances | 5/1 | 2,133 | 1,039 | 11,140 [10,663–11,758] | 11,593 | 11,758 | 0/5 (0%, 0–43%) | 1.74 | 85 |
| cascade/single-mbd200 · hesitation | 5/1 | 3,352 | 1,933 | 4,159 [3,978–4,255] | 4,243 | 4,255 | 0/5 (0%, 0–43%) | 1.06 | 91 |
| cascade/single-mbd200 · long | 5/1 | 2,726 | 1,710 | 10,409 [9,323–11,084] | 10,929 | 11,084 | 0/5 (0%, 0–43%) | 1.13 | 82 |
| cascade/single-mbd200 · medium | 15/3 | 2,849 | 1,138 | 4,195 [3,221–4,330] | 4,481 | 4,611 | 0/15 (0%, 0–20%) | 1.04 | 74 |
| cascade/single-mbd200 · noisy | 15/3 | 2,780 | 1,000 | 4,003 [3,410–4,402] | 4,494 | 9,067 | 0/15 (0%, 0–20%) | 1.01 | 74 |
| cascade/single-mbd200 · sentences | 30/6 | 2,841 | 1,038 | 4,146 [3,645–4,273] | 4,505 | 9,067 | 0/30 (0%, 0–11%) | 1.03 | 74 |
| cascade/single-mbd200 · short | 15/3 | 2,224 | 1,079 | 2,816 [2,095–2,924] | 2,960 | 2,988 | 0/15 (0%, 0–20%) | 1.13 | 100 |
| cascade/single-mbd200 · three_turns | 5/1 | 3,629 | 1,214 | 3,924 [3,262–3,977] | 3,963 | 3,977 | 0/5 (0%, 0–43%) | 0.68 | 64 |
| cascade/single-mbd200 · two_utterances | 5/1 | 2,203 | 1,004 | 3,140 [2,970–3,166] | 3,163 | 3,166 | 0/5 (0%, 0–43%) | 0.83 | 90 |

Per-clip paired difference vs round-1 phrase cascade (end lag at playback):

- cascade/baseline: median -56 ms over 10 clips (faster on 6).
- cascade/single-flush: median -79 ms over 10 clips (faster on 7).
- cascade/single-mbd0: median +5,952 ms over 10 clips (faster on 0).
- cascade/single-mbd200: median +14 ms over 10 clips (faster on 5).

### 5b

| Condition · category | runs/clips | start lag med | end lag arrival med | end lag playback med [95% CI] | p90 | max | stalls (rate, Wilson 95%) | output/source | chrF |
|---|---|---|---|---|---|---|---|---|---|
| cascade/single-flush · hesitation | 4/1 | 3,255 | 1,874 | 3,538 [3,428–3,772] | 3,703 | 3,772 | 0/4 (0%, 0–49%) | 0.96 | 91 |
| cascade/single-flush · long | 5/1 | 2,545 | 1,455 | 10,222 [8,872–12,316] | 11,533 | 12,316 | 0/5 (0%, 0–43%) | 1.13 | 80 |
| cascade/single-flush · medium | 15/3 | 2,633 | 1,053 | 3,644 [3,419–4,009] | 4,054 | 4,176 | 0/15 (0%, 0–20%) | 0.99 | 81 |
| cascade/single-flush · noisy | 15/3 | 2,641 | 930 | 3,586 [3,296–3,969] | 4,020 | 4,045 | 0/15 (0%, 0–20%) | 1.00 | 81 |
| cascade/single-flush · sentences | 30/6 | 2,640 | 1,025 | 3,594 [3,422–3,740] | 4,028 | 4,176 | 0/30 (0%, 0–11%) | 1.00 | 81 |
| cascade/single-flush · short | 15/3 | 2,145 | 1,135 | 2,858 [2,165–2,949] | 2,985 | 3,014 | 0/15 (0%, 0–20%) | 1.13 | 100 |
| cascade/single-flush · three_turns | 5/1 | 3,698 | 1,406 | 4,080 [4,026–4,434] | 4,351 | 4,434 | 0/5 (0%, 0–43%) | 0.69 | 67 |
| cascade/single-flush · two_utterances | 5/1 | 2,194 | 1,028 | 3,093 [2,955–3,562] | 3,386 | 3,562 | 0/5 (0%, 0–43%) | 0.83 | 90 |
| cascade/single-flush+concise · hesitation | 5/1 | 3,384 | 1,851 | 3,694 [3,460–4,121] | 3,967 | 4,121 | 0/5 (0%, 0–43%) | 0.95 | 86 |
| cascade/single-flush+concise · long | 5/1 | 2,530 | 1,466 | 9,120 [8,349–10,458] | 10,134 | 10,458 | 0/5 (0%, 0–43%) | 1.06 | 80 |
| cascade/single-flush+concise · medium | 15/3 | 2,569 | 1,021 | 3,391 [3,183–3,624] | 3,733 | 4,468 | 0/15 (0%, 0–20%) | 0.94 | 81 |
| cascade/single-flush+concise · noisy | 15/3 | 2,584 | 1,018 | 3,532 [3,332–3,884] | 4,677 | 4,723 | 0/15 (0%, 0–20%) | 0.91 | 81 |
| cascade/single-flush+concise · sentences | 30/6 | 2,583 | 1,019 | 3,500 [3,328–3,630] | 4,483 | 4,723 | 0/30 (0%, 0–11%) | 0.93 | 81 |
| cascade/single-flush+concise · short | 15/3 | 2,181 | 1,368 | 2,924 [2,248–3,229] | 3,251 | 3,589 | 0/15 (0%, 0–20%) | 1.13 | 100 |
| cascade/single-flush+concise · three_turns | 5/1 | 3,720 | 1,444 | 4,614 [4,449–4,858] | 4,831 | 4,858 | 0/5 (0%, 0–43%) | 0.75 | 64 |
| cascade/single-flush+concise · two_utterances | 4/1 | 2,066 | 1,067 | 2,899 [2,752–2,938] | 2,929 | 2,938 | 0/4 (0%, 0–49%) | 0.80 | 89 |
| cascade/single-flush+concise+adaptive · hesitation | 5/1 | 3,371 | 1,835 | 3,634 [3,630–3,909] | 3,860 | 3,909 | 0/5 (0%, 0–43%) | 0.96 | 91 |
| cascade/single-flush+concise+adaptive · long | 5/1 | 2,544 | 1,500 | 9,546 [8,395–10,929] | 10,401 | 10,929 | 0/5 (0%, 0–43%) | 1.09 | 81 |
| cascade/single-flush+concise+adaptive · medium | 15/3 | 2,512 | 1,086 | 3,801 [3,344–4,330] | 4,463 | 4,773 | 0/15 (0%, 0–20%) | 1.01 | 81 |
| cascade/single-flush+concise+adaptive · noisy | 15/3 | 2,637 | 921 | 3,454 [3,256–3,637] | 4,077 | 4,582 | 0/15 (0%, 0–20%) | 0.94 | 81 |
| cascade/single-flush+concise+adaptive · sentences | 30/6 | 2,547 | 1,013 | 3,552 [3,332–3,736] | 4,439 | 4,773 | 0/30 (0%, 0–11%) | 0.96 | 81 |
| cascade/single-flush+concise+adaptive · short | 15/3 | 2,228 | 1,316 | 2,882 [2,440–3,195] | 3,387 | 3,525 | 0/15 (0%, 0–20%) | 1.15 | 100 |
| cascade/single-flush+concise+adaptive · three_turns | 5/1 | 3,840 | 1,363 | 4,650 [3,943–4,971] | 4,944 | 4,971 | 0/5 (0%, 0–43%) | 0.74 | 66 |
| cascade/single-flush+concise+adaptive · two_utterances | 5/1 | 2,144 | 967 | 3,098 [2,854–3,283] | 3,230 | 3,283 | 0/5 (0%, 0–43%) | 0.80 | 94 |

Per-clip paired difference vs round-1 phrase cascade (end lag at playback):

- cascade/single-flush: median -74 ms over 10 clips (faster on 7).
- cascade/single-flush+concise: median -195 ms over 10 clips (faster on 7).
- cascade/single-flush+concise+adaptive: median -51 ms over 10 clips (faster on 7).

**Pass criterion:** monologue (long) end lag at playback under 5 s with chrF ≥ 85.

| Condition | long: end lag med | long: output/source | long: chrF | pass |
|---|---|---|---|---|
| cascade/baseline | 11,349 | 1.21 | 80 | fail |
| cascade/single-flush | 9,798 | 1.09 | 81 | fail |
| cascade/single-flush+concise | 9,120 | 1.06 | 80 | fail |
| cascade/single-flush+concise+adaptive | 9,546 | 1.09 | 81 | fail |
| cascade/single-mbd0 | 42,203 | 1.94 | 74 | fail |
| cascade/single-mbd200 | 10,409 | 1.13 | 82 | fail |

Adaptive synthesis speed changes per run (median): cascade/single-flush+concise+adaptive 1.0

## Experiment 6 · Translation host and pipelined fragments

**Partly blocked:** Groq serial / Groq pipelined: no GROQ_API_KEY; those conditions were skipped

| Condition · category | runs/clips | start lag med | end lag arrival med | end lag playback med [95% CI] | p90 | max | stalls (rate, Wilson 95%) | output/source | chrF |
|---|---|---|---|---|---|---|---|---|---|
| cascade/gpt41mini-pipelined · long | 5/1 | 2,624 | 1,508 | 9,855 [9,301–10,027] | 9,983 | 10,027 | 0/5 (0%, 0–43%) | 1.11 | 81 |
| cascade/gpt41mini-pipelined · medium | 15/3 | 2,520 | 1,033 | 3,749 [3,213–3,921] | 4,533 | 4,906 | 0/15 (0%, 0–20%) | 0.92 | 81 |
| cascade/gpt41mini-pipelined · noisy | 15/3 | 2,643 | 988 | 3,604 [3,410–4,546] | 4,641 | 8,173 | 1/15 (7%, 1–30%) | 0.94 | 81 |
| cascade/gpt41mini-pipelined · sentences | 30/6 | 2,625 | 1,000 | 3,691 [3,430–3,806] | 4,629 | 8,173 | 1/30 (3%, 1–17%) | 0.93 | 81 |
| cascade/gpt41mini-pipelined · short | 15/3 | 2,220 | 901 | 3,073 [2,132–3,138] | 3,215 | 3,329 | 0/15 (0%, 0–20%) | 1.16 | 100 |
| cascade/gpt41mini-serial · long | 5/1 | 2,525 | 1,419 | 9,910 [9,394–10,411] | 10,341 | 10,411 | 0/5 (0%, 0–43%) | 1.11 | 81 |
| cascade/gpt41mini-serial · medium | 15/3 | 2,647 | 1,074 | 3,741 [3,313–3,922] | 4,759 | 4,945 | 0/15 (0%, 0–20%) | 0.96 | 81 |
| cascade/gpt41mini-serial · noisy | 15/3 | 2,551 | 1,061 | 3,682 [3,240–3,754] | 4,298 | 4,894 | 0/15 (0%, 0–20%) | 0.90 | 81 |
| cascade/gpt41mini-serial · sentences | 30/6 | 2,632 | 1,062 | 3,690 [3,309–3,782] | 4,597 | 4,945 | 0/30 (0%, 0–11%) | 0.93 | 81 |
| cascade/gpt41mini-serial · short | 15/3 | 2,146 | 1,184 | 2,668 [2,119–3,102] | 3,127 | 3,145 | 0/15 (0%, 0–20%) | 1.07 | 100 |

LLM time per fragment (ms, from request):

| Condition | fragments | first token med | p90 | complete med | p90 |
|---|---|---|---|---|---|
| cascade/gpt41mini-pipelined | 200 | 452 | 585 | 526 | 666 |
| cascade/gpt41mini-serial | 200 | 436 | 553 | 511 | 647 |

Stage breakdown of the final fragment (medians, ms; sentences = medium + noisy):

| Condition | STT finalization | translation wait | TTS first byte | rest of TTS |
|---|---|---|---|---|
| cascade/gpt41mini-pipelined | 242 | -167 | 750 | 111 |
| cascade/gpt41mini-serial | 259 | -143 | 810 | 72 |

**Pass criterion:** sentence (medium + noisy) start lag ≤ 2,000 ms.

- cascade/gpt41mini-pipelined: sentence start lag median 2,625 ms fail.
- cascade/gpt41mini-serial: sentence start lag median 2,632 ms fail.

## Preparation for experiments 7 and 8 (built and unit-tested, not run)

- **Commit-rule parameters** can all be set per batch: `--set cascade.incremental.min_segment_words=3`, `holdback_words`,
  `stable_updates`, and the new `stable_ms` (a word is committed once it, and every word before it, has stood unchanged
  for that long; replaces the update-count rule when > 0). Round-2 plans pass the same overrides through `bench r2 run`.
- **Anticipation mode** (`translate_mode: anticipation`) sits behind a feature flag (`anticipation_enabled: true`;
  without it the cascade refuses to run) with two strategies:
  - `single_call`: the LLM returns only the translation prefix it is confident will not change given the partial source.
  - `sampled`: N short continuations of the partial source are generated in parallel, prefix + continuation is
    translated for each, and the longest common translated prefix beyond what was already spoken is committed.
  At end of turn the rest is translated as a continuation of what was already spoken.
- **Tests:** word-level longest-common-prefix (case/punctuation-insensitive, never splits words), stripping already
  committed text, and `stable_ms` timing including a revision restarting the clock (`tests/test_anticipation.py`).

## Decision gate

1. **Does flush-on-pause hold the median at or under 2.3 s with no stalls?** No.
2. **Does any cascade configuration reach 2.2 s at chrF 85 or better?** No (at a 1x listener; see experiment 1 for catch-up playback).

# Discussion

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

# Spend

| Batch | Experiment | OpenAI sessions | audio min | estimated $ (audio) | estimated $ if billed by session time |
|---|---|---|---|---|---|
| 20261013-222319 | smoke2a | 2 | 1.1 | 0.04 | 0.04 |
| 20261013-222434 | exp2a | 119 | 66.8 | 2.27 | 2.59 |
| 20261014-172417 | exp2a | 120 | 68.1 | 2.32 | 2.63 |
| 20261014-181904 | exp2b | 180 | 101.5 | 3.45 | 3.97 |
| 20261014-190148 | exp2c | 50 | 29.9 | 1.02 | 1.15 |
| 20261014-194721 | smoke3 | 4 | 0.2 | 0.01 | 0.02 |
| 20261014-194847 | exp3 | 664 | 55.9 | 1.90 | 4.59 |
| 20261014-213709 | smoke5 | 0 | 0.0 | 0.00 | 0.00 |
| 20261015-014128 | exp5a | 0 | 0.0 | 0.00 | 0.00 |
| 20261015-023140 | exp4 | 80 | 15.7 | 0.53 | 0.76 |
| 20261015-032059 | exp5b | 0 | 0.0 | 0.00 | 0.00 |
| 20261015-044648 | exp6 | 0 | 0.0 | 0.00 | 0.00 |

Estimated OpenAI translate spend: **$11.53** (audio minutes × $0.034); **$15.76** if sessions were billed by open time. LLM, Deepgram and Cartesia usage are small and not itemized here.
Actual OpenAI spend from the usage dashboard: **TO FILL IN (paste the figure from the OpenAI usage dashboard)**.

# Appendix · Data (JSON)

```json
{
 "exp1": {
  "summary": {
   "cascade[incremental]|short": {
    "n_runs": 15,
    "n_clips": 3,
    "none": {
     "median": 2821.437,
     "ci95": [
      2632.427,
      2904.959
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 2588.443,
     "ci95": [
      2451.601,
      2680.764
     ],
     "median_delta": -194.3040000000001,
     "delta_ci95": [
      -257.47800000000007,
      -187.77199999999993
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 2609.312,
     "ci95": [
      2461.601,
      2680.764
     ],
     "median_delta": -183.587,
     "delta_ci95": [
      -244.43499999999995,
      -172.65200000000004
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 2442.773,
     "ci95": [
      2355.16,
      2577.132
     ],
     "median_delta": -297.933,
     "delta_ci95": [
      -394.8000000000002,
      -287.9169999999999
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 2474.773,
     "ci95": [
      2370.493,
      2577.132
     ],
     "median_delta": -281.5,
     "delta_ci95": [
      -374.8000000000002,
      -264.7329999999997
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 2268.055,
     "ci95": [
      2236.331,
      2449.443
     ],
     "median_delta": -425.61900000000014,
     "delta_ci95": [
      -564.0,
      -411.3100000000002
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 2309.002,
     "ci95": [
      2258.236,
      2449.443
     ],
     "median_delta": -402.14300000000003,
     "delta_ci95": [
      -535.4290000000001,
      -378.19000000000005
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 2474.773,
     "ci95": [
      2370.493,
      2577.132
     ],
     "median_delta": -281.5,
     "delta_ci95": [
      -374.8000000000002,
      -264.7329999999997
     ]
    }
   },
   "cascade[incremental]|medium": {
    "n_runs": 15,
    "n_clips": 3,
    "none": {
     "median": 3771.15,
     "ci95": [
      3507.635,
      3953.158
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 3223.713,
     "ci95": [
      2926.733,
      3347.422
     ],
     "median_delta": -549.7449999999999,
     "delta_ci95": [
      -720.4409999999998,
      -450.56500000000005
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 3239.332,
     "ci95": [
      2943.885,
      3360.465
     ],
     "median_delta": -536.7019999999998,
     "delta_ci95": [
      -705.6090000000004,
      -437.52199999999993
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 2937.463,
     "ci95": [
      2708.667,
      3060.925
     ],
     "median_delta": -816.319,
     "delta_ci95": [
      -1084.6749999999997,
      -690.8670000000002
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 2976.801,
     "ci95": [
      2718.973,
      3068.616
     ],
     "median_delta": -804.875,
     "delta_ci95": [
      -1049.542,
      -670.8670000000002
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 2604.919,
     "ci95": [
      2347.852,
      2701.009
     ],
     "median_delta": -1040.437,
     "delta_ci95": [
      -1407.3690000000001,
      -960.9260000000004
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 2634.639,
     "ci95": [
      2389.979,
      2729.58
     ],
     "median_delta": -1040.437,
     "delta_ci95": [
      -1407.3690000000001,
      -935.0480000000002
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 3068.616,
     "ci95": [
      2809.485,
      3185.425
     ],
     "median_delta": -698.1500000000001,
     "delta_ci95": [
      -767.7329999999997,
      -640.5660000000003
     ]
    }
   },
   "cascade[incremental]|noisy": {
    "n_runs": 15,
    "n_clips": 3,
    "none": {
     "median": 3606.489,
     "ci95": [
      3486.72,
      3906.927
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 3152.449,
     "ci95": [
      2911.264,
      3352.915
     ],
     "median_delta": -532.326,
     "delta_ci95": [
      -613.4779999999996,
      -431.0659999999998
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 3165.492,
     "ci95": [
      2924.307,
      3381.394
     ],
     "median_delta": -506.8040000000001,
     "delta_ci95": [
      -605.9779999999996,
      -419.04399999999987
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 2852.028,
     "ci95": [
      2626.687,
      2974.904
     ],
     "median_delta": -795.3240000000001,
     "delta_ci95": [
      -940.6669999999999,
      -660.9670000000001
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 2856.028,
     "ci95": [
      2642.687,
      3006.571
     ],
     "median_delta": -755.366,
     "delta_ci95": [
      -929.1669999999999,
      -642.5340000000001
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 2514.985,
     "ci95": [
      2355.585,
      2668.417
     ],
     "median_delta": -1017.2950000000001,
     "delta_ci95": [
      -1343.81,
      -944.239
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 2560.223,
     "ci95": [
      2360.943,
      2689.443
     ],
     "median_delta": -984.2470000000003,
     "delta_ci95": [
      -1327.3809999999999,
      -917.9050000000002
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 2964.229,
     "ci95": [
      2791.754,
      3206.537
     ],
     "median_delta": -656.9670000000001,
     "delta_ci95": [
      -726.3669999999997,
      -641.1670000000004
     ]
    }
   },
   "cascade[incremental]|long": {
    "n_runs": 5,
    "n_clips": 1,
    "none": {
     "median": 11641.286,
     "ci95": [
      10545.98,
      12521.112
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 8209.59,
     "ci95": [
      7087.806,
      8812.503
     ],
     "median_delta": -3512.065999999999,
     "delta_ci95": [
      -3708.6089999999986,
      -3431.696
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 8248.72,
     "ci95": [
      7102.675,
      8825.546
     ],
     "median_delta": -3499.021999999999,
     "delta_ci95": [
      -3695.565999999999,
      -3392.5660000000007
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 6418.786,
     "ci95": [
      5243.447,
      6834.579
     ],
     "median_delta": -5385.1669999999995,
     "delta_ci95": [
      -5686.532999999999,
      -5222.5
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 6458.786,
     "ci95": [
      5266.247,
      6878.579
     ],
     "median_delta": -5341.1669999999995,
     "delta_ci95": [
      -5642.532999999999,
      -5182.5
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 4251.998,
     "ci95": [
      3239.801,
      4802.95
     ],
     "median_delta": -7389.2880000000005,
     "delta_ci95": [
      -7992.856999999999,
      -7170.183
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 4317.714,
     "ci95": [
      3329.992,
      4831.521
     ],
     "median_delta": -7323.572,
     "delta_ci95": [
      -7892.475999999999,
      -7141.612
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 9253.219,
     "ci95": [
      8374.247,
      9960.512
     ],
     "median_delta": -2388.067000000001,
     "delta_ci95": [
      -2560.5999999999985,
      -2171.733
     ]
    }
   },
   "cascade[incremental]|overlap": {
    "n_runs": 5,
    "n_clips": 1,
    "none": {
     "median": 9746.587,
     "ci95": [
      7674.117,
      10391.869
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 8035.804,
     "ci95": [
      6229.856,
      8639.782
     ],
     "median_delta": -1717.5650000000005,
     "delta_ci95": [
      -1752.0870000000014,
      -1444.2610000000004
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 8059.282,
     "ci95": [
      6258.378,
      8660.478
     ],
     "median_delta": -1679.7830000000013,
     "delta_ci95": [
      -1731.3910000000014,
      -1415.7390000000005
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 7140.19,
     "ci95": [
      5474.717,
      7705.336
     ],
     "median_delta": -2593.7660000000005,
     "delta_ci95": [
      -2686.5330000000004,
      -2199.4000000000005
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 7176.19,
     "ci95": [
      5510.451,
      7737.069
     ],
     "median_delta": -2557.7660000000005,
     "delta_ci95": [
      -2654.8,
      -2163.666
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 6069.312,
     "ci95": [
      4603.165,
      6553.964
     ],
     "median_delta": -3664.6440000000002,
     "delta_ci95": [
      -3837.9050000000007,
      -3070.952
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 6154.873,
     "ci95": [
      4665.593,
      6599.298
     ],
     "median_delta": -3591.714,
     "delta_ci95": [
      -3792.571000000001,
      -3008.5240000000003
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 7837.721,
     "ci95": [
      6170.117,
      8347.869
     ],
     "median_delta": -1920.1660000000002,
     "delta_ci95": [
      -2044.0,
      -1504.0
     ]
    }
   },
   "cascade[on_turn_end]|short": {
    "n_runs": 15,
    "n_clips": 3,
    "none": {
     "median": 2455.232,
     "ci95": [
      2217.544,
      2819.034
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 2250.493,
     "ci95": [
      2044.399,
      2624.229
     ],
     "median_delta": -183.87000000000012,
     "delta_ci95": [
      -195.61900000000014,
      -173.43499999999995
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 2263.536,
     "ci95": [
      2047.609,
      2639.881
     ],
     "median_delta": -179.15300000000025,
     "delta_ci95": [
      -189.69500000000016,
      -168.60799999999995
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 2141.298,
     "ci95": [
      1946.752,
      2520.334
     ],
     "median_delta": -281.9330000000002,
     "delta_ci95": [
      -299.9500000000003,
      -265.933
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 2161.298,
     "ci95": [
      1956.977,
      2544.334
     ],
     "median_delta": -274.7000000000003,
     "delta_ci95": [
      -290.866,
      -258.5329999999999
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 2006.756,
     "ci95": [
      1826.438,
      2392.319
     ],
     "median_delta": -402.76200000000017,
     "delta_ci95": [
      -428.5,
      -379.904
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 2035.327,
     "ci95": [
      1845.306,
      2426.605
     ],
     "median_delta": -392.4290000000001,
     "delta_ci95": [
      -415.5239999999999,
      -369.33299999999986
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 2161.298,
     "ci95": [
      1956.977,
      2544.334
     ],
     "median_delta": -274.7000000000003,
     "delta_ci95": [
      -290.866,
      -258.5329999999999
     ]
    }
   },
   "cascade[on_turn_end]|medium": {
    "n_runs": 15,
    "n_clips": 3,
    "none": {
     "median": 5084.217,
     "ci95": [
      4802.139,
      5230.402
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 4584.305,
     "ci95": [
      4363.443,
      4723.359
     ],
     "median_delta": -482.21800000000076,
     "delta_ci95": [
      -504.9339999999993,
      -438.6959999999999
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 4596.783,
     "ci95": [
      4372.4,
      4736.402
     ],
     "median_delta": -471.58699999999953,
     "delta_ci95": [
      -492.4559999999992,
      -429.7390000000005
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 4315.006,
     "ci95": [
      4129.472,
      4452.936
     ],
     "median_delta": -739.4010000000007,
     "delta_ci95": [
      -774.2329999999993,
      -672.6670000000004
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 4334.139,
     "ci95": [
      4143.205,
      4472.936
     ],
     "median_delta": -723.0999999999995,
     "delta_ci95": [
      -755.0999999999995,
      -658.9340000000002
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 3988.123,
     "ci95": [
      3841.186,
      4119.736
     ],
     "median_delta": -1056.286,
     "delta_ci95": [
      -1106.0469999999996,
      -960.953
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 4012.373,
     "ci95": [
      3860.805,
      4148.307
     ],
     "median_delta": -1033.0,
     "delta_ci95": [
      -1078.7139999999995,
      -941.3340000000003
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 4334.139,
     "ci95": [
      4143.205,
      4472.936
     ],
     "median_delta": -723.0999999999995,
     "delta_ci95": [
      -755.0999999999995,
      -658.9340000000002
     ]
    }
   },
   "cascade[on_turn_end]|noisy": {
    "n_runs": 15,
    "n_clips": 3,
    "none": {
     "median": 4821.807,
     "ci95": [
      4518.196,
      4968.496
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 4346.573,
     "ci95": [
      4092.239,
      4484.257
     ],
     "median_delta": -484.2390000000005,
     "delta_ci95": [
      -502.77700000000004,
      -437.174
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 4358.828,
     "ci95": [
      4105.282,
      4499.127
     ],
     "median_delta": -469.3689999999997,
     "delta_ci95": [
      -490.34799999999996,
      -421.23900000000003
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 4093.115,
     "ci95": [
      3865.062,
      4220.056
     ],
     "median_delta": -742.5,
     "delta_ci95": [
      -770.9249999999997,
      -670.3340000000003
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 4111.907,
     "ci95": [
      3885.062,
      4240.056
     ],
     "median_delta": -719.6999999999998,
     "delta_ci95": [
      -751.8670000000002,
      -645.9000000000001
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 3780.818,
     "ci95": [
      3585.148,
      3880.256
     ],
     "median_delta": -1060.714,
     "delta_ci95": [
      -1101.321,
      -957.6190000000001
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 3807.664,
     "ci95": [
      3613.719,
      3908.827
     ],
     "median_delta": -1028.143,
     "delta_ci95": [
      -1074.0950000000003,
      -922.7139999999999
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 4111.907,
     "ci95": [
      3885.062,
      4240.056
     ],
     "median_delta": -719.6999999999998,
     "delta_ci95": [
      -751.8670000000002,
      -645.9000000000001
     ]
    }
   },
   "cascade[on_turn_end]|long": {
    "n_runs": 5,
    "n_clips": 1,
    "none": {
     "median": 20533.412,
     "ci95": [
      20002.007,
      20708.853
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 18054.738,
     "ci95": [
      17584.659,
      18283.505
     ],
     "median_delta": -2445.826000000001,
     "delta_ci95": [
      -2478.673999999999,
      -2417.348000000002
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 18067.782,
     "ci95": [
      17598.311,
      18298.375
     ],
     "median_delta": -2432.7819999999992,
     "delta_ci95": [
      -2465.630000000001,
      -2403.696
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 16732.779,
     "ci95": [
      16295.407,
      16989.987
     ],
     "median_delta": -3750.2659999999996,
     "delta_ci95": [
      -3800.6330000000016,
      -3706.600000000002
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 16752.779,
     "ci95": [
      16316.34,
      17012.787
     ],
     "median_delta": -3730.2659999999996,
     "delta_ci95": [
      -3780.6330000000016,
      -3685.6670000000013
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 15103.936,
     "ci95": [
      14706.864,
      15396.187
     ],
     "median_delta": -5357.523999999999,
     "delta_ci95": [
      -5429.476000000001,
      -5295.143000000002
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 15132.507,
     "ci95": [
      14736.769,
      15428.758
     ],
     "median_delta": -5328.951999999999,
     "delta_ci95": [
      -5400.905000000001,
      -5265.238000000001
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 16752.779,
     "ci95": [
      16316.34,
      17012.787
     ],
     "median_delta": -3730.2659999999996,
     "delta_ci95": [
      -3780.6330000000016,
      -3685.6670000000013
     ]
    }
   },
   "cascade[on_turn_end]|overlap": {
    "n_runs": 5,
    "n_clips": 1,
    "none": {
     "median": 7560.75,
     "ci95": [
      5101.323,
      7728.947
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 6731.815,
     "ci95": [
      4596.312,
      7206.994
     ],
     "median_delta": -530.1310000000003,
     "delta_ci95": [
      -828.9350000000004,
      -505.0110000000004
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 6744.859,
     "ci95": [
      4607.04,
      7220.038
     ],
     "median_delta": -514.4780000000001,
     "delta_ci95": [
      -815.8909999999996,
      -494.28300000000036
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 6289.717,
     "ci95": [
      4326.973,
      6931.006
     ],
     "median_delta": -812.8670000000002,
     "delta_ci95": [
      -1271.0330000000004,
      -774.3500000000004
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 6309.717,
     "ci95": [
      4343.423,
      6951.006
     ],
     "median_delta": -788.8670000000002,
     "delta_ci95": [
      -1251.0330000000004,
      -757.9000000000005
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 5744.988,
     "ci95": [
      3995.108,
      6590.949
     ],
     "median_delta": -1161.2380000000003,
     "delta_ci95": [
      -1815.7619999999997,
      -1106.2150000000001
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 5773.559,
     "ci95": [
      4018.608,
      6619.52
     ],
     "median_delta": -1126.9530000000004,
     "delta_ci95": [
      -1787.1909999999998,
      -1082.7150000000001
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 6309.717,
     "ci95": [
      4343.423,
      6951.006
     ],
     "median_delta": -788.8670000000002,
     "delta_ci95": [
      -1251.0330000000004,
      -757.9000000000005
     ]
    }
   },
   "openai|short": {
    "n_runs": 14,
    "n_clips": 3,
    "none": {
     "median": 2267.9445,
     "ci95": [
      2064.723,
      2619.016
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 2038.0195,
     "ci95": [
      1880.383,
      2528.034
     ],
     "median_delta": -173.59099999999967,
     "delta_ci95": [
      -224.47199999999998,
      -142.6189999999997
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 2128.2219999999998,
     "ci95": [
      1967.546,
      2619.016
     ],
     "median_delta": -94.24099999999987,
     "delta_ci95": [
      -146.87499999999977,
      -36.52200000000016
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 2003.0625,
     "ci95": [
      1851.934,
      2513.393
     ],
     "median_delta": -194.52300000000014,
     "delta_ci95": [
      -271.70199999999977,
      -160.894
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 2117.2434999999996,
     "ci95": [
      1967.546,
      2619.016
     ],
     "median_delta": -106.58849999999995,
     "delta_ci95": [
      -170.817,
      -56.0
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 1971.8615,
     "ci95": [
      1819.38,
      2418.2470999999714
     ],
     "median_delta": -220.57799999999997,
     "delta_ci95": [
      -308.09900000000016,
      -173.94999999999982
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 2109.151,
     "ci95": [
      1967.546,
      2539.37702499997
     ],
     "median_delta": -96.13849999999991,
     "delta_ci95": [
      -188.57200000000012,
      -72.18499999999995
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 2115.3905,
     "ci95": [
      1936.723,
      2619.016
     ],
     "median_delta": -111.54999999999995,
     "delta_ci95": [
      -170.817,
      -56.0
     ]
    }
   },
   "openai|medium": {
    "n_runs": 15,
    "n_clips": 3,
    "none": {
     "median": 2738.075,
     "ci95": [
      2410.217,
      3452.079
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 2564.266,
     "ci95": [
      2292.447,
      3149.181
     ],
     "median_delta": -202.70800000000008,
     "delta_ci95": [
      -253.7837999999998,
      -160.92399999999998
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 2677.415,
     "ci95": [
      2334.565,
      3279.905
     ],
     "median_delta": -110.14799999999968,
     "delta_ci95": [
      -168.7070249999995,
      -64.49099999999999
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 2529.407,
     "ci95": [
      2288.859,
      3123.136
     ],
     "median_delta": -218.25300000000016,
     "delta_ci95": [
      -312.3929999999996,
      -164.08300000000008
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 2657.937,
     "ci95": [
      2350.217,
      3248.079
     ],
     "median_delta": -110.14799999999968,
     "delta_ci95": [
      -203.3399999999997,
      -64.0
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 2507.121,
     "ci95": [
      2284.437,
      3100.85
     ],
     "median_delta": -237.40700000000015,
     "delta_ci95": [
      -329.71799999999985,
      -175.23099999999977
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 2633.937,
     "ci95": [
      2329.427,
      3252.079
     ],
     "median_delta": -108.57200000000012,
     "delta_ci95": [
      -200.0,
      -85.71499999999992
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 2657.937,
     "ci95": [
      2350.217,
      3240.079
     ],
     "median_delta": -84.0,
     "delta_ci95": [
      -176.0,
      -60.0
     ]
    }
   },
   "openai|noisy": {
    "n_runs": 15,
    "n_clips": 3,
    "none": {
     "median": 2605.943,
     "ci95": [
      2045.449,
      2807.525
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 2428.073,
     "ci95": [
      1904.906,
      2655.402
     ],
     "median_delta": -166.38799999999992,
     "delta_ci95": [
      -213.51500000000033,
      -138.9770000000001
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 2530.291,
     "ci95": [
      1976.454,
      2704.793
     ],
     "median_delta": -75.65200000000004,
     "delta_ci95": [
      -137.50199999999995,
      -61.57999999999993
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 2426.418,
     "ci95": [
      1863.031,
      2633.06
     ],
     "median_delta": -181.46000000000004,
     "delta_ci95": [
      -241.73300000000017,
      -157.44800000000032
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 2545.943,
     "ci95": [
      1998.422,
      2704.793
     ],
     "median_delta": -92.95100000000002,
     "delta_ci95": [
      -144.8670000000002,
      -60.0
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 2424.379,
     "ci95": [
      1852.806,
      2596.303
     ],
     "median_delta": -200.03200000000015,
     "delta_ci95": [
      -243.45800000000008,
      -169.05100000000039
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 2520.229,
     "ci95": [
      1998.422,
      2679.789
     ],
     "median_delta": -91.42900000000009,
     "delta_ci95": [
      -149.35500000000002,
      -80.0
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 2545.943,
     "ci95": [
      1998.422,
      2704.793
     ],
     "median_delta": -64.0,
     "delta_ci95": [
      -144.0,
      -59.07500000000027
     ]
    }
   },
   "openai|long": {
    "n_runs": 5,
    "n_clips": 1,
    "none": {
     "median": 2752.999,
     "ci95": [
      2683.849,
      2992.799
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 2569.425,
     "ci95": [
      2447.209,
      2800.496
     ],
     "median_delta": -197.80499999999984,
     "delta_ci95": [
      -305.78999999999996,
      -71.58500000000004
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 2672.17,
     "ci95": [
      2574.954,
      2902.796
     ],
     "median_delta": -95.05999999999995,
     "delta_ci95": [
      -178.04499999999962,
      0.0
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 2568.878,
     "ci95": [
      2421.453,
      2764.626
     ],
     "median_delta": -228.17299999999977,
     "delta_ci95": [
      -331.5459999999998,
      -81.78200000000015
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 2672.17,
     "ci95": [
      2574.954,
      2881.927
     ],
     "median_delta": -103.8760000000002,
     "delta_ci95": [
      -178.04499999999962,
      0.0
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 2568.203,
     "ci95": [
      2405.449,
      2728.101
     ],
     "median_delta": -255.39900000000034,
     "delta_ci95": [
      -347.5499999999997,
      -94.346
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 2672.17,
     "ci95": [
      2574.954,
      2856.212
     ],
     "median_delta": -102.85699999999997,
     "delta_ci95": [
      -178.04499999999962,
      0.0
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 2672.17,
     "ci95": [
      2572.999,
      2888.799
     ],
     "median_delta": -95.05999999999995,
     "delta_ci95": [
      -180.0,
      0.0
     ]
    }
   },
   "openai|overlap": {
    "n_runs": 5,
    "n_clips": 1,
    "none": {
     "median": 721.773,
     "ci95": [
      318.988,
      2049.46
     ],
     "median_delta": 0.0,
     "delta_ci95": [
      0.0,
      0.0
     ]
    },
    "catchup@1.15x/300ms": {
     "median": 548.025,
     "ci95": [
      -6.151,
      1800.587
     ],
     "median_delta": -177.20199999999994,
     "delta_ci95": [
      -331.587,
      -33.91300000000001
     ]
    },
    "catchup@1.15x/500ms": {
     "median": 680.034,
     "ci95": [
      123.902,
      1834.5
     ],
     "median_delta": -83.47799999999995,
     "delta_ci95": [
      -258.26,
      0.0
     ]
    },
    "catchup@1.25x/300ms": {
     "median": 531.735,
     "ci95": [
      -23.733,
      1782.5
     ],
     "median_delta": -196.66299999999995,
     "delta_ci95": [
      -391.17499999999995,
      -52.0
     ]
    },
    "catchup@1.25x/500ms": {
     "median": 678.781,
     "ci95": [
      139.554,
      1834.5
     ],
     "median_delta": -123.99999999999994,
     "delta_ci95": [
      -304.0,
      0.0
     ]
    },
    "catchup@1.4x/300ms": {
     "median": 511.664,
     "ci95": [
      -45.398,
      1760.215
     ],
     "median_delta": -210.10900000000004,
     "delta_ci95": [
      -411.932,
      -74.28500000000008
     ]
    },
    "catchup@1.4x/500ms": {
     "median": 678.781,
     "ci95": [
      113.84,
      1834.5
     ],
     "median_delta": -97.14299999999997,
     "delta_ci95": [
      -325.71399999999994,
      0.0
     ]
    },
    "catchup_end@1.25x/500ms": {
     "median": 673.773,
     "ci95": [
      256.438,
      1834.5
     ],
     "median_delta": -62.55000000000001,
     "delta_ci95": [
      -292.0,
      0.0
     ]
    }
   },
   "openai|same_language": {
    "n_runs": 2,
    "n_clips": 1,
    "none": {
     "median": 4675.107,
     "ci95": null,
     "median_delta": 0.0,
     "delta_ci95": null
    },
    "catchup@1.15x/300ms": {
     "median": 4249.4655,
     "ci95": null,
     "median_delta": -425.6414999999997,
     "delta_ci95": null
    },
    "catchup@1.15x/500ms": {
     "median": 4322.6915,
     "ci95": null,
     "median_delta": -352.41550000000007,
     "delta_ci95": null
    },
    "catchup@1.25x/300ms": {
     "median": 4061.9219999999996,
     "ci95": null,
     "median_delta": -613.1850000000002,
     "delta_ci95": null
    },
    "catchup@1.25x/500ms": {
     "median": 4134.736,
     "ci95": null,
     "median_delta": -540.3710000000001,
     "delta_ci95": null
    },
    "catchup@1.4x/300ms": {
     "median": 3832.0915000000005,
     "ci95": null,
     "median_delta": -843.0154999999997,
     "delta_ci95": null
    },
    "catchup@1.4x/500ms": {
     "median": 3913.379,
     "ci95": null,
     "median_delta": -761.7279999999998,
     "delta_ci95": null
    },
    "catchup_end@1.25x/500ms": {
     "median": 4136.736,
     "ci95": null,
     "median_delta": -538.3710000000001,
     "delta_ci95": null
    }
   }
  },
  "listening_check": [
   {
    "system": "cascade[incremental]",
    "run_id": "20261013-211511_0048_cascade_es_en_medium_1_r0",
    "policy": "catchup@1.25x/500ms",
    "file": "20261013-211511_0048_cascade_es_en_medium_1_r0__catchup_1.25x_500ms.wav",
    "sped_up_content_s": 5.35,
    "chrf_vs_unstretched": 100.0,
    "unstretched_asr": "Tomorrow morning, I have to take someone to my daughter to the doctor before going to work.",
    "stretched_asr": "Tomorrow morning, I have to take someone to my daughter to the doctor before going to work."
   },
   {
    "system": "cascade[incremental]",
    "run_id": "20261013-211511_0048_cascade_es_en_medium_1_r0",
    "policy": "catchup@1.4x/300ms",
    "file": "20261013-211511_0048_cascade_es_en_medium_1_r0__catchup_1.4x_300ms.wav",
    "sped_up_content_s": 5.36,
    "chrf_vs_unstretched": 95.38,
    "unstretched_asr": "Tomorrow morning, I have to take someone to my daughter to the doctor before going to work.",
    "stretched_asr": "Tomorrow morning I have to take someone to my daughter to the doctor before going to work."
   },
   {
    "system": "cascade[incremental]",
    "run_id": "20261013-211511_0064_cascade_es_en_medium_2_r0",
    "policy": "catchup@1.25x/500ms",
    "file": "20261013-211511_0064_cascade_es_en_medium_2_r0__catchup_1.25x_500ms.wav",
    "sped_up_content_s": 3.94,
    "chrf_vs_unstretched": 100.0,
    "unstretched_asr": "Could you tell me if there is an open pharmacy nearby at this hour?",
    "stretched_asr": "Could you tell me if there is an open pharmacy nearby at this hour?"
   },
   {
    "system": "cascade[incremental]",
    "run_id": "20261013-211511_0064_cascade_es_en_medium_2_r0",
    "policy": "catchup@1.4x/300ms",
    "file": "20261013-211511_0064_cascade_es_en_medium_2_r0__catchup_1.4x_300ms.wav",
    "sped_up_content_s": 4.14,
    "chrf_vs_unstretched": 100.0,
    "unstretched_asr": "Could you tell me if there is an open pharmacy nearby at this hour?",
    "stretched_asr": "Could you tell me if there is an open pharmacy nearby at this hour?"
   },
   {
    "system": "cascade[incremental]",
    "run_id": "20261013-211511_0060_cascade_es_en_medium_3_r0",
    "policy": "catchup@1.25x/500ms",
    "file": "20261013-211511_0060_cascade_es_en_medium_3_r0__catchup_1.25x_500ms.wav",
    "sped_up_content_s": 4.11,
    "chrf_vs_unstretched": 100.0,
    "unstretched_asr": "The restaurant that we recommended was full, so we had dinner at the hotel.",
    "stretched_asr": "The restaurant that we recommended was full, so we had dinner at the hotel."
   },
   {
    "system": "cascade[incremental]",
    "run_id": "20261013-211511_0060_cascade_es_en_medium_3_r0",
    "policy": "catchup@1.4x/300ms",
    "file": "20261013-211511_0060_cascade_es_en_medium_3_r0__catchup_1.4x_300ms.wav",
    "sped_up_content_s": 4.31,
    "chrf_vs_unstretched": 100.0,
    "unstretched_asr": "The restaurant that we recommended was full, so we had dinner at the hotel.",
    "stretched_asr": "The restaurant that we recommended was full, so we had dinner at the hotel."
   },
   {
    "system": "cascade[on_turn_end]",
    "run_id": "20261013-201319_0072_cascade_es_en_medium_1_r0",
    "policy": "catchup@1.25x/500ms",
    "file": "20261013-201319_0072_cascade_es_en_medium_1_r0__catchup_1.25x_500ms.wav",
    "sped_up_content_s": 3.79,
    "chrf_vs_unstretched": 94.71,
    "unstretched_asr": "Tomorrow morning, I have to take my daughter to the doctor before going to work.",
    "stretched_asr": "Tomorrow morning I have to take my daughter to the doctor before going to work."
   },
   {
    "system": "cascade[on_turn_end]",
    "run_id": "20261013-201319_0072_cascade_es_en_medium_1_r0",
    "policy": "catchup@1.4x/300ms",
    "file": "20261013-201319_0072_cascade_es_en_medium_1_r0__catchup_1.4x_300ms.wav",
    "sped_up_content_s": 3.89,
    "chrf_vs_unstretched": 94.71,
    "unstretched_asr": "Tomorrow morning, I have to take my daughter to the doctor before going to work.",
    "stretched_asr": "Tomorrow morning I have to take my daughter to the doctor before going to work."
   },
   {
    "system": "cascade[on_turn_end]",
    "run_id": "20261013-201319_0069_cascade_es_en_medium_2_r0",
    "policy": "catchup@1.25x/500ms",
    "file": "20261013-201319_0069_cascade_es_en_medium_2_r0__catchup_1.25x_500ms.wav",
    "sped_up_content_s": 3.04,
    "chrf_vs_unstretched": 100.0,
    "unstretched_asr": "Could you tell me if there is a pharmacy open nearby at this hour?",
    "stretched_asr": "Could you tell me if there is a pharmacy open nearby at this hour?"
   },
   {
    "system": "cascade[on_turn_end]",
    "run_id": "20261013-201319_0069_cascade_es_en_medium_2_r0",
    "policy": "catchup@1.4x/300ms",
    "file": "20261013-201319_0069_cascade_es_en_medium_2_r0__catchup_1.4x_300ms.wav",
    "sped_up_content_s": 3.14,
    "chrf_vs_unstretched": 100.0,
    "unstretched_asr": "Could you tell me if there is a pharmacy open nearby at this hour?",
    "stretched_asr": "Could you tell me if there is a pharmacy open nearby at this hour?"
   },
   {
    "system": "cascade[on_turn_end]",
    "run_id": "20261013-201319_0109_cascade_es_en_medium_3_r0",
    "policy": "catchup@1.25x/500ms",
    "file": "20261013-201319_0109_cascade_es_en_medium_3_r0__catchup_1.25x_500ms.wav",
    "sped_up_content_s": 3.63,
    "chrf_vs_unstretched": 100.0,
    "unstretched_asr": "The restaurant you recommended was full, so we had dinner at the hotel.",
    "stretched_asr": "The restaurant you recommended was full, so we had dinner at the hotel."
   },
   {
    "system": "cascade[on_turn_end]",
    "run_id": "20261013-201319_0109_cascade_es_en_medium_3_r0",
    "policy": "catchup@1.4x/300ms",
    "file": "20261013-201319_0109_cascade_es_en_medium_3_r0__catchup_1.4x_300ms.wav",
    "sped_up_content_s": 3.72,
    "chrf_vs_unstretched": 100.0,
    "unstretched_asr": "The restaurant you recommended was full, so we had dinner at the hotel.",
    "stretched_asr": "The restaurant you recommended was full, so we had dinner at the hotel."
   },
   {
    "system": "openai",
    "run_id": "20261013-201319_0100_openai_es_en_medium_1_r0",
    "policy": "catchup@1.25x/500ms",
    "file": "20261013-201319_0100_openai_es_en_medium_1_r0__catchup_1.25x_500ms.wav",
    "sped_up_content_s": 16.8,
    "chrf_vs_unstretched": 93.82,
    "unstretched_asr": "Tomorrow morning I have to take my dar to the doctor before going to work.",
    "stretched_asr": "Tomorrow morning I have to take my car to the doctor before going to work."
   },
   {
    "system": "openai",
    "run_id": "20261013-201319_0100_openai_es_en_medium_1_r0",
    "policy": "catchup@1.4x/300ms",
    "file": "20261013-201319_0100_openai_es_en_medium_1_r0__catchup_1.4x_300ms.wav",
    "sped_up_content_s": 23.7,
    "chrf_vs_unstretched": 88.15,
    "unstretched_asr": "Tomorrow morning I have to take my dar to the doctor before going to work.",
    "stretched_asr": "Tomorrow morning, I have to take my doctor to the doctor before going to work."
   },
   {
    "system": "openai",
    "run_id": "20261013-201319_0037_openai_es_en_medium_2_r0",
    "policy": "catchup@1.25x/500ms",
    "file": "20261013-201319_0037_openai_es_en_medium_2_r0__catchup_1.25x_500ms.wav",
    "sped_up_content_s": 21.58,
    "chrf_vs_unstretched": 95.62,
    "unstretched_asr": "Could you tell me if there's a pharmacy open near here at this hour? You",
    "stretched_asr": "Could you tell me if there's a pharmacy open near here at this hour?"
   },
   {
    "system": "openai",
    "run_id": "20261013-201319_0037_openai_es_en_medium_2_r0",
    "policy": "catchup@1.4x/300ms",
    "file": "20261013-201319_0037_openai_es_en_medium_2_r0__catchup_1.4x_300ms.wav",
    "sped_up_content_s": 26.64,
    "chrf_vs_unstretched": 95.62,
    "unstretched_asr": "Could you tell me if there's a pharmacy open near here at this hour? You",
    "stretched_asr": "Could you tell me if there's a pharmacy open near here at this hour?"
   },
   {
    "system": "openai",
    "run_id": "20261013-201319_0022_openai_es_en_medium_3_r0",
    "policy": "catchup@1.25x/500ms",
    "file": "20261013-201319_0022_openai_es_en_medium_3_r0__catchup_1.25x_500ms.wav",
    "sped_up_content_s": 17.96,
    "chrf_vs_unstretched": 90.45,
    "unstretched_asr": "The restaurant you recommended was full, so we had dinner at the hotel.",
    "stretched_asr": "The restaurant you recommended was four, so we had dinner at the hotel."
   },
   {
    "system": "openai",
    "run_id": "20261013-201319_0022_openai_es_en_medium_3_r0",
    "policy": "catchup@1.4x/300ms",
    "file": "20261013-201319_0022_openai_es_en_medium_3_r0__catchup_1.4x_300ms.wav",
    "sped_up_content_s": 25.14,
    "chrf_vs_unstretched": 86.9,
    "unstretched_asr": "The restaurant you recommended was full, so we had dinner at the hotel.",
    "stretched_asr": "The restaurant you recommended was folk. So we had dinner at the hotel."
   }
  ]
 },
 "exp2ab": {
  "2a": {
   "openai/a20-rt60": {
    "n": 120,
    "failed": 0,
    "stalls": 4,
    "rate": 0.03333333333333333,
    "wilson": [
     0.01303755100780557,
     0.0825803380922108
    ],
    "held": 1,
    "start_lag_q": [
     1131.9104,
     1633.205,
     2509.8158000000003,
     37990.91
    ]
   },
   "openai/a20-rt60 \u00b7 medium": {
    "n": 60,
    "failed": 0,
    "stalls": 3,
    "rate": 0.05,
    "wilson": [
     0.01714950203090107,
     0.1370051656999034
    ],
    "held": 1,
    "start_lag_q": [
     1124.5516,
     1633.205,
     2913.9270000000024,
     37990.91
    ]
   },
   "openai/a20-rt60 \u00b7 short": {
    "n": 60,
    "failed": 0,
    "stalls": 1,
    "rate": 0.016666666666666666,
    "wilson": [
     0.0029481597621904,
     0.08855129817089587
    ],
    "held": 0,
    "start_lag_q": [
     1153.6732,
     1644.723,
     2379.9913,
     9810.523
    ]
   },
   "openai/a200-rt60": {
    "n": 119,
    "failed": 1,
    "stalls": 5,
    "rate": 0.04201680672268908,
    "wilson": [
     0.018078474345219454,
     0.09459894749060604
    ],
    "held": 0,
    "start_lag_q": [
     1163.9278,
     1899.735,
     3226.9496000000004,
     28757.109
    ]
   },
   "openai/a200-rt60 \u00b7 medium": {
    "n": 60,
    "failed": 0,
    "stalls": 4,
    "rate": 0.06666666666666667,
    "wilson": [
     0.02622869854224924,
     0.1592535740874143
    ],
    "held": 0,
    "start_lag_q": [
     1258.6506,
     2095.041,
     3755.2052000000003,
     28757.109
    ]
   },
   "openai/a200-rt60 \u00b7 short": {
    "n": 59,
    "failed": 1,
    "stalls": 1,
    "rate": 0.01694915254237288,
    "wilson": [
     0.002998233885205405,
     0.08995726054929187
    ],
    "held": 0,
    "start_lag_q": [
     1131.1764,
     1651.497,
     2343.1946000000003,
     26560.559
    ]
   }
  },
  "2a_best": "openai/a20-rt60",
  "2a_by_location": {
   "openai/a20-rt60 @ Philadelphia (inferred: before location was recorded)": {
    "n": 65,
    "failed": 0,
    "stalls": 1,
    "rate": 0.015384615384615385,
    "wilson": [
     0.0027209445719428432,
     0.08213284813120839
    ],
    "held": 0,
    "start_lag_q": [
     1142.0292,
     1682.264,
     2395.013,
     5581.718
    ]
   },
   "openai/a20-rt60 @ Pittsburgh, home Wi-Fi": {
    "n": 55,
    "failed": 0,
    "stalls": 3,
    "rate": 0.05454545454545454,
    "wilson": [
     0.018723215408912114,
     0.1485306064439705
    ],
    "held": 1,
    "start_lag_q": [
     1144.2731999999999,
     1618.0465,
     2713.3926000000006,
     37990.91
    ]
   },
   "openai/a200-rt60 @ Philadelphia (inferred: before location was recorded)": {
    "n": 54,
    "failed": 1,
    "stalls": 1,
    "rate": 0.018518518518518517,
    "wilson": [
     0.0032764876819977723,
     0.0977143758659639
    ],
    "held": 0,
    "start_lag_q": [
     1192.38,
     1988.577,
     3585.911100000001,
     20708.512
    ]
   },
   "openai/a200-rt60 @ Pittsburgh, home Wi-Fi": {
    "n": 65,
    "failed": 0,
    "stalls": 4,
    "rate": 0.06153846153846154,
    "wilson": [
     0.024189054798281123,
     0.14782151955218903
    ],
    "held": 0,
    "start_lag_q": [
     1138.7686,
     1666.321,
     2863.6716,
     28757.109
    ]
   }
  },
  "2b": {
   "openai/a20-nb50": {
    "n": 60,
    "failed": 0,
    "stalls": 0,
    "rate": 0.0,
    "wilson": [
     0.0,
     0.060171853034227174
    ],
    "held": 0,
    "start_lag_q": [
     1159.4252999999999,
     1629.6550000000002,
     2393.9293000000002,
     3101.352
    ]
   },
   "openai/a20-nb50 \u00b7 medium": {
    "n": 30,
    "failed": 0,
    "stalls": 0,
    "rate": 0.0,
    "wilson": [
     0.0,
     0.11351339476145204
    ],
    "held": 0,
    "start_lag_q": [
     1240.8825,
     1703.877,
     2481.3807,
     3101.352
    ]
   },
   "openai/a20-nb50 \u00b7 short": {
    "n": 30,
    "failed": 0,
    "stalls": 0,
    "rate": 0.0,
    "wilson": [
     0.0,
     0.11351339476145204
    ],
    "held": 0,
    "start_lag_q": [
     1156.0937000000001,
     1581.4705,
     2018.8386,
     2803.75
    ]
   },
   "openai/a20-rt60": {
    "n": 60,
    "failed": 0,
    "stalls": 3,
    "rate": 0.05,
    "wilson": [
     0.01714950203090107,
     0.1370051656999034
    ],
    "held": 0,
    "start_lag_q": [
     1250.3904,
     1701.801,
     2507.1926,
     24573.974
    ]
   },
   "openai/a20-rt60 \u00b7 medium": {
    "n": 30,
    "failed": 0,
    "stalls": 2,
    "rate": 0.06666666666666667,
    "wilson": [
     0.018477023624516822,
     0.21323458516874166
    ],
    "held": 0,
    "start_lag_q": [
     1295.7509,
     1983.8115,
     3407.7587000000017,
     24573.974
    ]
   },
   "openai/a20-rt60 \u00b7 short": {
    "n": 30,
    "failed": 0,
    "stalls": 1,
    "rate": 0.03333333333333333,
    "wilson": [
     0.005908590316197976,
     0.16670391146115726
    ],
    "held": 0,
    "start_lag_q": [
     1209.5067999999999,
     1639.54,
     2437.0152000000003,
     2508.863
    ]
   },
   "openai/a20-zero": {
    "n": 60,
    "failed": 0,
    "stalls": 1,
    "rate": 0.016666666666666666,
    "wilson": [
     0.0029481597621904,
     0.08855129817089587
    ],
    "held": 0,
    "start_lag_q": [
     1242.2063,
     1632.8665,
     2499.5503000000003,
     8995.29
    ]
   },
   "openai/a20-zero \u00b7 medium": {
    "n": 30,
    "failed": 0,
    "stalls": 0,
    "rate": 0.0,
    "wilson": [
     0.0,
     0.11351339476145204
    ],
    "held": 0,
    "start_lag_q": [
     1258.0533,
     1690.159,
     2789.4308,
     3650.219
    ]
   },
   "openai/a20-zero \u00b7 short": {
    "n": 30,
    "failed": 0,
    "stalls": 1,
    "rate": 0.03333333333333333,
    "wilson": [
     0.005908590316197976,
     0.16670391146115726
    ],
    "held": 0,
    "start_lag_q": [
     1213.8885,
     1631.3029999999999,
     2093.6554000000006,
     8995.29
    ]
   }
  },
  "2b_best": "openai/a20-nb50"
 },
 "exp2c": {
  "summary": [
   {
    "system": "openai/best-a20-noise_bed_m50",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2081.496,
    "start_lag_ci": [
     1187.452,
     2457.057
    ],
    "end_arrival_med": 2645.732,
    "end_play_med": 2750.692,
    "end_play_ci": [
     2592.936,
     7097.137
    ],
    "end_play_p90": 6705.3778,
    "end_play_max": 7097.137,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.766,
    "chrf_med": 78.78
   },
   {
    "system": "openai/best-a20-noise_bed_m50",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2011.42,
    "start_lag_ci": [
     1560.527,
     3663.94
    ],
    "end_arrival_med": 2557.628,
    "end_play_med": 2874.922,
    "end_play_ci": [
     2472.93,
     8937.21
    ],
    "end_play_p90": 16836.892799999994,
    "end_play_max": 58628.245,
    "stalls": 3,
    "stall_rate": 0.2,
    "stall_ci": [
     0.0704754929173392,
     0.45185448924055305
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.817,
    "chrf_med": 81.59
   },
   {
    "system": "openai/best-a20-noise_bed_m50",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 1880.332,
    "start_lag_ci": [
     1590.724,
     2287.426
    ],
    "end_arrival_med": 2096.639,
    "end_play_med": 2428.887,
    "end_play_ci": [
     1895.582,
     2814.844
    ],
    "end_play_p90": 19803.441999999992,
    "end_play_max": 61163.701,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.886,
    "chrf_med": 82.11
   },
   {
    "system": "openai/best-a20-noise_bed_m50",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 1937.2075,
    "start_lag_ci": [
     1619.381,
     2278.5245
    ],
    "end_arrival_med": 2213.1555,
    "end_play_med": 2575.5495,
    "end_play_ci": [
     2436.7625,
     2951.6465
    ],
    "end_play_p90": 22274.829400000013,
    "end_play_max": 61163.701,
    "stalls": 3,
    "stall_rate": 0.1,
    "stall_ci": [
     0.03459988847741868,
     0.256210827331743
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.863,
    "chrf_med": 82.11
   },
   {
    "system": "openai/best-a20-noise_bed_m50",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 1902.056,
    "start_lag_ci": [
     1519.604,
     2327.964
    ],
    "end_arrival_med": 2523.304,
    "end_play_med": 2961.922,
    "end_play_ci": [
     2351.346,
     29649.186
    ],
    "end_play_p90": 32431.7986,
    "end_play_max": 33016.794,
    "stalls": 2,
    "stall_rate": 0.13333333333333333,
    "stall_ci": [
     0.03736125876428492,
     0.37881983053980567
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.181,
    "chrf_med": 73.68
   }
  ],
  "paired_vs_round1": {
   "openai/best-a20-noise_bed_m50": {
    "per_clip_ms": {
     "es_en_long_1": -2.3,
     "es_en_medium_1": 4786.2,
     "es_en_medium_2": -973.8,
     "es_en_medium_3": 350.0,
     "es_en_noisy_1": -161.3,
     "es_en_noisy_2": -378.6,
     "es_en_noisy_3": -332.5,
     "es_en_short_1": 650.4,
     "es_en_short_2": 29280.4,
     "es_en_short_3": 113.3
    },
    "median_ms": 55.5,
    "n_clips": 10,
    "n_faster": 5
   }
  }
 },
 "exp3": {
  "pooled": {
   "openai/flush-none-a20": {
    "n_runs": 50,
    "n_clips": 10,
    "n_values": 75,
    "catchup_median": 2517.167,
    "catchup_max": 21992.185,
    "catchup_over_4s": 2,
    "overlap_s_med": null,
    "median": 2630.392,
    "max": 26026.412,
    "over_4s": 5,
    "stalls": 0,
    "chrf": 80.285,
    "start": 1793.5529999999999,
    "rotations_med": 0.0,
    "premature_hesitation": 0,
    "hesitation_runs": 5,
    "drain_med": null,
    "sessions_med": 1.0,
    "audio_min": 11.18,
    "pass": false,
    "chrf_ok": true,
    "sessions_text_without_speech": 0,
    "pass_catchup": false
   },
   "openai/flush-pause400-a20": {
    "n_runs": 50,
    "n_clips": 10,
    "n_values": 74,
    "catchup_median": 2297.331,
    "catchup_max": 11038.406,
    "catchup_over_4s": 5,
    "overlap_s_med": 0.0,
    "median": 2732.776,
    "max": 13598.406,
    "over_4s": 7,
    "stalls": 0,
    "chrf": 81.375,
    "start": 1824.048,
    "rotations_med": 1.0,
    "premature_hesitation": 5,
    "hesitation_runs": 5,
    "drain_med": 4724.3,
    "sessions_med": 3.0,
    "audio_min": 11.18,
    "pass": false,
    "chrf_ok": true,
    "sessions_text_without_speech": 1,
    "pass_catchup": false
   },
   "openai/flush-pause600-a20": {
    "n_runs": 50,
    "n_clips": 10,
    "n_values": 75,
    "catchup_median": 2294.472,
    "catchup_max": 12553.664,
    "catchup_over_4s": 7,
    "overlap_s_med": 0.0,
    "median": 2688.158,
    "max": 15554.705,
    "over_4s": 9,
    "stalls": 0,
    "chrf": 83.04,
    "start": 1668.0665,
    "rotations_med": 1.0,
    "premature_hesitation": 0,
    "hesitation_runs": 5,
    "drain_med": 4753.65,
    "sessions_med": 3.0,
    "audio_min": 11.18,
    "pass": false,
    "chrf_ok": true,
    "sessions_text_without_speech": 1,
    "pass_catchup": false
   },
   "openai/flush-pause800-a20": {
    "n_runs": 50,
    "n_clips": 10,
    "n_values": 73,
    "catchup_median": 2436.099,
    "catchup_max": 7616.015,
    "catchup_over_4s": 3,
    "overlap_s_med": 0.0,
    "median": 2748.587,
    "max": 7908.968,
    "over_4s": 4,
    "stalls": 1,
    "chrf": 83.04,
    "start": 1805.49,
    "rotations_med": 1.0,
    "premature_hesitation": 0,
    "hesitation_runs": 5,
    "drain_med": 4588.6,
    "sessions_med": 3.0,
    "audio_min": 11.18,
    "pass": false,
    "chrf_ok": true,
    "sessions_text_without_speech": 1,
    "pass_catchup": false
   },
   "openai/flush-watchdog1200-a20": {
    "n_runs": 50,
    "n_clips": 10,
    "n_values": 75,
    "catchup_median": 2560.494,
    "catchup_max": 21821.22,
    "catchup_over_4s": 2,
    "overlap_s_med": 0.1305,
    "median": 2591.832,
    "max": 25743.818,
    "over_4s": 6,
    "stalls": 0,
    "chrf": 83.99,
    "start": 1809.5275000000001,
    "rotations_med": 0.0,
    "premature_hesitation": 2,
    "hesitation_runs": 5,
    "drain_med": 5751.549999999999,
    "sessions_med": 2.0,
    "audio_min": 11.18,
    "pass": false,
    "chrf_ok": true,
    "sessions_text_without_speech": 0,
    "pass_catchup": false
   }
  },
  "control": "openai/flush-none-a20",
  "paired_vs_round1": {
   "openai/flush-none-a20": {
    "per_clip_ms": {
     "es_en_long_1": 382.9,
     "es_en_medium_1": 313.3,
     "es_en_medium_2": -848.9,
     "es_en_medium_3": -131.0,
     "es_en_noisy_1": 209.4,
     "es_en_noisy_2": -274.6,
     "es_en_noisy_3": -430.2
    },
    "median_ms": -131.0,
    "n_clips": 7,
    "n_faster": 4
   },
   "openai/flush-pause400-a20": {
    "per_clip_ms": {
     "es_en_long_1": 436.8,
     "es_en_medium_1": 306.9,
     "es_en_medium_2": -686.6,
     "es_en_medium_3": 272.8,
     "es_en_noisy_1": -48.4,
     "es_en_noisy_2": -228.8,
     "es_en_noisy_3": -6.1
    },
    "median_ms": -6.1,
    "n_clips": 7,
    "n_faster": 4
   },
   "openai/flush-pause600-a20": {
    "per_clip_ms": {
     "es_en_long_1": 563.1,
     "es_en_medium_1": -592.2,
     "es_en_medium_2": -1095.4,
     "es_en_medium_3": 584.7,
     "es_en_noisy_1": -388.8,
     "es_en_noisy_2": -315.4,
     "es_en_noisy_3": -113.9
    },
    "median_ms": -315.4,
    "n_clips": 7,
    "n_faster": 5
   },
   "openai/flush-pause800-a20": {
    "per_clip_ms": {
     "es_en_long_1": 240.4,
     "es_en_medium_1": 43.8,
     "es_en_medium_2": -346.0,
     "es_en_medium_3": 65.9,
     "es_en_noisy_1": 34.5,
     "es_en_noisy_2": -201.4,
     "es_en_noisy_3": 110.1
    },
    "median_ms": 43.8,
    "n_clips": 7,
    "n_faster": 2
   },
   "openai/flush-watchdog1200-a20": {
    "per_clip_ms": {
     "es_en_long_1": -21.0,
     "es_en_medium_1": 212.4,
     "es_en_medium_2": -1123.6,
     "es_en_medium_3": 15.8,
     "es_en_noisy_1": -295.6,
     "es_en_noisy_2": -734.8,
     "es_en_noisy_3": 1.8
    },
    "median_ms": -21.0,
    "n_clips": 7,
    "n_faster": 4
   }
  }
 },
 "exp4": {
  "summary": [
   {
    "system": "gemini",
    "category": "hesitation",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2997.837,
    "start_lag_ci": [
     2910.509,
     3400.828
    ],
    "end_arrival_med": 2906.383,
    "end_play_med": 3246.2435,
    "end_play_ci": [
     3240.5299999999997,
     3430.9925000000003
    ],
    "end_play_p90": 3368.4155,
    "end_play_max": 3430.9925000000003,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.033,
    "chrf_med": 91.14
   },
   {
    "system": "gemini",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3028.644,
    "start_lag_ci": [
     2978.02,
     3478.597
    ],
    "end_arrival_med": 2462.929,
    "end_play_med": 2868.755,
    "end_play_ci": [
     2612.501,
     3032.002
    ],
    "end_play_p90": 3021.164,
    "end_play_max": 3032.002,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.902,
    "chrf_med": 82.91
   },
   {
    "system": "gemini",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 3037.081,
    "start_lag_ci": [
     2913.891,
     3093.976
    ],
    "end_arrival_med": 2576.915,
    "end_play_med": 3099.678,
    "end_play_ci": [
     2789.397,
     3167.682
    ],
    "end_play_p90": 3228.386,
    "end_play_max": 3330.617,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.92,
    "chrf_med": 90.36
   },
   {
    "system": "gemini",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2973.899,
    "start_lag_ci": [
     2930.774,
     3065.945
    ],
    "end_arrival_med": 2396.852,
    "end_play_med": 2834.57,
    "end_play_ci": [
     2718.036,
     3038.68
    ],
    "end_play_p90": 3125.8361999999997,
    "end_play_max": 3260.19,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.898,
    "chrf_med": 93.22
   },
   {
    "system": "gemini",
    "category": "same_language",
    "n_runs": 10,
    "n_clips": 2,
    "n_failed": 0,
    "start_lag_med": 2933.966,
    "start_lag_ci": [
     2847.557,
     2997.321
    ],
    "end_arrival_med": 1657.0545,
    "end_play_med": 2064.4455,
    "end_play_ci": [
     1562.648,
     2385.139
    ],
    "end_play_p90": 2412.9169,
    "end_play_max": 2477.732,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.2775328030260577
    ],
    "held_until_flush": 0,
    "ratio_span_med": null,
    "chrf_med": null
   },
   {
    "system": "gemini",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 3007.2405,
    "start_lag_ci": [
     2961.709,
     3055.1865
    ],
    "end_arrival_med": 2435.067,
    "end_play_med": 2869.6105,
    "end_play_ci": [
     2795.2075,
     3073.5555
    ],
    "end_play_p90": 3186.3135,
    "end_play_max": 3330.617,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.11351339476145204
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.9125000000000001,
    "chrf_med": 90.36
   },
   {
    "system": "gemini",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2987.604,
    "start_lag_ci": [
     2936.606,
     3071.537
    ],
    "end_arrival_med": 3158.818,
    "end_play_med": 3576.043,
    "end_play_ci": [
     3405.307,
     3611.407
    ],
    "end_play_p90": 3736.415,
    "end_play_max": 3772.408,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.097,
    "chrf_med": 82.1
   },
   {
    "system": "gemini",
    "category": "silence",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": null,
    "start_lag_ci": null,
    "end_arrival_med": null,
    "end_play_med": null,
    "end_play_ci": null,
    "end_play_p90": null,
    "end_play_max": null,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": null,
    "chrf_med": null
   },
   {
    "system": "gemini",
    "category": "three_turns",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2920.55,
    "start_lag_ci": [
     2808.194,
     2961.275
    ],
    "end_arrival_med": 2417.257,
    "end_play_med": 2886.575,
    "end_play_ci": [
     2746.1059999999998,
     3056.099
    ],
    "end_play_p90": 3020.609,
    "end_play_max": 3056.099,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.686,
    "chrf_med": 65.7
   },
   {
    "system": "gemini",
    "category": "two_utterances",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3028.393,
    "start_lag_ci": [
     2892.519,
     3131.065
    ],
    "end_arrival_med": 2245.289,
    "end_play_med": 3410.933,
    "end_play_ci": [
     3296.2785000000003,
     3690.4694999999997
    ],
    "end_play_p90": 3590.0099,
    "end_play_max": 3690.4694999999997,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.774,
    "chrf_med": 91.49
   },
   {
    "system": "openai/none-control",
    "category": "hesitation",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2004.967,
    "start_lag_ci": [
     1285.683,
     2088.314
    ],
    "end_arrival_med": 2442.083,
    "end_play_med": 2700.9205,
    "end_play_ci": [
     2672.9815,
     2888.2844999999998
    ],
    "end_play_p90": 2826.5253,
    "end_play_max": 2888.2844999999998,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.869,
    "chrf_med": 81.09
   },
   {
    "system": "openai/none-control",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 1895.264,
    "start_lag_ci": [
     1586.998,
     20413.852
    ],
    "end_arrival_med": 2556.666,
    "end_play_med": 3041.917,
    "end_play_ci": [
     2799.949,
     19298.369
    ],
    "end_play_p90": 12889.237400000002,
    "end_play_max": 19298.369,
    "stalls": 1,
    "stall_rate": 0.2,
    "stall_ci": [
     0.03622410822563066,
     0.6244653729700049
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.76,
    "chrf_med": 76.02
   },
   {
    "system": "openai/none-control",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 1708.976,
    "start_lag_ci": [
     1564.335,
     2261.664
    ],
    "end_arrival_med": 2135.126,
    "end_play_med": 2340.261,
    "end_play_ci": [
     2033.842,
     2808.658
    ],
    "end_play_p90": 8375.920799999996,
    "end_play_max": 25468.941,
    "stalls": 1,
    "stall_rate": 0.06666666666666667,
    "stall_ci": [
     0.01186689536153232,
     0.2981653010887565
    ],
    "held_until_flush": 1,
    "ratio_span_med": 0.879,
    "chrf_med": 90.36
   },
   {
    "system": "openai/none-control",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 1678.995,
    "start_lag_ci": [
     1627.251,
     2276.91
    ],
    "end_arrival_med": 2236.743,
    "end_play_med": 2618.978,
    "end_play_ci": [
     2169.176,
     2968.15
    ],
    "end_play_p90": 3545.4604,
    "end_play_max": 27055.686,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.893,
    "chrf_med": 84.93
   },
   {
    "system": "openai/none-control",
    "category": "same_language",
    "n_runs": 10,
    "n_clips": 2,
    "n_failed": 0,
    "start_lag_med": null,
    "start_lag_ci": null,
    "end_arrival_med": null,
    "end_play_med": null,
    "end_play_ci": null,
    "end_play_p90": null,
    "end_play_max": null,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.2775328030260577
    ],
    "held_until_flush": 0,
    "ratio_span_med": null,
    "chrf_med": null
   },
   {
    "system": "openai/none-control",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 1693.9855,
    "start_lag_ci": [
     1629.2555,
     2024.8890000000001
    ],
    "end_arrival_med": 2158.584,
    "end_play_med": 2488.0785,
    "end_play_ci": [
     2251.809,
     2707.8605
    ],
    "end_play_p90": 4426.875400000012,
    "end_play_max": 27055.686,
    "stalls": 1,
    "stall_rate": 0.03333333333333333,
    "stall_ci": [
     0.005908590316197976,
     0.16670391146115726
    ],
    "held_until_flush": 1,
    "ratio_span_med": 0.886,
    "chrf_med": 88.08500000000001
   },
   {
    "system": "openai/none-control",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 1347.372,
    "start_lag_ci": [
     1166.42,
     2049.644
    ],
    "end_arrival_med": 1706.342,
    "end_play_med": 2225.002,
    "end_play_ci": [
     1930.868,
     2621.306
    ],
    "end_play_p90": 3101.3938,
    "end_play_max": 3826.702,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.252,
    "chrf_med": 75.12
   },
   {
    "system": "openai/none-control",
    "category": "silence",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": null,
    "start_lag_ci": null,
    "end_arrival_med": null,
    "end_play_med": null,
    "end_play_ci": null,
    "end_play_p90": null,
    "end_play_max": null,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": null,
    "chrf_med": null
   },
   {
    "system": "openai/none-control",
    "category": "three_turns",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 1743.339,
    "start_lag_ci": [
     1109.127,
     2054.202
    ],
    "end_arrival_med": 1886.069,
    "end_play_med": 2541.665,
    "end_play_ci": [
     2361.9395,
     2789.632
    ],
    "end_play_p90": 2705.8376,
    "end_play_max": 2789.632,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.514,
    "chrf_med": 59.55
   },
   {
    "system": "openai/none-control",
    "category": "two_utterances",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 1791.613,
    "start_lag_ci": [
     697.098,
     2155.082
    ],
    "end_arrival_med": 1360.747,
    "end_play_med": 2961.546,
    "end_play_ci": [
     2681.632,
     3244.9790000000003
    ],
    "end_play_p90": 3173.5988,
    "end_play_max": 3244.9790000000003,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.735,
    "chrf_med": 85.47
   }
  ]
 },
 "exp5": {
  "5a": [
   {
    "system": "cascade/baseline",
    "category": "hesitation",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3276.433,
    "start_lag_ci": [
     3131.532,
     3572.053
    ],
    "end_arrival_med": 1763.151,
    "end_play_med": 3704.6065,
    "end_play_ci": [
     3491.031,
     3949.6595
    ],
    "end_play_p90": 3854.4885,
    "end_play_max": 3949.6595,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.011,
    "chrf_med": 91.14
   },
   {
    "system": "cascade/baseline",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2537.583,
    "start_lag_ci": [
     2407.19,
     2766.301
    ],
    "end_arrival_med": 1482.893,
    "end_play_med": 11348.703,
    "end_play_ci": [
     10918.581,
     11961.988
    ],
    "end_play_p90": 11805.3252,
    "end_play_max": 11961.988,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.214,
    "chrf_med": 80.24
   },
   {
    "system": "cascade/baseline",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2606.733,
    "start_lag_ci": [
     2559.577,
     3067.268
    ],
    "end_arrival_med": 972.843,
    "end_play_med": 3743.166,
    "end_play_ci": [
     3337.728,
     3898.78
    ],
    "end_play_p90": 4031.4694,
    "end_play_max": 4090.899,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.019,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/baseline",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2591.659,
    "start_lag_ci": [
     2475.14,
     3074.041
    ],
    "end_arrival_med": 978.265,
    "end_play_med": 3762.137,
    "end_play_ci": [
     3425.802,
     3903.128
    ],
    "end_play_p90": 4062.307,
    "end_play_max": 4244.337,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.055,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/baseline",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 2600.7434999999996,
    "start_lag_ci": [
     2563.1887,
     2846.3345
    ],
    "end_arrival_med": 974.9925000000001,
    "end_play_med": 3752.6515,
    "end_play_ci": [
     3440.2005,
     3857.3509999999997
    ],
    "end_play_p90": 4038.1295,
    "end_play_max": 4244.337,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.11351339476145204
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.0415,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/baseline",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2168.171,
    "start_lag_ci": [
     1856.607,
     2590.531
    ],
    "end_arrival_med": 1211.723,
    "end_play_med": 2810.025,
    "end_play_ci": [
     2232.775,
     2888.799
    ],
    "end_play_p90": 2959.6994,
    "end_play_max": 3075.034,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.146,
    "chrf_med": 100.0
   },
   {
    "system": "cascade/baseline",
    "category": "three_turns",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3629.219,
    "start_lag_ci": [
     3535.537,
     3806.195
    ],
    "end_arrival_med": 1393.541,
    "end_play_med": 4416.1605,
    "end_play_ci": [
     3917.4195,
     4758.715
    ],
    "end_play_p90": 4645.3279999999995,
    "end_play_max": 4758.715,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.743,
    "chrf_med": 67.26
   },
   {
    "system": "cascade/baseline",
    "category": "two_utterances",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2180.223,
    "start_lag_ci": [
     2034.343,
     2289.751
    ],
    "end_arrival_med": 997.131,
    "end_play_med": 3117.495,
    "end_play_ci": [
     3045.4525000000003,
     3179.701
    ],
    "end_play_p90": 3164.5856,
    "end_play_max": 3179.701,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.846,
    "chrf_med": 89.79
   },
   {
    "system": "cascade/single-flush",
    "category": "hesitation",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3212.007,
    "start_lag_ci": [
     3142.568,
     3216.462
    ],
    "end_arrival_med": 1888.433,
    "end_play_med": 3458.877,
    "end_play_ci": [
     3385.1475,
     3625.435
    ],
    "end_play_p90": 3572.034,
    "end_play_max": 3625.435,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.942,
    "chrf_med": 91.14
   },
   {
    "system": "cascade/single-flush",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2565.957,
    "start_lag_ci": [
     2439.839,
     2654.349
    ],
    "end_arrival_med": 1416.272,
    "end_play_med": 9644.421,
    "end_play_ci": [
     9226.251,
     10783.162
    ],
    "end_play_p90": 10450.1664,
    "end_play_max": 10783.162,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.089,
    "chrf_med": 81.6
   },
   {
    "system": "cascade/single-flush",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2650.099,
    "start_lag_ci": [
     2544.547,
     3077.379
    ],
    "end_arrival_med": 1054.349,
    "end_play_med": 3633.927,
    "end_play_ci": [
     3468.496,
     4215.159
    ],
    "end_play_p90": 4269.3256,
    "end_play_max": 4337.673,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.982,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2650.044,
    "start_lag_ci": [
     2540.931,
     3086.02
    ],
    "end_arrival_med": 974.746,
    "end_play_med": 3674.203,
    "end_play_ci": [
     3340.845,
     4043.832
    ],
    "end_play_p90": 4803.344999999999,
    "end_play_max": 5799.58,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.996,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 2650.0715,
    "start_lag_ci": [
     2594.575,
     3009.0855
    ],
    "end_arrival_med": 1006.623,
    "end_play_med": 3654.065,
    "end_play_ci": [
     3449.7655,
     4022.4975
    ],
    "end_play_p90": 4279.4565,
    "end_play_max": 5799.58,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.11351339476145204
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.989,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2124.574,
    "start_lag_ci": [
     1926.36,
     2625.69
    ],
    "end_arrival_med": 1097.751,
    "end_play_med": 2716.425,
    "end_play_ci": [
     2269.457,
     2882.236
    ],
    "end_play_p90": 2923.4346,
    "end_play_max": 3145.63,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.07,
    "chrf_med": 100.0
   },
   {
    "system": "cascade/single-flush",
    "category": "three_turns",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3613.682,
    "start_lag_ci": [
     3570.737,
     3787.086
    ],
    "end_arrival_med": 1382.336,
    "end_play_med": 4249.916499999999,
    "end_play_ci": [
     3526.1805000000004,
     4460.8195
    ],
    "end_play_p90": 4393.8922999999995,
    "end_play_max": 4460.8195,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.674,
    "chrf_med": 66.63
   },
   {
    "system": "cascade/single-flush",
    "category": "two_utterances",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2165.665,
    "start_lag_ci": [
     2087.616,
     2499.73
    ],
    "end_arrival_med": 942.523,
    "end_play_med": 3238.1385,
    "end_play_ci": [
     2925.3695,
     3609.0375
    ],
    "end_play_p90": 3480.3718999999996,
    "end_play_max": 3609.0375,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.845,
    "chrf_med": 89.79
   },
   {
    "system": "cascade/single-mbd0",
    "category": "hesitation",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3200.356,
    "start_lag_ci": [
     3058.785,
     3452.583
    ],
    "end_arrival_med": 2056.718,
    "end_play_med": 7310.794,
    "end_play_ci": [
     7047.3715,
     7788.152
    ],
    "end_play_p90": 7664.982599999999,
    "end_play_max": 7788.152,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.743,
    "chrf_med": 91.14
   },
   {
    "system": "cascade/single-mbd0",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2538.88,
    "start_lag_ci": [
     2390.525,
     2546.293
    ],
    "end_arrival_med": 1612.35,
    "end_play_med": 42203.409,
    "end_play_ci": [
     39423.361,
     49706.126
    ],
    "end_play_p90": 47723.905999999995,
    "end_play_max": 49706.126,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.939,
    "chrf_med": 73.5
   },
   {
    "system": "cascade/single-mbd0",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2628.981,
    "start_lag_ci": [
     2462.153,
     3036.284
    ],
    "end_arrival_med": 1231.977,
    "end_play_med": 10043.872,
    "end_play_ci": [
     9613.153,
     11255.006
    ],
    "end_play_p90": 12434.934,
    "end_play_max": 12596.386,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.741,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-mbd0",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2678.082,
    "start_lag_ci": [
     2527.898,
     2964.703
    ],
    "end_arrival_med": 1177.105,
    "end_play_med": 9751.56,
    "end_play_ci": [
     9371.113,
     11897.001
    ],
    "end_play_p90": 12101.751600000001,
    "end_play_max": 12325.054,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.747,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-mbd0",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 2632.2405,
    "start_lag_ci": [
     2537.4719999999998,
     2958.7595
    ],
    "end_arrival_med": 1220.3065000000001,
    "end_play_med": 9986.1075,
    "end_play_ci": [
     9517.799500000001,
     11058.233499999998
    ],
    "end_play_p90": 12330.5466,
    "end_play_max": 12596.386,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.11351339476145204
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.7440000000000002,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-mbd0",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2084.427,
    "start_lag_ci": [
     1787.959,
     2581.984
    ],
    "end_arrival_med": 1148.276,
    "end_play_med": 4954.449,
    "end_play_ci": [
     4604.974,
     5164.143
    ],
    "end_play_p90": 6673.187999999999,
    "end_play_max": 6809.14,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 2.217,
    "chrf_med": 100.0
   },
   {
    "system": "cascade/single-mbd0",
    "category": "three_turns",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3587.466,
    "start_lag_ci": [
     3493.06,
     3744.377
    ],
    "end_arrival_med": 1739.906,
    "end_play_med": 8132.312,
    "end_play_ci": [
     7770.31,
     9779.109499999999
    ],
    "end_play_p90": 9578.331699999999,
    "end_play_max": 9779.109499999999,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.118,
    "chrf_med": 49.17
   },
   {
    "system": "cascade/single-mbd0",
    "category": "two_utterances",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2132.887,
    "start_lag_ci": [
     2076.211,
     2184.747
    ],
    "end_arrival_med": 1039.187,
    "end_play_med": 11140.444,
    "end_play_ci": [
     10663.3215,
     11757.9195
    ],
    "end_play_p90": 11592.8269,
    "end_play_max": 11757.9195,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.739,
    "chrf_med": 85.12
   },
   {
    "system": "cascade/single-mbd200",
    "category": "hesitation",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3351.957,
    "start_lag_ci": [
     3324.657,
     3458.321
    ],
    "end_arrival_med": 1932.971,
    "end_play_med": 4158.5795,
    "end_play_ci": [
     3977.5209999999997,
     4254.9169999999995
    ],
    "end_play_p90": 4243.2024,
    "end_play_max": 4254.9169999999995,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.058,
    "chrf_med": 91.14
   },
   {
    "system": "cascade/single-mbd200",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2726.14,
    "start_lag_ci": [
     2648.582,
     4190.345
    ],
    "end_arrival_med": 1710.204,
    "end_play_med": 10409.048,
    "end_play_ci": [
     9322.939,
     11083.813
    ],
    "end_play_p90": 10929.359,
    "end_play_max": 11083.813,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.131,
    "chrf_med": 81.83
   },
   {
    "system": "cascade/single-mbd200",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2848.669,
    "start_lag_ci": [
     2742.23,
     3223.638
    ],
    "end_arrival_med": 1137.533,
    "end_play_med": 4194.51,
    "end_play_ci": [
     3220.632,
     4329.969
    ],
    "end_play_p90": 4481.000599999999,
    "end_play_max": 4611.301,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.039,
    "chrf_med": 74.02
   },
   {
    "system": "cascade/single-mbd200",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2780.269,
    "start_lag_ci": [
     2636.945,
     3253.005
    ],
    "end_arrival_med": 1000.427,
    "end_play_med": 4002.615,
    "end_play_ci": [
     3410.302,
     4401.619
    ],
    "end_play_p90": 4494.1194,
    "end_play_max": 9067.144,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.014,
    "chrf_med": 74.02
   },
   {
    "system": "cascade/single-mbd200",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 2840.633,
    "start_lag_ci": [
     2723.197,
     3047.0969999999998
    ],
    "end_arrival_med": 1037.58,
    "end_play_med": 4145.773,
    "end_play_ci": [
     3644.5285000000003,
     4272.879
    ],
    "end_play_p90": 4505.322999999999,
    "end_play_max": 9067.144,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.11351339476145204
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.032,
    "chrf_med": 74.02
   },
   {
    "system": "cascade/single-mbd200",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2223.626,
    "start_lag_ci": [
     1726.806,
     2662.661
    ],
    "end_arrival_med": 1079.219,
    "end_play_med": 2816.152,
    "end_play_ci": [
     2094.894,
     2924.363
    ],
    "end_play_p90": 2959.7958,
    "end_play_max": 2988.453,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.13,
    "chrf_med": 100.0
   },
   {
    "system": "cascade/single-mbd200",
    "category": "three_turns",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3629.277,
    "start_lag_ci": [
     3509.805,
     3731.669
    ],
    "end_arrival_med": 1213.828,
    "end_play_med": 3923.73,
    "end_play_ci": [
     3261.5204999999996,
     3976.5805
    ],
    "end_play_p90": 3962.7003,
    "end_play_max": 3976.5805,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.679,
    "chrf_med": 64.32
   },
   {
    "system": "cascade/single-mbd200",
    "category": "two_utterances",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2203.0,
    "start_lag_ci": [
     2074.806,
     2334.579
    ],
    "end_arrival_med": 1003.627,
    "end_play_med": 3139.8639999999996,
    "end_play_ci": [
     2969.6405,
     3166.194
    ],
    "end_play_p90": 3163.2626,
    "end_play_max": 3166.194,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.827,
    "chrf_med": 89.79
   }
  ],
  "5a_paired": {
   "cascade/baseline": {
    "per_clip_ms": {
     "es_en_long_1": -292.6,
     "es_en_medium_1": 9.0,
     "es_en_medium_2": -97.5,
     "es_en_medium_3": -28.0,
     "es_en_noisy_1": 61.0,
     "es_en_noisy_2": -142.4,
     "es_en_noisy_3": 155.6,
     "es_en_short_1": 16.2,
     "es_en_short_2": -192.0,
     "es_en_short_3": -84.8
    },
    "median_ms": -56.4,
    "n_clips": 10,
    "n_faster": 6
   },
   "cascade/single-flush": {
    "per_clip_ms": {
     "es_en_long_1": -1996.9,
     "es_en_medium_1": 241.7,
     "es_en_medium_2": -15.8,
     "es_en_medium_3": -137.2,
     "es_en_noisy_1": 110.0,
     "es_en_noisy_2": -309.1,
     "es_en_noisy_3": 67.7,
     "es_en_short_1": -145.1,
     "es_en_short_2": -149.3,
     "es_en_short_3": -20.3
    },
    "median_ms": -78.75,
    "n_clips": 10,
    "n_faster": 7
   },
   "cascade/single-mbd0": {
    "per_clip_ms": {
     "es_en_long_1": 30562.1,
     "es_en_medium_1": 8357.8,
     "es_en_medium_2": 5899.3,
     "es_en_medium_3": 6272.7,
     "es_en_noisy_1": 7947.5,
     "es_en_noisy_2": 6004.7,
     "es_en_noisy_3": 5764.6,
     "es_en_short_1": 2184.0,
     "es_en_short_2": 2378.6,
     "es_en_short_3": 3578.5
    },
    "median_ms": 5952.0,
    "n_clips": 10,
    "n_faster": 0
   },
   "cascade/single-mbd200": {
    "per_clip_ms": {
     "es_en_long_1": -1232.2,
     "es_en_medium_1": 427.8,
     "es_en_medium_2": -213.7,
     "es_en_medium_3": 492.1,
     "es_en_noisy_1": 429.3,
     "es_en_noisy_2": -227.7,
     "es_en_noisy_3": 393.0,
     "es_en_short_1": 66.6,
     "es_en_short_2": -295.3,
     "es_en_short_3": -39.5
    },
    "median_ms": 13.549999999999997,
    "n_clips": 10,
    "n_faster": 5
   }
  },
  "5b": [
   {
    "system": "cascade/single-flush",
    "category": "hesitation",
    "n_runs": 4,
    "n_clips": 1,
    "n_failed": 1,
    "start_lag_med": 3254.6835,
    "start_lag_ci": [
     3177.01,
     3518.622
    ],
    "end_arrival_med": 1874.295,
    "end_play_med": 3538.05975,
    "end_play_ci": [
     3427.523,
     3772.1805
    ],
    "end_play_p90": 3702.7443,
    "end_play_max": 3772.1805,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.4898908403969213
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.9564999999999999,
    "chrf_med": 91.14
   },
   {
    "system": "cascade/single-flush",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2545.257,
    "start_lag_ci": [
     2521.482,
     2678.894
    ],
    "end_arrival_med": 1454.931,
    "end_play_med": 10222.295,
    "end_play_ci": [
     8872.217,
     12316.499
    ],
    "end_play_p90": 11533.3318,
    "end_play_max": 12316.499,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.129,
    "chrf_med": 79.77
   },
   {
    "system": "cascade/single-flush",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2632.878,
    "start_lag_ci": [
     2587.057,
     3031.727
    ],
    "end_arrival_med": 1052.636,
    "end_play_med": 3643.893,
    "end_play_ci": [
     3419.316,
     4009.345
    ],
    "end_play_p90": 4054.4344,
    "end_play_max": 4176.043,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.994,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2641.339,
    "start_lag_ci": [
     2574.811,
     3168.371
    ],
    "end_arrival_med": 929.904,
    "end_play_med": 3586.377,
    "end_play_ci": [
     3296.378,
     3969.026
    ],
    "end_play_p90": 4020.3364,
    "end_play_max": 4045.069,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.998,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 2640.4970000000003,
    "start_lag_ci": [
     2603.5640000000003,
     2828.667
    ],
    "end_arrival_med": 1024.853,
    "end_play_med": 3593.5885,
    "end_play_ci": [
     3422.0,
     3739.951
    ],
    "end_play_p90": 4028.3713000000002,
    "end_play_max": 4176.043,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.11351339476145204
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.996,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2145.095,
    "start_lag_ci": [
     1836.346,
     2662.8
    ],
    "end_arrival_med": 1135.174,
    "end_play_med": 2857.78,
    "end_play_ci": [
     2165.256,
     2949.349
    ],
    "end_play_p90": 2984.9658,
    "end_play_max": 3014.475,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.129,
    "chrf_med": 100.0
   },
   {
    "system": "cascade/single-flush",
    "category": "three_turns",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3697.996,
    "start_lag_ci": [
     3691.567,
     3734.542
    ],
    "end_arrival_med": 1405.803,
    "end_play_med": 4080.009,
    "end_play_ci": [
     4025.5945,
     4434.0785
    ],
    "end_play_p90": 4350.860699999999,
    "end_play_max": 4434.0785,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.69,
    "chrf_med": 67.26
   },
   {
    "system": "cascade/single-flush",
    "category": "two_utterances",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2193.614,
    "start_lag_ci": [
     2103.474,
     2707.059
    ],
    "end_arrival_med": 1028.417,
    "end_play_med": 3093.3875,
    "end_play_ci": [
     2954.768,
     3561.9094999999998
    ],
    "end_play_p90": 3385.5690999999997,
    "end_play_max": 3561.9094999999998,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.829,
    "chrf_med": 89.79
   },
   {
    "system": "cascade/single-flush+concise",
    "category": "hesitation",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3384.029,
    "start_lag_ci": [
     3341.856,
     3471.584
    ],
    "end_arrival_med": 1850.88,
    "end_play_med": 3693.9285,
    "end_play_ci": [
     3460.3885,
     4121.017
    ],
    "end_play_p90": 3967.2616,
    "end_play_max": 4121.017,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.948,
    "chrf_med": 85.51
   },
   {
    "system": "cascade/single-flush+concise",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2530.071,
    "start_lag_ci": [
     2487.181,
     2584.003
    ],
    "end_arrival_med": 1465.645,
    "end_play_med": 9120.169,
    "end_play_ci": [
     8349.187,
     10457.719
    ],
    "end_play_p90": 10134.297799999998,
    "end_play_max": 10457.719,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.059,
    "chrf_med": 79.89
   },
   {
    "system": "cascade/single-flush+concise",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2568.731,
    "start_lag_ci": [
     2481.378,
     3096.987
    ],
    "end_arrival_med": 1020.96,
    "end_play_med": 3391.309,
    "end_play_ci": [
     3183.009,
     3624.041
    ],
    "end_play_p90": 3732.7095999999997,
    "end_play_max": 4467.627,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.936,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush+concise",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2584.342,
    "start_lag_ci": [
     2556.494,
     3102.101
    ],
    "end_arrival_med": 1017.983,
    "end_play_med": 3531.81,
    "end_play_ci": [
     3332.217,
     3884.351
    ],
    "end_play_p90": 4676.5232,
    "end_play_max": 4723.485,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.91,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush+concise",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 2582.556,
    "start_lag_ci": [
     2520.947,
     2908.1130000000003
    ],
    "end_arrival_med": 1019.4715,
    "end_play_med": 3499.6620000000003,
    "end_play_ci": [
     3328.0744999999997,
     3630.114
    ],
    "end_play_p90": 4483.4516,
    "end_play_max": 4723.485,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.11351339476145204
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.9345000000000001,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush+concise",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2180.639,
    "start_lag_ci": [
     1880.021,
     2978.783
    ],
    "end_arrival_med": 1368.323,
    "end_play_med": 2924.026,
    "end_play_ci": [
     2247.907,
     3229.447
    ],
    "end_play_p90": 3250.7802,
    "end_play_max": 3589.172,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.126,
    "chrf_med": 100.0
   },
   {
    "system": "cascade/single-flush+concise",
    "category": "three_turns",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3719.66,
    "start_lag_ci": [
     3638.647,
     4042.66
    ],
    "end_arrival_med": 1444.235,
    "end_play_med": 4613.880999999999,
    "end_play_ci": [
     4448.56,
     4857.934
    ],
    "end_play_p90": 4830.9412,
    "end_play_max": 4857.934,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.75,
    "chrf_med": 63.99
   },
   {
    "system": "cascade/single-flush+concise",
    "category": "two_utterances",
    "n_runs": 4,
    "n_clips": 1,
    "n_failed": 1,
    "start_lag_med": 2066.3065,
    "start_lag_ci": [
     1955.488,
     2115.32
    ],
    "end_arrival_med": 1067.251,
    "end_play_med": 2898.70225,
    "end_play_ci": [
     2751.8595,
     2938.294
    ],
    "end_play_p90": 2929.0393,
    "end_play_max": 2938.294,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.4898908403969213
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.7985,
    "chrf_med": 89.14
   },
   {
    "system": "cascade/single-flush+concise+adaptive",
    "category": "hesitation",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3370.71,
    "start_lag_ci": [
     3361.291,
     3444.983
    ],
    "end_arrival_med": 1834.565,
    "end_play_med": 3633.8245,
    "end_play_ci": [
     3630.133,
     3908.881
    ],
    "end_play_p90": 3859.845,
    "end_play_max": 3908.881,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.964,
    "chrf_med": 91.14
   },
   {
    "system": "cascade/single-flush+concise+adaptive",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2544.411,
    "start_lag_ci": [
     2409.037,
     2811.896
    ],
    "end_arrival_med": 1499.503,
    "end_play_med": 9546.158,
    "end_play_ci": [
     8394.843,
     10928.752
    ],
    "end_play_p90": 10401.2484,
    "end_play_max": 10928.752,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.088,
    "chrf_med": 80.72
   },
   {
    "system": "cascade/single-flush+concise+adaptive",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2511.887,
    "start_lag_ci": [
     2434.397,
     3072.474
    ],
    "end_arrival_med": 1085.634,
    "end_play_med": 3801.269,
    "end_play_ci": [
     3343.89,
     4329.842
    ],
    "end_play_p90": 4463.406,
    "end_play_max": 4773.222,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.006,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush+concise+adaptive",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2637.042,
    "start_lag_ci": [
     2537.459,
     2996.88
    ],
    "end_arrival_med": 921.228,
    "end_play_med": 3453.898,
    "end_play_ci": [
     3256.286,
     3637.385
    ],
    "end_play_p90": 4076.7064,
    "end_play_max": 4581.532,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.944,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush+concise+adaptive",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 2547.0029999999997,
    "start_lag_ci": [
     2515.463,
     2883.31
    ],
    "end_arrival_med": 1013.015,
    "end_play_med": 3551.8720000000003,
    "end_play_ci": [
     3332.466,
     3736.0125
    ],
    "end_play_p90": 4439.336,
    "end_play_max": 4773.222,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.11351339476145204
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.963,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/single-flush+concise+adaptive",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2228.477,
    "start_lag_ci": [
     2043.601,
     3008.006
    ],
    "end_arrival_med": 1315.719,
    "end_play_med": 2882.231,
    "end_play_ci": [
     2439.846,
     3195.065
    ],
    "end_play_p90": 3386.5676,
    "end_play_max": 3525.115,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.15,
    "chrf_med": 100.0
   },
   {
    "system": "cascade/single-flush+concise+adaptive",
    "category": "three_turns",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 3840.467,
    "start_lag_ci": [
     3712.488,
     4040.728
    ],
    "end_arrival_med": 1362.573,
    "end_play_med": 4649.85,
    "end_play_ci": [
     3943.307,
     4971.296
    ],
    "end_play_p90": 4944.3332,
    "end_play_max": 4971.296,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.736,
    "chrf_med": 66.18
   },
   {
    "system": "cascade/single-flush+concise+adaptive",
    "category": "two_utterances",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2144.052,
    "start_lag_ci": [
     2028.888,
     2474.876
    ],
    "end_arrival_med": 967.394,
    "end_play_med": 3097.7870000000003,
    "end_play_ci": [
     2853.621,
     3283.0555
    ],
    "end_play_p90": 3229.6073,
    "end_play_max": 3283.0555,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.796,
    "chrf_med": 93.87
   }
  ],
  "5b_paired": {
   "cascade/single-flush": {
    "per_clip_ms": {
     "es_en_long_1": -1419.0,
     "es_en_medium_1": 2.2,
     "es_en_medium_2": -173.1,
     "es_en_medium_3": -127.3,
     "es_en_noisy_1": 38.8,
     "es_en_noisy_2": -132.0,
     "es_en_noisy_3": -20.1,
     "es_en_short_1": -3.8,
     "es_en_short_2": -297.6,
     "es_en_short_3": 13.6
    },
    "median_ms": -73.7,
    "n_clips": 10,
    "n_faster": 7
   },
   "cascade/single-flush+concise": {
    "per_clip_ms": {
     "es_en_long_1": -2521.1,
     "es_en_medium_1": -630.8,
     "es_en_medium_2": -283.7,
     "es_en_medium_3": -216.7,
     "es_en_noisy_1": 653.6,
     "es_en_noisy_2": -226.4,
     "es_en_noisy_3": 62.3,
     "es_en_short_1": -71.2,
     "es_en_short_2": -173.4,
     "es_en_short_3": 324.1
    },
    "median_ms": -195.05,
    "n_clips": 10,
    "n_faster": 7
   },
   "cascade/single-flush+concise+adaptive": {
    "per_clip_ms": {
     "es_en_long_1": -2095.1,
     "es_en_medium_1": 412.4,
     "es_en_medium_2": -161.0,
     "es_en_medium_3": 129.7,
     "es_en_noisy_1": -518.4,
     "es_en_noisy_2": -138.0,
     "es_en_noisy_3": -25.2,
     "es_en_short_1": -50.4,
     "es_en_short_2": -52.0,
     "es_en_short_3": 327.9
    },
    "median_ms": -51.2,
    "n_clips": 10,
    "n_faster": 7
   }
  },
  "long_pass": {
   "cascade/baseline": {
    "end": 11348.703,
    "ratio": 1.214,
    "chrf": 80.24,
    "pass": false
   },
   "cascade/single-flush": {
    "end": 9797.547,
    "ratio": 1.0925,
    "chrf": 81.0,
    "pass": false
   },
   "cascade/single-flush+concise": {
    "end": 9120.169,
    "ratio": 1.059,
    "chrf": 79.89,
    "pass": false
   },
   "cascade/single-flush+concise+adaptive": {
    "end": 9546.158,
    "ratio": 1.088,
    "chrf": 80.72,
    "pass": false
   },
   "cascade/single-mbd0": {
    "end": 42203.409,
    "ratio": 1.939,
    "chrf": 73.5,
    "pass": false
   },
   "cascade/single-mbd200": {
    "end": 10409.048,
    "ratio": 1.131,
    "chrf": 81.83,
    "pass": false
   }
  }
 },
 "exp6": {
  "summary": [
   {
    "system": "cascade/gpt41mini-pipelined",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2624.162,
    "start_lag_ci": [
     2506.792,
     2660.152
    ],
    "end_arrival_med": 1508.464,
    "end_play_med": 9855.003,
    "end_play_ci": [
     9300.652,
     10027.016
    ],
    "end_play_p90": 9983.254799999999,
    "end_play_max": 10027.016,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.106,
    "chrf_med": 81.42
   },
   {
    "system": "cascade/gpt41mini-pipelined",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2520.416,
    "start_lag_ci": [
     2478.255,
     3150.165
    ],
    "end_arrival_med": 1033.002,
    "end_play_med": 3749.251,
    "end_play_ci": [
     3213.434,
     3921.466
    ],
    "end_play_p90": 4533.465,
    "end_play_max": 4905.588,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.924,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/gpt41mini-pipelined",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2642.589,
    "start_lag_ci": [
     2572.874,
     3064.798
    ],
    "end_arrival_med": 988.387,
    "end_play_med": 3603.949,
    "end_play_ci": [
     3409.742,
     4545.996
    ],
    "end_play_p90": 4641.1872,
    "end_play_max": 8172.861,
    "stalls": 1,
    "stall_rate": 0.06666666666666667,
    "stall_ci": [
     0.01186689536153232,
     0.2981653010887565
    ],
    "held_until_flush": 1,
    "ratio_span_med": 0.945,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/gpt41mini-pipelined",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 2625.429,
    "start_lag_ci": [
     2534.934,
     2851.312
    ],
    "end_arrival_med": 1000.4780000000001,
    "end_play_med": 3690.634,
    "end_play_ci": [
     3429.76,
     3806.1585
    ],
    "end_play_p90": 4628.7645,
    "end_play_max": 8172.861,
    "stalls": 1,
    "stall_rate": 0.03333333333333333,
    "stall_ci": [
     0.005908590316197976,
     0.16670391146115726
    ],
    "held_until_flush": 1,
    "ratio_span_med": 0.9345000000000001,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/gpt41mini-pipelined",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2220.224,
    "start_lag_ci": [
     1765.617,
     2891.117
    ],
    "end_arrival_med": 901.491,
    "end_play_med": 3072.62,
    "end_play_ci": [
     2132.258,
     3137.843
    ],
    "end_play_p90": 3215.4584,
    "end_play_max": 3329.44,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.161,
    "chrf_med": 100.0
   },
   {
    "system": "cascade/gpt41mini-serial",
    "category": "long",
    "n_runs": 5,
    "n_clips": 1,
    "n_failed": 0,
    "start_lag_med": 2524.724,
    "start_lag_ci": [
     2414.427,
     2742.809
    ],
    "end_arrival_med": 1419.06,
    "end_play_med": 9910.061,
    "end_play_ci": [
     9394.327,
     10411.036
    ],
    "end_play_p90": 10340.9176,
    "end_play_max": 10411.036,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.43448246865939283
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.113,
    "chrf_med": 80.54
   },
   {
    "system": "cascade/gpt41mini-serial",
    "category": "medium",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2646.542,
    "start_lag_ci": [
     2581.573,
     3012.439
    ],
    "end_arrival_med": 1074.475,
    "end_play_med": 3740.656,
    "end_play_ci": [
     3313.217,
     3922.115
    ],
    "end_play_p90": 4758.5761999999995,
    "end_play_max": 4945.274,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.965,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/gpt41mini-serial",
    "category": "noisy",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2550.821,
    "start_lag_ci": [
     2517.684,
     3080.644
    ],
    "end_arrival_med": 1060.588,
    "end_play_med": 3681.776,
    "end_play_ci": [
     3239.918,
     3753.698199999986
    ],
    "end_play_p90": 4297.9652,
    "end_play_max": 4894.411,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.902,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/gpt41mini-serial",
    "category": "sentences",
    "n_runs": 30,
    "n_clips": 6,
    "n_failed": 0,
    "start_lag_med": 2631.951,
    "start_lag_ci": [
     2545.227,
     2851.9314999999997
    ],
    "end_arrival_med": 1061.6625,
    "end_play_med": 3690.224,
    "end_play_ci": [
     3308.6235,
     3781.773
    ],
    "end_play_p90": 4596.9827000000005,
    "end_play_max": 4945.274,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.11351339476145204
    ],
    "held_until_flush": 0,
    "ratio_span_med": 0.9285,
    "chrf_med": 80.52
   },
   {
    "system": "cascade/gpt41mini-serial",
    "category": "short",
    "n_runs": 15,
    "n_clips": 3,
    "n_failed": 0,
    "start_lag_med": 2145.614,
    "start_lag_ci": [
     1767.901,
     2889.776
    ],
    "end_arrival_med": 1184.492,
    "end_play_med": 2667.569,
    "end_play_ci": [
     2119.398,
     3102.094
    ],
    "end_play_p90": 3126.571,
    "end_play_max": 3144.744,
    "stalls": 0,
    "stall_rate": 0.0,
    "stall_ci": [
     0.0,
     0.20388330359648707
    ],
    "held_until_flush": 0,
    "ratio_span_med": 1.067,
    "chrf_med": 100.0
   }
  ],
  "llm": {
   "cascade/gpt41mini-pipelined": {
    "fragments": 200,
    "first_token_med": 451.53395,
    "first_token_p90": 585.08791,
    "complete_med": 526.2806,
    "complete_p90": 666.36537
   },
   "cascade/gpt41mini-serial": {
    "fragments": 200,
    "first_token_med": 436.4869,
    "first_token_p90": 552.94119,
    "complete_med": 511.1383,
    "complete_p90": 646.83245
   }
  },
  "pass": {
   "cascade/gpt41mini-pipelined": false,
   "cascade/gpt41mini-serial": false
  },
  "blocked": {
   "Groq serial / Groq pipelined": "no GROQ_API_KEY; those conditions were skipped"
  }
 },
 "prep78": {
  "anticipation": "implemented behind flag, not run",
  "stable_ms": "implemented, not run"
 },
 "decision_gate": {
  "flush_on_pause_holds_2_3s_no_stalls": {
   "answer": false,
   "conditions_meeting": [],
   "per_condition": {
    "openai/flush-pause400-a20": {
     "median": 2732.776,
     "stalls": 0,
     "max": 13598.406
    },
    "openai/flush-pause600-a20": {
     "median": 2688.158,
     "stalls": 0,
     "max": 15554.705
    },
    "openai/flush-pause800-a20": {
     "median": 2748.587,
     "stalls": 1,
     "max": 7908.968
    }
   }
  },
  "cascade_reaches_2_2s_at_chrf85": {
   "answer": false,
   "conditions_meeting": [],
   "per_condition_sentences": {
    "cascade/baseline": {
     "end_play_med": 3752.6515,
     "chrf_med": 80.52
    },
    "cascade/single-flush": {
     "end_play_med": 3593.5885,
     "chrf_med": 80.52
    },
    "cascade/single-mbd0": {
     "end_play_med": 9986.1075,
     "chrf_med": 80.52
    },
    "cascade/single-mbd200": {
     "end_play_med": 4145.773,
     "chrf_med": 74.02
    },
    "cascade/single-flush+concise": {
     "end_play_med": 3499.6620000000003,
     "chrf_med": 80.52
    },
    "cascade/single-flush+concise+adaptive": {
     "end_play_med": 3551.8720000000003,
     "chrf_med": 80.52
    },
    "cascade/gpt41mini-pipelined": {
     "end_play_med": 3690.634,
     "chrf_med": 80.52
    },
    "cascade/gpt41mini-serial": {
     "end_play_med": 3690.224,
     "chrf_med": 80.52
    }
   },
   "note": "sentences = medium + noisy pooled, end lag at playback, 1x listener"
  }
 },
 "costs": [
  {
   "batch_id": "20261013-222319",
   "experiment": "smoke2a",
   "note": "smoke test",
   "openai_sessions": 2,
   "openai_audio_min": 1.086,
   "openai_session_wall_min": 1.242,
   "openai_est_usd": 0.037,
   "openai_est_usd_if_billed_by_wall_time": 0.042,
   "other_provider_audio_min": {}
  },
  {
   "batch_id": "20261013-222434",
   "experiment": "exp2a",
   "note": "partial: stopped by user at 120/240",
   "openai_sessions": 119,
   "openai_audio_min": 66.794,
   "openai_session_wall_min": 76.248,
   "openai_est_usd": 2.271,
   "openai_est_usd_if_billed_by_wall_time": 2.592,
   "other_provider_audio_min": {}
  },
  {
   "batch_id": "20261014-172417",
   "experiment": "exp2a",
   "note": "resume of 20261014-164449",
   "openai_sessions": 120,
   "openai_audio_min": 68.149,
   "openai_session_wall_min": 77.446,
   "openai_est_usd": 2.317,
   "openai_est_usd_if_billed_by_wall_time": 2.633,
   "other_provider_audio_min": {}
  },
  {
   "batch_id": "20261014-181904",
   "experiment": "exp2b",
   "note": "stall study phase 2b; concurrency 4",
   "openai_sessions": 180,
   "openai_audio_min": 101.52,
   "openai_session_wall_min": 116.784,
   "openai_est_usd": 3.452,
   "openai_est_usd_if_billed_by_wall_time": 3.971,
   "other_provider_audio_min": {}
  },
  {
   "batch_id": "20261014-190148",
   "experiment": "exp2c",
   "note": "best exp2 condition at concurrency 1 for clean medians",
   "openai_sessions": 50,
   "openai_audio_min": 29.873,
   "openai_session_wall_min": 33.826,
   "openai_est_usd": 1.016,
   "openai_est_usd_if_billed_by_wall_time": 1.15,
   "other_provider_audio_min": {}
  },
  {
   "batch_id": "20261014-194721",
   "experiment": "smoke3",
   "note": "flush on pause / watchdog via session rotation; concurrency 1",
   "openai_sessions": 4,
   "openai_audio_min": 0.215,
   "openai_session_wall_min": 0.66,
   "openai_est_usd": 0.007,
   "openai_est_usd_if_billed_by_wall_time": 0.022,
   "other_provider_audio_min": {}
  },
  {
   "batch_id": "20261014-194847",
   "experiment": "exp3",
   "note": "flush on pause / watchdog via session rotation; concurrency 1",
   "openai_sessions": 664,
   "openai_audio_min": 55.908,
   "openai_session_wall_min": 134.992,
   "openai_est_usd": 1.901,
   "openai_est_usd_if_billed_by_wall_time": 4.59,
   "other_provider_audio_min": {}
  },
  {
   "batch_id": "20261014-213709",
   "experiment": "smoke5",
   "note": "cascade TTS context variants; phrase-by-phrase mode; concurrency 1",
   "openai_sessions": 0,
   "openai_audio_min": 0.0,
   "openai_session_wall_min": 0.0,
   "openai_est_usd": 0.0,
   "openai_est_usd_if_billed_by_wall_time": 0.0,
   "other_provider_audio_min": {
    "cascade": 0.75
   }
  },
  {
   "batch_id": "20261015-014128",
   "experiment": "exp5a",
   "note": "resume of 20261014-213910",
   "openai_sessions": 0,
   "openai_audio_min": 0.0,
   "openai_session_wall_min": 0.0,
   "openai_est_usd": 0.0,
   "openai_est_usd_if_billed_by_wall_time": 0.0,
   "other_provider_audio_min": {
    "cascade": 34.45
   }
  },
  {
   "batch_id": "20261015-023140",
   "experiment": "exp4",
   "note": "Gemini Live Translate vs OpenAI control in one randomized batch; concurrency 1",
   "openai_sessions": 80,
   "openai_audio_min": 15.682,
   "openai_session_wall_min": 22.353,
   "openai_est_usd": 0.533,
   "openai_est_usd_if_billed_by_wall_time": 0.76,
   "other_provider_audio_min": {
    "gemini": 15.68
   }
  },
  {
   "batch_id": "20261015-032059",
   "experiment": "exp5b",
   "note": "concise prompt and adaptive speed on the best single context",
   "openai_sessions": 0,
   "openai_audio_min": 0.0,
   "openai_session_wall_min": 0.0,
   "openai_est_usd": 0.0,
   "openai_est_usd_if_billed_by_wall_time": 0.0,
   "other_provider_audio_min": {
    "cascade": 39.46
   }
  },
  {
   "batch_id": "20261015-044648",
   "experiment": "exp6",
   "note": "translation host and pipelined fragments; phrase-by-phrase mode; concurrency 1",
   "openai_sessions": 0,
   "openai_audio_min": 0.0,
   "openai_session_wall_min": 0.0,
   "openai_est_usd": 0.0,
   "openai_est_usd_if_billed_by_wall_time": 0.0,
   "other_provider_audio_min": {
    "cascade": 19.75
   }
  }
 ],
 "estimated_openai_usd": 11.53
}
```
