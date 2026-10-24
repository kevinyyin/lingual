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
