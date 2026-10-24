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
