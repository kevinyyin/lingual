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
