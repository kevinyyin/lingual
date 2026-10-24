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
