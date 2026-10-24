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
