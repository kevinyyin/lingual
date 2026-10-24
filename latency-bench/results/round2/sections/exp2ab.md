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
