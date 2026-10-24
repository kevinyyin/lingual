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
