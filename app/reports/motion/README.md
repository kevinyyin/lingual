# Motion baselines ("before")

Measured at commit `6ce82cf` (the commit before the one that adds these files), default character Rocketbox
`Male_Adult_11` (no `--model` flag; the default was switched in the same working tree), contact data (`touch_from_asllex`) present.

Commands (from `app/`, backend :8000 and Vite :5173 running):

    uv run python -m training.audit_placement
    uv run --with playwright python scripts/motion_bench.py --label before
    uv run --with playwright python scripts/roundtrip.py --label before  --n 150
    uv run --with playwright python scripts/roundtrip.py --label before2 --n 150

## Placement audit (`placement_audit.json`)
World-landmark vs aspect-corrected 2D wrist placement relative to the nose, worst frame per clip with a wrist within 1.3
shoulder widths of the nose. **134 of 1976 clips (6.8%) exceed 0.3 shoulder widths** (median worst error 0.22). This share decides Task 7's path.

## Motion bench (`before.json`, `before.md`)
Total: **129 spikes**, 7624 frames, max step 179.7 deg, p99 step 165.9 deg, 316 twist-out, 0 backward elbow,
0 penetration, min arm gap 0.065 m, mean contact error 0.149 m.

## Round-trip, 150 signs (`roundtrip_before.json`, `roundtrip_before2.json`)
| run | top-1 | top-3 | top-5 |
|---|---|---|---|
| before  | 8.7% | 15.3% | 23.3% |
| before2 | 8.7% | 17.3% | 24.0% |

Same code, same signs: run-to-run drift is about 2 points on top-3/top-5 and none on top-1. Treat differences under ~2-3 points as noise.

# Core (Task 6: arm solver + rotation filters + turn-rate cap)

Same commands with `--label core` / `core2` (Male_Adult_11, 60 fps). Arm filter QuatFilter(3, 0.5), fingers (8, 0.5),
turn cap 1200 deg/s.

| metric | before | core |
|---|---|---|
| spikes | 129 / 7624 | 0 |
| max / p99 step (deg) | 179.7 / 165.9 | 20.1 / 20.1 |
| twist-out | 316 | 0 |
| backward elbow | 0 | 0 |
| max penetration / arm (m) | 0 / 0 | 0 / 0 |
| min arm gap (m) | 0.065 | 0.065 |
| contact error mean (m) | 0.149 | 0.144 |
| p95 frame ms | 1.2 | 1.7 |
| frames at the turn cap (>19.5 deg) | - | 143 / 7444 (1.9%) |

Round-trip 150 (after review fix round 1): core .073 / .147 / .227, core2 .080 / .133 / .233 (top-1/3/5) vs before .087 / .153-.173 / .233-.240:
below the baseline. `roundtrip_core_nofilter.json` (no filter, no cap, diagnostic) .093 / .147 / .227: the filtering costs ~1-2 points.

# Final: before vs after (Task 10)

`after.json` / `after.md` (`motion_bench.py --label after`) and `roundtrip_after.json` (`roundtrip.py --label after --n 150`),
Male_Adult_11, at 841a815; `after.json` re-measured after the final-review fixes (twist re-anchoring, time-based torso ease, head gap ease),
with a `CONTINUOUS (10 sentences)` item (all ten sentences back to back, no reset; kept out of TOTAL). Stages: `core`, `placement`, `head`, `arcs` JSON files hold the intermediate runs.

| metric | before | after |
|---|---|---|
| spikes | 129 / 7624 | 0 / 7624 |
| max / p99 step (deg) | 179.7 / 165.9 | 20.3 / 20.1 |
| backward elbow | 0 | 0 |
| twist out of range | 316 | 0 |
| max penetration / arm (m) | 0 / 0 | 0 / 0 |
| min arm gap (m) | 0.065 | 0.071 |
| contact error mean (m) | 0.149 | 0.139 |
| GRANDMOTHER@chin contact error mean / min (m) | 0.516 / - | 0.512 / 0.202 (gate 0.03 unmet, parked) |
| p95 frame ms | 1.2 | 1.6 |
| CONTINUOUS: spikes / twist out / backward elbow / max penetration (m) | - | 0 / 0 / 0 / 0.0023 (one frame, HELLO start) |

Round-trip, 150 signs (top-1 / top-3 / top-5):

| run | top-1 | top-3 | top-5 |
|---|---|---|---|
| before | 8.7% | 15.3% | 23.3% |
| before2 | 8.7% | 17.3% | 24.0% |
| core | 7.3% | 14.7% | 22.7% |
| core2 | 8.0% | 13.3% | 23.3% |
| placement (`roundtrip_placement.json`) | 8.0% | 14.0% | 20.0% |
| after | 8.7% | 14.0% | 22.7% |

The after round-trip is at or just below the before baseline (top-3 14.0% vs 15.3-17.3%, top-5 22.7% vs 23.3-24.0%; top-1 equal).
The drift between the two before runs is about 2 points, so top-1 and top-5 are within noise and top-3 is 1-3 points low.
The spec calls round-trip a weak signal (the recogniser was trained on real signers, not avatars).

# Readability (2026-10-21)

Toon body, 60 fps, the bench's 12 signs and 10 sentences (`readability_*.json`) and the 150-sign round trip (`roundtrip_readability_*.json`).
`before` is the code before this work; `a` adds the sign span + holds (`player.signSpan`) and the load-time cleaning
(`clipSmooth.ts`) with today's live smoothing; `b` also lightens the live filters (arm 6 Hz, fingers 12 Hz); `c` switches the
arm one-euro filter off (fingers 12 Hz). Edge/inner = the higher wrist's median per-frame speed and jerkiness within 40 ms of a
sign edge against elsewhere (TOTAL: mean over items).

| metric | before | a (kept) | b | c |
|---|---|---|---|---|
| frames at the speed limit (> 19.5 deg) | 15 / 7123 | 9 / 6455 | 9 | 10 |
| edge / inner speed | 11.0 / 5.8 = 1.89x | 1.26x | 1.14x | 0.85x |
| edge / inner jerkiness (mm) | 2.29 / 1.19 = 1.92x | 1.70 / 1.23 = 1.39x | 1.80 / 1.29 | 1.76 / 1.51 |
| spikes / backward elbow / twist out / arm penetration | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| max hand/finger penetration (m) | 0.0074 (MY) | 0.0083 (REMEMBER; MY 0.0023) | 0 | 0 |
| worst finger joint past its range (deg) | 4.4 | 9.3 (TOMORROW I GO STORE BUY APPLE) | 10.2 | 10.2 |
| contact error mean (m) | 0.171 | 0.183 | 0.182 | 0.171 |
| p95 frame (ms) | 2.7 | 2.4 | 2.4 | 2.6 |
| round trip top-1 / top-3 / top-5 | .060 / .140 / .213 | .067 / .147 / .180 | .093 / .160 / .187 | .067 / .153 / .167 |

Kept (a) by the spec's rule: (b) and (c) are each jerkier than (a) (rule 2), though (b) read best in the round trip (within its
~2-point noise on top-3). Criteria 1, 2 and 4 pass; criterion 3 misses by 0.8 mm: REMEMBER's thumb now dips 8.3 mm into the forehead
at its deepest frame (before: 0 for REMEMBER, worst MY 7.4 mm). The user watched the result on both screens and accepted it.
Contact error mean 0.171 -> 0.183 m: 79% of the rise (+0.152 of +0.193 m summed over items) is the two HELLO items, 0.177 -> 0.252 m
each. HELLO takes the fixed-trim fallback (142 frames before and after), so its change comes from the load-time cleaning alone, which
moves its landmarks by at most 0.3 cm at the forehead; something in the live solve magnifies that (cause not found). HELLO also goes
from 17.3 to 19.4 deg max step and 0.87 to 1.30 mm edge jerkiness. HELLO was in the set the user watched and signed off on both screens.

# Sign spans (2026-10-22)

Follow-up to the readability work, branch `avatar-sign-spans` (from `avatar-readability`). Same toon body, bench items and
150-sign round trip as above (`spans_*.json`, `roundtrip_spans_*.json`; commands as above with `--label spans_c` and
`--label spans_c --n 150`). `readability_a2` re-runs PR #5's code. Three candidates; `spans_c` is kept:

- `spans_a`: more signs get spans (below), and a sign's end is held only where the hand is still (under 0.1 m/s);
- `spans_b`: `spans_a` plus a fix for HELLO's twist (below);
- `spans_c` (kept): more signs get spans; holds at both ends as before.

**More signs get spans (kept).** The 40% cap now counts the frames the active hand is up, from its first rise past
0.25 m to its last drop below 0.2 m, not the whole clip: the rise, drop and rest around a sign no longer reject it. Below
0.25 m a slowdown ends the sign only where the hand really stops (under 0.1 m/s): the drop to rest wobbles there, and
without that check KEY's end moved to a frame at 117 cm/s. Over all 1,982 clips (cleaned, idle hands rested): spans
1,495 (75.4%) -> 1,760 (88.8%), 268 gained, 3 lost (CONNECT, E, SURGERY: the span would cut more than 40% of the time the
hand is up), 10 ends moved later, each to a real stop (1.3-8.3 cm/s). HELLO gets [9, 26): it was rejected for the hold,
drop and rest after its sign. THANK-YOU gets [9, 42): it was too short, because its last hold (palm out at the chest)
sits at 0.21 m, under 0.25.

**Holds only where the hand is still (measured, not kept).** Found starts move a median 12.8 cm/s (43% faster than 15)
against 4.1 for ends: signers flow from the reach into a sign but stop at its end. Holding an end only under 0.1 m/s
(39% of starts, 81% of ends) cost the round trip 4 top-3 hits net (22 -> 18), all lost in signs whose start hold was
dropped (DAY, HOME, SCARED, TIRED; SICK, newly spanned) and none won there, while signs treated as before only swapped by
noise (THIRSTY lost, FATHER and LEARN won). It also let a hand sink 16 mm into the body in the continuous run. The holds
help a sign's start read, so both ends stay held. MEET's doubled edge jerkiness in PR #5 (1.25 -> 2.59 mm) was not its
start hold: its start edge is as jerky with the old trim (6.3 vs 7.0 mm at the first edge frame, the hand still slowing
from the 450 ms reach out of rest), while the end hold made its end edge calmer (0.05-0.19 vs 0.19-0.37 mm) and the edge
window caught one frame fewer there, so the median of 9 edge frames landed on a start-edge value.

**HELLO's contact error: the cause.** Not the cleaning as such. When a hand appears, its turn eases in from the relaxed
hand (`rocketbox.ts`, `slerp` by 1 - idle), and that relaxed hand was solved at the forearm's current twist, which the
ease itself drives. HELLO's hand started 179.9 degrees from relaxed (at the clip's frame 6, the fixed trim), so the
shortest way round was a coin toss: the cleaning's 0.3 cm, the idle breathing (real time) or the torso turn left from the
item before could each pick the side. One side is stable (the angle falls to 97 degrees as the arm rises). The other
winds the twist demand to +232 degrees, the relaxed base turns with it until the angle passes 180 degrees at 233 ms, the
eased target jumps 55 degrees in a frame, the twist's palm-flip rule snaps the demand to -35 degrees, and the 20-degree
turn cap spreads that over the whole arm, which lags up to 23 cm behind for 170 ms: the 0.252 m contact error and the
19.4-degree steps. PR #5's code gave HELLO either result depending on that toss (`readability_a2`: 0.178 m, 17.1 deg).
With its span, HELLO now starts at frame 9, 175.2 degrees from relaxed and falling: no toss, and the same result on every
run, alone or after other items (0.180 m, 16.5 deg in `spans_c`; also in HELLO NICE MEET YOU).

**HELLO's twist fix (measured, not kept).** Easing from the relaxed hand with the forearm untwisted makes HELLO agree with
and without the cleaning (0.173 / 0.171 m). But it moved COOK's left hand, a palm-up base hand needing more forearm twist
than the arm has, onto the other branch: a palm-flip jump at 6.1 s put its demand at +117 degrees instead of -173, it
wound to +276 by the end of the sentence, and the unwind there turned the forearm 166 degrees through the turn cap: 5
frames at the speed limit and 23.5-degree steps in MY GRANDMOTHER TEACH I COOK (12 in total against 9). The general
problem, a twist demand running past what a forearm can reach, is there with either base; that is solver work for after
the demo.

| metric | a (PR #5) | spans_a | spans_b | spans_c (kept) |
|---|---|---|---|---|
| frames at the speed limit (> 19.5 deg) | 9 / 6455 | 7 / 5996 | 12 / 5996 | 7 / 6168 |
| edge / inner speed (mm) | 6.68 / 5.31 = 1.26x | 5.88 / 6.47 = 0.91x | 6.07 / 6.35 = 0.96x | 5.59 / 5.61 = 1.00x |
| edge / inner jerkiness (mm) | 1.70 / 1.23 = 1.39x | 1.62 / 1.51 = 1.07x | 1.70 / 1.50 = 1.13x | 1.59 / 1.39 = 1.14x |
| spikes / backward elbow / twist out / arm penetration | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| max hand/finger penetration (mm) | 8.3 | 8.9 | 8.3 | 8.3 |
| worst finger joint past its range (deg) | 9.3 | 9.2 | 9.2 | 9.3 |
| contact error mean (m) | 0.183 | 0.179 | 0.178 | 0.174 |
| max step (deg) | 20.7 | 20.5 | 23.5 | 20.7 |
| p95 frame (ms) | 2.4 | 2.4 | 2.4 | 2.6 |
| HELLO contact error (m) / max step (deg) | 0.252 / 19.4 | 0.209 / 16.5 | 0.204 / 15.7 | 0.180 / 16.5 |
| CONTINUOUS: frames at the limit / max penetration (mm) | 12 / 3.1 | 10 / 16.1 | 14 / 16.0 | 10 / 4.3 |
| round trip top-1 / top-3 / top-5 | .067 / .147 / .180 | .067 / .120 / .160 | - | .073 / .133 / .160 |

Kept `spans_c` by the brief's rules against `a`: frames at the speed limit 7 (no more than 9), edge/inner speed 1.00x and
jerkiness 1.14x (no worse than 1.26x and 1.39x; the sign edges themselves are slower and smoother, 5.59 vs 6.68 mm speed,
1.59 vs 1.70 mm jerkiness), round-trip top-3 1.4 points below (within 2; top-1 up 0.7, top-5 down 2.0). `spans_a` failed
the round trip (2.7 points below) and `spans_b` the speed limit. Nothing new breaks: no spikes, backward elbows, twist or
arm penetration, and the deepest hand point is REMEMBER's thumb as before (8.3 mm). The inner jerkiness rises because
HELLO, cut to its sign, is now mostly the two 450 ms reaches to and from rest (its inner clip frames are calmer, 0.92 vs
1.74 mm); signSpan runs once per clip, not per frame, so the p95 frame cost is run-to-run noise (PR #5's code measured
2.2 and 2.4 ms).
The continuous run's 4.3 mm is one frame, the first of I SORRY I UNDERSTAND NOT after LIVE OR DIE WHICH. To check by eye
before merging: HELLO, THANK-YOU and MEET on the laptop and the phone, with and without `?oldmotion`.

# Hand twist (2026-10-22)

Follow-up to the sign spans: which way a hand's turn takes the forearm as the hand appears. Branch
`claude/elegant-wiles-fce637` (from `avatar-readability` with PR #7). Same toon body, bench items and 150-sign round trip
as above (`twist_*.json`, `roundtrip_twist_b*.json`; `roundtrip_spans_c2` re-runs PR #7's code in this session). The
measurements below that needed HELLO's old trim or the breathing frozen used debug-only switches, not committed. Two
candidates; `twist_b` is kept:

- `twist_a`: the turn eases in joint space: the wrist swing by k, the forearm twist from the twist the forearm had as
  the hand appeared, on the side nearest it, waiting while that side is a toss; a leaving hand's forearm holds its twist;
- `twist_b` (kept): the slerp from the relaxed hand as before, its share waiting while the twist the hand asks of the
  forearm is within 10 degrees of half a turn (`armSolver.easeHand`).

**HELLO's toss, measured.** With HELLO's old fixed trim forced on, its first frame is the clip's raised hand on an arm
still easing up from rest: the hand bent 146 degrees back from the forearm and twisted exactly half a turn from it
(+179.6 degrees on one run, -179.8 on the next). The slerp from the relaxed hand went whichever way round that gave; the
losing side wound the forearm to its limit and the demand to +232, and the palm-flip rule snapped it back. PR #7's
code, 8 runs of HELLO: 0.252-0.253 m five times, 0.178 three times; HELLO NICE MEET YOU: 0.252 once, 0.178 three times.

**COOK, measured.** COOK's left base hand is bent 115-160 degrees back from the avatar's forearm for the whole sign,
where a hand's twist about the forearm is barely defined. Its demand follows the hand from -155 degrees as it appears
(65 from the forearm's -89.5), past -180 to -240 and back to -55; the forearm sits at its -89.5 limit throughout, then
the hand leaves. Nothing snaps, the same on every run (0.206 m, 17.8 degrees). Every branch on this path passes beyond
half a turn (-155, -240, -55 or +205, +120, +305), so no choice keeps the demand within reach without the forearm
turning over mid-sign. The first fix's untwisted base put the relaxed and the seen hand half a turn apart as the hand
passed -180 at k 0.88: the slerp's way round flipped, the twist rule read a palm flip and took the far branch (+117 up
to +276), and the unwind snapped it. Both candidates keep the forearm's own twist as the base, as PR #7 did.

**Joint-space ease (measured, not kept).** `twist_a` is deterministic too (old-trim HELLO 0.204-0.207 m over 8 runs) and
improves the continuous run (7 frames at the speed limit, not 10; 2.1 mm deepest, not 4.3). But it moves every hand bent
far from its forearm along a new path as it appears. In FATHER the contact gap's 24 mm one-frame step (there with
either ease) comes a frame earlier, inside the 40 ms edge window: FATHER's edge speed goes from 5.83 to 8.72 mm, and over
all items edge/inner speed is 1.03x and jerkiness 1.19x (against 1.00x and 1.14x). REMEMBER's deepest hand point goes
12.5 mm into the head (8.3).

**Kept (`twist_b`).** Each frame the turn's share advances by the ease's own step times how clearly the twist the hand
asks of the forearm (from the forearm's twist) is not half a turn: fully up to 170 degrees, not at all at 180. That is
exactly k for a hand appearing clear of 170 degrees, which is every appearance on the bench (the largest is today's
HELLO, at 165), and a unit test pins the plain slerp there: the bench matches `spans_c` in every column. The old-trim
HELLO waits out the toss, then turns the clear way: 0.184-0.186 m on all 8 runs (alone and in HELLO NICE MEET YOU),
HELLO's max step 17.2-18.0 degrees. Leaving hands ease out as before. `?oldmotion` keeps the plain slerp: with the
breathing frozen, PR #7's code and this one move every point of all 150 round-trip signs, COOK's sentence and HELLO NICE
MEET YOU identically under it. Per frame it adds one twist decomposition per arm while a hand eases in (about a second
after it appears), next to the relaxed solve already made there, and nothing otherwise.

**Round trip.** Two runs of `twist_b`: top-3 .100, then .120; PR #7's code gave .133 (`spans_c`) and .140 (`spans_c2`).
With the breathing frozen, 146 of the 150 signs move identically under both codes (every wrist, fingertip and elbow, on
every frame), among them all five that `twist_b`'s first run lost against both PR #7 runs (DAY, NICE, NINE, SEVEN,
TIRED). So the spread is in the renders: the breathing runs on real time, blinks are random, and the recognizer flips
those signs between a hit and a miss (DAY ranked 3rd, 1st, then missed twice). The four signs that move differently
(ALL, CHEESE, NEVER, NONE: a hand passing within 10 degrees of half a turn while it appears) are top-3 hits in none of
the four runs, and NEVER's largest per-frame jump falls from 238 to 204 mm.

| metric | spans_c | twist_a | twist_b (kept) |
|---|---|---|---|
| frames at the speed limit (> 19.5 deg) | 7 / 6168 | 7 / 6168 | 7 / 6168 |
| edge / inner speed (mm) | 5.59 / 5.61 = 1.00x | 5.77 / 5.62 = 1.03x | 5.59 / 5.61 = 1.00x |
| edge / inner jerkiness (mm) | 1.59 / 1.39 = 1.14x | 1.64 / 1.38 = 1.19x | 1.59 / 1.39 = 1.14x |
| spikes / backward elbow / twist out / arm penetration | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| max hand/finger penetration (mm) | 8.3 | 12.5 | 8.3 |
| contact error mean (m) | 0.174 | 0.174 | 0.174 |
| max step (deg) | 20.7 | 20.7 | 20.7 |
| p95 frame (ms) | 2.6 | 2.3 | 2.5 |
| HELLO contact error (m) / max step (deg) | 0.180 / 16.5 | 0.182 / 15.8 | 0.180 / 16.5 |
| MY GRANDMOTHER TEACH I COOK contact error (m) / max step (deg) | 0.206 / 17.8 | 0.206 / 17.8 | 0.206 / 17.8 |
| HELLO with its old trim, 8 runs: contact error (m) | 0.252-0.253 (5), 0.178 (3) | 0.204-0.207 | 0.184-0.186 |
| CONTINUOUS: frames at the limit / max penetration (mm) | 10 / 4.3 | 7 / 2.1 | 10 / 4.3 |
| round trip top-1 / top-3 / top-5 | .073 / .133 / .160; .060 / .140 / .160 (`spans_c2`) | - | .053 / .100 / .147; .053 / .120 / .140 |

Kept `twist_b` by the brief's rules against `spans_c`: frames at the speed limit and edge/inner ratios the same; the
round trip's top-3 1.3 points below on the second run and 3.3 on the first, a spread the frozen-breathing comparison puts
in the renders, not the motion. The 7 frames still at the speed limit are steady-state twist jumps, not eases: the
tracked palm flipping (97-154 degrees in a frame in TOMORROW I GO STORE BUY APPLE, TRAIN LEAVE WHAT TIME and YOUR CAR
NEW) and YOUR CAR NEW's left demand unwinding from 271 degrees. To check by eye before merging: HELLO and MY GRANDMOTHER
TEACH I COOK on the laptop and the phone, with and without `?oldmotion` (the default motion should look as in PR #7).
