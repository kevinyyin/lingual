# Lead analyst (adjudication)

You get the failures where the two analysts' tag sets differ: each with its `sheet`, the item fields from `analyst.md`
and both analysts' answers. Open each sheet, decide the tag set yourself under the same rules, and append one line per
item to `app/data/teaching/analysis/results/lead.jsonl` in the analyst schema plus `"adjudicated": true`.

# Synthesis: the failure report

Inputs (read-only): `app/data/teaching/eval/summary*.json` (accuracy), `app/data/teaching/analysis/stats.json`
(feature statistics, within-sign gaps, models, confusion phonology), `app/data/teaching/analysis/analysts.json` (tag
shares, per-tag agreement between analysts, which tags the measurements back up), `app/data/teaching/verify/summary.json`
(label accuracy, control catch rate), and the analyst result lines for example clips.

Write `app/reports/teaching_videos_failures.md` (numbers only, no images):
1. Ranked failure modes. For each: share of failures tagged with it (and analyst agreement κ), its measured evidence
   (within-sign gap and p, quartile failure rates, Cliff's delta, boosting importance), 2-3 example clip ids with their
   one-line causes, and a concrete fix to try in this codebase (for example re-key hands to the pose wrists in
   `server/features.py`; training augmentation for X; segmenter thresholds in `server/segmenter.py`; camera guidance in
   the page). Mark every claim as **measured** (statistics support it) or **hypothesis** (analysts only).
2. What the confusions say (minimal pairs vs unrelated signs; which parameter differs most).
3. Caveats: agents are not fluent signers; agreement levels; sample sizes; what a Deaf advisor should check first.
Keep it under 900 words. Do not invent numbers: every number must come from the input files.
