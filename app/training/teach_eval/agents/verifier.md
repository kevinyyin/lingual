# Verifier: does each clip really show the sign its label says?

You review one batch file of ASL sign clips. You are checking the **video's label**, not the app.

For each item in the batch (JSON list: `item_id`, `sheet`, `gloss`, `description`):
1. Open the image at `sheet` (use the Read tool on the .jpg). Top strip: 6 frames from the **clip under review**, left to
   right through the sign. Bottom strip: 6 frames from a **reference** video of `gloss` by a different signer (ASL Citizen).
2. Decide whether the clip shows the sign `gloss`. Use the reference and `description` (ASL-LEX: one- or two-handed,
   location, handshape, movement). Different signers vary: a mirrored (left-handed) version, a slightly different
   location or a small size difference is still the same sign. A regional/variant form that differs in handshape or
   movement from the reference is still the gloss if it plausibly means the same word; say so in
   `same_variant_as_reference`.
3. Be honest about uncertainty. Frames are a sampling; when you cannot tell, answer `unsure`. Some items are deliberately
   mislabelled controls; do not assume every label is right.

Write one JSON object per item, one per line, to the results file you are given
(`app/data/teaching/verify/results/<batch name>_<your reviewer id>.jsonl`):

```json
{"item_id": "...", "matches_label": "yes|no|unsure", "same_variant_as_reference": "yes|no|unsure",
 "single_sign": true, "issues": ["cut_off|multiple_signs|fingerspelled|mirrored|small_in_frame|occluded|low_quality|text_overlay|other"],
 "note": "one short sentence: what you saw that decided it", "confidence": 0.0}
```

Do not open other batches, the key file, or any other files. Finish with a one-line summary: items done, yes/no/unsure counts.
