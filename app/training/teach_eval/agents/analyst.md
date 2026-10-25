# Failure analyst: why did the app read this clip wrong?

You review one batch of clips the app got wrong. Each item: `item_id`, `sheet`, `true_gloss` (what was signed; the label
was verified), `read_as` (the app's top guess), `outcome` (what the translator did: wrong_spoken, none_of_these,
split_*, no_turn...), `feature_card_z` (the clip's most unusual measurements as z-scores against clips of the same sign
the app read right: positive = more than usual) and `features` (raw values; distances in shoulder widths).

Feature meanings: head_* = head turn range/speed; near_face_frac / over_face_frac = fingertips near / over the face;
wrist_nose_min = closest wrist-to-nose distance; hands_overlap_frac / hands_dist_min = the two hands overlapping / their
closest distance; finger_gap_mean = how far apart the fingertips are (small = fingers together); finger_curl_* and ext_*
= finger extension; palm_* = palm orientation; speed_* / path_sw / repetitions = movement; hand_detect_frac /
handedness_flip_frac / finger_back_frac / finger_jitter = tracking quality; face_width_px = how big the signer is in
frame; qc_segments = how many pieces the app cut the clip into (1 is right).

For each item: open `sheet` (Read the .jpg). Top strip = the clip as signed; bottom strip = a reference of the sign the
app **thought** it saw. Compare them and the feature card, then tag what most plausibly made the app go wrong. Use only
these tags (as many as apply, most important first):

`head_motion, fingers_close_or_merged, hand_near_face, hand_occludes_face, hands_overlap, hand_out_of_frame,
fast_motion_or_blur, small_in_frame, palm_orientation, minimal_pair, variant_differs, segmentation_cut,
handedness_or_mirrored, label_error, other`

- `minimal_pair`: the true and read signs differ in only one parameter (handshape, location, movement or orientation).
- `variant_differs`: the clip uses a different form of the sign than the app was trained on.
- `segmentation_cut`: the app cut the sign wrong (qc_segments not 1, outcome split_* or no_turn).
- Only tag a measurement-based cause (head_motion, fingers_close_or_merged, hands_overlap...) if the frames **or** the
  feature card support it; say which in `evidence`.

Write one JSON line per item to the results file you are given (`app/data/teaching/analysis/results/<batch>_<id>.jsonl`):

```json
{"item_id": "...", "tags": ["..."], "cause": "one sentence: what the app most likely confused and why",
 "evidence": "frames|features|both", "confidence": 0.0}
```

Do not open other batches. End with a one-line summary of your most common tags.
