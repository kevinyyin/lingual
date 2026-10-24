# Round 2 status (2026-10-16)

Pre-event research spike, round 2; not product code. Code is in `spike2/`, results in `results2/`, logs in
`logs2/`. Nothing is committed. Run everything from `asl-spike/`.

## Done
R0, R1 (+ framing, aspect), R2, R3 (round-1 base model), P1, V1, P2, P3, P4 (reranker), P5 (+ pooled RWTH),
V2, V3, V4, R5. F2 skipped (no DAI access). Report generator: `spike2/report2.py` → `REPORT-SPIKE-2.md`
(every number is read from `results2/`; rerun it after any result changes). `NEXT.md` updated.

## Still running / to finish
1. **R4** (5 variants × 3 seeds, ~15–35 min per run): `bash spike2/run_r4.sh > logs2/r4.log 2>&1`.
   It skips runs whose `results2/r4_<variant>_s<seed>.json` exists, so rerunning resumes.
2. **R3 with the best R4 model** (after R4):
   `uv run python spike2/r3_calibrate.py cache/s2r2/kaggle_feats_aspect.npz cache/s2r2/r4_<best>_s0.pt best`
   → `results2/r3_calibration_best.json` (the report picks it up automatically).
3. **F1** face size: done (`results2/f1_face_size.json`).
4. `a_bi` re-scored with per-signer CIs (done). After R4 + R3-best: `uv run python spike2/report2.py` and fill the R4 line in `NEXT.md`.
