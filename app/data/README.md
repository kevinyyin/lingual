# data/ (contents gitignored except this file)

- `asl_citizen/` — ASL Citizen dataset (Microsoft Research, download id 105253, ASL_Citizen.zip, 42.8 GB).
  **License: use "solely for non-commercial, non-revenue generating, research purposes."** Models and
  avatar clips derived from it inherit that limit; keep them private. Commercial use needs our own recordings.
- `landmarks/asl_citizen/{train,val,test}/{GLOSS}/*.npz` — extracted landmarks (training/extract_asl_citizen.py).
- `ours/{signer}/{GLOSS}/*.npz` — our own recordings (landmarks only, never video). Recorded with consent.
- Holdout signer (never trained on): passed to `training.eval` as `--holdout-signer`.
