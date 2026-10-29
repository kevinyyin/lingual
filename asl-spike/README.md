# asl-spike: early research spike; not product code

This folder holds throwaway research code for a go/no-go feasibility check, run in October 2026 before the main
build in `app/`. The app does not import this code, though `app/server/spike_classifier.py` can load the spike's trained sign
model as an optional extra. The findings carried over: see `REPORT-SPIKE.md` and `NEXT.md`.

The MVP being de-risked is a two-way ASL bridge for one café scenario:

- **Sign → speech:** isolated-sign recognition over about 150 signs, a face grammar/affect parser,
  LLM gloss → English, and Cartesia Sonic speech.
- **Speech → sign:** Cartesia Ink speech-to-text, LLM English → gloss with facial grammar, and a browser avatar.

No recordings of people were made. Every test uses public datasets of fluent Deaf signers.

## Layout

| Path | What |
|---|---|
| `REPORT-SPIKE.md` | All results: gates table first, then human steps and per-experiment tables |
| `NEXT.md` | What the main build should reuse as knowledge, in priority order |
| `scenario.md`, `vocab.json` | S1 café dialogue and the customer vocabulary with dataset coverage |
| `results/*.json(l)` | Raw results and configs for every experiment |
| `spike/` | Experiment code (S1–S8) |
| `listening/index.html` | Blind listening test for the S4 voices (local only; open via a local server) |
| `avatar/index.html` | S6 avatar prototype (needs `avatar/signs.json`, built locally from ASL Citizen) |
| `data/` | Datasets (git-ignored; never redistributed) |
| `cache/` | Landmarks, audio and model caches (git-ignored) |
| `.env` | API keys (git-ignored) |

## Running

Python 3.11 with uv. On this Windows machine, Application Control blocks `.exe` shims, so always use the
module form:

```
uv sync
uv run --env-file .env python spike/s4_tts.py          # any experiment script; see each file's docstring
uv run python spike/avatar_server.py 8766              # then open http://127.0.0.1:8766/index.html
uv run python -m http.server 8767 --directory listening   # listening sheet
```

## Data licenses

Each dataset's license is recorded in REPORT-SPIKE.md. In short:

- **ASL Citizen:** non-commercial research only. No distribution of the data or of modified data, and no transfer to third parties.
- **How2Sign and RWTH-BOSTON-104:** CC BY-NC.
- **PopSign:** CC BY 4.0.
- **Kaggle competition data:** see the competition rules.
- **NCSLGR:** BU/ASLLRP terms. Cite the ASLLRP.
