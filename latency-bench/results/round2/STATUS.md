# Round 2 status (15 Oct 2026): complete

- Experiments 1–6 are done: 1,435 new runs, 3 failed on connection errors. Batch ids are in `batches.json`.
- Experiment 6's Groq conditions are blocked because there is no GROQ_API_KEY.
- Experiments 7–8 are prepared and unit-tested but have not been run.
- Report: `REPORT-ROUND2.md` (built by `write_report.py`). Interactive plots: `results/report.html`.
- Estimated OpenAI spend for round 2 is $11.53, based on audio minutes. Please confirm it on the dashboard.
- Checks:
  - Self-test passed.
  - pytest: 49 passed, 1 failed. The pacing p99 test measured 6.5 ms against a 5 ms limit under background CPU load.
- On this machine, Windows Application Control blocks `bench.exe`. Run the harness with `uv run python -m bench.cli ...`.
