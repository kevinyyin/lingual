## Preparation for experiments 7 and 8 (built and unit-tested, not run)

- **Commit-rule parameters** can all be set per batch: `--set cascade.incremental.min_segment_words=3`, `holdback_words`,
  `stable_updates`, and the new `stable_ms` (a word is committed once it, and every word before it, has stood unchanged
  for that long; replaces the update-count rule when > 0). Round-2 plans pass the same overrides through `bench r2 run`.
- **Anticipation mode** (`translate_mode: anticipation`) sits behind a feature flag (`anticipation_enabled: true`;
  without it the cascade refuses to run) with two strategies:
  - `single_call`: the LLM returns only the translation prefix it is confident will not change given the partial source.
  - `sampled`: N short continuations of the partial source are generated in parallel, prefix + continuation is
    translated for each, and the longest common translated prefix beyond what was already spoken is committed.
  At end of turn the rest is translated as a continuation of what was already spoken.
- **Tests:** word-level longest-common-prefix (case/punctuation-insensitive, never splits words), stripping already
  committed text, and `stable_ms` timing including a revision restarting the clock (`tests/test_anticipation.py`).
