# Adjudicator: settle verifier disagreements

You get a list of items where two verifiers disagreed on `matches_label`: each has `item_id`, `sheet`, `gloss`,
`description`, and both verifiers' JSON answers. Open each `sheet` (Read the .jpg), weigh the two answers and their notes,
and decide yourself (same rules as `verifier.md`: variants and mirroring are still the sign; when you truly cannot tell,
`unsure`). Append one line per item to `app/data/teaching/verify/results/adjudication.jsonl` in the verifier schema, plus
`"adjudicated": true` and in `note` which verifier you sided with and why. Summarise counts at the end.
