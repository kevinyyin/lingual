# Plan C: scripted demo sentences (snap a signed sentence to the nearest demo sentence)

**Goal:** short signed sentences read right more than 80% of the time in the demo.

**Effort:** about 1–2 h.

**Owner:** _(teammate)_. Plan E (sign-by-sign mode) is being built separately on branch `plan-E`.

## Why

We replayed 300 short sentences (2–4 signs, spliced from real ASL Citizen and ASL Signbank clips) through the translator:

- **Exactly right:** 53% of sentences; 59% if the signer pauses about 0.25 s between signs.
- **The bottleneck is cutting, not reading.** The turn is cut into the wrong *number* of signs 34–40% of the time; usually two signs get merged into one.
- **When the count is right, signs read well:** each piece's top guess is right 90% of the time, and the right sign is in its top 3 for 97–98%.
- **So no smarter word choice gets past about 60%,** because the piece count limits it.

**The fix:** for the demo, we know which sentences will be signed. Keep a list of 15–20 demo sentences. After a turn is read, pick the demo sentence that best matches what was read. Matching tolerates a merged or split piece. When nothing matches well, fall back to normal translation.

**Offline estimate** (`training/teach_eval/sentence_eval.py` outputs): 96–98% of sentences snap to the right one. Those test distractors barely overlapped, though. Real demo sentences share words, so expect about **85–95%**. That's why the sentences must differ from each other (rule below).

## What to build

### 1. `vocab/demo_sentences.yaml` (new)

Each entry is the glosses in signing order plus the exact English to speak:

```yaml
- signs: [HELLO, NICE, MEET]
  english: "Hello, nice to meet you!"
- signs: [WANT, COFFEE, PLEASE]
  english: "I'd like a coffee, please."
- signs: [NEED, BATHROOM, WHERE]
  english: "Where is the bathroom? I need it."
# ... 15-20 total
```

Rules for picking sentences:

- **Use only signs the model reads well.** From our eval, these read right at least 83% of the time unless a rate is noted:
  - greetings and social: HELLO, NICE, MEET, WELCOME, SORRY, FINE, OK, THANK-YOU (0.71);
  - questions: NAME (0.71), HOW, WHERE (0.71);
  - wants and needs: YOUR, WANT, NEED, PLEASE, HELP, LIKE, FAVORITE, READY, FINISH;
  - food and drink: EAT, DRINK, THIRSTY, COFFEE, JUICE;
  - places and travel: BATHROOM, HOSPITAL, RESTAURANT, HOME, GO, CAR, BUS;
  - people and feelings: FAMILY, FEEL, HURT, EXCITED;
  - time: TOMORROW, YESTERDAY, LATER, NOW;
  - learning: LEARN, SIGN, KNOW, REMEMBER.
- **Avoid weak signs:** GOOD (0.33; it's read as THANK-YOU), MY (0.5), WHAT (0.5), YOU (0.6), WATER (0.67), HUNGRY (0.67), MINUTE/HOUR.
- **There is no ME sign.** I exists. Usually just drop the pronoun, as ASL often does.
- **Every pair of sentences differs in at least 2 signs.** Not both WANT COFFEE PLEASE and WANT JUICE PLEASE; make one of them FAVORITE JUICE.
- **Use 2–4 signs per sentence.** 2–3 is safest.
- **Rehearse the exact sentences.** Sign them the way the demo signer will.

Suggested starting list: HELLO NICE MEET · YOUR NAME HOW · WANT COFFEE PLEASE · NEED BATHROOM WHERE · THANK-YOU HELP · FEEL HURT NEED HOSPITAL · TOMORROW GO RESTAURANT · YESTERDAY LEARN SIGN · FAVORITE JUICE · FAMILY HOME LATER · SORRY REMEMBER · WELCOME HOME · BUS NOW GO · EXCITED MEET FAMILY.

### 2. `server/demo_sentences.py` (new; keep all logic here)

Keep everything in this new module so the hook in `session.py` stays tiny. `plan-E` also edits `session.py`, so this avoids merge conflicts.

```python
def load(path) -> list[DemoSentence]           # empty list if the file is missing (feature off)
def snap(tops: list[list[tuple[str, float]]], sentences) -> tuple[DemoSentence | None, float, float]
    # tops = per-piece ranked (gloss, p) lists; returns (best, best_score, runner_up_score)
```

How `snap` scores a candidate sentence S against the pieces:

- **Same number of pieces as signs in S:** score = Σ log p(S[i] in piece i's top-k), with p = 1e-3 if the sign isn't in that top-k. Use k = 5–10, since the right sign is in the top 3 for 97% of pieces.
- **Different number** (merged or split pieces): align with a small DP, an edit distance whose costs are:
  - substitution: −log p of that sign in the piece, or a large penalty if it's not there;
  - one piece covering two consecutive signs of S (a merge): the better of the two signs' −log p, plus a penalty of about 1.0;
  - an extra piece (a split): a penalty of about 1.5.

  A plain edit distance on each piece's top guess against S also works as a first version. It was tested at about 96%.
- **Normalise by sentence length,** so short sentences aren't favoured.
- **Accept the best sentence only if both hold:**
  - its normalised score clears a floor;
  - it beats the runner-up by a margin.

  Otherwise return None and the normal translation runs. Tune both on rehearsal recordings; start with a generous floor.

### 3. The hook in `server/session.py` `_sign_turn` (about 10 lines)

Put it right after `segments, tops = ...` (currently about line 382) and before the picker loop. Only do it when demo mode is on, with an env var `LINGUAL_DEMO_SENTENCES=1` and/or `?demo=1` sent in the mode message.

- **Snapping needs the top 10 per piece, not the top 3.** Re-read each piece with `self.recognizer.topk(s.context or s.frames, 10)` inside `asyncio.to_thread`. That's one model call per piece, about a few ms each. Or widen `_read_turn` to return the top 10.
- **On a confident snap:**
  - skip the per-sign picker;
  - send the sign chips (`YouSaidMsg(read=[SignToken(kind="sign", gloss=g) for g in S.signs])`);
  - speak `S.english` directly, skipping Claude, so it's faster and always exactly the scripted English;
  - log `turn.result["demo_sentence"] = {...}` with the score and margin.
- **On no snap:** fall through to today's code, unchanged.

**Optional nicer UX:** when the top two sentences are close, reuse the picker to offer them as two or three *sentence* options. `PickOption.english` can carry the sentence text, and the gloss field a sentence id.

### 4. Tests (`tests/test_demo_sentences.py`)

- **Exact pieces:** pieces whose top-1 equals the sentence snap to it.
- **One merged piece:** a turn where two signs were read as one piece still snaps to the right sentence.
- **A substitution inside the top 5:** snaps.
- **Garbage:** returns None, so the fallback runs.
- **Missing yaml:** the feature is off and nothing changes. The existing `tests/test_session.py` must stay green.

## How to check it

- **Live:** run the app, sign each demo sentence 5 times, and count right snaps. Target 4 out of 5 or better per sentence. Swap out any sentence that keeps failing.
- **Offline (only on the laptop that ran the evaluation):** the evaluation data in `app/data/teaching/` is not in git. Ask for a run of `uv run python -m training.teach_eval.sentence_eval --n 300 --patch <patch>` with sentences restricted to the demo list. The harness already supports `--patch` and per-sentence output.
- **Regression:** with the flag off, nothing changes. `uv run pytest -q` must stay green.

## Be upfront in the pitch

This is a demo mode: it only recognises the scripted sentences. Present it as "demo phrases"; outside them, the app falls back to open translation, which gets about 53–59% of short sentences exactly right.
