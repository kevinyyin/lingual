# corpus/audio — source recordings

Real human recordings are preferred over synthetic speech: synthetic speech is cleaner,
more evenly paced and has crisp onsets/offsets, which can flatter every provider.

## Add your own recording for an existing manifest entry

1. Record the sentence from the entry's `text` in `../manifest.yaml` (mono, any sample rate
   ≥ 16 kHz, a quiet room, speak naturally — do not over-enunciate).
2. Save it as `audio/<id>.wav`, e.g. `audio/es_en_medium_1.wav`. Leading/trailing silence is
   trimmed automatically; `prepare` adds the standard room-tone padding.
3. If `audio/<id>.synth.json` exists (left by an earlier synthetic run), delete it — that
   file is what marks a clip `synthetic: true`.
4. Run `uv run bench prepare`. Derived clips (noisy, overlap, same_language) built from it
   are rebuilt too and become `real` as long as all their inputs are real.

## Add a new clip

Append an entry under `clips:` in `../manifest.yaml`:

```yaml
- id: es_en_medium_4          # unique; the wav is audio/<id>.wav
  pair: es-en                 # session direction it is streamed into
  category: medium            # short | medium | long | verb_final
  source_lang: es             # language actually spoken
  text: "¿A qué hora sale el próximo tren a Sevilla?"
  reference: "What time does the next train to Seville leave?"
  synthetic: false
```

Reports break results out by `real` vs `synthetic`, so mixing is safe.
