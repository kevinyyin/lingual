# Two-Way ASL

A Deaf ASL signer and a hearing person talk face to face on one laptop. Signing becomes spoken English in a voice
that carries the signer's emotion; speech becomes a 3D avatar signing with facial grammar and the speaker's tone,

**Limits:** not yet reviewed by a fluent Deaf signer (required before anyone outside the team uses it); isolated
signs plus fingerspelling, not continuous ASL; needs steady front lighting. Models and clips derived from ASL Citizen,
WLASL, ASL-LEX and the Kaggle competitions are for **non-commercial research only** and must not be redistributed.

## Models (all published or trained on public data)

| Part | Model | Where it comes from |
|---|---|---|
| Sign recognition | ASL Citizen ST-GCN (Microsoft Research): 2,731 signs from 27 pose + hand keypoints, ensembled with the asl-spike model (143 signs) | `models/asl-citizen/` (GitHub release `checkpoints_v1`), `models/asl-sign-models/` (teammate bundle) |
| Fingerspelling | CTC reader trained here on Google's Kaggle ASL Fingerspelling landmarks (Deaf signers) | `models/fingerspell_ctc.pt` (`training.train_fingerspell_ctc`) |
| English -> sign plan | Claude (`claude-haiku-5-5`), prompted with ASL grammar rules and the signs the avatar can show | `server/gloss.py` |
| Signs -> English | Claude, from recognized signs + facial grammar + emotion | `server/compose.py` |
| Avatar | Our stylized "toon" character is the default: rounded shapes, flat colours, big eyes, painted-on face with the 20 ARKit shapes the app drives, on MPFB's rig (`scripts/build_toon_avatar.py`, no Blender needed). `?model=Male_Adult_11` loads Microsoft Rocketbox (MIT); `?model=mpfb` loads our own MPFB character (MakeHuman for Blender, CC0: V-taper build, hooded eyes, charcoal tee, 52 ARKit face shapes; `scripts/build_avatar.py`, needs Blender 4.5 + MPFB); `?model=Male_Adult_07` the other Rocketbox | `web/public/avatar/toon/`, `web/public/avatar/mpfb/`, `web/public/avatar/rocketbox/` |
| Avatar motion | Clips from WLASL dictionary videos (MediaPipe), the most typical of up to 4 takes per sign. Wrists placed where the signer's were relative to their body (relative to the nose near the face), elbows by two-bone IK, every finger bone aimed; hands kept out of the body | `web/public/signs/` (`training.clips_from_wlasl`), `web/src/avatar/rocketbox.ts` |
| Fingerspelling handshapes | A-Z from Deaf signers in the Kaggle data: the reader's surest frame per letter, depth and knuckle bend corrected | `web/public/letters/` (`training.letters_from_kaggle`) |
| Voice, speech-to-text | Cartesia Sonic 3.6, Ink 2 | `.env` keys |

The vocabulary is the 160 hand-checked signs in `vocab/vocab.yaml` plus every ASL Citizen and WLASL sign
(~2,900) in the generated `vocab/vocab_extended.yaml`, with English meanings from ASL-LEX 2.0.

Measured (details in the execution record): sign recognition on signers no model trained on, choosing among all
~2,900 signs, top-1 0.60 / top-3 0.81, holding up under 240p, blur, low light and half frame rate; on WLASL's mixed
online videos top-1 ~0.25 / top-3 ~0.49 (the top-3 picker covers the gap). English -> sign plans: 182/182 checks
passed. Fingerspelling: 0.36 letter error rate on held-out Deaf signers.

## Setup

```bash
cd app
uv sync                       # Python 3.12 backend
cp .env.example .env          # add ANTHROPIC_API_KEY and CARTESIA_API_KEY
cd web && npm install && cd ..   # also fetches MediaPipe wasm + models into web/public

# published sign models (MIT repo; weights trained on ASL Citizen)
mkdir -p models/asl-citizen && curl -L -o models/asl-citizen/w.zip \
  https://github.com/microsoft/ASL-citizen-code/releases/download/checkpoints_v1/ASL_citizen_stgcn_weights.zip
uv run python -c "import zipfile; zipfile.ZipFile('models/asl-citizen/w.zip').extractall('models/asl-citizen')"
uv run python -m training.remote_zip ASL_Citizen/splits/train.csv ASL_Citizen/splits/val.csv ASL_Citizen/splits/test.csv
#   ^ reads just the label CSVs out of the 42.8 GB ASL Citizen zip with HTTP range requests (label order of the model)

uv run python -m training.build_vocab        # ~2,900-sign vocabulary (ASL Citizen + ASL-LEX + WLASL)
uv run python -m training.clips_from_wlasl   # avatar clips from WLASL videos (resumable; --workers 8)
uv run python -m training.touch_from_asllex   # contact data; rerun after every clip rebuild
```

Optional: unzip the asl-spike bundle (`asl-sign-models-*.zip`) into `models/` for the ensemble, and train the
fingerspelling reader (below). Without keys the app still runs: no Cartesia key means no voice or mic (open the page
with `?dev=1` to type what the hearing person says); no Anthropic key means English is the plain sign words and plans
are fingerspelled. Without any sign model a scripted demo recognizer stands in.

## Run

```bash
uv run asl-server             # backend on :8000 (from app/)
npm --prefix web run dev      # page on :5173 (from app/)
```

Open http://localhost:5173, sit head-to-waist in frame with hands resting and a neutral face; it calibrates by itself
in 2 s (**Calibrate** redoes it). Sign one sign per movement, pause briefly between signs, lower your hands for about
half a second to finish.
If the picker appears, press 1-3 (Esc: none of these). `?dev=1` adds a box to type what the hearing person says;
`?dev=1&camera=0` is speech-to-sign only. `?mode=review` plays each avatar clip beside the video it came from.

## Run on iPhone (the plushie)

The iOS app (`web/ios/`, Capacitor) is a full-screen web view that opens this laptop's backend through a Cloudflare
quick tunnel; recognition, Claude and Cartesia stay on the laptop. The backend also serves the production build of the
page (`web/dist`) and `web/public`, so the phone gets one bundled page instead of the dev server's modules. Needs Xcode
and `brew install cloudflared`.

```bash
npm --prefix web run build                               # after every web change (dist/ is what the phone gets)
cloudflared tunnel --url http://localhost:8000          # prints https://….trycloudflare.com
ALLOWED_ORIGINS=https://….trycloudflare.com uv run asl-server   # from app/: lets that origin open /ws
cd web && npx cap sync ios && npx cap open ios           # first time, or after changing web/shell or ios/
```

In Xcode pick the iPhone and Run. In the app, paste the tunnel URL and tap Open: teaching mode opens with the phone's
camera, mic and speaker. The URL changes every time the tunnel restarts: paste the new one (the app remembers the last)
and restart the server with the new `ALLOWED_ORIGINS`. Installs from a free Apple ID expire after 7 days. Anyone who
learns the tunnel URL can use the app while it runs: stop the tunnel after the demo. (`npm --prefix web run dev:tunnel`
with the tunnel on :5173 instead gives live edits, but loads much slower on the phone.)

## Fingerspelling

```bash
# real fingerspelling from Deaf signers (accept the competition rules on kaggle.com first; KAGGLE_USERNAME/KAGGLE_API_TOKEN)
uvx kaggle competitions download -c asl-fingerspelling -f train.csv -p data/kaggle_fs
uvx kaggle competitions download -c asl-fingerspelling -f train_landmarks/1019715464.parquet -p data/kaggle_fs  # + more files
uv run --with pyarrow --with pandas python -m training.train_fingerspell_ctc   # -> models/fingerspell_ctc.pt
```

## Evaluate

```bash
uv run python -m training.eval_recognition          # real online videos (ASL Citizen test signers + WLASL) x 7 quality conditions
uv run python -m training.eval_gloss                # ~180 live English -> sign plans checked for grammar, signs and ASL order
uv run --with pyarrow --with pandas python -m training.eval_fingerspell   # pretrained letter model on Kaggle signers
uv run python -m training.eval latency --last 30    # turn delays from logs/turns.jsonl (each direction <= 2000 ms)
uv run python scripts/probe_cartesia.py             # live Sonic/Ink API shapes
uv run --with playwright python scripts/record_demo.py   # screen recordings of both directions + an avatar tour
uv run pytest && (cd web && npx vitest run)          # unit tests
```

Reports land in `data/eval/`. Results, decisions and their evidence are in the execution record (linked above).

## Our own data (optional)

Record our signing at `http://localhost:5173/?mode=record&signer=NAME` (five takes per sign; `&grid=1` emotion
grid, `&letters=1` alphabet, `&names=1` names), then `training.train_signs` / `training.train_fingerspell` train
`models/signs.pt` / `models/letters.pt`, which take priority over the published models when present. Tune
`UP_T`/`DOWN_T` in `server/grammar.py` against the grid, `PICK_THRESHOLD` in `server/session.py` so no wrong sign is
spoken without a pick, and `MAPPING` in `server/voice.py` with a listening test.
