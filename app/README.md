# ASL Dragon: Two-Way ASL

A Deaf ASL signer and a hearing person talk face to face on one laptop. Signing becomes spoken English in a voice
that carries the signer's emotion; speech becomes a 3D avatar signing with facial grammar and the speaker's tone,

**Limits:** not yet reviewed by a fluent Deaf signer (required before anyone outside the team uses it); single signs,
short sentences of everyday signs and fingerspelling, not fluent continuous ASL; needs steady front lighting. Models
and clips derived from ASL Citizen, WLASL, ASL-LEX and the Kaggle competitions are for **non-commercial research
only**; `models/signs.pt` (trained on ASL Citizen) is published in the repo at its owner's decision.

## Models (all published or trained on public data)

| Part | Model | Where it comes from |
|---|---|---|
| Sign recognition | Our own model for the camera's signs (BiGRU over shoulder-normalized landmarks), trained on ASL Citizen videos of 155 of them, also as pieces of sentences: 69% of new WLASL signers' single signs right (ST-GCN 58%), 53% inside sentences (48%); replayed sentences through the translator, signs wrong 75% -> 60% with `server/recut.py`. Without it: the ASL Citizen ST-GCN (Microsoft Research, 2,731 signs from 27 keypoints), ensembled with the asl-spike model when present | `models/signs.pt` (`training.extract_asl_citizen --remote`, then `training.train_signs`; research use only), `models/asl-citizen/` (GitHub release `checkpoints_v1`), `models/asl-sign-models/` (teammate bundle) |
| Fingerspelling | Pretrained CTC reader on MediaPipe Holistic landmarks, run by Core ML on the Mac (Apache-2.0; trained on Google's Kaggle ASL Fingerspelling corpus, CC-BY 4.0): ~17% of letters wrong on Kaggle names fed as the app does; else a CTC reader trained here (~43%) | `models/fingerspell_coreml/` (Hugging Face `Skunk0/openglasses-fingerspelling-ctc`), `models/fingerspell_ctc.pt` (`training.train_fingerspell_ctc`) |
| English -> sign plan | Claude (`claude-haiku-5-5`), prompted with ASL grammar rules and the signs the avatar can show | `server/gloss.py` |
| Signs -> English | Claude, from recognized signs + facial grammar + emotion | `server/compose.py` |
| Avatar | Our stylized "toon" character is the default: rounded shapes, flat colours, big eyes, painted-on face with the 20 ARKit shapes the app drives, on MPFB's rig (`scripts/build_toon_avatar.py`, no Blender needed), as six signing buddies picked on the Profile tab, each with its own Cartesia voice (`web/src/buddies.ts`, `server/voice.py`). `?model=Male_Adult_11` loads Microsoft Rocketbox (MIT); `?model=mpfb` loads our own MPFB character (MakeHuman for Blender, CC0: V-taper build, hooded eyes, charcoal tee, 52 ARKit face shapes; `scripts/build_avatar.py`, needs Blender 4.5 + MPFB); `?model=Male_Adult_07` the other Rocketbox | `web/public/avatar/toon/`, `web/public/avatar/mpfb/`, `web/public/avatar/rocketbox/` |
| Avatar motion | Clips from WLASL dictionary videos (MediaPipe), the most typical of up to 4 takes per sign. Wrists placed where the signer's were relative to their body (relative to the nose near the face), elbows by two-bone IK, every finger bone aimed; hands kept out of the body | `web/public/signs/` (`training.clips_from_wlasl`), `web/src/avatar/rocketbox.ts` |
| Fingerspelling handshapes | A-Z from Deaf signers in the Kaggle data: the reader's surest frame per letter, depth and knuckle bend corrected | `web/public/letters/` (`training.letters_from_kaggle`) |
| Voice, speech-to-text | Cartesia Sonic 3.6, Ink 2 | `.env` keys |

The vocabulary is the 160 hand-checked signs in `vocab/vocab.yaml` plus every ASL Citizen and WLASL sign
(~2,900) in the generated `vocab/vocab_extended.yaml`, with English meanings from ASL-LEX 2.0. The camera reads only
the hand-checked signs (a lesson only its course's signs): 155 of the 160, since HOW-MANY, FIVE, TEN, FOOD and FORGET
have no ASL Citizen videos. The avatar shows any sign it has a clip for.

Measured: `models/signs.pt` reads 85% of unseen ASL Citizen signers' signs (84% inside made-up sentences). On other
people's WLASL videos replayed through the live path at 15 fps: 69% of single signs right (the ST-GCN it replaced:
58%), 53% of signs inside sentences (48%); whole replayed sentences, with `server/recut.py` cutting them, 60% of signs
wrong (75%); 94% of correct signing passes a lesson's camera check. Which signs read best (measured with the ST-GCN):
`reports/sign_reliability.md`. English -> sign plans: 182/182 checks passed. Fingerspelling (the Core ML reader):
17% of letters wrong on Kaggle names fed the way the app sends frames.

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

# our sign model: landmarks of the ASL Citizen videos of the camera's signs (~5,600, fetched one at a time out of the
# 42.8 GB zip, ~1.5 h), then ~7 min of training -> models/signs.pt
uv run python -m training.extract_asl_citizen --remote
uv run python -m training.train_signs

# fingerspelling reader (Core ML, Macs only; 18 MB)
uv run --with huggingface_hub python -c "from huggingface_hub import snapshot_download; snapshot_download('Skunk0/openglasses-fingerspelling-ctc', local_dir='models/fingerspell_coreml', allow_patterns=['*.mlpackage/*', 'vocab.txt', 'LICENSE'])"

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

Open http://localhost:5173 and pick **Voice** (hold Space or the Hold to talk button and talk; the avatar signs) or **Signing** (you sign, it speaks). In
signing mode the camera tells you where to sit and calibrates by itself once you're in position (and again if you
move). Sign one sign per movement, pause briefly between signs, lower your hands for about half a second to finish.
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

## Plushie arms and neck display (optional)

An Arduino Uno R4 WiFi in the plushie reacts to each teaching-mode answer: right shows a check on the MAX7219 8x8 dot
matrix on its neck, lights the right LED and flaps both arms; wrong shows an X and lights the wrong LED; a first missed sign (a second try follows) shows a !. Wiring: servo
signals on pins 6 and 9, LEDs on pins 2 (right) and 3 (wrong) through 220 ohm resistors, matrix DIN 11, CS 10, CLK 13;
everything's 5V and GND on the breadboard rails. Upload `arduino/plush_arms/` once (Arduino IDE, or
`arduino-cli compile --upload -p <port> --fqbn arduino:renesas_uno:unor4wifi arduino/plush_arms`) and leave the board
plugged into the laptop that runs `asl-server`: the server finds it by itself (`server/plush.py`) and skips it when it
isn't there. Tune the arm angles, `BRIGHTNESS` and `MIRROR` at the top of the sketch on the real plush.

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
