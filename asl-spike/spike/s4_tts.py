"""S4: Cartesia Sonic 3.6 expressive voice. Generates audio over one WebSocket, sequentially
(concurrency 1), one fresh context per request, and records raw perf_counter_ns timestamps.

Conditions per sentence and voice:
  a  no emotion (one clip per sentence/voice; scored against each intended emotion)
  b  generation_config.emotion
  c  emotion + speed/volume from the S3 mapping (to_voice with emulated kinematics)
  mm mismatch texts: emotion contradicting the words, plus the text-consistent emotion for contrast

Outputs: cache/s4/<id>.wav (git-ignored), results/s4_tts_runs.jsonl, results/s4_config.json.
Stops (and records why) on an API failure that looks like billing or auth (402/401/403/429/credits).
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import websockets

sys.path.insert(0, os.path.dirname(__file__))
from voice_map import EMULATED_KINEMATICS, to_voice  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache" / "s4"
RES = ROOT / "results"
URL = "wss://api.cartesia.ai/tts/websocket?cartesia_version=2026-08-14"
MODEL = "sonic-3.6"
SR = 24000
SEED = 13

VOICES = {"Leo": "0834f3df-e650-4766-a20c-5a93a43aa6e3", "Maya": "cbaf8084-f009-4838-a096-07ee2e6612b1",
          "Gavin": "f4a3a8e4-694c-4c45-9ca0-27caf97901b5"}
EMOTIONS = ["neutral", "happy", "sad", "angry"]
SENTENCES = [
    "Hi! I'd like a coffee, please.",
    "Large, please.",
    "Do you have oat milk?",
    "Oat milk, and no sugar.",
    "What kinds of muffins do you have?",
    "One blueberry. Can you warm it up?",
    "My name is Maya.",
    "Sorry, how much? Again, please.",
    "Can I pay with a card?",
    "No receipt, thanks.",
    "Where? I don't understand.",
    "Okay, thank you! Have a good day.",
]
# (text, contradicting emotion, text-consistent emotion)
MISMATCH = [
    ("Thank you so much, this is perfect!", "angry", "happy"),
    ("I'm really upset, this is the wrong order.", "happy", "angry"),
    ("Yes! That's amazing, I love it!", "sad", "happy"),
    ("Have a wonderful day, see you tomorrow!", "angry", "happy"),
    ("Sorry, I'm not feeling well today.", "happy", "sad"),
    ("This is ridiculous, I've been waiting twenty minutes!", "sad", "angry"),
]
FAIL_WORDS = ("402", "401", "403", "429", "credit", "quota", "payment", "billing", "insufficient")


def jobs() -> list[dict]:
    js = []
    for si, s in enumerate(SENTENCES):
        q = s.rstrip().endswith("?")
        for v in VOICES:
            js.append({"cond": "a", "sent": si, "text": s, "voice": v, "emotion": None, "intended": None})
            for e in EMOTIONS:
                js.append({"cond": "b", "sent": si, "text": s, "voice": v, "emotion": e, "intended": e})
                vm = to_voice({"type": "yn" if q else "statement"}, {"label": e, "intensity": 0.7},
                              EMULATED_KINEMATICS[e], text=s)
                js.append({"cond": "c", "sent": si, "text": vm["text_punctuation"], "voice": v, "emotion": e,
                           "intended": e, "speed": vm["speed"], "volume": vm["volume"]})
    for mi, (t, bad, good) in enumerate(MISMATCH):
        for v in VOICES:
            js.append({"cond": "mm_contra", "sent": 100 + mi, "text": t, "voice": v, "emotion": bad, "intended": bad,
                       "text_emotion": good})
            js.append({"cond": "mm_match", "sent": 100 + mi, "text": t, "voice": v, "emotion": good, "intended": good,
                       "text_emotion": good})
    random.Random(SEED).shuffle(js)
    for i, j in enumerate(js):
        j["id"] = f"{i:04d}_{j['cond']}_{j['voice']}_s{j['sent']}_{j['emotion'] or 'none'}"
    # condition d (added after the first pass; ids kept separate so earlier ids stay stable):
    # emotion as an inline SSML-style tag in the transcript instead of generation_config
    dj = [{"cond": "d", "sent": si, "text": f'<emotion value="{e}"/> {s}', "voice": v, "emotion": None,
           "intended": e, "tag": e} for si, s in enumerate(SENTENCES) for v in VOICES for e in EMOTIONS]
    random.Random(SEED + 1).shuffle(dj)
    for i, j in enumerate(dj):
        j["id"] = f"d{i:03d}_{j['cond']}_{j['voice']}_s{j['sent']}_{j['tag']}"
    # conditions e/f: LLM-rewritten emotional wording (results/s4_rewrites.json), with (e) and without (f) the tag
    ef = []
    rp = ROOT / "results" / "s4_rewrites.json"
    if rp.exists():
        rw = json.load(open(rp))
        for si in range(len(SENTENCES)):
            for v in ("Leo", "Maya"):
                for e in EMOTIONS:
                    t = rw[f"{si}|{e}"]
                    ef.append({"cond": "e", "sent": si, "text": t, "voice": v, "emotion": e, "intended": e})
                    ef.append({"cond": "f", "sent": si, "text": t, "voice": v, "emotion": None, "intended": e})
        random.Random(SEED + 2).shuffle(ef)
        for i, j in enumerate(ef):
            j["id"] = f"{j['cond']}{i:03d}_{j['voice']}_s{j['sent']}_{j['intended']}"
    return js + dj + ef


async def synth(ws, j: dict) -> dict:
    ctx = str(uuid.uuid4())
    msg = {"model_id": MODEL, "transcript": j["text"], "voice": {"mode": "id", "id": VOICES[j["voice"]]},
           "output_format": {"container": "raw", "encoding": "pcm_s16le", "sample_rate": SR},
           "language": "en", "context_id": ctx, "continue": False}
    gc = {}
    if j.get("emotion"):
        gc["emotion"] = j["emotion"]
    if "speed" in j:
        gc["speed"], gc["volume"] = j["speed"], j["volume"]
    if gc:
        msg["generation_config"] = gc
    chunks, t_first, err = [], None, None
    t_send = time.perf_counter_ns()
    await ws.send(json.dumps(msg))
    while True:
        r = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        if r.get("context_id") not in (None, ctx):
            continue
        if r.get("type") == "chunk" and r.get("data"):
            if t_first is None:
                t_first = time.perf_counter_ns()
            chunks.append(base64.b64decode(r["data"]))
        elif r.get("type") == "done":
            break
        elif r.get("type") == "error" or r.get("error"):
            err = str(r.get("error") or r)[:300]
            break
    t_done = time.perf_counter_ns()
    pcm = np.frombuffer(b"".join(chunks), dtype=np.int16)
    out = dict(j, t_send_ns=t_send, t_first_audio_ns=t_first, t_done_ns=t_done, error=err,
               ttfb_ms=(t_first - t_send) / 1e6 if t_first else None, total_ms=(t_done - t_send) / 1e6,
               audio_s=len(pcm) / SR, chars=len(j["text"]), generation_config=gc)
    if len(pcm):
        sf.write(OUT / f"{j['id']}.wav", pcm, SR)
    return out


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RES.mkdir(exist_ok=True)
    js = jobs()
    done = set()
    runs_path = RES / "s4_tts_runs.jsonl"
    if runs_path.exists():
        done = {json.loads(x)["id"] for x in runs_path.read_text().splitlines() if x.strip() and not json.loads(x).get("error")}
    json.dump({"model": MODEL, "cartesia_version": "2026-08-14", "sample_rate": SR, "voices": VOICES,
               "emotions": EMOTIONS, "sentences": SENTENCES, "mismatch": MISMATCH, "seed": SEED,
               "n_jobs": len(js), "concurrency": 1}, open(RES / "s4_config.json", "w"), indent=1)
    key = os.environ.get("CARTESIA_API_KEY")
    if not key:
        print("skipped: no key")
        return
    todo = [j for j in js if j["id"] not in done]
    print(f"{len(todo)} of {len(js)} jobs to run, ~{sum(len(j['text']) for j in todo)} chars")
    async with websockets.connect(URL, additional_headers={"X-API-Key": key}, max_size=None) as ws:
        with open(runs_path, "a", encoding="utf-8") as f:
            for i, j in enumerate(todo):
                try:
                    r = await synth(ws, j)
                except Exception as e:  # record, never drop
                    r = dict(j, error=f"{type(e).__name__}: {e}"[:300])
                f.write(json.dumps(r) + "\n")
                f.flush()
                if r.get("error") and any(w in r["error"].lower() for w in FAIL_WORDS):
                    print("API FAILURE, stopping:", r["error"])
                    sys.exit(2)
                if (i + 1) % 25 == 0:
                    print(f"{i + 1}/{len(todo)}")
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
