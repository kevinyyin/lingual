"""FastAPI app: one WebSocket per browser page, one Session per socket."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
import csv
import json
import os

import anthropic

from urllib.parse import urlparse

from fastapi import Body, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import session as S
from .classifier import DEFAULT_MODEL, SignPredictor
from .citizen_classifier import DEFAULT_MODEL as CITIZEN_MODEL, TRAIN_CSV as CITIZEN_CSV, CitizenSignPredictor
from .ensemble import EnsembleRecognizer
from .prior import PriorRecognizer
from .spike_classifier import DEFAULT_MODEL as SPIKE_MODEL, SpikeSignPredictor
from .compose import Composer
from .fingerspell import DEFAULT_MODEL as LETTER_MODEL, SpellReader
from .fingerspell_ctc import DEFAULT_MODEL as CTC_MODEL, CtcSpellReader
from .gloss import Gloss
from .protocol import RecordMsg, RecordedMsg, parse_client
from .phone_audio import ClientMic, WsPlayer
from .readers import FaceReaders
from .record import save_recording
from .segmenter import Segmenter
from .stt import MicTranscriber
from .voice import Voice
from .vocab import load_vocab

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
log = logging.getLogger("asl")


async def warm_up():
    """First calls of a fresh process ran 2-2.6 s; pay that before anyone signs."""
    warm_voice = getattr(VOICE, "warm", None)  # StubVoice has nothing to warm
    await asyncio.gather(Composer(CLAUDE, VOCAB).warm(), *([warm_voice()] if warm_voice else []))


@asynccontextmanager
async def lifespan(app):
    await warm_up()
    yield


app = FastAPI(lifespan=lifespan)
VOCAB = load_vocab()
_current: S.Session | None = None  # the open page's session (single-user app)
_current_ws: WebSocket | None = None


# vs 1.0 for the spike model in the ensemble. training.eval_recognition, 4,286 runs (online videos x 7 qualities),
# top-1 on the live-like span: 1:1 0.471, 2:1 0.461, ST-GCN alone 0.463.
CITIZEN_WEIGHT = 1.0
# The camera reads only the 160 hand-checked everyday signs (the avatar still shows ~2,000): among all ~2,900, rare
# look-alikes won. One-sign turns of an everyday conversation (WLASL videos, live path): right 24% -> 41%.
CAMERA_VOCAB = load_vocab(extended=None)


def _recognizer():
    """Our own trained model if present; else the published ASL Citizen ST-GCN (limited to the camera's vocabulary),
    ensembled with the asl-spike model when both exist; else whichever exists; else the scripted demo."""
    if DEFAULT_MODEL.exists():
        return SignPredictor(DEFAULT_MODEL)
    members = []
    if CITIZEN_MODEL.exists() and CITIZEN_CSV.exists():
        members.append((CitizenSignPredictor(CAMERA_VOCAB, restrict_to_vocab=True), CITIZEN_WEIGHT))
    if SPIKE_MODEL.exists():
        members.append((SpikeSignPredictor(SPIKE_MODEL), 1.0))
    if members:
        log.warning("sign recognizer: %s (curated prior)", " + ".join(type(m).__name__ for m, _ in members))
        return PriorRecognizer(members[0][0] if len(members) == 1 else EnsembleRecognizer(members, CAMERA_VOCAB), VOCAB)
    log.warning("no sign model (models/asl-citizen or the asl-spike bundle): demo recognizer reads each sign as the "
                "next word of a scripted conversation")
    return S.DemoRecognizer()


RECOGNIZER = _recognizer()


def _speller():
    """Our own letter model if trained; else the CTC reader trained on Kaggle's real fingerspelling (letter CER ~0.43
    on held-out signers vs 0.95 for the best pretrained letter model); None sends spelled segments to the sign model."""
    if LETTER_MODEL.exists():
        return SpellReader(LETTER_MODEL)
    if CTC_MODEL.exists():
        return CtcSpellReader(CTC_MODEL)
    return None


SPELLER = _speller()
VOICE = Voice(api_key=os.environ["CARTESIA_API_KEY"]) if os.environ.get("CARTESIA_API_KEY") else S.StubVoice()
CLAUDE = anthropic.AsyncAnthropic(max_retries=0)  # a retry would blow the 2 s budget; callers retry once themselves


def build_session(send, voice=None) -> S.Session:
    return S.Session(
        send=send,
        vocab=VOCAB,
        make_segmenter=Segmenter,
        recognizer=RECOGNIZER,
        readers=FaceReaders(),
        composer=Composer(CLAUDE, VOCAB),
        voice=voice or VOICE,
        gloss=Gloss(CLAUDE, VOCAB),
        log_path=ROOT / "logs" / "turns.jsonl",
        spell_reader=SPELLER,
    )


def origin_ok(origin: str | None) -> bool:
    """Only pages served from this machine may connect: browsers don't apply CORS to WebSockets,
    so without this any website could open /ws and read live captions of the room. ALLOWED_ORIGINS adds exact
    origins, e.g. the phone's tunnel (https://….trycloudflare.com)."""
    if not origin:
        return False
    if origin.rstrip("/") in [o.strip().rstrip("/") for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]:
        return True
    u = urlparse(origin)
    return u.scheme in ("http", "https") and u.hostname in ("localhost", "127.0.0.1", "::1")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global _current, _current_ws
    if not origin_ok(ws.headers.get("origin")):
        await ws.close(code=1008)
        return
    await ws.accept()
    if _current_ws is not None:  # one page owns the session and the mic; a second open tab also heard and signed
        try:
            await _current_ws.close(code=4001, reason="opened in another tab")
        except Exception:
            pass
    _current_ws = ws
    lock = asyncio.Lock()

    async def send(msg):
        async with lock:
            await ws.send_text(msg.model_dump_json())

    # The iOS app's page brings its own mic and speaker (/ws?audio=client): audio rides this socket as binary frames.
    client_mic = ClientMic() if ws.query_params.get("audio") == "client" else None
    voice = VOICE
    if client_mic and isinstance(VOICE, Voice):
        loop = asyncio.get_running_loop()

        async def send_audio(b: bytes):
            async with lock:
                await ws.send_bytes(b)

        voice = Voice(client=VOICE.client, voice_id=VOICE.voice_id, player_factory=lambda: WsPlayer(send_audio, loop))
        warm = asyncio.create_task(voice.warm())
    sess = _current = build_session(send, voice)
    await send(S.StateMsg(state="calibrating"))
    mic = None
    if os.environ.get("CARTESIA_API_KEY"):
        mic = asyncio.create_task(MicTranscriber(
            sess.on_transcript, sess.hears_speech,
            api_key=os.environ["CARTESIA_API_KEY"], **({"mic": client_mic} if client_mic else {})).run())
    else:
        log.warning("CARTESIA_API_KEY missing: no speech input (use ?dev=1 to type what the hearing person says)")
    try:
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                if client_mic:
                    client_mic.push(m["bytes"])
                continue
            raw = m.get("text") or ""
            try:
                msg = parse_client(raw)
            except ValueError as e:
                log.warning("bad client message: %s", e)
                continue
            if isinstance(msg, RecordMsg):
                try:
                    path = await asyncio.to_thread(save_recording, ROOT / "data" / "ours", msg, VOCAB)
                    await send(RecordedMsg(ok=True, detail=str(path.relative_to(ROOT))))
                except ValueError as e:
                    await send(RecordedMsg(ok=False, detail=str(e)))
                continue
            await sess.on_client(msg)
    except WebSocketDisconnect:
        pass
    finally:
        if mic:
            mic.cancel()
        await sess.close()
        if voice is not VOICE:
            warm.cancel()
            await voice.close()  # its own TTS socket; otherwise every reload leaks one until Cartesia times it out
        if _current is sess:
            _current = None
        if _current_ws is ws:
            _current_ws = None


@app.get("/vocab")
def vocab():
    return [{"gloss": s.gloss, "english": s.english, "avatar_ok": s.avatar_ok} for s in VOCAB.active]


ASL_CITIZEN = ROOT / "data" / "asl_citizen" / "ASL_Citizen"


WLASL_CLIPS = ROOT / "data" / "wlasl" / "clips.json"


@app.get("/reference/{gloss}")
def reference(gloss: str):
    """Reference video for a vocab gloss (recorder and avatar review): the WLASL video its avatar clip was built from
    (training.clips_from_wlasl), else the first ASL Citizen train video."""
    sign = VOCAB.get(gloss)
    if not sign:
        raise HTTPException(404)
    if WLASL_CLIPS.exists():
        src = json.loads(WLASL_CLIPS.read_text(encoding="utf-8")).get(gloss)
        video = ROOT / "data" / "wlasl" / "videos" / f"{gloss}_{src['video_id']}.mp4" if src else None
        if video and video.exists():
            return FileResponse(video, media_type="video/mp4")
    try:
        with open(ASL_CITIZEN / "splits" / "train.csv", newline="") as fh:
            for row in list(csv.reader(fh))[1:]:
                if row[2].strip().upper() == sign.asl_citizen.upper():
                    return FileResponse(ASL_CITIZEN / "videos" / row[1])
    except FileNotFoundError:
        pass
    raise HTTPException(404, "ASL Citizen not downloaded or gloss missing")


@app.post("/dev/transcript")
async def dev_transcript(request: Request, text: str = Body(..., embed=True)):
    """Dev stand-in for speech (page opened with ?dev=1): types what the hearing person says."""
    if not origin_ok(request.headers.get("origin")):
        raise HTTPException(403)
    if _current:
        await _current.on_transcript(text, True)
    return {"ok": _current is not None}


class _Hashed(StaticFiles):
    """Vite's built assets have content hashes in their names: cache them for good."""

    def file_response(self, *a, **kw):
        r = super().file_response(*a, **kw)
        r.headers["cache-control"] = "public, max-age=31536000, immutable"
        return r


def mount_web(app: FastAPI, dist: Path, public: Path) -> None:
    """Serve the production build (npm run build) and web/public from this server, for the phone: one bundled,
    minified page instead of the dev server compiling dozens of modules through the tunnel. Call after the routes.
    shortcut: only public's folders (signs, avatar, models, wasm, letters), not its loose files; add any the page needs."""
    if not (dist / "teach.html").exists():
        return
    # level 1: sign clips come 2.3x smaller in ~4 ms through the phone's tunnel (level 9 takes 10x as long for 2.5x)
    app.add_middleware(GZipMiddleware, compresslevel=1)
    if (dist / "assets").is_dir():
        app.mount("/assets", _Hashed(directory=dist / "assets"), name="assets")
    for d in sorted(public.iterdir()) if public.is_dir() else []:
        if d.is_dir():
            app.mount(f"/{d.name}", StaticFiles(directory=d), name=f"public-{d.name}")
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")


mount_web(app, ROOT / "web" / "dist", ROOT / "web" / "public")


def run():
    import uvicorn
    # uvloop rejects happy_eyeballs_delay (voice.WS_OPTIONS); asyncio has it
    uvicorn.run("server.main:app", host="127.0.0.1", port=int(os.environ.get("PORT", 8000)), loop="asyncio")
