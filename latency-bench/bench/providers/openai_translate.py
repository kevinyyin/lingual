"""OpenAI gpt-realtime-translate over WebSocket.

Per the docs (developers.openai.com/api/docs/guides/realtime-translation and the
translation client/server event references; read 2026-10-13, unchanged when re-read for
round 2):
  * wss://api.openai.com/v1/realtime/translations?model=gpt-realtime-translate
  * headers: Authorization: Bearer <key>, OpenAI-Safety-Identifier: <hashed id>
  * first server event: session.created
  * client: session.update {session: {audio: {output: {language}, input: {transcription, noise_reduction}}}}
            -> server: session.updated
  * client: session.input_audio_buffer.append {audio: base64 PCM16 24 kHz mono}
    (engine consumes 200 ms frames; shorter appends are buffered server-side)
  * server: session.output_audio.delta {delta, sample_rate?, channels?, format?, elapsed_ms?}
            session.output_transcript.delta {delta, elapsed_ms?}
            session.input_transcript.delta {delta, elapsed_ms?}   (only if transcription configured)
            error {error: {type, code, message, param, event_id}}  (most are recoverable)
  * client: session.close -> server flushes pending input and emits the remaining
    translation, then session.closed. There is no commit event in translation sessions.
Unknown event types are recorded as stage marks (stage="openai.<type>") so a changed
API shows up in the raw data instead of being silently dropped.

Round 2 adds flush policies (session rotation), all off by default:
  flush_policy: none      round-1 behaviour, one session
                on_pause  a causal VAD on the sent stream fires speech_end after
                          min_silence_ms; the active session is sent session.close (its
                          remaining translation keeps draining into the output) and new
                          source audio goes to a pre-opened, pre-configured standby
                          session, which becomes active; a new standby is opened at once
                watchdog  at speech_end arm a timer; if no audible translated audio arrives
                          within watchdog_ms, rotate exactly as above
Standby sessions receive no audio until promoted (no double billing); audio sent before a
promoted session finished its setup is buffered and flushed in order once it is ready.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from typing import Any

import numpy as np
import websockets

from bench.clock import now_ns
from bench.events import ProviderEvent
from bench.providers.base import Provider

AUDIBLE_PEAK = int(10 ** (-45 / 20) * 32768)  # same gate as analysis (output_gate_dbfs -45)


class _Session:
    def __init__(self, idx: int):
        self.idx = idx
        self.ws: websockets.ClientConnection | None = None
        self.reader: asyncio.Task | None = None
        self.created = asyncio.Event()
        self.updated = asyncio.Event()
        self.closed = asyncio.Event()
        self.state = "opening"      # opening | standby | active | closing | closed
        self.audio_bytes = 0
        self.t_open: int | None = None
        self.session_id: str | None = None
        self.error: str | None = None
        self.pending: list[bytes] = []   # audio routed here before it was ready
        self.cost_emitted = False
        self.close_sent = False


class OpenAITranslateProvider(Provider):
    name = "openai"
    hosts = ("api.openai.com",)

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.input_sample_rate = int(cfg.get("input_sample_rate", 24000))
        self.policy = cfg.get("flush_policy", "none")
        self.watchdog_ms = float(cfg.get("watchdog_ms", 1200))
        self.sessions: list[_Session] = []
        self.active: _Session | None = None
        self.standby: _Session | None = None
        self._tasks: list[asyncio.Task] = []
        self._ready_emitted = False
        self._target = "en"
        self._wd_task: asyncio.Task | None = None
        self._wd_armed_at: int | None = None
        self._last_audible_ns = 0
        self.vad = cfg.get("_vad")  # tests may inject a scripted VAD
        if self.vad is None and self.policy in ("on_pause", "watchdog"):
            from bench.causal_vad import CausalVAD

            self.vad = CausalVAD(self.input_sample_rate, int(cfg.get("min_silence_ms", 600)))

    @classmethod
    def check_available(cls, cfg: dict[str, Any]) -> str | None:
        return None if os.environ.get("OPENAI_API_KEY") else "OPENAI_API_KEY not set in latency-bench/.env"

    # ------------------------------------------------------------ sessions
    async def _open(self, idx: int, role: str) -> _Session:
        s = _Session(idx)
        self.sessions.append(s)
        url = f"{self.cfg.get('url', 'wss://api.openai.com/v1/realtime/translations')}?model={self.cfg.get('model', 'gpt-realtime-translate')}"
        sid = hashlib.sha256(str(self.cfg.get("safety_identifier", "latency-bench")).encode()).hexdigest()[:32]
        headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "OpenAI-Safety-Identifier": sid}
        s.ws = await websockets.connect(url, additional_headers=headers, max_size=None,
                                        compression=None, open_timeout=20, ping_interval=20)
        s.t_open = now_ns()
        self._mark("openai.ws_open", s.t_open, session=idx, role=role)
        s.reader = asyncio.create_task(self._read_loop(s))
        await asyncio.wait_for(self._wait(s, s.created), 20)
        audio: dict[str, Any] = {"output": {"language": self._target}}
        inp: dict[str, Any] = {}
        if self.cfg.get("input_transcription_model"):
            inp["transcription"] = {"model": self.cfg["input_transcription_model"]}
        if "noise_reduction" in self.cfg:
            nr = self.cfg.get("noise_reduction")
            inp["noise_reduction"] = {"type": nr} if nr else None
        if inp:
            audio["input"] = inp
        await s.ws.send(json.dumps({"type": "session.update", "session": {"audio": audio}}))
        await asyncio.wait_for(self._wait(s, s.updated), 20)
        if s.state == "opening":
            s.state = role
        if s.pending and s.state in ("active", "closing"):
            await self._flush_pending(s)
        return s

    async def _wait(self, s: _Session, ev: asyncio.Event) -> None:
        while not ev.is_set():
            if s.error and not self._ready_emitted:
                raise RuntimeError(s.error)
            if s.reader and s.reader.done():
                raise RuntimeError(s.error or "connection closed during setup")
            await asyncio.sleep(0.005)

    async def connect(self, source_lang: str, target_lang: str) -> None:
        self._target = target_lang
        self.active = await self._open(0, "active")
        if self.policy != "none":
            self._open_standby()

    def _open_standby(self) -> None:
        idx = len(self.sessions)
        task = asyncio.create_task(self._open(idx, "standby"))
        self._tasks.append(task)

        def done(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            if t.exception() is not None:
                self._emit(ProviderEvent("error", now_ns(), text=f"standby open failed: {t.exception()}",
                                         meta={"fatal": False, "session": idx}))
            else:
                self._mark("standby.ready", session=idx)

        task.add_done_callback(done)

    def _standby_session(self) -> _Session | None:
        """The newest session that is neither active nor closing (may still be opening)."""
        for s in reversed(self.sessions):
            if s is not self.active and s.state in ("opening", "standby"):
                return s
        return None

    async def _read_loop(self, s: _Session) -> None:
        assert s.ws is not None
        try:
            async for raw in s.ws:
                t = now_ns()  # stamp before any parsing
                self._handle(s, raw, t)
                if s.closed.is_set():
                    break
        except websockets.ConnectionClosed as e:
            if not s.closed.is_set():
                fatal = s is self.active or not self._ready_emitted
                self._emit(ProviderEvent("error", now_ns(), text=f"connection closed: {e}",
                                         meta={"fatal": fatal, "session": s.idx}))
                s.error = s.error or f"connection closed: {e}"
        finally:
            self._emit_cost(s)
            s.closed.set()
            if s.state != "closed":
                s.state = "closed"

    def _handle(self, s: _Session, raw: str | bytes, t: int) -> None:
        size = len(raw)
        ev = json.loads(raw)
        typ = ev.get("type", "")
        meta: dict[str, Any] = {k: ev[k] for k in ("elapsed_ms", "event_id") if k in ev}
        meta["session"] = s.idx
        if s.state == "closing":
            meta["after_promotion"] = True  # drain of a session we already rotated away from
        if typ == "session.output_audio.delta":
            pcm = base64.b64decode(ev.get("delta", ""))
            if pcm:
                a = np.frombuffer(pcm, "<i2")
                if a.size and int(np.max(np.abs(a.astype(np.int32)))) > AUDIBLE_PEAK:
                    self._last_audible_ns = t
                    if self._wd_armed_at is not None and t >= self._wd_armed_at:
                        self._mark("watchdog.disarm", t, session=s.idx)
                        self._wd_armed_at = None
            self._emit(ProviderEvent("audio_out", t, size, pcm=pcm, sample_rate=int(ev.get("sample_rate") or 24000), meta=meta))
        elif typ == "session.output_transcript.delta":
            self._emit(ProviderEvent("text_out_delta", t, size, text=ev.get("delta", ""), meta=meta))
        elif typ == "session.input_transcript.delta":
            self._emit(ProviderEvent("source_text_delta", t, size, text=ev.get("delta", ""), meta=meta))
        elif typ == "session.created":
            s.session_id = ev.get("session", {}).get("id")
            self._mark("openai.session.created", t, session_id=s.session_id, session=s.idx)
            s.created.set()
        elif typ == "session.updated":
            self._mark("openai.session.updated", t, audio=ev.get("session", {}).get("audio"), session=s.idx)
            if not s.updated.is_set():
                s.updated.set()
                if not self._ready_emitted:
                    self._ready_emitted = True
                    self._emit(ProviderEvent("session_ready", t, size))
        elif typ == "session.closed":
            self._mark("openai.session.closed", t, session=s.idx)
            self._emit_cost(s)
            s.closed.set()
        elif typ == "error":
            err = ev.get("error", {})
            msg = f"{err.get('type')}/{err.get('code')}: {err.get('message')}"
            fatal = not self._ready_emitted  # errors before the first session is configured abort the run
            self._emit(ProviderEvent("error", t, size, text=msg, meta={"error": err, "fatal": fatal, "session": s.idx}))
            if not s.updated.is_set():
                s.error = msg
        else:
            self._mark(f"openai.{typ}", t, payload_size=size, session=s.idx)

    # ------------------------------------------------------------ audio
    async def _send(self, s: _Session, chunk: bytes) -> None:
        assert s.ws is not None
        await s.ws.send(json.dumps({"type": "session.input_audio_buffer.append",
                                    "audio": base64.b64encode(chunk).decode()}))
        s.audio_bytes += len(chunk)

    async def _flush_pending(self, s: _Session) -> None:
        n = len(s.pending)
        while s.pending:
            await self._send(s, s.pending.pop(0))
        self._mark("rotation.pending_flushed", session=s.idx, chunks=n)

    async def send_audio(self, pcm16_chunk: bytes, t_sent_ns: int) -> None:
        if self.vad is not None:
            for ev in self.vad.push(pcm16_chunk, t_sent_ns):
                self._mark(f"vad.{ev.kind}", ev.t_detect_ns, sample_16k=ev.sample_16k)
                if ev.kind == "speech_end":
                    if self.policy == "on_pause":
                        await self._rotate("pause", ev.t_detect_ns)
                    elif self.policy == "watchdog":
                        self._arm_watchdog(ev.t_detect_ns)
        s = self.active
        assert s is not None
        if s.updated.is_set() and not s.pending:
            await self._send(s, pcm16_chunk)
        else:
            s.pending.append(pcm16_chunk)

    def _arm_watchdog(self, t: int) -> None:
        self._wd_armed_at = t
        self._mark("watchdog.arm", t)

        async def fire() -> None:
            from bench.clock import sleep_until_ns

            armed = t
            await sleep_until_ns(armed + int(self.watchdog_ms * 1e6))
            if self._wd_armed_at == armed and self._last_audible_ns < armed:
                self._wd_armed_at = None
                await self._rotate("watchdog", now_ns())

        if self._wd_task and not self._wd_task.done():
            self._wd_task.cancel()
        self._wd_task = asyncio.create_task(fire())
        self._tasks.append(self._wd_task)

    async def _rotate(self, reason: str, t_trigger: int) -> None:
        old = self.active
        new = self._standby_session()
        if old is None or new is None:
            self._mark("rotation.skipped", reason=reason, why="no standby")
            return
        self._mark("rotation.trigger", t_trigger, reason=reason, session=old.idx)
        old.state = "closing"
        if old.updated.is_set():
            if old.pending:
                await self._flush_pending(old)
            await old.ws.send(json.dumps({"type": "session.close"}))
            old.close_sent = True
            self._mark("rotation.close_sent", session=old.idx)
        new.state = "active"
        self.active = new
        self._mark("rotation.promote", session=new.idx, ready=new.updated.is_set())
        self._open_standby()

    # ------------------------------------------------------------ finish
    async def finish(self) -> None:
        for s in list(self.sessions):
            if s.ws is None or s.closed.is_set() or s.close_sent:
                continue
            if s.pending and s.updated.is_set():
                await self._flush_pending(s)
            if s.updated.is_set():
                await s.ws.send(json.dumps({"type": "session.close"}))
                s.close_sent = True
                stage = "openai.session.close_sent" if s is self.active else "openai.standby.close_sent"
                self._mark(stage, session=s.idx)
        timeout = float(self.cfg.get("close_timeout_s", 30))
        try:
            await asyncio.wait_for(asyncio.gather(*(s.closed.wait() for s in self.sessions if s.close_sent)), timeout)
        except asyncio.TimeoutError:
            self._emit(ProviderEvent("error", now_ns(), text="timed out waiting for session.closed",
                                     meta={"fatal": False}))
        await self.aclose()

    def _emit_cost(self, s: _Session) -> None:
        if s.cost_emitted or s.t_open is None:
            return
        s.cost_emitted = True
        self._mark("cost.session", provider="openai", session=s.idx, session_id=s.session_id,
                   audio_s=round(s.audio_bytes / 2 / self.input_sample_rate, 3),
                   wall_s=round((now_ns() - s.t_open) / 1e9, 3))

    async def aclose(self) -> None:
        for t in self._tasks:
            if not t.done():
                t.cancel()
        for s in self.sessions:
            self._emit_cost(s)
            if s.ws is not None:
                try:
                    await s.ws.close()
                except Exception:
                    pass
            if s.reader is not None:
                try:
                    await asyncio.wait_for(s.reader, 5)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    s.reader.cancel()
        self._emit_closed()
