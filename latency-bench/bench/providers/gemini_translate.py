"""Google Gemini Live Translate over the Live API WebSocket.

Per the docs (ai.google.dev/gemini-api/docs/live-api/live-translate and the Live API
reference ai.google.dev/api/live, read 2026-10-13):
  * wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=<key>
  * first client message (field placement per the Live API reference; see note in connect()):
        {"setup": {"model": "models/gemini-3.5-live-translate-preview",
        "inputAudioTranscription": {}, "outputAudioTranscription": {},
        "generationConfig": {"responseModalities": ["AUDIO"],
        "translationConfig": {"targetLanguageCode": "en", "echoTargetLanguage": false}}}}
  * wait for {"setupComplete": {}} before sending anything else
  * audio in: {"realtimeInput": {"audio": {"data": <b64 PCM16 16 kHz mono LE>, "mimeType": "audio/pcm;rate=16000"}}}
    (docs recommend 100 ms chunks; we send the harness's 20 ms chunks, see README)
  * audio out: serverContent.modelTurn.parts[].inlineData.data (b64 PCM16 24 kHz mono LE)
  * text: serverContent.inputTranscription.text / serverContent.outputTranscription.text
  * end of input: {"realtimeInput": {"audioStreamEnd": true}}; translation is a continuous
    stream with no single terminal event, so we drain until no output for drain_idle_s.
No source-language field exists in the translation config; the model detects it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import Any

import numpy as np
import websockets

from bench.clock import now_ns
from bench.events import ProviderEvent
from bench.providers.base import Provider


AUDIBLE_PEAK = int(10 ** (-45 / 20) * 32768)  # same gate as analysis (output_gate_dbfs -45)


class GeminiTranslateProvider(Provider):
    name = "gemini"
    hosts = ("generativelanguage.googleapis.com",)

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.input_sample_rate = int(cfg.get("input_sample_rate", 16000))
        self.ws: websockets.ClientConnection | None = None
        self._reader: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._last_output_ns = 0
        self._setup_error: str | None = None
        self._ws_closed = asyncio.Event()

    @classmethod
    def check_available(cls, cfg: dict[str, Any]) -> str | None:
        return None if os.environ.get("GEMINI_API_KEY") else "GEMINI_API_KEY not set in latency-bench/.env"

    def _lang(self, code: str) -> str:
        return dict(self.cfg.get("language_codes", {})).get(code, code)

    async def connect(self, source_lang: str, target_lang: str) -> None:
        url = f"{self.cfg['url']}?key={os.environ['GEMINI_API_KEY']}"
        self.ws = await websockets.connect(url, max_size=None, compression=None, open_timeout=20, ping_interval=20)
        self._mark("gemini.ws_open")
        self._reader = asyncio.create_task(self._read_loop())
        # Live API reference (BidiGenerateContentSetup): inputAudioTranscription and
        # outputAudioTranscription are top-level setup fields; translationConfig lives in
        # generationConfig. The Live Translate guide's raw-WebSocket example nests the
        # transcription fields inside generationConfig, which the server rejects (1007).
        setup = {"setup": {
            "model": f"models/{self.cfg.get('model', 'gemini-3.5-live-translate-preview')}",
            "inputAudioTranscription": {},
            "outputAudioTranscription": {},
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "translationConfig": {
                    "targetLanguageCode": self._lang(target_lang),
                    "echoTargetLanguage": bool(self.cfg.get("echo_target_language", False)),
                },
            },
        }}
        await self.ws.send(json.dumps(setup))
        while not self._ready.is_set():
            if self._setup_error or self._reader.done():
                raise RuntimeError(self._setup_error or "connection closed during setup")
            await asyncio.sleep(0.005)

    async def _read_loop(self) -> None:
        assert self.ws is not None
        try:
            async for raw in self.ws:
                t = now_ns()
                self._handle(raw, t)
        except websockets.ConnectionClosed as e:
            reason = f"connection closed: code={e.rcvd.code if e.rcvd else None} reason={e.rcvd.reason if e.rcvd else ''}"
            if not self._ready.is_set():
                self._setup_error = reason
            self._mark("gemini.ws_closed", detail=reason)
        finally:
            self._ws_closed.set()

    def _handle(self, raw: str | bytes, t: int) -> None:
        size = len(raw)
        msg = json.loads(raw)
        if "setupComplete" in msg:
            self._mark("gemini.setupComplete", t)
            self._ready.set()
            self._emit(ProviderEvent("session_ready", t, size))
            return
        sc = msg.get("serverContent")
        if sc:
            for part in (sc.get("modelTurn") or {}).get("parts", []) or []:
                inline = part.get("inlineData")
                if inline and inline.get("data"):
                    pcm = base64.b64decode(inline["data"])
                    a = np.frombuffer(pcm, "<i2")
                    audible = bool(a.size) and int(np.max(np.abs(a.astype(np.int32)))) > AUDIBLE_PEAK
                    rate = 24000
                    mt = inline.get("mimeType", "")
                    if "rate=" in mt:
                        rate = int(mt.split("rate=")[1].split(";")[0])
                    if audible:  # silent filler frames do not count as activity for the drain
                        self._last_output_ns = t
                    self._emit(ProviderEvent("audio_out", t, size, pcm=pcm, sample_rate=rate, meta={"mime": mt}))
            if sc.get("outputTranscription"):
                tr = sc["outputTranscription"]
                if tr.get("text"):  # Gemini also sends periodic empty, languageCode-only messages
                    self._last_output_ns = t
                self._emit(ProviderEvent("text_out_delta", t, size, text=tr.get("text", ""),
                                         meta={"languageCode": tr.get("languageCode")}))
            if sc.get("inputTranscription"):
                tr = sc["inputTranscription"]
                self._emit(ProviderEvent("source_text_delta", t, size, text=tr.get("text", ""),
                                         meta={"languageCode": tr.get("languageCode")}))
            for flag in ("generationComplete", "turnComplete", "interrupted"):
                if sc.get(flag):
                    self._mark(f"gemini.{flag}", t, interactionStatus=sc.get("interactionStatus"))
            return
        if "goAway" in msg:
            self._mark("gemini.goAway", t, detail=msg["goAway"])
        elif "error" in msg:
            self._emit(ProviderEvent("error", t, size, text=json.dumps(msg["error"])[:500], meta={"fatal": True}))
        else:
            keys = [k for k in msg if k != "usageMetadata"]
            if keys:
                self._mark("gemini." + ",".join(keys), t, payload_size=size)

    async def send_audio(self, pcm16_chunk: bytes, t_sent_ns: int) -> None:
        assert self.ws is not None
        await self.ws.send(json.dumps({"realtimeInput": {"audio": {
            "data": base64.b64encode(pcm16_chunk).decode(), "mimeType": f"audio/pcm;rate={self.input_sample_rate}",
        }}}))

    async def finish(self) -> None:
        assert self.ws is not None
        await self.ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
        t_end = now_ns()
        self._mark("gemini.audioStreamEnd_sent", t_end)
        idle_ns = int(float(self.cfg.get("drain_idle_s", 3.0)) * 1e9)
        max_ns = int(float(self.cfg.get("drain_max_s", 20.0)) * 1e9)
        while not self._ws_closed.is_set():
            now = now_ns()
            if now - max(self._last_output_ns, t_end) >= idle_ns or now - t_end >= max_ns:
                break
            await asyncio.sleep(0.05)
        self._mark("gemini.drained")
        await self.aclose()

    async def aclose(self) -> None:
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
        if self._reader is not None:
            try:
                await asyncio.wait_for(self._reader, 5)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self._reader.cancel()
        self._emit_closed()
