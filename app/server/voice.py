"""Emotion -> Cartesia Sonic generation_config, and streaming playback on the laptop speaker."""

import asyncio
import logging
import threading
from typing import AsyncIterator, Callable

log = logging.getLogger("asl.voice")

MODEL = "sonic-3.6"
DEFAULT_VOICE_ID = "0834f3df-e650-4766-a20c-5a93a43aa6e3"  # "Leo", listed by Cartesia among the most emotive voices
SAMPLE_RATE = 24000
STALL_S = 5.0  # no TTS event this long = a half-open socket; give up so the session can recover
# Race IPv4 against IPv6 (RFC 8305) like browsers and curl: on a network with broken IPv6, asyncio otherwise tries each
# of Cartesia's 8 IPv6 addresses in turn and every handshake hits websockets' 10 s open timeout.
WS_OPTIONS = {"happy_eyeballs_delay": 0.25}

# label -> (mild emotion, strong emotion, speed delta at full intensity, volume delta at full intensity)
# Starting values; tuned by the listening test (plan Task 18).
MAPPING = {
    "happy": ("happy", "excited", +0.15, +0.30),
    "sad": ("sad", "disappointed", -0.30, -0.35),
    "angry": ("frustrated", "angry", +0.20, +0.60),
}
STRONG = 0.6


def voice_controls(e) -> dict:
    """Sonic generation_config for an Emotion; neutral leaves the voice alone."""
    m = MAPPING.get(e["label"])
    if not m:
        return {"speed": 1.0, "volume": 1.0}
    i = min(max(float(e["intensity"]), 0.0), 1.0)
    mild, strong, ds, dv = m
    return {
        "emotion": strong if i >= STRONG else mild,
        "speed": round(min(max(1.0 + ds * i, 0.6), 1.5), 3),
        "volume": round(min(max(1.0 + dv * i, 0.5), 2.0), 3),
    }


class VoiceError(Exception):
    pass


class _Speaker:
    """Blocking PCM16 mono writer on the default output device. Voice writes from a worker thread: the lock keeps a
    cut-off line's abort from racing a chunk still being written (PortAudio is not safe for that)."""

    def __init__(self):
        import sounddevice as sd

        self.stream = sd.RawOutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        self.stream.start()
        self.lock, self.ended = threading.Lock(), False

    def write(self, b: bytes) -> None:
        with self.lock:
            if not self.ended:
                self.stream.write(b)

    def close(self) -> None:
        self._end(self.stream.stop)  # plays out what is buffered

    def abort(self) -> None:
        self._end(self.stream.abort)  # cut off: drops it

    def _end(self, how: Callable[[], None]) -> None:
        with self.lock:
            if not self.ended:
                self.ended = True
                how()
                self.stream.close()


class Voice:
    def __init__(self, client=None, voice_id: str = DEFAULT_VOICE_ID, player_factory: Callable = _Speaker, api_key: str | None = None):
        if client is None:
            from cartesia import AsyncCartesia

            client = AsyncCartesia(api_key=api_key)
        self.client, self.voice_id, self.player_factory = client, voice_id, player_factory
        self._conn = None  # one TTS socket kept open across utterances; setup costs ~100+ ms of the 2 s budget

    async def _connect(self):
        if self._conn is None:
            self._conn = await self.client.tts.websocket_connect(websocket_connection_options=WS_OPTIONS).enter()
        return self._conn

    async def _drop(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                await asyncio.wait_for(conn.close(), 1.0)
            except Exception:
                pass

    async def close(self) -> None:
        """Release the kept socket (a phone page's own Voice, when the page goes away)."""
        await self._drop()

    async def warm(self) -> None:
        """Open the socket before the first utterance; failures are left for speak() to retry."""
        try:
            await self._connect()
        except Exception as e:
            log.warning("TTS warm-up failed: %s", e)

    async def speak(self, text: str, emotion) -> AsyncIterator[str]:
        player = self.player_factory()
        stop, ctx = player.close, None
        try:
            while True:
                reused, heard = self._conn is not None, False
                try:
                    conn = await self._connect()
                    ctx = conn.context(model_id=MODEL, voice=self.voice_id, language="en",
                                       output_format={"container": "raw", "encoding": "pcm_s16le", "sample_rate": SAMPLE_RATE},
                                       generation_config=voice_controls(emotion))
                    await ctx.push(text)
                    await ctx.no_more_inputs()
                    events = ctx.receive().__aiter__()
                    while True:
                        try:
                            ev = await asyncio.wait_for(anext(events), STALL_S)
                        except StopAsyncIteration:
                            break
                        if ev.type == "chunk" and ev.audio:
                            await asyncio.to_thread(player.write, ev.audio)
                            if not heard:
                                heard = True
                                yield "first_audio"
                        elif ev.type == "error":
                            raise VoiceError(str(getattr(ev, "error", ev)))
                        elif ev.type == "done":
                            break
                    break
                except Exception as e:  # socket drops, stalls, auth errors: surface as VoiceError so the session recovers
                    await self._drop()
                    if reused and not heard and not isinstance(e, VoiceError):
                        continue  # the kept socket died while idle: one retry on a fresh one
                    if isinstance(e, VoiceError):
                        raise
                    raise VoiceError(f"{type(e).__name__}: {e}") from e
        except asyncio.CancelledError:  # a newer line cut this one off: stop generating it and stop its sound now
            stop = getattr(player, "abort", player.close)
            if ctx is not None:
                try:
                    await ctx.cancel()  # Cartesia drops the rest of the reply; the kept socket serves the next line
                except Exception:
                    await self._drop()
            raise
        finally:
            await asyncio.to_thread(stop)  # close: the phone's player waits out the playback tail
        yield "done"
