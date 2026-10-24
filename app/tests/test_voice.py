from types import SimpleNamespace

import pytest

from server.voice import Voice, VoiceError, voice_controls

LABELS = ["happy", "sad", "angry", "neutral"]


def test_neutral_is_plain():
    c = voice_controls({"label": "neutral", "intensity": 0.9})
    assert c == {"speed": 1.0, "volume": 1.0}


def test_intensity_monotonic():
    lo = voice_controls({"label": "sad", "intensity": 0.3})
    hi = voice_controls({"label": "sad", "intensity": 0.9})
    assert hi["speed"] < lo["speed"] and hi["volume"] < lo["volume"]
    assert hi["emotion"] in ("sad", "disappointed", "dejected")


def test_controls_within_api_bounds():
    for label in LABELS:
        for i in (0.0, 0.5, 1.0):
            c = voice_controls({"label": label, "intensity": i})
            assert 0.6 <= c["speed"] <= 1.5 and 0.5 <= c["volume"] <= 2.0


class FakeCtx:
    def __init__(self, events):
        self.events, self.pushed, self.kw = events, [], None

    async def push(self, text):
        self.pushed.append(text)

    async def no_more_inputs(self):
        pass

    async def receive(self):
        for e in self.events:
            if isinstance(e, Exception):
                raise e
            yield e


class FakeConn:
    def __init__(self, make_ctx):
        self.make_ctx, self.closed = make_ctx, False

    def context(self, **kw):
        ctx = self.make_ctx()
        ctx.kw = kw
        return ctx

    async def close(self):
        self.closed = True


def client_of(make_ctx):
    """Fake AsyncCartesia; .conns lists every socket opened."""
    conns = []

    class Manager:
        def __init__(self, **kw):
            pass

        async def enter(self):
            conns.append(FakeConn(make_ctx))
            return conns[-1]

    return SimpleNamespace(tts=SimpleNamespace(websocket_connect=Manager), conns=conns)


def fake_client(events):
    ctx = FakeCtx(events)
    return client_of(lambda: ctx), ctx


class FakePlayer:
    def __init__(self):
        self.data, self.closed = bytearray(), False

    def write(self, b):
        self.data += b

    def close(self):
        self.closed = True


def chunk(b):
    return SimpleNamespace(type="chunk", audio=b)


async def test_speak_streams_audio_with_controls():
    client, ctx = fake_client([chunk(b"\x01\x00"), chunk(b"\x02\x00"), SimpleNamespace(type="done")])
    player = FakePlayer()
    v = Voice(client=client, voice_id="v1", player_factory=lambda: player)
    events = [e async for e in v.speak("Hi there.", {"label": "happy", "intensity": 0.8})]
    assert events == ["first_audio", "done"]
    assert bytes(player.data) == b"\x01\x00\x02\x00" and player.closed
    assert ctx.pushed == ["Hi there."]
    assert ctx.kw["generation_config"]["emotion"] in ("happy", "excited")


async def test_socket_drop_raises_after_first_audio():
    client, _ = fake_client([chunk(b"\x01\x00"), ConnectionError("closed")])
    player = FakePlayer()
    v = Voice(client=client, voice_id="v1", player_factory=lambda: player)
    seen = []
    with pytest.raises(VoiceError):
        async for e in v.speak("Hi.", {"label": "neutral", "intensity": 0}):
            seen.append(e)
    assert seen == ["first_audio"] and player.closed


async def test_error_event_raises():
    client, _ = fake_client([SimpleNamespace(type="error", error="bad voice")])
    v = Voice(client=client, voice_id="v1", player_factory=FakePlayer)
    with pytest.raises(VoiceError):
        [e async for e in v.speak("Hi.", {"label": "neutral", "intensity": 0})]


async def test_stalled_stream_times_out(monkeypatch):
    import asyncio

    import server.voice as V

    monkeypatch.setattr(V, "STALL_S", 0.05)

    class Stalled(FakeCtx):
        async def receive(self):
            yield chunk(b"\x01\x00")
            await asyncio.sleep(10)
            yield SimpleNamespace(type="done")

    v = Voice(client=client_of(lambda: Stalled([])), voice_id="v", player_factory=FakePlayer)
    with pytest.raises(VoiceError):
        async with asyncio.timeout(2):
            [e async for e in v.speak("Hi.", {"label": "neutral", "intensity": 0})]


NEUTRAL = {"label": "neutral", "intensity": 0}
OK = [chunk(b"\x01\x00"), SimpleNamespace(type="done")]


async def say(v):
    return [e async for e in v.speak("Hi.", NEUTRAL)]


async def test_socket_reused_across_utterances():
    client = client_of(lambda: FakeCtx(OK))
    v = Voice(client=client, voice_id="v", player_factory=FakePlayer)
    assert await say(v) == ["first_audio", "done"]
    assert await say(v) == ["first_audio", "done"]
    assert len(client.conns) == 1


async def test_warm_opens_socket_ahead():
    client = client_of(lambda: FakeCtx(OK))
    v = Voice(client=client, voice_id="v", player_factory=FakePlayer)
    await v.warm()
    assert len(client.conns) == 1
    await say(v)
    assert len(client.conns) == 1


async def test_reconnects_after_failure():
    scripts = [[chunk(b"\x01\x00"), ConnectionError("closed")], OK]
    client = client_of(lambda: FakeCtx(scripts.pop(0)))
    v = Voice(client=client, voice_id="v", player_factory=FakePlayer)
    with pytest.raises(VoiceError):
        await say(v)
    assert client.conns[0].closed
    assert await say(v) == ["first_audio", "done"]
    assert len(client.conns) == 2


async def test_dead_reused_socket_retried_before_audio():
    """A socket the server closed while idle fails before any audio: retry once on a fresh one, unseen by the caller."""
    scripts = [OK, [ConnectionError("idle close")], OK]
    client = client_of(lambda: FakeCtx(scripts.pop(0)))
    v = Voice(client=client, voice_id="v", player_factory=FakePlayer)
    await say(v)
    assert await say(v) == ["first_audio", "done"]
    assert len(client.conns) == 2


async def test_fresh_socket_failure_not_retried():
    scripts = [[ConnectionError("down")], OK]
    client = client_of(lambda: FakeCtx(scripts.pop(0)))
    v = Voice(client=client, voice_id="v", player_factory=FakePlayer)
    with pytest.raises(VoiceError):
        await say(v)
    assert len(client.conns) == 1


async def test_close_drops_the_kept_socket():
    client, _ = fake_client([])
    v = Voice(client=client, voice_id="v1", player_factory=FakePlayer)
    await v.warm()
    await v.close()
    assert client.conns[0].closed and v._conn is None
