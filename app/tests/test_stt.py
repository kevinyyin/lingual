from contextlib import asynccontextmanager
from types import SimpleNamespace

from server.stt import MicTranscriber


class FakeWS:
    def __init__(self, events):
        self.events, self.sent = events, []

    async def send_raw(self, b):
        self.sent.append(bytes(b))

    async def send(self, msg):
        pass

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self.events:
            yield e


def fake_client(ws):
    @asynccontextmanager
    async def websocket(**kw):
        yield ws

    return SimpleNamespace(stt=SimpleNamespace(auto_finalize=SimpleNamespace(websocket=websocket)))


async def mic(chunks):
    for c in chunks:
        yield c


async def test_audio_not_sent_when_closed():
    ws = FakeWS([])
    t = MicTranscriber(on_transcript=None, is_open=lambda: False, client=fake_client(ws), mic=lambda: mic([b"\x07\x07"] * 3))
    await t.connection()
    assert ws.sent and all(b == b"\x00\x00" for b in ws.sent)  # silence keeps the socket alive, voice never reaches Ink


async def test_audio_sent_when_open():
    ws = FakeWS([])
    t = MicTranscriber(on_transcript=None, is_open=lambda: True, client=fake_client(ws), mic=lambda: mic([b"\x07\x07"]))
    await t.connection()
    assert ws.sent == [b"\x07\x07"]


async def test_final_transcript_forwarded_once():
    calls = []

    async def on_transcript(text, final):
        calls.append((text, final))

    ws = FakeWS([SimpleNamespace(type="turn.start"), SimpleNamespace(type="turn.update", transcript="hi"),
                 SimpleNamespace(type="turn.eager_end", transcript="hi there"), SimpleNamespace(type="turn.end", transcript="hi there")])
    t = MicTranscriber(on_transcript=on_transcript, is_open=lambda: True, client=fake_client(ws), mic=lambda: mic([]))
    await t.connection()
    assert calls == [("hi", False), ("hi there", True)]


async def test_connection_drop_returns_session_to_ready():
    import asyncio

    calls = []

    async def on_transcript(text, final):
        calls.append((text, final))

    @asynccontextmanager
    async def broken(**kw):
        raise ConnectionError("ink dropped")
        yield

    client = SimpleNamespace(stt=SimpleNamespace(auto_finalize=SimpleNamespace(websocket=broken)))
    t = MicTranscriber(on_transcript=on_transcript, is_open=lambda: True, client=client, mic=lambda: mic([]))
    task = asyncio.create_task(t.run())
    await asyncio.sleep(0.05)
    task.cancel()
    assert ("", True) in calls  # an empty final ends any half-heard turn
