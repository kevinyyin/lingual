import asyncio
import threading

import pytest

from server.phone_audio import ClientMic, WsPlayer


def _loop():
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


async def test_client_mic_yields_pushed_then_silence():
    m = ClientMic()
    it = m().__aiter__()
    asyncio.get_running_loop().call_soon(m.push, b"\x01\x00" * 10)
    assert await it.__anext__() == b"\x01\x00" * 10
    assert await it.__anext__() == bytes(3200)  # 100 ms of 16 kHz PCM16 silence keeps Ink alive


def test_player_paces_to_real_time():
    t, sent = [0.0], []

    def sleep(s):
        t[0] += s

    async def send(b):
        sent.append(b)

    loop = _loop()
    p = WsPlayer(send, loop, ahead_s=0.3, clock=lambda: t[0], sleep=sleep)
    for _ in range(3):
        p.write(bytes(48000))  # 1 s of 24 kHz PCM16 each
    p.close()
    assert len(sent) == 3
    assert t[0] >= 3.0 - 0.3 - 1e-6  # never more than ahead_s ahead of the phone's playback
    loop.call_soon_threadsafe(loop.stop)


def test_player_write_raises_when_send_fails():
    async def send(b):
        raise ConnectionError("closed")

    loop = _loop()
    p = WsPlayer(send, loop, sleep=lambda s: None)
    with pytest.raises(ConnectionError):
        p.write(bytes(4800))
    loop.call_soon_threadsafe(loop.stop)


async def test_client_mic_never_pads_a_128ms_cadence():
    """The phone sends 2048-sample (128 ms) chunks; silence between them would break Ink's turn detection."""
    m = ClientMic()
    it = m().__aiter__()
    loop = asyncio.get_running_loop()
    for i in range(3):
        loop.call_later(0.128, m.push, bytes([i + 1]) * 4096)
        assert await it.__anext__() == bytes([i + 1]) * 4096


def test_player_close_waits_for_the_phone_to_finish():
    t, slept = [0.0], []

    def sleep(s):
        slept.append(s)
        t[0] += s

    async def send(b):
        pass

    loop = _loop()
    p = WsPlayer(send, loop, ahead_s=0.3, clock=lambda: t[0], sleep=sleep)
    p.write(bytes(48000))  # 1 s of audio; write returns ~0.3 s before it ends
    p.close()
    assert t[0] >= 1.0 + WsPlayer.TAIL_S - 1e-6  # echo still in flight through the tunnel and the mic buffer
    loop.call_soon_threadsafe(loop.stop)


async def test_client_mic_drops_stale_audio_when_ink_reconnects():
    """Audio queued while Ink was down is old (maybe the plushie's own voice): a new connection starts fresh."""
    m = ClientMic()
    for _ in range(20):
        m.push(b"\x07\x00" * 2048)  # 2.5 s piled up during the outage
    m.push(b"\x09\x00" * 2048)
    it = m().__aiter__()  # MicTranscriber reconnected: a new iteration
    asyncio.get_running_loop().call_later(0.05, m.push, b"\x01\x00" * 2048)
    assert await it.__anext__() == b"\x01\x00" * 2048


def test_client_mic_backlog_is_bounded():
    m = ClientMic()
    for _ in range(100):
        m.push(bytes(4096))
    assert m.q.qsize() <= ClientMic.MAX_CHUNKS
