from server.main import origin_ok


def test_local_origins_allowed():
    for o in ("http://localhost:5173", "http://127.0.0.1:8000", "http://localhost", "http://[::1]:5173"):
        assert origin_ok(o), o


def test_other_sites_rejected():
    for o in ("https://evil.example", "http://localhost.evil.example", "null", "", None):
        assert not origin_ok(o), o


async def test_startup_warms_llm_and_tts(monkeypatch):
    from types import SimpleNamespace

    import server.main as M

    calls = []

    class Msgs:
        async def create(self, **kw):
            calls.append("llm")
            raise TypeError("no credentials")

    class FakeVoice:
        async def warm(self):
            calls.append("tts")

    monkeypatch.setattr(M, "CLAUDE", SimpleNamespace(messages=Msgs()))
    monkeypatch.setattr(M, "VOICE", FakeVoice())
    await M.warm_up()
    assert sorted(calls) == ["llm", "tts"]


def test_allowed_origins_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://abc.trycloudflare.com, https://x.example")
    assert origin_ok("https://abc.trycloudflare.com")
    assert origin_ok("https://x.example")
    assert not origin_ok("https://other.trycloudflare.com")
    assert not origin_ok("http://abc.trycloudflare.com")  # exact origin, scheme included
    monkeypatch.delenv("ALLOWED_ORIGINS")
    assert not origin_ok("https://abc.trycloudflare.com")


def test_allowed_origin_trailing_slash(monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://abc.trycloudflare.com/")
    assert origin_ok("https://abc.trycloudflare.com")
