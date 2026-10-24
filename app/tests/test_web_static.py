import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.main import mount_web


def test_serves_the_built_pages_and_the_public_assets(tmp_path):
    dist, public = tmp_path / "dist", tmp_path / "public"
    (dist / "assets").mkdir(parents=True)
    (dist / "teach.html").write_text("<html>teach</html>")
    (dist / "assets" / "teach-abc.js").write_text("js")
    (public / "signs").mkdir(parents=True)
    clip = [0.123456789] * 100  # a sign clip's floats: big enough to compress
    (public / "signs" / "HELLO.json").write_text(json.dumps(clip))
    app = FastAPI()

    @app.get("/vocab")
    def vocab():
        return ["api"]

    mount_web(app, dist, public)
    c = TestClient(app)
    assert c.get("/teach.html").text == "<html>teach</html>"
    assert c.get("/assets/teach-abc.js").text == "js"
    assert "immutable" in c.get("/assets/teach-abc.js").headers["cache-control"]  # hashed names never change
    r = c.get("/signs/HELLO.json")
    assert r.json() == clip and r.headers["content-encoding"] == "gzip"  # the phone fetches clips through a tunnel
    assert c.get("/vocab").json() == ["api"]  # API routes still win
    assert c.get("/nope.txt").status_code == 404


def test_no_build_mounts_nothing(tmp_path):
    app = FastAPI()
    mount_web(app, tmp_path / "missing", tmp_path / "public")
    assert TestClient(app).get("/teach.html").status_code == 404
