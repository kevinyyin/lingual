"""Local static server for the avatar prototype (127.0.0.1 only) that also accepts POST /upload?name=...
to save a screen capture or timing file into avatar/. Research spike only."""
import http.server
import os
import socketserver
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent / "avatar"
ALLOWED = {"capture.webm", "timings.json", "capture2.webm"}


class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT), **k)

    def do_POST(self):
        q = parse_qs(urlparse(self.path).query)
        name = (q.get("name") or [""])[0]
        if urlparse(self.path).path != "/upload" or name not in ALLOWED:
            self.send_error(400)
            return
        n = int(self.headers.get("Content-Length", 0))
        (ROOT / name).write_bytes(self.rfile.read(n))
        self.send_response(200)
        self.end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), H) as s:
        print(f"serving {ROOT} on http://127.0.0.1:{port}", flush=True)
        s.serve_forever()
