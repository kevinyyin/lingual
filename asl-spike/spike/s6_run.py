"""S6: drive the avatar prototype in the system Microsoft Edge (Playwright, GPU via ANGLE/D3D11), play each
demo sentence several times, and collect timings (gloss JSON received -> first avatar motion), paint cost,
key-frame screenshots and a WebM screen capture (MediaRecorder on the canvas, uploaded to the local server).

Needs spike/avatar_server.py running on 127.0.0.1:8766. Results: results/s6_avatar.json, avatar/shots/*.jpg,
avatar/capture.webm (git-ignored; derived from ASL Citizen poses, local research use only).
"""
from __future__ import annotations

import base64
import json
import statistics
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8766/index.html?hq=1"
REPS = 5


def main(headless: bool = True):
    shots = ROOT / "avatar" / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    out = {"browser": "Microsoft Edge via Playwright", "headless": headless, "runs": []}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=headless,
                              args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"])
        pg = b.new_page(viewport={"width": 1100, "height": 720})
        logs = []
        pg.on("console", lambda m: logs.append(m.text))
        t0 = time.perf_counter()
        pg.goto(URL)
        pg.wait_for_function("window.ready === true", timeout=120000)
        out["page_ready_s"] = round(time.perf_counter() - t0, 2)
        out["gpu_renderer"] = pg.evaluate("""() => { const gl = document.createElement('canvas').getContext('webgl2');
            const e = gl && gl.getExtension('WEBGL_debug_renderer_info'); return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : null; }""")
        presets = pg.evaluate("fetch('sentences.json').then(r => r.json())")
        for rep in range(REPS):
            for i, pr in enumerate(presets):
                if rep == 0:
                    # first run of each preset: also grab key frames
                    pg.evaluate("s => { window._p = playSentence(s); }", pr["sentence"])
                    for k, at in enumerate((300, 1200, 2400)):
                        pg.wait_for_timeout(at if k == 0 else at - (300, 1200, 2400)[k - 1])
                        img = pg.evaluate("window.paintNow()")
                        (shots / f"preset{i}_{k}.jpg").write_bytes(base64.b64decode(img.split(",")[1]))
                    m = pg.evaluate("window._p")
                else:
                    m = pg.evaluate("s => playSentence(s)", pr["sentence"])
                out["runs"].append({"preset": pr["label"], "rep": rep, "cold_lexicon": rep == 0 and i == 0, **m,
                                    "render_ms": pg.evaluate("window.renderMs")})
                print(pr["label"][:40], {k: round(v) for k, v in m.items()}, flush=True)
        # screen capture of all presets
        pg.evaluate("document.getElementById('rec').click()")
        pg.wait_for_function("window.recordDone !== undefined", timeout=180000)
        out["console_tail"] = logs[-20:]
        b.close()
    fm = [r["first_motion_ms"] for r in out["runs"]]
    warm = [r["first_motion_ms"] for r in out["runs"] if not r["cold_lexicon"]]
    out["first_motion_ms"] = {"median": round(statistics.median(fm), 1), "p90": round(sorted(fm)[int(0.9 * (len(fm) - 1))], 1),
                              "max": round(max(fm), 1), "n": len(fm),
                              "warm_median": round(statistics.median(warm), 1) if warm else None,
                              "cold_first_run": round([r["first_motion_ms"] for r in out["runs"] if r["cold_lexicon"]][0], 1)}
    out["render_ms_median"] = round(statistics.median([r["render_ms"] for r in out["runs"] if r["render_ms"]]), 2)
    json.dump(out, open(ROOT / "results/s6_avatar.json", "w"), indent=1)
    print(out["first_motion_ms"], out["render_ms_median"], out["gpu_renderer"])


if __name__ == "__main__":
    main(headless="--headed" not in sys.argv)
