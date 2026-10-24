"""Screenshot helper for the round-2 avatar (Edge via Playwright, GPU).
Usage:
  python v_shots.py hands <out.jpg> <shape,shape,...> [query]   handshape grid
  python v_shots.py face <out.jpg> [query]                      neutral / raise / lower brows (+ bangs clip)
"""
from __future__ import annotations

import base64
import sys

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8766/index2.html?render=1"


def dec(u):
    return cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",")[1]), np.uint8), 1)


def page(p, query="", w=1280, h=720):
    b = p.chromium.launch(channel="msedge", headless=True, args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": w, "height": h})
    pg.goto(URL + ("&" + query if query else ""))
    pg.wait_for_function("window.ready === true", timeout=120000)
    return b, pg


def grid(ims, labels, cols):
    h = min(i.shape[0] for i in ims)
    ims = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h)) for i in ims]
    for i, l in zip(ims, labels):
        cv2.putText(i, l, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    rows = [np.hstack(ims[i:i + cols] + [np.full_like(ims[0], 255)] * (cols - len(ims[i:i + cols]))) for i in range(0, len(ims), cols)]
    return np.vstack(rows)


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    with sync_playwright() as p:
        if mode == "hands":
            shapes = sys.argv[3].split(",")
            b, pg = page(p, sys.argv[4] if len(sys.argv) > 4 else "", 640, 640)
            ims = [dec(pg.evaluate(f"window.shot(window.handshapePose('{s}'))")) for s in shapes]
            cv2.imwrite(out, grid(ims, shapes, 4))
        else:
            b, pg = page(p, "face=1&" + (sys.argv[3] if len(sys.argv) > 3 else ""), 600, 600)
            ims = [dec(pg.evaluate(f"window.shot({{bones:{{}}, nmm:{{brows:'{m}'}}}})")) for m in ("neutral", "raise", "lower")]
            cv2.imwrite(out, grid(ims, ["neutral", "raise", "lower"], 3))
        b.close()
