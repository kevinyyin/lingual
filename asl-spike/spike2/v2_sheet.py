"""V2: handshape contact sheet - avatar canonical handshape next to a reference crop from an ASL Citizen
training clip with that handshape (middle of the P1 active segment, dominant hand). Local only (dataset
frames): results2/v2_sheets/handshapes_*.jpg."""
from __future__ import annotations

import base64
import json
import os
import sys

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, citizen_class_map, citizen_rows  # noqa: E402
import trim as TR  # noqa: E402


def reference(hs, links, trims, rows, cmap):
    for r in rows.itertuples():
        e = (links.get(cmap[r.Gloss]) or {}).get("primary") or {}
        if e.get("handshape") != hs or e.get("flexion_change") or r.video not in trims:
            continue
        d = np.load(ROOT / "cache/s2/citizen" / f"{os.path.splitext(r.video)[0]}.npz")
        a, b = trims[r.video]["a"], trims[r.video]["b"]
        mid = (a + b) // 2
        h = d["lh"] if TR._motion(d["lh"]) > TR._motion(d["rh"]) else d["rh"]
        if np.isnan(h[mid, 0, 0]):
            continue
        cap = cv2.VideoCapture(str(ROOT / "data/asl-citizen/videos_vocab" / r.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
        ok, f = cap.read()
        if not ok:
            continue
        H, W = f.shape[:2]
        xs, ys = h[mid, :, 0] * W, h[mid, :, 1] * H
        cx, cy, s = xs.mean(), ys.mean(), 1.6 * max(np.ptp(xs), np.ptp(ys))
        x0, y0 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2))
        crop = f[y0:int(min(H, cy + s / 2)), x0:int(min(W, cx + s / 2))]
        if crop.size:
            return cv2.resize(crop, (300, 300)), cmap[r.Gloss]
    return np.full((300, 300, 3), 200, np.uint8), "-"


def main():
    links = json.load(open(R2 / "asllex_links.json"))["classes"]
    lib = json.load(open(R2 / "handshapes.json"))["library"]
    trims = json.load(open(CACHE2 / "trim_citizen.json"))
    cmap = citizen_class_map()
    rows = citizen_rows()
    rows = rows[(rows["Gloss"].isin(cmap)) & (rows["split"] == "train")].sample(frac=1, random_state=0)
    shapes = sorted(lib, key=lambda k: -lib[k]["frames"])
    tiles = []
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True, args=["--use-angle=d3d11"])
        pg = b.new_page(viewport={"width": 600, "height": 600})
        pg.goto("http://127.0.0.1:8766/index2.html?render=1&handcam=1")
        pg.wait_for_function("window.ready === true", timeout=180000)
        for hs in shapes:
            u = pg.evaluate(f"window.shot(window.handshapePose('{hs}'))")
            av = cv2.resize(cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",")[1]), np.uint8), 1), (300, 300))
            ref, cls = reference(hs, links, trims, rows, cmap)
            t = np.hstack([ref, av])
            cv2.putText(t, f"{hs} (ref: {cls})", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            tiles.append(t)
        b.close()
    out = R2 / "v2_sheets"
    out.mkdir(exist_ok=True)
    for k in range(0, len(tiles), 12):
        grp = tiles[k:k + 12]
        while len(grp) % 3:
            grp.append(np.full_like(tiles[0], 255))
        sheet = np.vstack([np.hstack(grp[i:i + 3]) for i in range(0, len(grp), 3)])
        cv2.imwrite(str(out / f"handshapes_{k // 12 + 1}.jpg"), sheet)
    print("sheets for", len(tiles), "handshapes")


if __name__ == "__main__":
    main()
