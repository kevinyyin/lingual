"""P1 validation and trim table.

1. Grid over RISE x SPEED on ASL Citizen *validation* signers: choose the setting that removes the most
   frames while the round-1 classifier's top-1 on trimmed clips stays within 1 point of untrimmed.
2. Report on *test* signers: removal distribution, untrimmed vs trimmed top-1/top-3.
3. Save trim points (a, b) for every ASL Citizen clip of the 143 classes: cache/s2r2/trim_citizen.json.
4. 20 random test-clip contact sheets (lead-in, onset, middle, offset, lead-out): results2/p1_contact/ (local).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, citizen_class_map, citizen_rows, jdump, load_model, pstats  # noqa: E402
from features import from_holistic, normalize  # noqa: E402
import trim as TR  # noqa: E402


def load_all():
    cmap = citizen_class_map()
    rows = citizen_rows()
    rows = rows[rows["Gloss"].isin(cmap)]
    links = json.load(open(R2 / "asllex_links.json"))["classes"]
    items = []
    for r in rows.itertuples():
        p = ROOT / "cache/s2/citizen" / f"{os.path.splitext(r.video)[0]}.npz"
        if not p.exists():
            continue
        d = dict(np.load(p))
        c = cmap[r.Gloss]
        items.append({"video": r.video, "signer": r.signer, "split": r.split, "cls": c, "d": d,
                      "static": (links[c]["primary"] or {}).get("path_movement") == "None",
                      "aspect": float(d["width"]) / float(d["height"]), "fps": float(d["fps"]) or 30.0})
    return items


def feats(it, a=None, b=None):
    d = it["d"]
    sl = slice(a, b)
    return normalize(d["lh"][sl, :, :2], d["rh"][sl, :, :2], d["pose"][sl][:, [0, 11, 12, 13, 14, 15, 16, 23], :2], d["face"][sl][:, [8, 1, 2, 11, 14, 20], :2])


def score(model, classes, items, trims=None):
    X = np.stack([feats(it, *(trims[i] if trims else (None, None))) for i, it in enumerate(items)])
    y = np.array([classes.index(it["cls"]) for it in items])
    with torch.no_grad():
        lg = torch.cat([model(torch.tensor(X[i:i + 512])) for i in range(0, len(X), 512)])
    top = lg.topk(3, 1).indices.numpy()
    return float((top[:, 0] == y).mean()), float((top == y[:, None]).any(1).mean())


def main():
    model, classes = load_model()
    items = load_all()
    val = [it for it in items if it["split"] == "val"]
    test = [it for it in items if it["split"] == "test"]
    base_val = score(model, classes, val)
    grid = []
    for rise in (0.15, 0.25, 0.35, 0.5):
        for spd in (0.3, 0.6, 1.0, 1.5):
            tr = [TR.trim(it["d"]["lh"], it["d"]["rh"], it["d"]["pose"], it["fps"], it["aspect"], it["static"], rise=rise, speed=spd)[:2] for it in val]
            rem = float(np.mean([1 - (b - a) / len(it["d"]["lh"]) for (a, b), it in zip(tr, val)]))
            t1, t3 = score(model, classes, val, tr)
            grid.append({"rise": rise, "speed": spd, "removed_mean": round(rem, 3), "val_top1": round(t1, 4), "val_top3": round(t3, 4)})
            print(grid[-1], flush=True)
    ok = [g for g in grid if g["val_top1"] >= base_val[0] - 0.01]
    best = max(ok, key=lambda g: g["removed_mean"]) if ok else max(grid, key=lambda g: g["val_top1"])
    rise, spd = best["rise"], best["speed"]
    all_tr = {}
    for it in items:
        a, b, info = TR.trim(it["d"]["lh"], it["d"]["rh"], it["d"]["pose"], it["fps"], it["aspect"], it["static"], rise=rise, speed=spd)
        all_tr[it["video"]] = {"a": a, "b": b, "T": len(it["d"]["lh"]), "fps": it["fps"], **info}
    json.dump(all_tr, open(CACHE2 / "trim_citizen.json", "w"))
    tt = [(all_tr[it["video"]]["a"], all_tr[it["video"]]["b"]) for it in test]
    base_t = score(model, classes, test)
    trim_t = score(model, classes, test, tt)
    rem = [1 - (b - a) / len(it["d"]["lh"]) for (a, b), it in zip(tt, test)]
    secs_before = [len(it["d"]["lh"]) / it["fps"] for it in test]
    secs_after = [(b - a) / it["fps"] for (a, b), it in zip(tt, test)]
    # contact sheets
    cs = R2 / "p1_contact"
    cs.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    for k in rng.choice(len(test), 20, replace=False):
        it = test[k]
        a, b = tt[k]
        T = len(it["d"]["lh"])
        idx = [max(0, a // 2), a, (a + b) // 2, max(a, b - 1), min(T - 1, (b + T) // 2)]
        cap = cv2.VideoCapture(str(ROOT / "data/asl-citizen/videos_vocab" / it["video"]))
        fr, i = {}, 0
        while True:
            ok_, f = cap.read()
            if not ok_:
                break
            if i in idx:
                fr[i] = cv2.resize(f, (240, int(240 * f.shape[0] / f.shape[1])))
            i += 1
        tiles = []
        for lab, j in zip(("lead-in", "onset", "middle", "offset", "lead-out"), idx):
            t = fr.get(j, np.zeros((135, 240, 3), np.uint8)).copy()
            cv2.putText(t, f"{lab} {j}", (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            tiles.append(t)
        h = min(t.shape[0] for t in tiles)
        cv2.imwrite(str(cs / f"{k:04d}_{it['cls']}.jpg"), np.hstack([t[:h] for t in tiles]))
    res = {"params": {"rise_sw": rise, "speed_sw_per_s": spd, "hysteresis_s": TR.HYST_S, "min_hold_s_static": TR.MIN_HOLD_S,
                      "pad_frames": TR.PAD, "rest_height": "10th percentile of the clip's dominant-wrist height",
                      "aspect": "x multiplied by width/height before geometry"},
           "grid_val": grid, "val_untrimmed": {"top1": base_val[0], "top3": base_val[1]},
           "test": {"n": len(test), "signers": len({it["signer"] for it in test}),
                    "untrimmed_top1": base_t[0], "untrimmed_top3": base_t[1], "trimmed_top1": trim_t[0], "trimmed_top3": trim_t[1],
                    "removed_frac": pstats(rem), "seconds_before": pstats(secs_before), "seconds_after": pstats(secs_after)},
           "contact_sheets": "results2/p1_contact/ (ASL Citizen frames; local only)"}
    jdump(res, "p1_trimmer.json")
    print(json.dumps({k: res[k] for k in ("params", "val_untrimmed", "test")}, indent=1))


if __name__ == "__main__":
    main()
