"""V2: data-driven canonical handshapes from ASL Citizen training clips, keyed by ASL-LEX handshape label.

Frames: dominant hand, middle 40% of the P1 active segment, signs whose ASL-LEX FlexionChange is 0, training
split only, all signers. Coordinates: MediaPipe image landmarks with x (and z, same scale as x) made
pixel-proportional (x*w/h); left hands are mirrored to right-hand geometry. (Hand world landmarks were not
cached in round 1; image landmarks with aspect correction are used instead - an approximation.)

Per frame, in a hand-local frame (x: wrist -> middle MCP, z: palm normal from index/pinky MCPs):
  MCP/PIP/DIP flexion for index..pinky (angle between successive bone vectors, deg)
  spread: angle between adjacent proximal phalanges projected on the palm plane (index-middle, middle-ring,
          ring-pinky)
  thumb: CMC flexion (wrist->CMC vs CMC->MCP), MCP flexion, IP flexion, abduction (angle between thumb
         metacarpal and index metacarpal in the palm plane), opposition (thumb metacarpal elevation out of
         the palm plane)
Per handshape: median and IQR over frames; counts of frames, clips, signers. Flexion-change signs get a
data-derived end handshape: median over the last 20% of their active segment.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, citizen_class_map, citizen_rows, jdump  # noqa: E402
import trim as TR  # noqa: E402

FING = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
ANGLE_NAMES = ([f"{f}_{j}" for f in FING for j in ("mcp", "pip", "dip")] + ["spread_im", "spread_mr", "spread_rp"] +
               ["thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_abd", "thumb_opp"])


def ang(u, v):
    cu = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cu, -1, 1)))


def hand_angles(h):
    """h: (T,21,3) right-hand geometry -> (T, 20) angles in degrees."""
    w = h[:, 0]
    x = h[:, 9] - w
    x /= np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
    n = np.cross(h[:, 5] - w, h[:, 17] - w)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9

    def plane(v):
        return v - (v * n).sum(-1, keepdims=True) * n
    out = []
    for f, (m, p, d, t) in FING.items():
        out += [ang(h[:, m] - w, h[:, p] - h[:, m]), ang(h[:, p] - h[:, m], h[:, d] - h[:, p]), ang(h[:, d] - h[:, p], h[:, t] - h[:, d])]
    prox = [plane(h[:, FING[f][1]] - h[:, FING[f][0]]) for f in FING]
    out += [ang(prox[0], prox[1]), ang(prox[1], prox[2]), ang(prox[2], prox[3])]
    out += [ang(h[:, 1] - w, h[:, 2] - h[:, 1]), ang(h[:, 2] - h[:, 1], h[:, 3] - h[:, 2]), ang(h[:, 3] - h[:, 2], h[:, 4] - h[:, 3])]
    tm = h[:, 2] - h[:, 1]
    out.append(ang(plane(tm), plane(h[:, 5] - w)))
    out.append(np.degrees(np.arcsin(np.clip((tm * n).sum(-1) / (np.linalg.norm(tm, axis=-1) + 1e-9), -1, 1))))
    return np.stack(out, 1)


def dominant_right(d, aspect):
    lh, rh = d["lh"].copy(), d["rh"].copy()
    for a in (lh, rh):
        a[..., 0] *= aspect
        a[..., 2] *= aspect
    if TR._motion(lh) > TR._motion(rh):
        lh[..., 0] = -lh[..., 0]  # mirror left hand into right-hand geometry
        return lh
    return rh


def main():
    links = json.load(open(R2 / "asllex_links.json"))["classes"]
    trims = json.load(open(CACHE2 / "trim_citizen.json"))
    cmap = citizen_class_map()
    rows = citizen_rows()
    rows = rows[(rows["Gloss"].isin(cmap)) & (rows["split"] == "train")]
    # ASL-LEX entry per ASL Citizen gloss (variants can differ in handshape)
    entry_of = {}
    for c, v in links.items():
        for e in [v["primary"]] + v["variants"]:
            if e:
                for g in e["citizen_glosses"]:
                    entry_of[g] = e
    by_hs = defaultdict(list)
    meta = defaultdict(lambda: {"clips": 0, "signers": set(), "classes": set()})
    ends = defaultdict(list)
    for r in rows.itertuples():
        e = entry_of.get(r.Gloss)
        if not e or r.video not in trims:
            continue
        p = ROOT / "cache/s2/citizen" / f"{os.path.splitext(r.video)[0]}.npz"
        if not p.exists():
            continue
        d = dict(np.load(p))
        h = dominant_right(d, float(d["width"]) / float(d["height"]))
        a, b = trims[r.video]["a"], trims[r.video]["b"]
        L = b - a
        if L < 5:
            continue
        ok = ~np.isnan(h[:, 0, 0])
        if not e.get("flexion_change"):
            sl = np.arange(a + int(0.3 * L), a + int(0.7 * L))
            sl = sl[ok[sl]]
            if len(sl):
                by_hs[e["handshape"]].append(hand_angles(h[sl]))
                m = meta[e["handshape"]]
                m["clips"] += 1
                m["signers"].add(r.signer)
                m["classes"].add(cmap[r.Gloss])
        else:
            sl = np.arange(b - max(1, int(0.2 * L)), b)
            sl = sl[ok[sl]]
            if len(sl):
                ends[cmap[r.Gloss]].append(hand_angles(h[sl]))
    lib = {}
    for hs, arrs in by_hs.items():
        A = np.concatenate(arrs)
        lib[hs] = {"median": dict(zip(ANGLE_NAMES, np.round(np.nanmedian(A, 0), 1).tolist())),
                   "iqr": dict(zip(ANGLE_NAMES, np.round(np.nanpercentile(A, 75, 0) - np.nanpercentile(A, 25, 0), 1).tolist())),
                   "frames": int(len(A)), "clips": meta[hs]["clips"], "signers": len(meta[hs]["signers"]), "classes": sorted(meta[hs]["classes"])}
    end_hs = {c: {"median": dict(zip(ANGLE_NAMES, np.round(np.nanmedian(np.concatenate(v), 0), 1).tolist())), "clips": len(v)}
              for c, v in ends.items()}
    needed = sorted({(links[c]["primary"] or {}).get("handshape") for c in links} - {None})
    nd_needed = sorted({(links[c]["primary"] or {}).get("nondominant_handshape") for c in links} - {None})
    res = {"handshapes_needed_dominant": needed, "handshapes_needed_nondominant": nd_needed,
           "library": lib, "missing_from_library": [h for h in needed + nd_needed if h not in lib],
           "flexion_change_end_shapes": end_hs, "angle_names": ANGLE_NAMES,
           "note": "image landmarks with aspect correction (hand world landmarks not cached); left hands mirrored"}
    jdump(res, "handshapes.json")
    print("needed", len(needed), "in library", len(lib), "missing", res["missing_from_library"])
    for hs in sorted(lib, key=lambda k: -lib[k]["frames"])[:40]:
        m = lib[hs]["median"]
        print(f"{hs:12s} frames {lib[hs]['frames']:6d} signers {lib[hs]['signers']:3d} idx {m['index_mcp']:5.0f}/{m['index_pip']:5.0f} mid {m['middle_mcp']:5.0f}/{m['middle_pip']:5.0f} pinky {m['pinky_mcp']:5.0f}/{m['pinky_pip']:5.0f} th {m['thumb_mcp']:4.0f} abd {m['thumb_abd']:4.0f}")


if __name__ == "__main__":
    main()
