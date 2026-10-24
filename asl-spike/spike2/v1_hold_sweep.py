"""V1 follow-up: sweep the hold-speed threshold for hold compression (path length must stay >= 90%)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, jdump
import v1_avatar as V
lex = json.load(open(CACHE2 / "lexicon_v1.json")); trims = json.load(open(CACHE2 / "trim_citizen.json"))
links = json.load(open(R2 / "asllex_links.json"))["classes"]
out = []
for hs in (0.3, 0.6, 1.0, 1.5, 2.0):
    V.HOLD_SPEED = hs
    ratios, durs = [], []
    for c, e in lex.items():
        d = dict(np.load(ROOT / "cache/s2/citizen" / f"{os.path.splitext(e['video'])[0]}.npz"))
        a, b = trims[e["video"]]["a"], trims[e["video"]]["b"]
        fps = float(d["fps"]) or 30.0
        w = V.dom_wrist(d, float(d["width"]) / float(d["height"]))
        keep = a + V.compress_holds(w[a:b], fps, e["static"])
        pa = V.path_len(w[a:b]); ratios.append(V.path_len(w[keep]) / pa if pa > 0 else 1.0); durs.append(len(keep) / fps)
    out.append({"hold_speed": hs, "path_ratio_min": round(min(ratios), 3), "below_90": int(sum(r < 0.9 for r in ratios)),
                "median_s": round(float(np.median(durs)), 3), "mean_s": round(float(np.mean(durs)), 3)})
    print(out[-1], flush=True)
jdump(out, "v1_hold_sweep.json")
