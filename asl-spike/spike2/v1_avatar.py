"""V1: avatar playback length - trim, compress holds, direct minimum-jerk transitions, speed up.

Lexicon: one ASL Citizen training clip per class, preferring signer P11 (round-1 avatar signer), then the
clip with the best hand detection. Each clip is cut to its P1 active segment; internal holds (dominant-wrist
speed < HOLD_SPEED for longer than 150 ms) are compressed to 100 ms, except for static-hold signs (ASL-LEX
path movement "None"). Between signs: a direct minimum-jerk transition of TRANS_MS (no return to rest);
rest only at the end (REST_MS). Playback speeds 1.0/1.2/1.4x.

Checks per lexicon sign: dominant-wrist path length of the processed clip vs the uncompressed active
segment (>= 90% required), and the round-1 classifier on the processed landmarks (no rendering).
Durations: round-1 method (whole clips + 200 ms blends) vs new, for the 3 round-1 sentences and the 30 S5
barista lines (glosses the lexicon lacks are skipped). Writes results2/v1_avatar.json and
cache/s2r2/lexicon_v1.json (local only: derived from ASL Citizen).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, citizen_class_map, citizen_rows, jdump, load_model  # noqa: E402
from features import normalize  # noqa: E402
import trim as TR  # noqa: E402

HOLD_SPEED = 0.3   # shoulder widths / s
HOLD_MIN_S = 0.15
HOLD_KEEP_S = 0.10
TRANS_MS = 160
REST_MS = 300
R1_BLEND_MS = 200


def dom_wrist(d, aspect):
    lh, rh = d["lh"].copy(), d["rh"].copy()
    lh[..., 0] *= aspect
    rh[..., 0] *= aspect
    dom = lh if TR._motion(lh) > TR._motion(rh) else rh
    sh = d["pose"][:, 11:13, :2].copy()
    sh[..., 0] *= aspect
    sw = float(np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1)))
    w = dom[:, 0, :2].copy()
    for k in range(2):
        col = w[:, k]
        ok = ~np.isnan(col)
        if ok.sum() >= 2:
            col[~ok] = np.interp(np.where(~ok)[0], np.where(ok)[0], col[ok])
    return w / sw


def path_len(w):
    v = np.diff(w, axis=0)
    v = v[~np.isnan(v).any(1)]
    return float(np.linalg.norm(v, axis=1).sum())


def compress_holds(w, fps, static):
    """frame indices to keep after compressing internal holds."""
    n = len(w)
    if static or n < 3:
        return np.arange(n)
    sp = np.r_[0, np.linalg.norm(np.diff(w, axis=0), axis=1)] * fps
    slow = sp < HOLD_SPEED
    keep = np.ones(n, bool)
    i = 0
    while i < n:
        if slow[i]:
            j = i
            while j < n and slow[j]:
                j += 1
            if (j - i) / fps > HOLD_MIN_S and i > 0 and j < n:  # internal hold only
                k = max(1, int(round(HOLD_KEEP_S * fps)))
                keep[i + k:j] = False
            i = j
        else:
            i += 1
    return np.where(keep)[0]


def build_lexicon():
    cmap = citizen_class_map()
    rows = citizen_rows()
    rows = rows[(rows["Gloss"].isin(cmap)) & (rows["split"] == "train")]
    trims = json.load(open(CACHE2 / "trim_citizen.json"))
    lex = {}
    for c, g in rows.groupby(rows["Gloss"].map(cmap)):
        g = pd.concat([g[g["signer"] == "P11"], g[g["signer"] != "P11"]])
        best, score = None, -1
        for r in g.itertuples():
            p = ROOT / "cache/s2/citizen" / f"{os.path.splitext(r.video)[0]}.npz"
            if not p.exists() or r.video not in trims:
                continue
            with np.load(p) as d:
                rate = float(np.mean(~np.isnan(d["rh"][:, 0, 0]) | ~np.isnan(d["lh"][:, 0, 0])))
            s = rate + (0.5 if r.signer == "P11" else 0)
            if s > score:
                best, score = r, s
        if best is not None:
            lex[c] = {"video": best.video, "signer": best.signer}
    return lex


def main():
    model, classes = load_model()
    links = json.load(open(R2 / "asllex_links.json"))["classes"]
    trims = json.load(open(CACHE2 / "trim_citizen.json"))
    lex = build_lexicon()
    rows, lexout = [], {}
    for c, e in sorted(lex.items()):
        d = dict(np.load(ROOT / "cache/s2/citizen" / f"{os.path.splitext(e['video'])[0]}.npz"))
        aspect = float(d["width"]) / float(d["height"])
        fps = float(d["fps"]) or 30.0
        t = trims[e["video"]]
        a, b = t["a"], t["b"]
        static = (links[c]["primary"] or {}).get("path_movement") == "None"
        w = dom_wrist(d, aspect)
        keep = a + compress_holds(w[a:b], fps, static)
        pl_active, pl_kept = path_len(w[a:b]), path_len(w[keep])
        sl = lambda arr: arr[keep]  # noqa: E731
        x_full = normalize(d["lh"][..., :2], d["rh"][..., :2], d["pose"][:, [0, 11, 12, 13, 14, 15, 16, 23], :2], d["face"][:, [8, 1, 2, 11, 14, 20], :2])
        x_proc = normalize(sl(d["lh"])[..., :2], sl(d["rh"])[..., :2], sl(d["pose"])[:, [0, 11, 12, 13, 14, 15, 16, 23], :2], sl(d["face"])[:, [8, 1, 2, 11, 14, 20], :2])
        with torch.no_grad():
            lg = model(torch.tensor(np.stack([x_full, x_proc])))
        top = lg.topk(3, 1).indices.numpy()
        ci = classes.index(c)
        rows.append({"class": c, "signer": e["signer"], "static": static, "full_s": len(w) / fps, "active_s": (b - a) / fps,
                     "processed_s": len(keep) / fps, "path_active": pl_active, "path_processed": pl_kept,
                     "path_ratio": pl_kept / pl_active if pl_active > 0 else None,
                     "full_top1": bool(top[0, 0] == ci), "full_top3": bool(ci in top[0]),
                     "proc_top1": bool(top[1, 0] == ci), "proc_top3": bool(ci in top[1])})
        lexout[c] = {"video": e["video"], "signer": e["signer"], "fps": fps, "frames": [int(i) for i in keep], "static": static}
    json.dump(lexout, open(CACHE2 / "lexicon_v1.json", "w"))
    df = pd.DataFrame(rows).set_index("class")

    def dur(glosses, method, speed=1.0):
        gl = [g for g in glosses if g in df.index]
        if not gl:
            return None, 0
        if method == "round1":
            return float(df.loc[gl, "full_s"].sum() + R1_BLEND_MS / 1000 * (len(gl) - 1)), len(gl)
        return float(df.loc[gl, "processed_s"].sum() / speed + TRANS_MS / 1000 * (len(gl) - 1) / speed + REST_MS / 1000), len(gl)
    sents = []
    for p in json.load(open(ROOT / "avatar/sentences.json")):
        sents.append(("round1:" + p["label"], [s["gloss"] for s in p["sentence"]["signs"]]))
    for r in map(json.loads, open(ROOT / "results/s5_e2g_v2.jsonl")):
        sents.append(("s5:" + r["english"], [s["gloss"] for s in (r.get("output") or {}).get("signs", [])]))
    table = []
    for name, gl in sents:
        r1, n = dur(gl, "round1")
        row = {"sentence": name, "glosses": gl, "n_signs_played": n, "skipped": [g for g in gl if g not in df.index], "round1_s": r1}
        for sp in (1.0, 1.2, 1.4):
            d_, _ = dur(gl, "new", sp)
            row[f"new_{sp}x_s"] = d_
            row[f"signs_per_s_{sp}x"] = round(n / d_, 2) if d_ else None
        table.append(row)
    six = [r for r in table if r["n_signs_played"] == 6]
    # 6-sign sentence built from median processed duration (for the gate)
    med_sign = float(df["processed_s"].median())
    six_synth = {sp: round((6 * med_sign + 5 * TRANS_MS / 1000) / sp + REST_MS / 1000, 2) for sp in (1.0, 1.2, 1.3, 1.4)}
    res = {"params": {"hold_speed_sw_per_s": HOLD_SPEED, "hold_min_s": HOLD_MIN_S, "hold_keep_s": HOLD_KEEP_S,
                      "transition_ms": TRANS_MS, "transition_easing": "minimum jerk (10t^3 - 15t^4 + 6t^5)", "rest_ms_at_end": REST_MS,
                      "trimmer": {"rise": TR.RISE, "speed": TR.SPEED, "pad": TR.PAD}},
           "lexicon": {"n_signs": len(df), "signers": df["signer"].value_counts().to_dict(),
                       "median_full_s": round(float(df["full_s"].median()), 2), "median_active_s": round(float(df["active_s"].median()), 2),
                       "median_processed_s": round(med_sign, 2),
                       "path_ratio_min": round(float(df["path_ratio"].min()), 3), "path_ratio_median": round(float(df["path_ratio"].median()), 3),
                       "signs_below_90pct_path": df.index[df["path_ratio"] < 0.9].tolist(),
                       "classifier_full_top1": round(float(df["full_top1"].mean()), 3), "classifier_full_top3": round(float(df["full_top3"].mean()), 3),
                       "classifier_processed_top1": round(float(df["proc_top1"].mean()), 3), "classifier_processed_top3": round(float(df["proc_top3"].mean()), 3),
                       "note": "lexicon clips are training-set clips, so classifier accuracy here is optimistic; the comparison full vs processed is what matters"},
           "six_sign_estimate_s": six_synth, "sentences": table, "per_sign": rows}
    jdump(res, "v1_avatar.json")
    print(json.dumps({k: res[k] for k in ("lexicon", "six_sign_estimate_s")}, indent=1))
    for r in table[:3] + six[:3]:
        print(r["sentence"][:50], r["n_signs_played"], round(r["round1_s"], 1), [round(r[f"new_{s}x_s"], 2) for s in (1.0, 1.2, 1.4)])


if __name__ == "__main__":
    main()
