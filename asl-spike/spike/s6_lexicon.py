"""S6: build avatar/signs.json (one pose sequence per gloss) and avatar/sentences.json (demo sentences
from S5 outputs) for the avatar prototype.

Per gloss we take one ASL Citizen training video, preferring a single signer for consistent proportions
(the signer who covers most needed glosses), then the highest hand-detection rate. Frames carry pose world
landmarks (pw), normalized pose (p2), and hand landmarks (lh/rh, or null when not detected), resampled
to 30 fps. LOCAL RESEARCH USE ONLY: ASL Citizen's license forbids distributing the data or derived data.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def frames_of(d):
    out = []
    T = len(d["pose"])
    for i in range(T):
        def hand(a):
            return None if np.isnan(a[i]).all() else np.round(a[i], 4).tolist()
        pw = d["pose_world"][i]
        out.append({"pw": None if np.isnan(pw).all() else np.round(pw, 4).tolist(),
                    "p2": None if np.isnan(d["pose"][i]).all() else np.round(np.nan_to_num(d["pose"][i, :, :4]), 4).tolist(),
                    "lh": hand(d["lh"]), "rh": hand(d["rh"])})
    # trim leading/trailing frames without hands (keep 3 frames of margin)
    idx = [k for k, f in enumerate(out) if f["lh"] or f["rh"]]
    if idx:
        out = out[max(0, idx[0] - 3): idx[-1] + 4]
    return out


def main():
    vocab = {e["gloss"]: e for e in json.load(open(ROOT / "vocab.json"))}
    rows = pd.read_csv(ROOT / "results/s2_citizen_subset.csv")
    rows = rows[rows["split"] == "train"]
    c2v = {g: e["gloss"] for e in vocab.values() for g in e["asl_citizen_class_videos"]}
    rows = rows.assign(vg=rows["Gloss"].map(c2v)).dropna(subset=["vg"])
    # demo sentences: S5 v2 outputs fully covered by the lexicon
    have = set(rows["vg"])
    cands = []
    for r in map(json.loads, open(ROOT / "results/s5_e2g_v2.jsonl")):
        signs = (r.get("output") or {}).get("signs") or []
        keep = [s for s in signs if s["gloss"] in have]
        if len(keep) < 3 or len(keep) < len(signs) - 1:
            continue
        kind = "wh" if any(s["nmm"]["brows"] == "lower" for s in keep) else \
            "yn" if any(s["nmm"]["brows"] == "raise" for s in keep) else \
            "neg" if any(s["nmm"]["headshake"] for s in keep) else "statement"
        cands.append({"kind": kind, "english": r["english"], "signs": keep, "dropped": [s["gloss"] for s in signs if s not in keep]})
    picks = []
    for kind in ("wh", "yn", "neg"):
        ks = sorted([c for c in cands if c["kind"] == kind], key=lambda c: (-len(c["signs"]), c["english"]))
        if ks:
            picks.append(ks[0])
    need = sorted({s["gloss"] for p in picks for s in p["signs"]})
    sub = rows[rows["vg"].isin(need)]
    best_signer = Counter(sub.groupby("Participant ID")["vg"].nunique().to_dict()).most_common(1)[0][0]
    lex, src = {}, {}
    for g in need:
        cand = sub[sub["vg"] == g]
        cand = pd.concat([cand[cand["Participant ID"] == best_signer], cand[cand["Participant ID"] != best_signer]])
        best, bscore = None, -1
        for v in cand["Video file"]:
            p = ROOT / "cache/s2/citizen" / f"{Path(v).stem}.npz"
            if not p.exists():
                continue
            d = dict(np.load(p))
            rate = float(np.mean(~np.isnan(d["rh"][:, 0, 0]) | ~np.isnan(d["lh"][:, 0, 0])))
            score = rate + (0.5 if v in set(cand[cand["Participant ID"] == best_signer]["Video file"]) else 0)
            if score > bscore:
                best, bscore = (v, d), score
        if best is None and len(cand):
            # not extracted yet by S2: extract the preferred exemplar now
            import sys as _s
            _s.path.insert(0, str(ROOT / "spike"))
            from extract import holistic_video
            v = cand["Video file"].iloc[0]
            d = holistic_video(str(ROOT / "data/asl-citizen/videos_vocab" / v))
            best = (v, d)
        if best:
            v, d = best
            lex[g] = {"fps": float(d["fps"]) or 30.0, "frames": frames_of(d)}
            src[g] = v
    json.dump(lex, open(ROOT / "avatar/signs.json", "w"))
    presets = [{"label": f"{p['kind']}: {p['english']}", "sentence": {"signs": p["signs"]}, "dropped_glosses": p["dropped"]}
               for p in picks]
    json.dump(presets, open(ROOT / "avatar/sentences.json", "w"), indent=1)
    json.dump({"signer": best_signer, "sources": src, "presets": presets,
               "license": "ASL Citizen (Microsoft Research License Terms): research use only; no distribution of data or derived data"},
              open(ROOT / "results/s6_lexicon.json", "w"), indent=1)
    print("presets:", [(p["label"], [s["gloss"] for s in p["sentence"]["signs"]]) for p in presets])
    print("lexicon", len(lex), "signs; signer", best_signer, "; size", (ROOT / "avatar/signs.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
