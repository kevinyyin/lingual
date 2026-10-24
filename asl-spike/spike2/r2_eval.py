"""R2: round-1 ASL-Citizen-only model on WLASL (new signers, mostly two-handed-capable dictionary video), no
retraining, argmax over all 143 classes. Two feature variants: (1) the identical round-1 pipeline;
(2) with x rescaled to ASL Citizen's pixel geometry (x * (w/h) / 1.333), the aspect fix found in R1.
Breakdowns: ASL-LEX sign type, variation_id (0 vs >=1), source site, YouTube vs not, per sign; signer
bootstrap CIs. Also builds and saves the signer-disjoint WLASL split for R4 (results2/wlasl_split.json).
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, acc_with_ci, jdump, load_model  # noqa: E402
from features import normalize  # noqa: E402

CIT_ASPECT = 4 / 3
POSE8 = [0, 11, 12, 13, 14, 15, 16, 23]
FACE6 = [8, 1, 2, 11, 14, 20]


def wl_features(d, aspect_fix):
    lh, rh, pose, face = (d["lh"][..., :2].copy(), d["rh"][..., :2].copy(), d["pose"][:, POSE8, :2].copy(), d["face"][:, FACE6, :2].copy())
    if aspect_fix:
        s = (float(d["width"]) / float(d["height"])) / CIT_ASPECT
        for a in (lh, rh, pose, face):
            a[..., 0] *= s
    return normalize(lh, rh, pose, face)


def load_items():
    dl = pd.DataFrame(json.load(open(R2 / "r2_downloads.json"))["rows"])
    items = []
    for r in dl.itertuples():
        p = CACHE2 / "wlasl" / f"{r.video_id}.npz"
        if r.status != "ok" or not p.exists():
            continue
        d = dict(np.load(p))
        items.append({"video_id": r.video_id, "class": r._3, "signer": str(r.signer_id), "source": r.source,
                      "variation_id": int(r.variation_id), "youtube": bool(r.youtube), "d": d,
                      "hand_rate": float(np.mean(~np.isnan(d["rh"][:, 0, 0]) | ~np.isnan(d["lh"][:, 0, 0]))),
                      "aspect": float(d["width"]) / float(d["height"])})
    return items, dl


def make_split(items, seed=0, test_frac=0.3, val_frac=0.1, tries=400):
    signers = sorted({it["signer"] for it in items})
    by_s = {s: {it["class"] for it in items if it["signer"] == s} for s in signers}
    all_c = {it["class"] for it in items}
    rng = np.random.default_rng(seed)
    best, bscore = None, -1
    nt, nv = round(test_frac * len(signers)), round(val_frac * len(signers))
    for _ in range(tries):
        perm = list(rng.permutation(signers))
        test, val = perm[:nt], perm[nt:nt + nv]
        cov = len(set().union(*[by_s[s] for s in test])) / len(all_c)
        trc = len(set().union(*[by_s[s] for s in perm[nt + nv:]])) / len(all_c)
        sc = cov + 0.5 * trc
        if sc > bscore:
            best, bscore = (test, val, perm[nt + nv:]), sc
    test, val, train = best
    return {"seed": seed, "test_signers": sorted(test), "val_signers": sorted(val), "train_signers": sorted(train),
            "test_class_coverage": round(len(set().union(*[by_s[s] for s in test])) / len(all_c), 3),
            "n_classes": len(all_c), "method": f"{tries} seeded random signer permutations; maximize test class coverage + 0.5 x train coverage"}


def main():
    model, classes = load_model()
    links = json.load(open(R2 / "asllex_links.json"))["classes"]
    items, dl = load_items()
    y = np.array([classes.index(it["class"]) for it in items])
    sig = np.array([it["signer"] for it in items])
    res = {"downloads": {"instances": int(len(dl)), "ok": int((dl["status"] == "ok").sum()),
                         "extracted": len(items), "failed_reasons": dict(Counter(dl[dl["status"] != "ok"]["reason"].str.slice(0, 60)).most_common(12)),
                         "success_by_source": {s: f"{int((g['status'] == 'ok').sum())}/{len(g)}" for s, g in dl.groupby("source")}},
           "n": len(items), "signers": int(len(set(sig))), "classes": int(len(set(y)))}
    for variant in ("round1_pipeline", "aspect_fix"):
        X = torch.tensor(np.stack([wl_features(it["d"], variant == "aspect_fix") for it in items]))
        with torch.no_grad():
            lg = model(X)
        top = lg.topk(5, 1).indices.numpy()
        h1, h3, h5 = top[:, 0] == y, (top[:, :3] == y[:, None]).any(1), (top == y[:, None]).any(1)
        df = pd.DataFrame({"class": [it["class"] for it in items], "signer": sig, "h1": h1, "h3": h3, "h5": h5,
                           "type": [links[it["class"]]["primary"]["sign_type"] for it in items],
                           "var": ["0" if it["variation_id"] == 0 else ">=1" for it in items],
                           "source": [it["source"] for it in items], "yt": [it["youtube"] for it in items]})

        def br(col):
            return {str(k): {"top1": acc_with_ci(g["h1"], g["signer"]), "top3": round(float(g["h3"].mean()), 4)} for k, g in df.groupby(col)}
        res[variant] = {"top1": acc_with_ci(h1, sig), "top3": acc_with_ci(h3, sig), "top5": acc_with_ci(h5, sig),
                        "by_sign_type": br("type"), "by_variation": br("var"), "by_source": br("source"), "by_youtube": br("yt"),
                        "per_sign": {c: {"top1": round(float(g["h1"].mean()), 3), "top3": round(float(g["h3"].mean()), 3),
                                         "instances": len(g), "signers": int(g["signer"].nunique())} for c, g in df.groupby("class")}}
        print(variant, res[variant]["top1"], res[variant]["top3"]["acc"], flush=True)
    res["hand_detection_rate_median"] = float(np.median([it["hand_rate"] for it in items]))
    res["aspect_values"] = dict(Counter(round(it["aspect"], 2) for it in items).most_common(8))
    split = make_split(items)
    split["items"] = {it["video_id"]: ("test" if it["signer"] in split["test_signers"] else "val" if it["signer"] in split["val_signers"] else "train") for it in items}
    jdump(split, "wlasl_split.json")
    res["split_summary"] = {k: len(split[k]) for k in ("test_signers", "val_signers", "train_signers")} | {"test_class_coverage": split["test_class_coverage"]}
    jdump(res, "r2_wlasl_eval.json")
    print(json.dumps({k: res[k] for k in ("downloads", "n", "signers", "classes", "split_summary")}, indent=1))


if __name__ == "__main__":
    main()
