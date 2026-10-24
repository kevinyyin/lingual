"""R1: explain the round-1 PopSign gap (no new training).

1. Re-score the round-1 ASL-Citizen-only model on the Kaggle (PopSign) features, per sequence, and slice by
   the class's ASL-LEX sign type (primary entry). Signer-bootstrapped 95% CIs (21 PopSign signers).
2. Both-hands detection fraction per class, PopSign vs ASL Citizen (from r1_stats.py).
3. For classes under 15% top-1: dominant-hand location (rel. to the nose, shoulder widths) and PIP flexion,
   PopSign vs ASL Citizen; flag likely variants.
4. Decompose the drop with counterfactuals: set the accuracy of each group of affected classes to the
   accuracy of the unaffected classes and see how much of the gap closes.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, acc_with_ci, jdump, load_model, vocab  # noqa: E402

LOW_BOTH = 0.35      # PopSign both-hands fraction this much below ASL Citizen => "signed one-handed"
VAR_LOC = 0.6        # shoulder widths
VAR_FLEX = 20.0      # degrees


def popsign_predictions(model, classes):
    v = {e["gloss"]: e for e in vocab()}
    signs = {k for e in vocab() if e["asl_citizen_class_videos"] for k in e["kaggle_asl_signs"]}
    meta = pd.read_csv(ROOT / "data/kaggle-asl-signs/train.csv")
    meta = meta[meta["sign"].isin(signs)].reset_index(drop=True)
    F = dict(np.load(ROOT / "cache/s2/kaggle_feats.npz", allow_pickle=True))
    assert len(meta) == len(F["X"]) and (meta["sign"].to_numpy() == F["sign"]).all(), "feature order mismatch"
    k2v = {}
    for g in classes:
        for k in v[g]["kaggle_asl_signs"]:
            k2v.setdefault(k, g)
    keep = meta["sign"].isin(k2v).to_numpy()
    X = torch.tensor(F["X"][keep])
    with torch.no_grad():
        logits = torch.cat([model(X[i:i + 512]) for i in range(0, len(X), 512)])
    m = meta[keep].reset_index(drop=True)
    y = np.array([classes.index(k2v[s]) for s in m["sign"]])
    top = logits.topk(3, 1).indices.numpy()
    m["class"] = [classes[i] for i in y]
    m["top1"] = top[:, 0] == y
    m["top3"] = (top == y[:, None]).any(1)
    m["pred"] = [classes[i] for i in top[:, 0]]
    return m


def main():
    model, classes = load_model()
    links = json.load(open(R2 / "asllex_links.json"))["classes"]
    st = {c: links[c]["primary"]["sign_type"] for c in classes}
    m = popsign_predictions(model, classes)
    m["sign_type"] = m["class"].map(st)
    m["two_handed"] = m["sign_type"] != "OneHanded"
    res = {"n_sequences": len(m), "n_signers": int(m["participant_id"].nunique()), "n_classes": int(m["class"].nunique()),
           "overall": {"top1": acc_with_ci(m["top1"], m["participant_id"]), "top3": acc_with_ci(m["top3"], m["participant_id"])}}
    res["by_sign_type"] = {t: {"classes": int(g["class"].nunique()), "top1": acc_with_ci(g["top1"], g["participant_id"]),
                               "top3": acc_with_ci(g["top3"], g["participant_id"])} for t, g in m.groupby("sign_type")}
    res["one_vs_two_handed"] = {("two_handed" if k else "one_handed"): {"classes": int(g["class"].nunique()),
                                "top1": acc_with_ci(g["top1"], g["participant_id"]), "top3": acc_with_ci(g["top3"], g["participant_id"])}
                                for k, g in m.groupby("two_handed")}
    # ASL Citizen test accuracy on the same classes (round-1 per-sign results)
    cit = json.load(open(ROOT / "results/s2_citizen_eval.json"))
    cit_acc = {c: v["acc"] for c, v in cit["per_sign"].items()}
    # hand statistics
    hs = [json.loads(x) for x in open(CACHE2 / "r1_handstats.jsonl") if x.strip()]
    hs = pd.DataFrame([h for h in hs if "error" not in h])
    agg = hs.groupby(["dataset", "class"]).agg(both=("both_frac", "mean"), lx=("dom_loc_x", "mean"), ly=("dom_loc_y", "mean"),
                                               flex=("dom_flex", "mean"), n=("both_frac", "size")).reset_index()
    pa = agg[agg.dataset == "popsign"].set_index("class")
    ca = agg[agg.dataset == "citizen"].set_index("class")
    per_class = {}
    for c, g in m.groupby("class"):
        row = {"sign_type": st[c], "popsign_top1": round(float(g["top1"].mean()), 3), "popsign_top3": round(float(g["top3"].mean()), 3),
               "n": len(g), "citizen_test_top1": cit_acc.get(c),
               "both_frac_popsign": round(float(pa.loc[c, "both"]), 3) if c in pa.index else None,
               "both_frac_citizen": round(float(ca.loc[c, "both"]), 3) if c in ca.index else None,
               "top_confusions": g[~g["top1"]]["pred"].value_counts().head(3).to_dict()}
        if c in pa.index and c in ca.index:
            row["dom_loc_popsign"] = [round(float(pa.loc[c, "lx"]), 2), round(float(pa.loc[c, "ly"]), 2)]
            row["dom_loc_citizen"] = [round(float(ca.loc[c, "lx"]), 2), round(float(ca.loc[c, "ly"]), 2)]
            row["dom_flex_popsign"] = round(float(pa.loc[c, "flex"]), 1)
            row["dom_flex_citizen"] = round(float(ca.loc[c, "flex"]), 1)
            dloc = float(np.hypot(pa.loc[c, "lx"] - ca.loc[c, "lx"], pa.loc[c, "ly"] - ca.loc[c, "ly"]))
            dflex = abs(float(pa.loc[c, "flex"] - ca.loc[c, "flex"]))
            row["delta_loc_sw"] = round(dloc, 2)
            row["delta_flex_deg"] = round(dflex, 1)
            row["signed_one_handed"] = bool(st[c] != "OneHanded" and ca.loc[c, "both"] - pa.loc[c, "both"] > LOW_BOTH)
            row["likely_variant"] = bool(row["popsign_top1"] < 0.15 and (dloc > VAR_LOC or dflex > VAR_FLEX))
        per_class[c] = row
    res["per_class"] = per_class
    # both-hands fraction summary by sign type
    res["both_frac_by_type"] = {}
    for t in sorted(set(st[c] for c in per_class)):
        cs = [c for c in per_class if st[c] == t and per_class[c].get("both_frac_popsign") is not None]
        res["both_frac_by_type"][t] = {"classes": len(cs), "popsign": round(float(np.mean([per_class[c]["both_frac_popsign"] for c in cs])), 3),
                                       "citizen": round(float(np.mean([per_class[c]["both_frac_citizen"] for c in cs])), 3)}
    # decomposition (class-level counterfactuals, weighted by PopSign sequence counts)
    w = {c: per_class[c]["n"] for c in per_class}
    acc = {c: per_class[c]["popsign_top1"] for c in per_class}
    tot = sum(w.values())
    actual = sum(acc[c] * w[c] for c in acc) / tot
    cit_same = sum((cit_acc.get(c) or 0) * w[c] for c in acc) / tot
    one = [c for c in acc if per_class[c].get("signed_one_handed")]
    var = [c for c in acc if per_class[c].get("likely_variant") and c not in one]
    clean = [c for c in acc if c not in one and c not in var]
    clean_acc = sum(acc[c] * w[c] for c in clean) / max(sum(w[c] for c in clean), 1)

    def cf(fix):
        return sum((clean_acc if c in fix else acc[c]) * w[c] for c in acc) / tot
    gap = cit_same - actual
    res["decomposition"] = {
        "citizen_test_top1_same_classes_weighted": round(cit_same, 3), "popsign_top1_actual": round(actual, 3), "gap": round(gap, 3),
        "classes_signed_one_handed": one, "classes_likely_variant": var, "clean_classes_popsign_top1": round(clean_acc, 3),
        "explained_by_one_handed": round(cf(one) - actual, 3), "explained_by_variants": round(cf(var) - actual, 3),
        "remaining_gap_after_both": round(cit_same - cf(one + var), 3),
        "method": ("class-level counterfactual: replace the PopSign accuracy of each affected group by the accuracy of the unaffected "
                   "('clean') classes, weighted by sequence counts. 'signed one-handed' = two-handed in ASL-LEX and PopSign "
                   f"both-hands fraction more than {LOW_BOTH} below ASL Citizen; 'likely variant' = under 15% top-1 and dominant-hand "
                   f"location differs by > {VAR_LOC} shoulder widths or PIP flexion by > {VAR_FLEX} deg")}
    jdump(res, "r1_popsign_gap.json")
    print(json.dumps({k: res[k] for k in ("overall", "one_vs_two_handed", "both_frac_by_type", "decomposition")}, indent=1)[:4000])


if __name__ == "__main__":
    main()
