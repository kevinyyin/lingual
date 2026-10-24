"""R5: core vocabulary proposal (~50 signs) - every proposal needs Deaf advisor review.

Robustness score per class = min(ASL Citizen test top-1, WLASL top-1 (no retraining, R2), post-calibration top-1
at k = 3 on the 6 PopSign stand-in signers (fine-tune, 1 draw each; only the 53 overlapping signs have this))
minus a confusability penalty (0.15 x the share of the class's test errors that go to its single most-confused
partner, from round-1 ASL Citizen + R2 WLASL confusions). Classes with too few WLASL instances (< 3) use
ASL Citizen + calibration only.

Core set: (1) every sign the 14 customer lines need (flagged if weak, i.e. score < 0.6), (2) required dialogue
functions (greeting, ordering, yes/no, a wh-question, negation, thank-you), then (3) the highest-scoring
remaining café signs up to ~50. Weak needed signs get a proposed rewording with stronger signs.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, jdump, load_model  # noqa: E402

REQUIRED = {"greeting": ["HELLO"], "ordering": ["WANT"], "yes/no question": ["HAVE", "CAN"], "wh-question": ["WHAT", "WHERE", "HOW-MUCH"],
            "negation": ["NOT", "NONE"], "thank-you": ["THANK-YOU"], "answer": ["YES", "NO"]}
# candidate rewordings for weak signs (English-gloss level; needs advisor review)
ALTERNATIVES = {"HOW-MUCH": ["COST"], "WARM": ["HOT"], "KIND": ["WHAT"], "LARGE": ["BIG"], "SWEET": ["SUGAR"], "FINE": ["GOOD"],
                "DAY": ["GOOD"], "SORRY": ["PLEASE"], "AGAIN": ["PLEASE"], "UNDERSTAND": ["KNOW"], "NONE": ["NOT"], "NEED": ["WANT"],
                "MY": ["IX-1P"], "PAY": ["CARD"], "HERE": ["SIT"], "BLUEBERRY": ["<fingerspell>"], "COFFEE": ["<keep>"]}


def calib_per_sign():
    import r3_calibrate as R3
    F = dict(np.load(CACHE2 / "kaggle_feats_aspect.npz", allow_pickle=True))
    X, y, pid = F["X"].astype(np.float32), F["y"], F["pid"].astype(str)
    model, classes = load_model()
    sys.path.insert(0, str(ROOT / "spike"))
    from s2_train import load_features
    CX, Cy, Cs, *_ = load_features()
    replay = (CX[Cs == "train"], Cy[Cs == "train"])
    tuned = json.load(open(R2 / "r3_calibration_base.json"))["chosen"]["finetune"]
    hits = defaultdict(list)
    for s in R3.TEST_SIGNERS:
        m = pid == s
        Xs, ys = X[m], y[m]
        cal, te = R3.draw(ys, 3, np.random.default_rng(7))
        ft = R3.finetune(model, Xs[cal], ys[cal], replay, tuned["epochs"], tuned["lr"], 0)
        with torch.no_grad():
            pred = ft(torch.tensor(Xs[te])).argmax(1).numpy()
        for t, p in zip(ys[te], pred):
            hits[classes[t]].append(t == p)
    return {c: float(np.mean(v)) for c, v in hits.items()}


def main():
    _, classes = load_model()
    cit = json.load(open(ROOT / "results/s2_citizen_eval.json"))
    cit_acc = {c: v["acc"] for c, v in cit["per_sign"].items()}
    wl = json.load(open(R2 / "r2_wlasl_eval.json"))["aspect_fix"]["per_sign"]
    cal = calib_per_sign()
    # confusions: round-1 citizen test predictions
    conf = defaultdict(Counter)
    for p in cit["test_predictions"]:
        if p["true"] != p["pred"]:
            conf[p["true"]][p["pred"]] += 1
    rows = {}
    for c in classes:
        parts = {"citizen": cit_acc.get(c)}
        if c in wl and wl[c]["instances"] >= 3:
            parts["wlasl"] = wl[c]["top1"]
        if c in cal:
            parts["calibrated_k3"] = round(cal[c], 3)
        errs = sum(conf[c].values())
        top_partner, top_n = (conf[c].most_common(1)[0] if errs else (None, 0))
        pen = 0.15 * (top_n / errs) if errs else 0.0
        vals = [v for v in parts.values() if v is not None]
        rows[c] = {**parts, "score": round(min(vals) - pen, 3) if vals else None, "most_confused_with": top_partner,
                   "penalty": round(pen, 3), "wlasl_instances": wl.get(c, {}).get("instances", 0)}
    import p2_gen as G
    lines = G.script_lines()[:14]
    needed = [t for l in lines for t in l if t != "<unk>"]
    need_set = list(dict.fromkeys(needed))
    core = list(need_set)
    for fn, opts in REQUIRED.items():
        best = max((o for o in opts if o in rows), key=lambda o: rows[o]["score"] or -1, default=None)
        if best and best not in core:
            core.append(best)
    rest = sorted((c for c in classes if c not in core and rows[c]["score"] is not None), key=lambda c: -rows[c]["score"])
    core += rest[: max(0, 50 - len(core))]
    weak = [c for c in need_set if (rows[c]["score"] or 0) < 0.6]
    edits = []
    for i, l in enumerate(lines):
        bad = [t for t in l if t in weak]
        if not bad:
            continue
        prop = []
        for t in l:
            alt = [a for a in ALTERNATIVES.get(t, []) if a in rows and (rows[a]["score"] or 0) > (rows[t]["score"] or 0)]
            prop.append(alt[0] if t in weak and alt else t)
        edits.append({"line": i, "glosses": l, "weak": {t: rows[t]["score"] for t in bad},
                      "proposal": prop if prop != l else None, "status": "needs Deaf advisor review"})
    out = {"method": __doc__, "core": [{"gloss": c, **rows[c], "needed_by_script": c in need_set} for c in core],
           "n_core": len(core), "weak_needed_signs": {c: rows[c] for c in weak},
           "script_edits": edits, "all": rows, "status": "every proposal needs Deaf advisor review"}
    jdump(out, "vocab_core.json")
    print("core", len(core), "needed", len(need_set), "weak needed", {c: rows[c]["score"] for c in weak})
    for e in edits:
        print(e["glosses"], "->", e["proposal"], e["weak"])


if __name__ == "__main__":
    main()
