"""S3 evaluation on RWTH-BOSTON-104 matched to NCSLGR non-manual annotations (source a), with
leave-one-signer-out (3 signers). Also the naive facial-expression baseline (HSEmotion, frame by frame)
and the affect layer's label distribution per sentence type.

Labels: from results/s3_rwth_ncslgr_match.json. Primary set = exact gloss matches; secondary = exact + near
(edit distance 1). RWTH-BOSTON-104 contains no yes/no questions, so sentence type here is statement vs wh;
rhetorical questions (rhq) are reported separately.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from s3_extract import HSE_CLASSES  # noqa: E402
from s3_face import affect_layer, signals, signer_norms, signing_window, utterance_features  # noqa: E402
from stats import wilson  # noqa: E402
from voice_map import Kinematics, to_voice  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRAM_FEATS = ["brow_up_mean", "brow_up_sus_max", "brow_up_end", "brow_down_mean", "brow_down_sus_max", "brow_down_end",
              "brow_h_mean", "brow_h_sus_max", "brow_h_sus_min", "brow_h_end", "pitch_mean"]
NEG_FEATS = ["hs_amp_rel", "hs_reversals_per_s", "hs_band_ratio"]


def sent_type(nmm):
    if nmm is None:
        return None
    if "wh" in nmm:
        return "wh"
    if "yn" in nmm:
        return "yn"
    if "rhq" in nmm:
        return "rhq"
    return "statement"


def load(src="rwth", match_levels=("exact",)):
    m = json.load(open(ROOT / "results/s3_rwth_ncslgr_match.json"))["rows"]
    items = []
    for r in m:
        if r["match"] not in match_levels:
            continue
        p = ROOT / "cache/s3" / src / f"{r['name']}.npz"
        if not p.exists():
            continue
        d = dict(np.load(p, allow_pickle=True))
        items.append({"name": r["name"], "signer": r["speaker"], "type": sent_type(r["nmm"]),
                      "neg": "negative" in r["nmm"], "nmm": r["nmm"], "orth": r["orth"], "translation": r["translation"],
                      "d": d, "fps": float(d["fps"]) if float(d["fps"]) > 0 else 30.0})
    return items


def featurize(items):
    by = defaultdict(list)
    for it in items:
        it["sig"] = signals(it["d"])
        by[it["signer"]].append(it["sig"])
    norms = {s: signer_norms(v) for s, v in by.items()}
    for it in items:
        it["f"] = utterance_features(it["sig"], norms[it["signer"]], it["fps"])
    # headshake amplitude relative to the signer's median utterance
    med = {s: np.median([it["f"]["hs_amp"] for it in items if it["signer"] == s]) for s in by}
    for it in items:
        it["f"]["hs_amp_rel"] = it["f"]["hs_amp"] / max(med[it["signer"]], 1e-9)
    kin = {s: {k: float(np.median([it["f"][k] for it in items if it["signer"] == s])) for k in ("kin_speed", "kin_amp")}
           for s in by}
    for it in items:
        it["affect"] = affect_layer(it["f"], kin[it["signer"]])
    return items


def loso(items, feats, label_fn, positive=None, C=1.0):
    """Leave-one-signer-out logistic regression. Returns predictions aligned with items."""
    signers = sorted({it["signer"] for it in items})
    pred = [None] * len(items)
    for s in signers:
        tr = [i for i, it in enumerate(items) if it["signer"] != s]
        te = [i for i, it in enumerate(items) if it["signer"] == s]
        Xtr = np.array([[items[i]["f"][k] for k in feats] for i in tr])
        ytr = np.array([label_fn(items[i]) for i in tr])
        if len(set(ytr)) < 2:
            for i in te:
                pred[i] = ytr[0]
            continue
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000).fit(sc.transform(Xtr), ytr)
        Xte = np.array([[items[i]["f"][k] for k in feats] for i in te])
        for i, p in zip(te, clf.predict(sc.transform(Xte))):
            pred[i] = p
    return pred


def loso_threshold(items, key, label_fn, sign=+1):
    """One-feature threshold tuned on training signers (balanced accuracy), applied to the held-out signer."""
    signers = sorted({it["signer"] for it in items})
    pred = [None] * len(items)
    for s in signers:
        tr = [it for it in items if it["signer"] != s]
        vals = sorted({sign * it["f"][key] for it in tr})
        best, bt = -1, 0.0
        for t in vals:
            tp = sum(1 for it in tr if label_fn(it) and sign * it["f"][key] >= t)
            fn = sum(1 for it in tr if label_fn(it) and sign * it["f"][key] < t)
            tn = sum(1 for it in tr if not label_fn(it) and sign * it["f"][key] < t)
            fp = sum(1 for it in tr if not label_fn(it) and sign * it["f"][key] >= t)
            bal = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
            if bal > best:
                best, bt = bal, t
        for i, it in enumerate(items):
            if it["signer"] == s:
                pred[i] = sign * it["f"][key] >= bt
    return pred


def confusion(y, p, labels):
    cm = {a: {b: 0 for b in labels} for a in labels}
    for a, b in zip(y, p):
        if a in cm and b in cm[a]:
            cm[a][b] += 1
    return cm


def naive_baseline(it):
    a, b = signing_window(it["sig"], it["fps"])
    h = it["d"]["hse"][a:b]
    h = h[~np.isnan(h).any(1)]
    if len(h) == 0:
        return {"majority": None, "frame_share": {}}
    lab = [HSE_CLASSES[i] for i in h.argmax(1)]
    c = Counter(lab)
    return {"majority": c.most_common(1)[0][0], "frame_share": {k: round(v / len(lab), 3) for k, v in c.items()},
            "mean_prob": {k: round(float(v), 3) for k, v in zip(HSE_CLASSES, h.mean(0))}}


def cramers_v(a, b):
    la, lb = sorted(set(a)), sorted(set(b))
    if len(la) < 2 or len(lb) < 2:
        return None
    t = np.array([[sum(1 for x, y in zip(a, b) if x == i and y == j) for j in lb] for i in la], float)
    n = t.sum()
    e = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / n
    chi = ((t - e) ** 2 / np.where(e == 0, 1, e)).sum()
    return float(np.sqrt(chi / n / (min(len(la), len(lb)) - 1)))


def evaluate(match_levels):
    items = featurize(load("rwth", match_levels))
    st = [it for it in items if it["type"] in ("statement", "wh")]
    y = [it["type"] for it in st]
    res = {"source": "RWTH-BOSTON-104 camera0 video + NCSLGR SignStream 2 annotations (decoded), matched by gloss sequence",
           "match_levels": list(match_levels), "n_utterances": len(items),
           "class_counts": dict(Counter(it["type"] for it in items)),
           "by_signer": {s: dict(Counter(it["type"] for it in items if it["signer"] == s)) for s in sorted({it["signer"] for it in items})},
           "negation_count": sum(it["neg"] for it in items)}
    # grammar layer: logistic regression on brow features (LOSO), and a single-feature threshold rule
    p_lr = loso(st, GRAM_FEATS, lambda it: it["type"])
    p_th = ["wh" if x else "statement" for x in loso_threshold(st, "brow_down_sus_max", lambda it: it["type"] == "wh")]
    for name, p in (("logreg_brows", p_lr), ("threshold_brow_down", p_th)):
        k = sum(a == b for a, b in zip(y, p))
        per_signer = {s: round(np.mean([a == b for a, b, it in zip(y, p, st) if it["signer"] == s]), 3)
                      for s in sorted({it["signer"] for it in st})}
        bal = np.mean([np.mean([a == b for a, b in zip(y, p) if a == c]) for c in ("statement", "wh")])
        res[f"sentence_type_{name}"] = {"accuracy": round(k / len(y), 3), "wilson": wilson(k, len(y)), "n": len(y),
                                        "balanced_accuracy": round(float(bal), 3), "per_signer": per_signer,
                                        "confusion": confusion(y, p, ["statement", "wh"])}
    # threshold-free separability (AUC, with bootstrap CI over utterances)
    from sklearn.metrics import roc_auc_score
    def auc(pos, neg, key):
        a = [it["f"][key] for it in pos]; b = [it["f"][key] for it in neg]
        if not a or not b:
            return None
        yv = [1] * len(a) + [0] * len(b); sv = a + b
        rng = np.random.default_rng(0); bs = []
        for _ in range(2000):
            ia = rng.integers(0, len(a), len(a)); ib = rng.integers(0, len(b), len(b))
            bs.append(roc_auc_score([1] * len(a) + [0] * len(b), [a[i] for i in ia] + [b[i] for i in ib]))
        return {"auc": round(float(roc_auc_score(yv, sv)), 3), "ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                "n_pos": len(a), "n_neg": len(b)}
    for it in items:
        it["f"]["brow_contrast"] = it["f"]["brow_down_sus_max"] - it["f"]["brow_up_sus_max"]
    stm = [it for it in items if it["type"] == "statement"]
    res["auc"] = {"wh_vs_statement_brow_down_sus_max": auc([it for it in items if it["type"] == "wh"], stm, "brow_down_sus_max"),
                  "wh_vs_statement_brow_contrast": auc([it for it in items if it["type"] == "wh"], stm, "brow_contrast"),
                  "rhq_vs_statement_brow_up_sus_max": auc([it for it in items if it["type"] == "rhq"], stm, "brow_up_sus_max"),
                  "raised_vs_lowered_questions_brow_contrast": auc([it for it in items if it["type"] == "wh"], [it for it in items if it["type"] in ("rhq", "yn")], "brow_contrast"),
                  "negation_vs_rest_hs_amp_rel": auc([it for it in items if it["neg"]], [it for it in items if not it["neg"]], "hs_amp_rel")}
    # three-way contrast rule (lowered / raised / neutral), thresholds tuned on training signers (LOSO)
    three = [it for it in items if it["type"] in ("statement", "wh", "rhq", "yn")]
    lab3 = lambda it: {"wh": "lowered", "rhq": "raised", "yn": "raised"}.get(it["type"], "neutral")
    signers = sorted({it["signer"] for it in three})
    p3 = {}
    grid = np.linspace(-1, 3, 41)
    for s_ in signers:
        tr = [it for it in three if it["signer"] != s_]
        best, bt = -1, (1.0, 1.0)
        for tl in grid:
            for tu in grid:
                pr = ["lowered" if it["f"]["brow_contrast"] >= tl else "raised" if -it["f"]["brow_contrast"] >= tu else "neutral" for it in tr]
                bal = np.mean([np.mean([p == lab3(it) for p, it in zip(pr, tr) if lab3(it) == c]) for c in ("neutral", "lowered", "raised") if any(lab3(it) == c for it in tr)])
                if bal > best:
                    best, bt = bal, (tl, tu)
        for it in three:
            if it["signer"] == s_:
                c = it["f"]["brow_contrast"]
                p3[it["name"]] = "lowered" if c >= bt[0] else "raised" if -c >= bt[1] else "neutral"
    y3 = [lab3(it) for it in three]; q3 = [p3[it["name"]] for it in three]
    k3 = sum(a == b for a, b in zip(y3, q3))
    res["brow_rule_3way"] = {"accuracy": round(k3 / len(y3), 3), "wilson": wilson(k3, len(y3)), "n": len(y3),
                             "balanced_accuracy": round(float(np.mean([np.mean([a == b for a, b in zip(y3, q3) if a == c]) for c in sorted(set(y3))])), 3),
                             "confusion": confusion(y3, q3, ["neutral", "lowered", "raised"]),
                             "note": "classes: neutral = statement, lowered = wh, raised = rhq (no yes/no questions in RWTH-104)"}
    # topic/conditional confound: raised brows also mark topics and conditionals in statements
    def grp(it):
        if it["type"] != "statement":
            return it["type"]
        return "statement+topic/cond" if ("topic" in it["nmm"] or "cond" in it["nmm"]) else "statement plain"
    res["brow_by_group_topic_confound"] = {g: {"n": sum(1 for it in items if grp(it) == g),
                                               **{k: round(float(np.mean([it["f"][k] for it in items if grp(it) == g])), 3)
                                                  for k in ("brow_up_sus_max", "brow_down_sus_max", "brow_contrast")}}
                                           for g in sorted({grp(it) for it in items})}
    res["normalization_note"] = ("signals are z-scored per signer using all of that signer's utterances without labels "
                                 "(transductive calibration; a product would calibrate on the first seconds of a conversation)")
    # negation: headshake features
    yn = [it["neg"] for it in items]
    p_neg = loso(items, NEG_FEATS, lambda it: it["neg"])
    p_neg_th = loso_threshold(items, "hs_amp_rel", lambda it: it["neg"])
    for name, p in (("logreg_headshake", p_neg), ("threshold_headshake_amp", p_neg_th)):
        tp = sum(1 for a, b in zip(yn, p) if a and b)
        fp = sum(1 for a, b in zip(yn, p) if not a and b)
        fn = sum(1 for a, b in zip(yn, p) if a and not b)
        res[f"negation_{name}"] = {"precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3),
                                   "tp": tp, "fp": fp, "fn": fn, "n_neg": sum(yn), "n": len(yn),
                                   "recall_wilson": wilson(tp, tp + fn), "precision_wilson": wilson(tp, tp + fp)}
    # brow statistics per class (z-scored per signer), incl. rhq
    res["brow_by_class"] = {c: {k: round(float(np.mean([it["f"][k] for it in items if it["type"] == c])), 3)
                                for k in ("brow_up_sus_max", "brow_down_sus_max", "brow_h_mean", "pitch_mean")}
                            for c in sorted({it["type"] for it in items})}
    # naive baseline
    nb = {}
    for it in items:
        it["naive"] = naive_baseline(it)
    for c in sorted({it["type"] for it in items}):
        maj = [it["naive"]["majority"] for it in items if it["type"] == c and it["naive"]["majority"]]
        nb[c] = {"n": len(maj), "majority_label_counts": dict(Counter(maj))}
    wh = [it for it in items if it["type"] == "wh" and it["naive"]["majority"]]
    raised = [it for it in items if it["type"] in ("yn", "rhq") and it["naive"]["majority"]]
    nb["headline"] = {
        "wh_labelled_anger_or_disgust": f"{sum(it['naive']['majority'] in ('anger', 'disgust') for it in wh)}/{len(wh)}",
        "wh_labelled_anger_disgust_or_contempt": f"{sum(it['naive']['majority'] in ('anger', 'disgust', 'contempt') for it in wh)}/{len(wh)}",
        "raised_brow_questions_labelled_surprise (rhq; no yn in RWTH)": f"{sum(it['naive']['majority'] == 'surprise' for it in raised)}/{len(raised)}",
        "statements_labelled_neutral": f"{sum(it['naive']['majority'] == 'neutral' for it in items if it['type'] == 'statement' and it['naive']['majority'])}/{sum(1 for it in items if it['type'] == 'statement' and it['naive']['majority'])}",
        "frame_share_mean_by_class": {c: {k: round(float(np.mean([it['naive']['frame_share'].get(k, 0) for it in items if it['type'] == c])), 3)
                                          for k in HSE_CLASSES} for c in sorted({it['type'] for it in items})}}
    res["naive_baseline_hsemotion"] = nb
    # our grammar layer's labels on the same items (LOSO predictions, statement/wh model) for comparison
    gl = dict(zip([it["name"] for it in st], p_lr))
    res["grammar_layer_labels"] = {c: dict(Counter(gl.get(it["name"], "n/a (not in statement/wh model)") for it in items if it["type"] == c))
                                   for c in sorted({it["type"] for it in items})}
    # affect layer distribution and independence from sentence type
    res["affect_by_type"] = {c: dict(Counter(it["affect"]["label"] for it in items if it["type"] == c))
                             for c in sorted({it["type"] for it in items})}
    res["affect_vs_type_cramers_v"] = cramers_v([it["type"] for it in items], [it["affect"]["label"] for it in items])
    res["naive_vs_type_cramers_v"] = cramers_v([it["type"] for it in items if it["naive"]["majority"]],
                                               [it["naive"]["majority"] for it in items if it["naive"]["majority"]])
    # example to_voice outputs
    ex = []
    for it in items[:12]:
        g = {"type": gl.get(it["name"], "statement"), "negation": bool(p_neg[items.index(it)])}
        a = it["affect"]
        ex.append({"name": it["name"], "translation": it["translation"], "true": it["type"], "grammar": g,
                   "affect": {k: a[k] for k in ("label", "intensity")},
                   "voice": to_voice(g, a, Kinematics(a["tempo_ratio"], a["amplitude_ratio"]), text=it["translation"])})
    res["to_voice_examples"] = ex
    res["per_utterance"] = [{"name": it["name"], "signer": it["signer"], "type": it["type"], "neg": it["neg"],
                             "orth": it["orth"], "features": {k: round(v, 4) for k, v in it["f"].items()},
                             "naive": it["naive"]["majority"], "affect": it["affect"]["label"]} for it in items]
    return res


def main():
    out = {"primary_exact": evaluate(("exact",)), "secondary_exact_plus_near": evaluate(("exact", "near"))}
    json.dump(out, open(ROOT / "results/s3_rwth_eval.json", "w"), indent=1, default=str)
    for k, r in out.items():
        print("==", k, r["n_utterances"], r["class_counts"], "neg", r["negation_count"])
        print("auc", r["auc"]); print("3way", {x: r["brow_rule_3way"][x] for x in ("accuracy", "balanced_accuracy", "wilson", "confusion")})
        for kk in ("sentence_type_logreg_brows", "sentence_type_threshold_brow_down", "negation_logreg_headshake",
                   "negation_threshold_headshake_amp"):
            print(kk, {x: r[kk][x] for x in r[kk] if x != "confusion"})
        print("brow by class", r["brow_by_class"])
        print("naive", r["naive_baseline_hsemotion"]["headline"])
        print("naive majority by class", {c: v for c, v in r["naive_baseline_hsemotion"].items() if c != "headline"})
        print("affect", r["affect_by_type"], "V", r["affect_vs_type_cramers_v"], "naive V", r["naive_vs_type_cramers_v"])


if __name__ == "__main__":
    main()
