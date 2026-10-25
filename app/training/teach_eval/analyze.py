"""Step 3: which motions and conditions go with wrong reads.

  uv run --with scikit-learn --with scipy python -m training.teach_eval.analyze stats
  uv run python -m training.teach_eval.analyze failures [--n 300]     # sheets + feature cards -> analyst batches
  uv run python -m training.teach_eval.analyze merge-analysts

A clip "fails" when the translator turn is not right (right straight away or after picking). Statistics:
  - each feature: failure rate per quartile (Wilson CI) and Cliff's delta (failing vs passing clips)
  - within-sign: for signs with both passing and failing clips, the mean z-scored gap (failing - passing) per feature,
    with a permutation test that shuffles pass/fail inside each sign (sign difficulty cannot explain it)
  - models: logistic regression with sign fixed effects; gradient boosting with permutation importance, cross-validated
    by signer
  - confusions: which ASL-LEX parameters differ between the true sign and the sign read (one = a minimal pair)
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict

import numpy as np

from training.landmarks import load_npz
from training.teach_eval.clips import DATA, load_clips
from training.teach_eval.run import _kappa, _wilson
from training.teach_eval.sheets import SHEETS, pair_sheet, signer_box
from training.touch_from_asllex import lexicon

OUT = DATA / "analysis"
SKIP = {"t_active0", "t_active1", "frames", "active_frames", "empty", "loc_bin"}
TAXONOMY = ["head_motion", "fingers_close_or_merged", "hand_near_face", "hand_occludes_face", "hands_overlap",
            "hand_out_of_frame", "fast_motion_or_blur", "small_in_frame", "palm_orientation", "minimal_pair",
            "variant_differs", "segmentation_cut", "handedness_or_mirrored", "label_error", "other"]
# agent tag -> (feature, direction): the measurement that should back the tag up
EVIDENCE = {"head_motion": ("head_speed_max", 1), "fingers_close_or_merged": ("finger_gap_mean", -1),
            "hand_near_face": ("near_face_frac", 1), "hand_occludes_face": ("over_face_frac", 1),
            "hands_overlap": ("hands_overlap_frac", 1), "fast_motion_or_blur": ("speed_peak", 1),
            "small_in_frame": ("face_width_px", -1), "handedness_or_mirrored": ("handedness_flip_frac", 1)}


def table() -> list[dict]:
    auto = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "verify" / "auto.jsonl").read_text().splitlines()}
    lab_path = DATA / "verify" / "labels.jsonl"  # step 1's verdicts: clips whose label is wrong or in doubt are left out
    labels = {json.loads(l)["id"]: json.loads(l)["status"] for l in lab_path.read_text().splitlines()} if lab_path.exists() else {}
    rows = []
    for p in sorted((DATA / "eval").glob("results*.jsonl")):
        for l in p.read_text().splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            a = auto.get(r["id"])
            if not a or not r["in_model"] or labels.get(r["id"]) in ("rejected", "uncertain", "unreviewed_flagged"):
                continue
            feats = {k: v for k, v in a["features"].items() if k not in SKIP and isinstance(v, (int, float)) and v == v}
            feats.update({f"qc_{k}": a["qc"][k] for k in ("blur", "brightness", "caption_band", "segments") if a["qc"].get(k) == a["qc"].get(k)})
            rows.append({"id": r["id"], "gloss": r["gloss"], "signer": r["signer"], "source": r["source"],
                         "fail": r["translator"]["outcome"] not in ("right_direct", "right_picked"),
                         "outcome": r["translator"]["outcome"], "read_as": (r["iso_15_span_top"] or [[None]])[0][0], "f": feats})
    return rows


def cliffs(a: np.ndarray, b: np.ndarray) -> float:
    if not len(a) or not len(b):
        return float("nan")
    return float((np.sign(a[:, None] - b[None, :])).mean())


def stats() -> None:
    rows = table()
    names = sorted({k for r in rows for k in r["f"]})
    fail = np.array([r["fail"] for r in rows])
    X = np.array([[r["f"].get(k, np.nan) for k in names] for r in rows], float)
    res = {"clips": len(rows), "failures": int(fail.sum()), "features": {}}
    for j, k in enumerate(names):
        x = X[:, j]
        ok = ~np.isnan(x)
        if ok.sum() < 20:
            continue
        qs = np.nanpercentile(x, [25, 50, 75])
        bins = np.digitize(x[ok], qs)
        res["features"][k] = {
            "cliffs_delta_fail_vs_pass": cliffs(x[ok & fail], x[ok & ~fail]),
            "quartiles": [{"fail_rate": float(fail[ok][bins == b].mean()) if (bins == b).any() else None,
                           "n": int((bins == b).sum()), "ci95": _wilson(int(fail[ok][bins == b].sum()), int((bins == b).sum()))}
                          for b in range(4)], "cuts": qs.tolist()}
    # within-sign gaps with a permutation test (pass/fail shuffled inside each sign)
    by_sign = defaultdict(list)
    for i, r in enumerate(rows):
        by_sign[r["gloss"]].append(i)
    mixed = [ix for ix in by_sign.values() if 0 < fail[ix].sum() < len(ix)]
    Z = (X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-9)

    def gap(lbl):
        g = []
        for ix in mixed:
            ix = np.array(ix)
            f, p = ix[lbl[ix]], ix[~lbl[ix]]
            g.append(np.nanmean(Z[f], 0) - np.nanmean(Z[p], 0))
        return np.nanmean(np.array(g), 0)

    obs, rng, null = gap(fail), np.random.default_rng(0), []
    for _ in range(500):
        perm = fail.copy()
        for ix in mixed:
            perm[ix] = rng.permutation(perm[ix])
        null.append(gap(perm))
    null = np.array(null)
    p = (np.sum(np.abs(null) >= np.abs(obs), 0) + 1) / (len(null) + 1)
    res["within_sign"] = {"signs": len(mixed), "gaps": sorted(({"feature": names[j], "gap_sd": float(obs[j]), "p": float(p[j])}
                                                             for j in range(len(names)) if obs[j] == obs[j]), key=lambda d: d["p"])}
    try:  # models (scikit-learn is not an app dependency: run with uv --with scikit-learn)
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold

        Xf = np.where(np.isnan(Z), 0.0, Z)
        signs = sorted(by_sign)
        S = np.zeros((len(rows), len(signs)))
        for i, r in enumerate(rows):
            S[i, signs.index(r["gloss"])] = 1
        lr = LogisticRegression(C=0.5, max_iter=2000).fit(np.hstack([Xf, S]), fail)
        res["logistic_sign_fixed_effects"] = sorted(({"feature": names[j], "coef_per_sd": float(lr.coef_[0][j])} for j in range(len(names))),
                                                    key=lambda d: -abs(d["coef_per_sd"]))[:25]
        groups = [r["signer"] or r["id"] for r in rows]
        aucs, imps = [], np.zeros(len(names))
        for tr, te in GroupKFold(n_splits=5).split(X, fail, groups):
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300).fit(X[tr], fail[tr])
            if 0 < fail[te].sum() < len(te):
                aucs.append(roc_auc_score(fail[te], m.predict_proba(X[te])[:, 1]))
                imps += permutation_importance(m, X[te], fail[te], scoring="roc_auc", n_repeats=5, random_state=0).importances_mean
        res["boosting"] = {"auc_by_signer_cv": float(np.mean(aucs)) if aucs else None,
                           "importance": sorted(({"feature": names[j], "auc_drop": float(imps[j] / 5)} for j in range(len(names))),
                                                key=lambda d: -d["auc_drop"])[:20]}
    except ImportError:
        res["models"] = "scikit-learn not installed: run with uv --with scikit-learn"
    # confusions against phonology
    lex = lexicon(sorted({r["gloss"] for r in rows} | {r["read_as"] for r in rows if r["read_as"]}))
    params = ("SignType.2.0", "MajorLocation.2.0", "Handshape.2.0", "SelectedFingers.2.0", "Flexion.2.0", "Movement.2.0")
    diffs, n_diff = Counter(), Counter()
    for r in rows:
        if r["read_as"] and r["read_as"] != r["gloss"] and r["gloss"] in lex and r["read_as"] in lex:
            d = [p.split(".")[0] for p in params if lex[r["gloss"]].get(p) != lex[r["read_as"]].get(p)]
            diffs.update(d)
            n_diff[len(d)] += 1
    total = sum(n_diff.values())
    res["confusion_phonology"] = {"wrong_reads_with_both_in_asllex": total,
                                  "parameters_that_differ": {k: v / total for k, v in diffs.most_common()} if total else {},
                                  "number_of_parameters_differing": {k: v / total for k, v in sorted(n_diff.items())} if total else {}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"clips": res["clips"], "failures": res["failures"], "top_within_sign": res["within_sign"]["gaps"][:10],
                      "boosting": res.get("boosting", {}).get("auc_by_signer_cv"), "confusion_phonology": res["confusion_phonology"]}, indent=1))


def failures(n: int, seed: int = 0) -> None:
    """A stratified sample of failures (by sign and outcome) with sheets and feature cards for the analyst agents."""
    rows, rng = table(), random.Random(seed)
    by_id = {c.id: c for c in load_clips()}
    names = sorted({k for r in rows for k in r["f"]})
    passing = defaultdict(list)
    for r in rows:
        if not r["fail"]:
            passing[r["gloss"]].append(r)
    allpass = [r for r in rows if not r["fail"]]
    strata = defaultdict(list)
    for r in rows:
        if r["fail"]:
            strata[(r["gloss"], r["outcome"])].append(r)
    keys, picked = list(strata), []
    rng.shuffle(keys)
    while len(picked) < n and any(strata.values()):
        for k in keys:
            if strata[k] and len(picked) < n:
                picked.append(strata[k].pop(rng.randrange(len(strata[k]))))
    auto = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "verify" / "auto.jsonl").read_text().splitlines()}
    items = []
    for r in picked:
        ref = passing[r["gloss"]] if len(passing[r["gloss"]]) >= 3 else allpass  # z-scores against passing clips of the sign
        card = {}
        for k in names:
            vals = np.array([x["f"][k] for x in ref if k in x["f"]], float)
            if k in r["f"] and len(vals) >= 3:
                card[k] = round(float((r["f"][k] - vals.mean()) / (vals.std() + 1e-9)), 2)
        c, f = by_id[r["id"]], auto[r["id"]]["features"]
        sheet = pair_sheet(c.path, f.get("t_active0"), f.get("t_active1"), r["read_as"] or r["gloss"],
                           f"CLIP: signed {r['gloss']}, app outcome {r['outcome']}", f"REFERENCE: {r['read_as']} (what the app read)",
                           SHEETS / "failures" / f"{c.key}.jpg",
                           box=signer_box(load_npz(c.frames_path("native")), f.get("t_active0"), f.get("t_active1")))
        items.append({"item_id": c.key, "sheet": str(sheet), "true_gloss": r["gloss"], "read_as": r["read_as"], "outcome": r["outcome"],
                      "feature_card_z": dict(sorted(card.items(), key=lambda kv: -abs(kv[1]))[:14]), "features": {k: round(v, 3) for k, v in r["f"].items()}})
    out = OUT / "batches"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for rep in range(2):
        order = items[:]
        random.Random(seed + 7 + rep).shuffle(order)
        for b in range(math.ceil(len(order) / 15)):
            (out / f"batch_{rep}{b:02d}.json").write_text(json.dumps(order[b * 15:(b + 1) * 15], indent=1))
    print(f"{len(items)} failures x 2 analysts -> {len(list(out.glob('*.json')))} batches in {out}")


def merge_analysts() -> None:
    votes = defaultdict(list)
    for p in sorted((OUT / "results").glob("*.jsonl")):
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                v = json.loads(l)
                votes[v["item_id"]].append(v)
    feats = {r["id"].replace(":", "_").replace("/", "_"): r["f"] for r in table()}
    tags, agree, evidence = Counter(), {}, {}
    for t in TAXONOMY:
        a = [int(t in vs[0].get("tags", [])) for vs in votes.values() if len(vs) >= 2]
        b = [int(t in vs[1].get("tags", [])) for vs in votes.values() if len(vs) >= 2]
        agree[t] = _kappa([str(x) for x in a], [str(x) for x in b]) if a else None
    final = {}
    for iid, vs in votes.items():
        if len(vs) >= 3:  # the lead analyst's adjudication
            final[iid] = set(vs[2].get("tags", []))
        else:
            final[iid] = set.intersection(*[set(v.get("tags", [])) for v in vs]) if len(vs) >= 2 else set(vs[0].get("tags", []))
        tags.update(final[iid])
    for t, (k, sign) in EVIDENCE.items():
        with_t = [feats[i][k] for i, ts in final.items() if t in ts and i in feats and k in feats[i]]
        without = [feats[i][k] for i, ts in final.items() if t not in ts and i in feats and k in feats[i]]
        if with_t and without:
            evidence[t] = {"feature": k, "cliffs_delta_tagged_vs_not": sign * cliffs(np.array(with_t), np.array(without)), "n_tagged": len(with_t)}
    res = {"failures_reviewed": len(final), "tag_share": {t: n / max(len(final), 1) for t, n in tags.most_common()},
           "kappa_per_tag": agree, "tags_backed_by_features": evidence,
           "disagreements_for_lead": [i for i, vs in votes.items() if len(vs) == 2 and set(vs[0].get("tags", [])) != set(vs[1].get("tags", []))]}
    (OUT / "analysts.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "disagreements_for_lead"}, indent=1),
          f"\n{len(res['disagreements_for_lead'])} items need the lead analyst")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    f = sub.add_parser("failures")
    f.add_argument("--n", type=int, default=300)
    sub.add_parser("merge-analysts")
    a = ap.parse_args()
    {"stats": stats, "failures": lambda: failures(a.n), "merge-analysts": merge_analysts}[a.cmd]()


if __name__ == "__main__":
    main()
