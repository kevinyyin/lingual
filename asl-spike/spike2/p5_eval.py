"""P5: phrase-mode evaluation.

Sets: synthetic free sentences (test split signers, 2,000; validation 500 for tuning), synthetic script
lines (14 customer lines x test signers), RWTH-BOSTON-104 (the 75 round-1 sentences with >= 2 vocabulary
signs). Metrics: gloss WER (OOV -> <unk>), vocabulary-sign recall/precision (order-preserving, round-1
method), sentence top-1/top-3 for script lines, signer-bootstrapped CIs. Delay for the streaming model.
Usage: python p5_eval.py <model_tag> [--rwth-fold K] [--tune]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, model_classes, signer_bootstrap  # noqa: E402
import p4_decode as D  # noqa: E402
from p3_model import PhraseNet, UNK_ID, label_ids, rwth_items  # noqa: E402

HOLD_MS = 300


def load(tag):
    ck = torch.load(CACHE2 / f"p3_{tag}.pt")
    net = PhraseNet(ck["mode"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net


def logprobs(net, X):
    with torch.no_grad():
        return net(torch.tensor(np.asarray(X, np.float32))[None]).log_softmax(-1)[0]


def script_lm_and_lines(classes):
    import p2_gen as G
    lines = G.script_lines()
    ids = [label_ids(l, classes) for l in lines]
    # generator grammar: bigram from the lines (same as p2_gen) - the LM is estimated on the same lines
    return D.BigramLM(ids), ids


def summarize_sets(rows, key_signer="signer"):
    def ci(col):
        by = {}
        for r in rows:
            by.setdefault(str(r.get(key_signer, "x")), []).append(r[col])
        return signer_bootstrap(by)
    e = sum(r["err"] for r in rows)
    n = sum(r["n_ref"] for r in rows)
    tp = sum(r["tp"] for r in rows)
    nr = sum(r["nr"] for r in rows)
    nh = sum(r["nh"] for r in rows)
    for r in rows:
        r["_wer"] = r["err"] / max(r["n_ref"], 1)
        r["_rec"] = r["tp"] / max(r["nr"], 1)
    return {"wer": round(e / max(n, 1), 4), "wer_ci": ci("_wer"), "recall": round(tp / max(nr, 1), 4), "recall_ci": ci("_rec"),
            "precision": round(tp / max(nh, 1), 4), "n_sentences": len(rows), "signers": len({r.get(key_signer) for r in rows})}


def eval_free(net, data, classes, lm, decoders=("greedy", "beam")):
    out = {d: [] for d in decoders}
    ops = Counter()
    for item in data:
        lp = logprobs(net, item["X"])
        ref = label_ids(item["labels"], classes)
        for d in decoders:
            hyp = D.greedy(lp.numpy()) if d == "greedy" else D.beam_search(lp.numpy(), lm)[0][0]
            err, n = D.wer(ref, hyp)
            tp, nr, nh = D.vocab_pr(ref, hyp)
            out[d].append({"err": err, "n_ref": n, "tp": tp, "nr": nr, "nh": nh, "signer": item.get("signer", "?")})
            if d == "beam":
                for op in D.align_ops(ref, hyp):
                    ops[(op[0], classes[op[1]] if op[1] is not None and op[1] < UNK_ID else ("<unk>" if op[1] == UNK_ID else None),
                         classes[op[2]] if op[2] is not None and op[2] < UNK_ID else ("<unk>" if op[2] == UNK_ID else None))] += 1
    res = {d: summarize_sets(v) for d, v in out.items()}
    res["errors"] = {k: [{"ref": a[1], "hyp": a[2], "count": c} for a, c in ops.most_common() if a[0] == k][:15] for k in ("sub", "del", "ins")}
    return res


def eval_script(net, data, classes, lm, lines, tau):
    rows = []
    for item in data:
        lp = logprobs(net, item["X"])
        free = D.beam_search(lp.numpy(), lm)[0][0]
        order, ll, used = D.script_decode(lp, lines, free, tau)
        rows.append({"signer": item["signer"], "line": item["line"], "top1": order[0] == item["line"] and not used,
                     "top3": item["line"] in order, "fallback": used})
    by1, by3 = {}, {}
    for r in rows:
        by1.setdefault(r["signer"], []).append(r["top1"])
        by3.setdefault(r["signer"], []).append(r["top3"])
    return {"top1": round(float(np.mean([r["top1"] for r in rows])), 4), "top1_ci": signer_bootstrap(by1),
            "top3": round(float(np.mean([r["top3"] for r in rows])), 4), "top3_ci": signer_bootstrap(by3),
            "fallback_rate": round(float(np.mean([r["fallback"] for r in rows])), 4), "n": len(rows), "signers": len(by1),
            "per_line_top1": {i: round(float(np.mean([r["top1"] for r in rows if r["line"] == i])), 3) for i in sorted({r["line"] for r in rows})}}


def tune_tau(net, val_script, val_free, classes, lm, lines):
    """tau maximizing (script top-1 on script lines) + (fallback rate on free sentences that are not script lines)."""
    lps_s = [(logprobs(net, it["X"]), it["line"]) for it in val_script]
    lps_f = [logprobs(net, it["X"]) for it in val_free[:200]]
    pre_s = [(lp, D.beam_search(lp.numpy(), lm)[0][0], line) for lp, line in lps_s]
    pre_f = [(lp, D.beam_search(lp.numpy(), lm)[0][0]) for lp in lps_f]
    ll_s = [(D.ctc_loglik(lp, lines), D.ctc_loglik(lp, [fr])[0], line) for lp, fr, line in pre_s]
    ll_f = [(D.ctc_loglik(lp, lines), D.ctc_loglik(lp, [fr])[0]) for lp, fr in pre_f]
    grid = []
    for tau in (0, 1, 2, 4, 6, 8, 12, 16, 24, 1e9):
        s1 = np.mean([(np.argmax(ll) == line) and not (fl - ll.max() > tau) for ll, fl, line in ll_s])
        ff = np.mean([fl - ll.max() > tau for ll, fl in ll_f])
        grid.append({"tau": tau, "script_top1": round(float(s1), 4), "free_fallback": round(float(ff), 4)})
    # smallest tau whose script top-1 stays within 3 points of no fallback (keeps the most off-script detection)
    target = grid[-1]["script_top1"] - 0.03
    bt = min(g["tau"] for g in grid if g["script_top1"] >= target)
    return bt, grid


def delay(net, data, classes, lm, n=300):
    """streaming: frame at which the greedy prefix first equals the final hypothesis (and stays), relative to
    the last sign's end; final = max(that, end + hold) + compute of the last chunk. bi: end + hold + full compute."""
    rows = []
    fps = 30.0
    for item in data[:n]:
        X = np.asarray(item["X"], np.float32)
        end = item["ends"][-1]
        t0 = time.perf_counter_ns()
        lp = logprobs(net, X)
        comp_ms = (time.perf_counter_ns() - t0) / 1e6
        final = D.greedy(lp.numpy())
        if net.mode == "stream":
            # causal model: logits at frame t depend only on frames <= t, so prefixes are exact
            arr = lp.numpy().argmax(1)
            first_ok = None
            for t in range(end - 10, len(X) + 1):
                if t <= 0:
                    continue
                pre = D.greedy(lp.numpy()[:t])
                if pre == final:
                    first_ok = t
                    break
            model_ms = max(0.0, (first_ok - end) / fps * 1000) if first_ok is not None else None
            per_frame_ms = comp_ms / len(X)
            final_ms = None if model_ms is None else max(model_ms, HOLD_MS) + per_frame_ms * 2
        else:
            model_ms = None
            final_ms = HOLD_MS + comp_ms
        ref = label_ids(item["labels"], classes)
        rows.append({"model_ms": model_ms, "final_ms": final_ms, "compute_ms": comp_ms, "frames": len(X),
                     "correct_final": final == ref})
    def ps(xs):
        xs = [x for x in xs if x is not None]
        return {"median": round(float(np.median(xs)), 1), "p90": round(float(np.percentile(xs, 90)), 1), "max": round(float(np.max(xs)), 1), "n": len(xs)} if xs else None
    return {"final_ms": ps([r["final_ms"] for r in rows]), "model_emission_ms": ps([r["model_ms"] for r in rows]),
            "compute_ms_per_utterance": ps([r["compute_ms"] for r in rows]), "n": len(rows), "hold_ms": HOLD_MS}


def eval_rwth(net, classes, lm, fold=None, calib=False):
    """RWTH: the 75 round-1 sentences (>= 2 vocabulary signs); with fold, only the held-out signer's ones."""
    if fold is None:
        items, _ = rwth_items()
    else:
        items, _ = rwth_items(fold, calib, "test")
    items = [o for o in items if o["n_vocab"] >= 2]
    rows = []
    for o in items:
        lp = logprobs(net, o["X"])
        ref = label_ids(o["labels"], classes)
        for d in ("greedy", "beam"):
            hyp = D.greedy(lp.numpy()) if d == "greedy" else D.beam_search(lp.numpy(), lm)[0][0]
            tp, nr, nh = D.vocab_pr(ref, hyp)
            err, n = D.wer(ref, hyp)
            rows.append({"dec": d, "tp": tp, "nr": nr, "nh": nh, "err": err, "n_ref": n, "signer": o["signer"]})
    return {d: summarize_sets([r for r in rows if r["dec"] == d]) for d in ("greedy", "beam")}


def main():
    a = sys.argv
    tag = a[1]
    classes = model_classes()
    net = load(tag)
    lm, lines = script_lm_and_lines(classes)
    res = {"model": tag, "mode": net.mode}
    if "--rwth-only" not in a:
        val = pickle.load(open(CACHE2 / "p2_val.pkl", "rb"))
        test = pickle.load(open(CACHE2 / "p2_test.pkl", "rb"))
        val_s = pickle.load(open(CACHE2 / "p2_val_script.pkl", "rb"))
        test_s = pickle.load(open(CACHE2 / "p2_test_script.pkl", "rb"))

        tau, grid = tune_tau(net, val_s, val, classes, lm, lines)
        res["tau"] = tau
        res["tau_grid_val"] = grid
        res["synthetic_free"] = eval_free(net, test, classes, lm)
        res["script_lines"] = eval_script(net, test_s, classes, lm, lines, tau)
        res["script_lines_no_fallback"] = eval_script(net, test_s, classes, lm, lines, None)
        res["delay"] = delay(net, test, classes, lm)
        print(json.dumps({k: res[k] for k in ("tau",)} | {"free": {d: res["synthetic_free"][d] for d in ("greedy", "beam")},
                          "script": {k: res["script_lines"][k] for k in ("top1", "top3", "fallback_rate")}, "delay": res["delay"]}, indent=1), flush=True)
    fold = int(a[a.index("--rwth-fold") + 1]) if "--rwth-fold" in a else None
    res["rwth"] = eval_rwth(net, classes, lm, fold, "--rwth-calib" in a)
    print("rwth", json.dumps(res["rwth"]), flush=True)
    json.dump(res, open(R2 / f"p5_eval_{tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
