"""P5: pooled RWTH-BOSTON-104 results over the 3 leave-one-signer-out folds (exact counts), greedy decoding,
for (a) synthetic-only, (b) + 2 RWTH signers, (c) + half of the held-out signer's sentences (tested on the
other half). Signer-bootstrap CIs over the 3 held-out signers."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import R2, jdump, model_classes, signer_bootstrap
import p4_decode as D
from p3_model import label_ids, rwth_items
from p5_eval import load, logprobs
classes = model_classes()
out = {}
for var in ("a", "b", "c"):
    rows = []
    for k in range(3):
        tag = "a_bi2" if var == "a" else f"{var}_bi_f{k}"
        net = load(tag)
        items, signers = rwth_items(k, var == "c", "test")
        for o in items:
            if o["n_vocab"] < 2:
                continue
            hyp = D.greedy(logprobs(net, o["X"]).numpy())
            ref = label_ids(o["labels"], classes)
            tp, nr, nh = D.vocab_pr(ref, hyp)
            rows.append({"signer": o["signer"], "tp": tp, "nr": nr, "nh": nh})
    tp, nr, nh = (sum(r[k] for r in rows) for k in ("tp", "nr", "nh"))
    by = {}
    for r in rows:
        by.setdefault(r["signer"], []).append(r["tp"] / max(r["nr"], 1))
    out[var] = {"recall": round(tp / nr, 4), "precision": round(tp / max(nh, 1), 4), "tp": tp, "n_ref": nr, "n_hyp": nh,
                "n_sentences": len(rows), "signers": len(by), "recall_ci_signer": signer_bootstrap(by)}
    print(var, out[var], flush=True)
jdump(out, "p5_rwth_pooled.json")
