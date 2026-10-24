"""S1: match the candidate café vocabulary against ASL Citizen, WLASL and the Kaggle asl-signs 250-sign
list; write vocab.json (gloss, dataset ids, substitute, notes) and results/s1_coverage.json.

Matching: names are normalized (uppercase, non-alphanumerics removed); an alias matches a dataset gloss
when the normalized forms are equal, or equal after dropping a trailing variant digit (e.g. ASL Citizen
"COFFEE1"). Every match is a candidate only: the same English word can name different signs.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from vocab_src import IN_DIALOGUE, V  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", s.upper())


def index(names):
    idx = defaultdict(list)
    for n in names:
        k = norm(n)
        idx[k].append(n)
        k2 = re.sub(r"\d+$", "", k)
        if k2 and k2 != k:
            idx[k2].append(n)
    return idx


def citizen_glosses() -> dict[str, int]:
    """gloss -> number of videos (all splits)."""
    sp = ROOT / "data/asl-citizen/ASL_Citizen/splits"
    counts: dict[str, int] = defaultdict(int)
    for f in ("train.csv", "val.csv", "test.csv"):
        if (sp / f).exists():
            for g in pd.read_csv(sp / f)["Gloss"]:
                counts[str(g)] += 1
    return dict(counts)


def main() -> None:
    cit = citizen_glosses()
    wl = [e["gloss"] for e in json.load(open(ROOT / "data/wlasl/WLASL_v0.3.json"))]
    kg = list(json.load(open(ROOT / "data/kaggle-asl-signs/sign_to_prediction_index_map.json")))
    ic, iw, ik = index(cit), index(wl), index(kg)
    out = []
    for gloss, (cat, aliases, sub) in V.items():
        def find(ix):
            hits = []
            for a in [gloss] + aliases:
                for h in ix.get(norm(a), []):
                    if h not in hits:
                        hits.append(h)
            return hits
        c, w, k = find(ic), find(iw), find(ik)
        # an alias that is a different sign (e.g. COST for HOW-MUCH) is noted as a substitute, not a match
        exact = [h for h in c if re.sub(r"\d+$", "", norm(h)) == norm(gloss) or norm(h) in [norm(a) for a in aliases[:1]]]
        out.append({"gloss": gloss, "category": cat, "in_dialogue": gloss in IN_DIALOGUE,
                    "asl_citizen": c, "asl_citizen_videos": sum(cit.get(h, 0) for h in c), "asl_citizen_exact": exact,
                    "wlasl": w, "kaggle_asl_signs": k,
                    "substitute": None if c else (sub or f"FS-{re.sub('[^A-Z]', '', gloss)}"),
                    "notes": ""})
    # ownership: one ASL Citizen gloss trains exactly one class (the entry where it is an exact name match first)
    owner = {}
    for e in out:
        for g in e["asl_citizen_exact"]:
            owner.setdefault(g, e["gloss"])
    for e in out:
        for g in e["asl_citizen"]:
            owner.setdefault(g, e["gloss"])
    for e in out:
        mine = [g for g in e["asl_citizen"] if owner[g] == e["gloss"]]
        others = sorted({owner[g] for g in e["asl_citizen"] if owner[g] != e["gloss"]})
        e["asl_citizen_class_videos"] = mine
        e["recognized_as"] = e["gloss"] if mine else (others[0] if others else None)
        if not mine and others:
            e["substitute"] = others[0]
            e["notes"] = f"shares its ASL Citizen sign with {', '.join(others)}; recognized as that class"
    for e in out:
        if e["asl_citizen"] and not e["asl_citizen_exact"] and not e["notes"]:
            e["notes"] = f"matched through an alias ({', '.join(e['asl_citizen'])}); confirm it is the same sign"
        if e["gloss"] in ("IX-1P", "YOU", "MY", "YOUR"):
            e["notes"] = (e["notes"] + "; " if e["notes"] else "") + "pointing/indexing sign; in continuous signing often a point, not a citation form"
        if e["category"] == "number":
            e["notes"] = (e["notes"] + "; " if e["notes"] else "") + "number handshapes vary (e.g. 6-9, 16-19 use twisting movements); check citation form"
    cov = {"n_candidates": len(out), "asl_citizen": sum(1 for e in out if e["asl_citizen"]),
           "asl_citizen_exact": sum(1 for e in out if e["asl_citizen_exact"]),
           "wlasl": sum(1 for e in out if e["wlasl"]), "kaggle_asl_signs": sum(1 for e in out if e["kaggle_asl_signs"]),
           "dialogue_glosses": len(IN_DIALOGUE),
           "dialogue_in_asl_citizen": sum(1 for e in out if e["in_dialogue"] and e["asl_citizen"]),
           "asl_citizen_total_glosses": len(cit),
           "model_classes": sum(1 for e in out if e["asl_citizen_class_videos"]), "missing_in_asl_citizen": [e["gloss"] for e in out if not e["asl_citizen"]],
           "in_citizen_and_kaggle": [e["gloss"] for e in out if e["asl_citizen_class_videos"] and e["kaggle_asl_signs"]]}
    json.dump(out, open(ROOT / "vocab.json", "w"), indent=1)
    (ROOT / "results").mkdir(exist_ok=True)
    json.dump(cov, open(ROOT / "results/s1_coverage.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in cov.items() if not isinstance(v, list)}, indent=0))
    print("missing:", cov["missing_in_asl_citizen"])
    print("overlap with kaggle:", len(cov["in_citizen_and_kaggle"]), cov["in_citizen_and_kaggle"])


if __name__ == "__main__":
    main()
