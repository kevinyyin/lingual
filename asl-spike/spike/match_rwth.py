"""Match RWTH-BOSTON-104 sentences to NCSLGR utterances (decoded SignStream 2 files)
by normalized gloss sequence: exact first, then edit distance <= 1 token. Writes
results/s3_rwth_ncslgr_match.json."""
import csv, glob, json, os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
from ss2 import parse

ROOT = os.path.join(os.path.dirname(__file__), "..")
RW = f"{ROOT}/data/rwth-boston-104/rwth-boston-104/corpus"


def norm_ncslgr(g: str) -> str | None:
    g = g.strip()
    if not g or g.startswith('"') or g.startswith("("):
        g = re.sub(r"^\([^)]*\)", "", g)
    g = re.sub(r"^fs-", "", g)
    g = re.sub(r"^#", "", g)
    g = re.sub(r"[+]+$", "", g)
    g = re.sub(r":[a-z0-9]+$", "", g)
    g = g.upper()
    if g.startswith("IX-1P"):
        return "IX-1P"
    if g.startswith("IX") or g.startswith("POSS") or g.startswith("PART"):
        return "IX" if g.startswith("IX") else g.split("-")[0]
    return re.sub(r"[^A-Z0-9-]", "", g) or None


def norm_rwth(g: str) -> str:
    g = re.sub(r"\d+$", "", g)  # GO1 -> GO (pronunciation variants)
    return g


def edit(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[len(b)]


def main():
    nc = []
    for f in sorted(glob.glob(f"{ROOT}/data/ncslgr/ssdb/*")):
        if os.path.isdir(f):
            continue
        for k, u in enumerate(parse(f)):
            seq = [x for x in (norm_ncslgr(g) for g in u.glosses) if x]
            nc.append({"file": os.path.basename(f), "idx": k, "seq": seq, "english": u.english,
                       "nmm": sorted(u.tracks), "spans": u.tracks})
    rows = []
    for split, fn in (("train", "train.sentences.pronunciations.multi.translations.csv"),
                      ("test", "test.sentences.multi.translations.csv")):
        for r in csv.DictReader(open(f"{RW}/{fn}", encoding="latin-1"), delimiter=";"):
            seq = [norm_rwth(g) for g in r["orth"].split() if not g.startswith("[")]
            best = min(nc, key=lambda u: edit(seq, u["seq"]))
            dist = edit(seq, best["seq"])
            exact = [u for u in nc if u["seq"] == seq]
            labs = {tuple(u["nmm"]) for u in exact}
            if len(labs) > 1:
                # same glosses, different non-manuals: keep the NCSLGR utterance whose English
                # translation is closest to RWTH's (RWTH translations come from NCSLGR)
                import difflib
                tr = r["translation"].strip().lower()
                exact = [max(exact, key=lambda u: difflib.SequenceMatcher(None, u["english"].lower(), tr).ratio())]
            rows.append({"name": r["name"], "split": split, "speaker": r["speaker"], "orth": " ".join(seq),
                         "translation": r["translation"].strip(), "match": "exact" if exact else ("near" if dist <= 1 else "none"),
                         "dist": dist, "ncslgr": f"{best['file']}#{best['idx']}", "ncslgr_seq": " ".join(best["seq"]),
                         "nmm": sorted(set().union(*[set(u["nmm"]) for u in exact])) if exact else (best["nmm"] if dist <= 1 else None),
                         "ambiguous_exact": len(labs) > 1})
    json.dump({"ncslgr_utterances": len(nc), "rows": rows}, open(f"{ROOT}/results/s3_rwth_ncslgr_match.json", "w"), indent=1)
    from collections import Counter
    print("ncslgr utterances", len(nc))
    print(Counter(r["match"] for r in rows))
    def cls(n):
        if n is None: return "unmatched"
        if "wh" in n: return "wh"
        if "yn" in n: return "yn"
        if "negative" in n: return "neg"
        if "rhq" in n: return "rhq"
        return "statement"
    print(Counter((cls(r["nmm"]), r["speaker"]) for r in rows))
    print("ambiguous", sum(r["ambiguous_exact"] for r in rows))


if __name__ == "__main__":
    main()
