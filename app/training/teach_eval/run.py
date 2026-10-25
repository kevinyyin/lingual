"""Step 1 driver: automated checks, the review sample for the verifier agents, and merging their verdicts.

  uv run python -m training.teach_eval.run checks            # qc + features + ASL-LEX agreement for every clip
  uv run python -m training.teach_eval.run sample [--n 400]  # stratified sample + flagged clips -> sheets + batches
  uv run python -m training.teach_eval.run merge-verify      # agents' verdicts -> verify/labels.jsonl + summary

Review batches (verify/batches/batch_NN.json) hold items with a random id each; every clip appears in two different
batches (two independent reviewers), and about 10% of items are planted controls (a clip shown with a wrong sign's
label and that sign's reference). Which items are controls is kept in verify/key.json, never in a batch.
"""
import argparse
import json
import math
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

from server.vocab import load_vocab
from training.landmarks import load_npz
from training.teach_eval import phonology, qc
from training.teach_eval.clips import DATA, Clip, load_clips
from training.teach_eval.features import features
from training.teach_eval.sheets import SHEETS, pair_sheet, signer_box
from training.touch_from_asllex import lexicon

VERIFY = DATA / "verify"
AUTO = VERIFY / "auto.jsonl"
BATCH = 20
CONTROL_SHARE = 0.10
LABEL_FLAGS = {"not_one_sign", "few_hands", "too_short", "too_long", "small_in_frame"}  # can hide a wrong label;
# caption_band / dark / blurry are camera conditions (kept for the failure analysis), not reasons to doubt the label


def _flagged(r: dict) -> bool:
    return bool(LABEL_FLAGS & set(r["qc"]["flags"])) or (r["phonology"]["score"] is not None and r["phonology"]["score"] < 0.5)


def _auto_one(c: Clip) -> dict | None:
    if not c.frames_path("native").exists():
        return None
    meta = json.loads(c.frames_path("native").with_suffix(".json").read_text())
    f = features(load_npz(c.frames_path("native")), meta)
    q = qc.check(c)
    p = phonology.check(c.gloss, f, _LEX)
    return {"id": c.id, "source": c.source, "gloss": c.gloss, "signer": c.signer, "qc": q, "features": f, "phonology": p}


_LEX: dict = {}


def _init(lex):
    global _LEX
    _LEX = lex


def checks(workers: int) -> None:
    clips = load_clips()
    lex = lexicon(sorted({c.gloss for c in clips}))
    VERIFY.mkdir(parents=True, exist_ok=True)
    rows = []
    with Pool(workers, initializer=_init, initargs=(lex,)) as pool:
        for r in pool.imap_unordered(_auto_one, clips, chunksize=4):
            if r:
                rows.append(r)
    AUTO.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    flagged = sum(_flagged(r) for r in rows)
    print(f"{len(rows)} clips checked, {flagged} flagged -> {AUTO}")
    print("flags:", Counter(f for r in rows for f in r["qc"]["flags"]).most_common())


def _describe(gloss: str, lex: dict, vocab) -> str:
    e = lex.get(gloss) or {}
    s = vocab.get(gloss)
    bits = [f"English: {s.english}" if s else "", f"ASL-LEX: {e.get('SignType.2.0')}, location {e.get('MajorLocation.2.0')}/"
            f"{e.get('MinorLocation.2.0')}, handshape {e.get('Handshape.2.0')}, movement {e.get('Movement.2.0')}" if e else ""]
    return "; ".join(b for b in bits if b)


def sample(n: int, seed: int = 0) -> None:
    rng = random.Random(seed)
    rows = [json.loads(line) for line in AUTO.read_text(encoding="utf-8").splitlines()]
    by_id = {c.id: c for c in load_clips()}
    flagged = [r for r in rows if _flagged(r)]
    rest = [r for r in rows if r not in flagged]
    strata = defaultdict(list)  # source x sign, so every source and as many signs as possible are covered
    for r in rest:
        strata[(r["source"], r["gloss"])].append(r)
    picked = list(flagged)
    keys = list(strata)
    rng.shuffle(keys)
    while len(picked) < n and keys:
        for k in list(keys):
            if strata[k]:
                picked.append(strata[k].pop(rng.randrange(len(strata[k]))))
            else:
                keys.remove(k)
            if len(picked) >= n:
                break
    vocab, lex = load_vocab(extended=None), lexicon(sorted({r["gloss"] for r in rows}))
    glosses = sorted({r["gloss"] for r in rows})
    items, key = [], {}
    for r in picked:
        c, f = by_id[r["id"]], r["features"]
        for shown, control in [(r["gloss"], False)] + ([(rng.choice([g for g in glosses if g != r["gloss"]]), True)]
                                                         if rng.random() < CONTROL_SHARE else []):
            iid = f"{rng.getrandbits(40):010x}"
            sheet = pair_sheet(c.path, f.get("t_active0"), f.get("t_active1"), shown, "CLIP under review",
                               f"REFERENCE: {shown} (ASL Citizen training video)", SHEETS / "verify" / f"{iid}.jpg",
                               box=signer_box(load_npz(c.frames_path("native")), f.get("t_active0"), f.get("t_active1")))
            items.append({"item_id": iid, "sheet": str(sheet), "gloss": shown, "description": _describe(shown, lex, vocab)})
            key[iid] = {"clip_id": c.id, "true_gloss": r["gloss"], "shown_gloss": shown, "control": control}
    out = VERIFY / "batches"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    for rep in range(2):  # two independent reviewers: each item in two different batches, different orders
        order = items[:]
        random.Random(seed + 1 + rep).shuffle(order)
        for b in range(math.ceil(len(order) / BATCH)):
            (out / f"batch_{rep}{b:02d}.json").write_text(json.dumps(order[b * BATCH:(b + 1) * BATCH], indent=1))
    (VERIFY / "key.json").write_text(json.dumps(key, indent=1))
    print(f"{len(picked)} clips ({len(flagged)} flagged), {sum(v['control'] for v in key.values())} controls, "
          f"{len(items)} items x 2 reviewers -> {len(list(out.glob('*.json')))} batches in {out}")


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def _kappa(a: list[str], b: list[str]) -> float:
    cats = sorted(set(a) | set(b))
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def merge_verify() -> None:
    key = json.loads((VERIFY / "key.json").read_text())
    votes: dict[str, list[dict]] = defaultdict(list)
    for p in sorted((VERIFY / "results").glob("*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                votes[v["item_id"]].append(v)
    pairs_a, pairs_b, controls, labels = [], [], [], {}
    for iid, info in key.items():
        vs = sorted(votes.get(iid, []), key=lambda v: bool(v.get("adjudicated")))  # the adjudicator's verdict last
        verdicts = [v.get("matches_label", "unsure") for v in vs]
        if len(verdicts) >= 2:
            pairs_a.append(verdicts[0])
            pairs_b.append(verdicts[1])
        if info["control"]:
            controls.append(Counter(verdicts).most_common(1)[0][0] if verdicts else "missing")
            continue
        final = verdicts[2] if len(verdicts) >= 3 else (verdicts[0] if len(set(verdicts[:2])) == 1 and verdicts else "unsure")
        labels[info["clip_id"]] = {"status": {"yes": "verified", "no": "rejected"}.get(final, "uncertain"), "votes": vs}
    auto = {json.loads(line)["id"]: json.loads(line) for line in AUTO.read_text(encoding="utf-8").splitlines()}
    out = []
    for cid, r in auto.items():
        rev = labels.get(cid)
        flagged = _flagged(r)
        status = rev["status"] if rev else ("unreviewed_flagged" if flagged else "unreviewed_clean")
        out.append({"id": cid, "source": r["source"], "gloss": r["gloss"], "status": status, "reviewed": bool(rev)})
    (VERIFY / "labels.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
    summary = {"kappa": _kappa(pairs_a, pairs_b), "controls": len(controls),
               "control_catch_rate": (controls.count("no") / len(controls)) if controls else None, "by_source": {}}
    for src in sorted({o["source"] for o in out}):
        rev = [o for o in out if o["source"] == src and o["reviewed"]]
        ok = sum(o["status"] == "verified" for o in rev)
        summary["by_source"][src] = {"clips": sum(o["source"] == src for o in out), "reviewed": len(rev), "verified": ok,
                                     "rejected": sum(o["status"] == "rejected" for o in rev),
                                     "label_accuracy": ok / len(rev) if rev else None, "ci95": _wilson(ok, len(rev))}
    (VERIFY / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("checks")
    c.add_argument("--workers", type=int, default=6)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=400)
    sub.add_parser("merge-verify")
    a = ap.parse_args()
    if a.cmd == "checks":
        checks(a.workers)
    elif a.cmd == "sample":
        sample(a.n)
    else:
        merge_verify()


if __name__ == "__main__":
    main()
