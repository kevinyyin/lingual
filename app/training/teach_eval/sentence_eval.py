"""Short sentences from the teaching clips: 2-4 signs by the same signer, spliced with the hands kept up between signs
(each middle sign cut to its active span, an optional hold between), replayed as ONE translator turn. Only clips the
app reads right on their own are used, so every error here comes from signing them in a row.

  uv run python -m training.teach_eval.sentence_eval [--n 300] [--hold 0] [--gap still|lowered] [--patch p.py] [--out name]

Scripted signer: at a picker it picks the sentence's sign for that position if offered, else any sentence sign
offered, else "None of these" (the turn is dropped). Metrics: exact sentence, word error rate (edit distance / words),
pieces vs words, picker prompts per sentence. Per-sentence rows go to eval/sentences_<out>.jsonl.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

from training.eval_live import lowered, run_turn
from training.landmarks import load_npz
from training.teach_eval import replay
from training.teach_eval.clips import DATA, load_clips

RIGHT = ("right_direct", "right_picked")
MARGIN = 150.0  # ms kept around each middle sign's active span
_R = {}


def _init(patch):
    if patch:
        exec(compile(open(patch, encoding="utf-8").read(), patch, "exec"), {"__name__": "teach_patch"})
    from server.classifier import SignPredictor
    _R["session"] = replay.make_session(SignPredictor())


def splice(parts: list[tuple[list, float, float]], hold_ms: float, gap: str = "still") -> list:
    """parts: (frames, t0, t1) per sign. First sign keeps its lead-in, last its ending; middles are cut to their span."""
    out, t = [], 0.0
    for i, (fs, t0, t1) in enumerate(parts):
        lo = -1e9 if i == 0 else t0 - MARGIN
        hi = 1e9 if i == len(parts) - 1 else t1 + MARGIN
        seg = [f for f in fs if lo <= f.t <= hi] or fs
        dt = 1000.0 / 15
        if out and hold_ms > 0:  # hold the last pose with a hand in sight (recordings lose the hands as a sign ends,
            held = next((f for f in reversed(out) if f.left or f.right), out[-1])  # and a held lost hand ends a turn)
            for _ in range(int(hold_ms / dt)):
                t += dt
                out.append(held.model_copy(update={"t": t}) if gap == "still" else lowered(held, t))
        base = seg[0].t
        for f in seg:
            out.append(f.model_copy(update={"t": t + (f.t - base) + (dt if out else 0.0)}))
        t = out[-1].t
    return out


def _edit(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def _one(job) -> dict:
    sid, glosses, parts, hold, gap = job
    session = _R["session"]
    frames = splice([(load_npz(p), t0, t1) for p, t0, t1 in parts], hold, gap)
    ev = run_turn(frames)
    row = {"id": sid, "truth": glosses, "hold": hold}
    if ev is None or not ev.segments:
        return {**row, "outcome": "no_turn", "said": [], "pieces": 0, "picks": 0, "reads": []}
    read = [(s, t) for s, t in session._read_turn(ev) if t]
    said, picks = [], 0
    for i, (s, t) in enumerate(read):
        ask, t = session._needs_pick(t, s)
        if ask:
            offered = [g for g, _ in t[:3]]
            want = glosses[i] if i < len(glosses) and glosses[i] in offered else next((g for g in glosses if g in offered), None)
            if want is None:
                return {**row, "outcome": "none_of_these", "said": said, "pieces": len(read), "picks": picks, "reads": replay._reads(read)}
            said.append(want)
            picks += 1
        else:
            said.append(t[0][0])
    return {**row, "outcome": "exact" if said == glosses else "wrong", "said": said, "pieces": len(read), "picks": picks,
            "wer": _edit(said, glosses) / len(glosses), "reads": replay._reads(read)}


def jobs(n: int, hold: float, seed: int = 0, gap: str = "still") -> list:
    labels = {json.loads(l)["id"]: json.loads(l)["status"] for l in (DATA / "verify" / "labels.jsonl").read_text().splitlines()}
    auto = {json.loads(l)["id"]: json.loads(l)["features"] for l in (DATA / "verify" / "auto.jsonl").read_text().splitlines()}
    ok = {json.loads(l)["id"] for l in (DATA / "eval" / "results.jsonl").read_text().splitlines()
          if l.strip() and json.loads(l)["translator"]["outcome"] in RIGHT}
    by_signer = defaultdict(list)
    for c in load_clips():
        f = auto.get(c.id) or {}
        if c.id in ok and labels.get(c.id) in ("verified", "unreviewed_clean") and f.get("t_active0") is not None:
            by_signer[c.signer].append((c.gloss, str(c.frames_path("15fps")), f["t_active0"], f["t_active1"]))
    rng, out = random.Random(seed), []
    signers = sorted(s for s, v in by_signer.items() if len({g for g, *_ in v}) >= 4)
    for k in range(n):
        pool = by_signer[signers[k % len(signers)]]
        picked, seen = [], set()
        for g, p, t0, t1 in rng.sample(pool, len(pool)):
            if g not in seen:
                picked.append((g, p, t0, t1))
                seen.add(g)
            if len(picked) == 2 + k % 3:
                break
        out.append((f"s{k:04d}", [g for g, *_ in picked], [(p, t0, t1) for _, p, t0, t1 in picked], hold, gap))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--hold", type=float, default=0.0, help="ms the hands hold still between signs")
    ap.add_argument("--gap", default="still", choices=["still", "lowered"], help="between signs: hold still, or drop the hands")
    ap.add_argument("--patch", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="base")
    a = ap.parse_args()
    with Pool(a.workers, initializer=_init, initargs=(a.patch,)) as pool:
        rows = list(pool.imap_unordered(_one, jobs(a.n, a.hold, gap=a.gap), chunksize=4))
    (DATA / "eval" / f"sentences_{a.out}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    words = sum(len(r["truth"]) for r in rows)
    by_len = defaultdict(list)
    for r in rows:
        by_len[len(r["truth"])].append(r["outcome"] == "exact")
    print(json.dumps({
        "sentences": len(rows), "hold_ms": a.hold,
        "exact": round(sum(r["outcome"] == "exact" for r in rows) / len(rows), 3),
        "exact_by_length": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_len.items())},
        "wer_scored": round(sum(r.get("wer", 1.0) * len(r["truth"]) for r in rows) / words, 3),
        "outcomes": dict(Counter(r["outcome"] for r in rows)),
        "pieces_minus_words": dict(sorted(Counter(r["pieces"] - len(r["truth"]) for r in rows).items())),
        "picks_per_sentence": round(sum(r["picks"] for r in rows) / len(rows), 2)}, indent=1))


if __name__ == "__main__":
    main()
