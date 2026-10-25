"""Step 2: how the app reads the verified clips, three ways (replay.py), with signer-bootstrap confidence intervals.

The recognizer is the one the app runs: server.classifier.SignPredictor on models/signs.pt (main._recognizer picks it
when the file is there). Clips whose gloss the model has no class for are scored but reported apart.

  uv run python -m training.teach_eval.evaluate [--status verified] [--workers 6] [--shard SOURCE]
Writes eval/results[_<shard>].jsonl and eval/summary[_<shard>].json.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from multiprocessing import Pool

import numpy as np

from server.classifier import SignPredictor
from training.landmarks import load_npz
from training.teach_eval import replay
from training.teach_eval.clips import DATA, load_clips

EVAL = DATA / "eval"
_R = {}


def _init():
    _R["rec"] = SignPredictor()
    _R["session"] = replay.make_session(_R["rec"])


def _one(c) -> dict:
    rec, session = _R["rec"], _R["session"]
    native, browser = load_npz(c.frames_path("native")), load_npz(c.frames_path("15fps"))
    iso_n, iso_b = replay.isolated(native, rec), replay.isolated(browser, rec)
    rank = lambda top: next((i for i, (g, _) in enumerate(top) if g == c.gloss), None)  # noqa: E731
    return {"id": c.id, "source": c.source, "gloss": c.gloss, "signer": c.signer, "in_model": c.gloss in rec.labels,
            "iso_native_whole": rank(iso_n["whole"]), "iso_native_span": rank(iso_n["span"]),
            "iso_15_whole": rank(iso_b["whole"]), "iso_15_span": rank(iso_b["span"]),
            "iso_15_span_top": iso_b["span"], "translator": replay.translator(browser, c.gloss, session),
            "lesson": replay.lesson(browser, c.gloss, session)}


def _boot(rows: list[dict], stat, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% interval of stat over signers resampled with replacement (clips of one signer move together)."""
    by = defaultdict(list)
    for r in rows:
        by[r["signer"] or r["id"]].append(r)
    keys, rng, vals = list(by), random.Random(seed), []
    for _ in range(n):
        sample = [r for k in (rng.choice(keys) for _ in keys) for r in by[k]]
        vals.append(stat(sample))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def summarize(rows: list[dict]) -> dict:
    rows = [r for r in rows if r["in_model"]]
    top = lambda k, key: (lambda rs: float(np.mean([r[key] is not None and r[key] < k for r in rs])) if rs else float("nan"))  # noqa: E731
    out = {"clips": len(rows), "signers": len({r["signer"] for r in rows}), "signs": len({r["gloss"] for r in rows})}
    for key in ("iso_native_span", "iso_15_span", "iso_15_whole"):
        for k in (1, 3, 5):
            f = top(k, key)
            out[f"{key}_top{k}"] = {"value": f(rows), "ci95": _boot(rows, f)}
    outcomes = Counter(r["translator"]["outcome"] for r in rows)
    out["translator_outcomes"] = {k: v / len(rows) for k, v in outcomes.most_common()} if rows else {}
    right = lambda rs: float(np.mean([r["translator"]["outcome"] in ("right_direct", "right_picked") for r in rs]))  # noqa: E731
    out["translator_right"] = {"value": right(rows), "ci95": _boot(rows, right)} if rows else {}
    course = replay.course_signs()  # a lesson only ever asks for course signs: the rest are not lesson attempts
    lesson = lambda rs: float(np.mean([r["lesson"]["pass"] for r in rs if r["gloss"] in course])) if any(r["gloss"] in course for r in rs) else float("nan")  # noqa: E731
    in_course = [r for r in rows if r["gloss"] in course]
    out["lesson_pass"] = {"value": lesson(in_course), "ci95": _boot(in_course, lesson), "clips": len(in_course),
                          "signs": len({r["gloss"] for r in in_course})} if in_course else {}
    for group in ("gloss", "source", "signer"):
        g = defaultdict(list)
        for r in rows:
            g[r[group]].append(r)
        out[f"by_{group}"] = {k: {"n": len(v), "translator_right": right(v), "lesson_pass": lesson(v),
                                  "iso_15_span_top1": top(1, "iso_15_span")(v)} for k, v in sorted(g.items())}
    conf = Counter((r["gloss"], r["iso_15_span_top"][0][0]) for r in rows if r["iso_15_span_top"] and r["iso_15_span_top"][0][0] != r["gloss"])
    out["top_confusions"] = [{"true": a, "read_as": b, "n": n} for (a, b), n in conf.most_common(40)]
    bins = defaultdict(list)  # calibration of the first read's top score against being right
    for r in rows:
        reads = r["translator"].get("reads") or []
        if reads:
            p, g = reads[0]["top"][0][1], reads[0]["top"][0][0]
            bins[min(9, int(p * 10))].append(g == r["gloss"])
    out["calibration"] = {f"{b / 10:.1f}-{(b + 1) / 10:.1f}": {"n": len(v), "right": float(np.mean(v))} for b, v in sorted(bins.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default="verified", help="label statuses to include (comma list), or 'all'")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--shard", default="", help="only this source")
    ap.add_argument("--rescore", action="store_true", help="re-summarise saved eval/results.jsonl for --status, no replay")
    a = ap.parse_args()
    if a.rescore:  # e.g. after step 1's verdicts arrive: the same reads, only the labelled-right clips
        keep = set(a.status.split(","))
        labels = {json.loads(l)["id"]: json.loads(l)["status"] for l in (DATA / "verify" / "labels.jsonl").read_text().splitlines()}
        rows = [json.loads(l) for l in (EVAL / "results.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in rows if (a.status == "all" or labels.get(r["id"]) in keep) and (not a.shard or r["source"] == a.shard)]
        s = summarize(rows)
        tag = "_" + a.status.replace(",", "+") + (f"_{a.shard}" if a.shard else "")
        (EVAL / f"summary{tag}.json").write_text(json.dumps(s, indent=1))
        print(json.dumps({k: v for k, v in s.items() if not k.startswith("by_") and k != "top_confusions"}, indent=1))
        return
    clips = [c for c in load_clips() if c.frames_path("15fps").exists() and (not a.shard or c.source == a.shard)]
    if a.status != "all":
        labels = {json.loads(l)["id"]: json.loads(l)["status"] for l in (DATA / "verify" / "labels.jsonl").read_text().splitlines()}
        keep = set(a.status.split(","))
        clips = [c for c in clips if labels.get(c.id) in keep]
    EVAL.mkdir(parents=True, exist_ok=True)
    with Pool(a.workers, initializer=_init) as pool:
        rows = list(pool.imap_unordered(_one, clips, chunksize=4))
    tag = f"_{a.shard.replace(':', '_')}" if a.shard else ""
    (EVAL / f"results{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    s = summarize(rows)
    (EVAL / f"summary{tag}.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({k: v for k, v in s.items() if not k.startswith("by_") and k != "top_confusions"}, indent=1))


if __name__ == "__main__":
    main()
