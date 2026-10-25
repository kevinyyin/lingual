"""A/B check of a proposed recognition change without touching the code: a patch file (plain Python, run in every
worker before the recogniser and session are built) monkeypatches whatever it changes; the replay is compared with
the saved baseline (eval/results.jsonl) clip by clip.

  uv run python -m training.teach_eval.quick_eval --patch path/to/patch.py [--all] [--workers 4]

Default subset: every failure among the trusted clips + 250 passing ones (seeded), so a run takes ~2 minutes. --all
replays all 995 trusted clips. Prints: translator right / lesson pass / single-sign top-1 before and after, the
clips it fixes and the clips it breaks (regressions), and replay time per clip (efficiency).
"""
import argparse
import json
import random
import time
from multiprocessing import Pool

from training.landmarks import load_npz
from training.teach_eval import replay
from training.teach_eval.clips import DATA, load_clips

RIGHT = ("right_direct", "right_picked")
_R = {}


def _init(patch: str | None):
    if patch:
        exec(compile(open(patch, encoding="utf-8").read(), patch, "exec"), {"__name__": "teach_patch"})
    from server.classifier import SignPredictor
    _R["rec"] = SignPredictor()
    _R["session"] = replay.make_session(_R["rec"])


def _one(c) -> dict:
    rec, session = _R["rec"], _R["session"]
    browser = load_npz(c.frames_path("15fps"))
    t0 = time.perf_counter()
    tr = replay.translator(browser, c.gloss, session)
    les = replay.lesson(browser, c.gloss, session)
    span = replay.isolated(browser, rec)["span"]
    return {"id": c.id, "gloss": c.gloss, "outcome": tr["outcome"], "lesson": les["pass"],
            "top1": bool(span) and span[0][0] == c.gloss, "ms": (time.perf_counter() - t0) * 1000}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    labels = {json.loads(l)["id"]: json.loads(l)["status"] for l in (DATA / "verify" / "labels.jsonl").read_text().splitlines()}
    base = {r["id"]: r for r in (json.loads(l) for l in (DATA / "eval" / "results.jsonl").read_text().splitlines() if l.strip())
            if labels.get(r["id"]) in ("verified", "unreviewed_clean") and r["in_model"]}
    fails = sorted(i for i, r in base.items() if r["translator"]["outcome"] not in RIGHT)
    passes = sorted(i for i in base if i not in fails)
    ids = set(base) if a.all else set(fails) | set(random.Random(0).sample(passes, 250))
    clips = [c for c in load_clips() if c.id in ids]
    with Pool(a.workers, initializer=_init, initargs=(a.patch,)) as pool:
        rows = {r["id"]: r for r in pool.imap_unordered(_one, clips, chunksize=4)}
    course = replay.course_signs()
    b_right = sum(base[i]["translator"]["outcome"] in RIGHT for i in rows)
    n_right = sum(r["outcome"] in RIGHT for r in rows.values())
    fixed = sorted(i for i, r in rows.items() if r["outcome"] in RIGHT and base[i]["translator"]["outcome"] not in RIGHT)
    broke = sorted(i for i, r in rows.items() if r["outcome"] not in RIGHT and base[i]["translator"]["outcome"] in RIGHT)
    les = [i for i in rows if rows[i]["gloss"] in course]
    out = {"clips": len(rows), "translator_right": [b_right, n_right], "fixed": len(fixed), "broke": len(broke),
           "lesson_pass": [sum(base[i]["lesson"]["pass"] for i in les), sum(rows[i]["lesson"] for i in les), len(les)],
           "lesson_broke": sorted(i for i in les if base[i]["lesson"]["pass"] and not rows[i]["lesson"]),
           "top1": [sum(base[i]["iso_15_span"] == 0 for i in rows), sum(r["top1"] for r in rows.values())],
           "ms_per_clip_median": sorted(r["ms"] for r in rows.values())[len(rows) // 2],
           "outcomes_after": {o: sum(r["outcome"] == o for r in rows.values()) for o in sorted({r["outcome"] for r in rows.values()})},
           "fixed_ids": fixed, "broke_ids": broke}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
