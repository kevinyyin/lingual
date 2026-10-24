"""Live-pipeline recognition: one-sign videos through the app's own segmenter and recognizer, as the browser feeds them.

eval_recognition scores the recognizer on a clean hands-visible span; live, the segmenter decides what the recognizer
sees, and a sign cut into pieces is read piece by piece. This replays each cached landmark run (eval_recognition's
half_fps condition is the browser's ~15 frames/s) through server.segmenter.Segmenter, then ends the turn with the hands
lowered, and scores what the signer would get:
  - segments: how many pieces one sign became (1 is right)
  - turn top-1: the sign spoken without a pick, i.e. exactly one segment whose top guess is right
  - any top-k: some segment lists the sign in its top k (what the picker could offer)
Variants sweep the segmenter's pause and the recognizer's prior over curated (everyday) vs generated signs.

Test videos: ASL Citizen test signers and WLASL for the curated signs, plus ASL Citizen test videos for a random sample
of generated (vocab_extended) signs, so a prior favouring everyday signs shows what it costs the rest. Landmarks are
extracted and cached as in eval_recognition.
  uv run python -m training.eval_live [--generated 200] [--pause 90,200,300,450] [--curated-prior 1,3,10,30]
Writes data/eval/live_report.md.
"""

import argparse
import random
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from server import segmenter as seg_mod
from server.protocol import Frame
from server.segmenter import Segmenter
from server.session import Baseline, TurnEnd
from server.vocab import load_vocab
from server.vocab import Vocab
from training.eval_recognition import EVAL, _cache, _extract, citizen_test_videos, wlasl_videos
from training.landmarks import load_npz

REST_MARGIN = 0.05
TAIL_MS = 1000.0  # hands lowered after the clip: long enough for REST_MS to end the turn


def baseline(frames: list[Frame]) -> Baseline:
    """Dictionary videos have no calibration: the rest line sits just above the hips (wrists in the lap), as live."""
    hips = np.mean([[f.pose[23][1], f.pose[24][1]] for f in frames])
    return Baseline(rest_y=float(hips) - REST_MARGIN, brow={}, motion_floor=0.0)


def lowered(f: Frame, t: float) -> Frame:
    pose = [list(p) for p in f.pose]
    for i in (15, 16, 17, 18, 19, 20, 21, 22):  # wrists and hand points of the pose model, dropped into the lap
        pose[i][1] = max(pose[23][1], pose[24][1]) + 0.1
    return Frame(t=t, pose=pose, left=None, right=None, blend=f.blend, head=f.head, face=f.face)


def run_turn(frames: list[Frame]) -> TurnEnd | None:
    s = Segmenter(baseline(frames))
    end = None
    dt = float(np.median(np.diff([f.t for f in frames]))) if len(frames) > 1 else 66.0
    tail = [lowered(frames[-1], frames[-1].t + dt * (i + 1)) for i in range(int(TAIL_MS / dt) + 1)]
    for f in frames + tail:
        ev = s.push(f)
        if isinstance(ev, TurnEnd):
            end = ev
            break
    return end


class Prior:
    """Recognizer probabilities times a per-sign prior, renormalized: curated signs get `curated` x the weight."""

    def __init__(self, recognizer, vocab, curated: float):
        self.r, self.vocab, self.curated = recognizer, vocab, curated

    def ranked(self, frames: list[Frame]) -> list[tuple[str, float]]:
        scores = []
        for g, p in self.r.topk(frames, 10**6):
            s = self.vocab.get(g)
            scores.append((g, p * (self.curated if s is not None and s.curated else 1.0)))
        z = sum(p for _, p in scores) or 1.0
        return sorted(((g, p / z) for g, p in scores), key=lambda kv: -kv[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="half_fps")
    ap.add_argument("--per-gloss", type=int, default=2, help="ASL Citizen test videos per curated sign")
    ap.add_argument("--wlasl-per-gloss", type=int, default=2)
    ap.add_argument("--generated", type=int, default=200, help="random generated signs, one test video each")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pause", default="90,200,300,450", help="segmenter PAUSE_MS values to sweep")
    ap.add_argument("--curated-prior", default="1,3,10,30", help="prior weight of curated signs vs generated ones")
    a = ap.parse_args()

    from server.citizen_classifier import CitizenSignPredictor

    vocab = load_vocab()
    rec = CitizenSignPredictor(vocab, restrict_to_vocab=True)  # the live recognizer (main._recognizer)
    curated = load_vocab(extended=None)
    generated = sorted((s for s in vocab.active if not s.curated and s.gloss in rec.labels), key=lambda s: s.gloss)
    sample = Vocab(random.Random(0).sample(generated, min(a.generated, len(generated))))
    items = (citizen_test_videos(curated, a.per_gloss) + wlasl_videos(curated, a.wlasl_per_gloss)
             + citizen_test_videos(sample, 1))
    items = [it for it in items if it["gloss"] in rec.labels]
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(_extract, [(it, a.condition) for it in items if not _cache(it, a.condition).exists()]))
    clips = [(it, load_npz(_cache(it, a.condition))) for it in items if _cache(it, a.condition).exists()]
    print(f"{len(clips)} one-sign videos ({a.condition})", flush=True)

    rows = []
    default_pause = seg_mod.PAUSE_MS
    for pause in [float(x) for x in a.pause.split(",")]:
        seg_mod.PAUSE_MS = pause
        turns = [(it, run_turn(fs)) for it, fs in clips]
        for w in [float(x) for x in a.curated_prior.split(",")]:
            prior = Prior(rec, vocab, w)
            for it, turn in turns:
                segs = turn.segments if turn else []
                tops = [[g for g, _ in prior.ranked(s.context or s.frames)[:5]] for s in segs]
                curated = bool(vocab.get(it["gloss"]) and vocab.get(it["gloss"]).curated)
                rows.append({"pause": pause, "prior": w, "curated": curated, "n": len(segs),
                             "turn1": len(segs) == 1 and tops[0][0] == it["gloss"],
                             "any1": any(t[0] == it["gloss"] for t in tops),
                             "any3": any(it["gloss"] in t[:3] for t in tops),
                             "whole1": bool(segs) and prior.ranked([f for s in segs for f in (s.context or s.frames)])[0][0] == it["gloss"]})
            print(f"  pause {pause:g} prior {w:g} done", flush=True)
    seg_mod.PAUSE_MS = default_pause

    agg = defaultdict(list)
    for r in rows:
        agg[(r["pause"], r["prior"], "curated" if r["curated"] else "generated")].append(r)
    lines = ["| pause ms | curated prior | signs | n | segments (mean) | one segment | turn top-1 | any top-1 | any top-3 "
             "| whole turn top-1 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (pause, w, grp), rs in sorted(agg.items()):
        m = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        lines.append(f"| {pause:g} | {w:g} | {grp} | {len(rs)} | {np.mean([r['n'] for r in rs]):.2f} | "
                     f"{np.mean([r['n'] == 1 for r in rs]):.2f} | {m('turn1'):.3f} | {m('any1'):.3f} | {m('any3'):.3f} | {m('whole1'):.3f} |")
    report = "\n".join(lines)
    (EVAL / "live_report.md").write_text(f"# Live-pipeline recognition ({a.condition})\n\n{report}\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
