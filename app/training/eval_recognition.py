"""Sign recognition on real online videos of varying quality, through the same MediaPipe pipeline the browser uses.

Sources: ASL Citizen *test* videos (signers no model trained on; fetched one by one from the 42.8 GB zip with range
requests) and WLASL dictionary videos (many sites, cameras and signers). Every video is run clean and under six
degradations; landmarks are cached per (video, condition), so re-runs only re-score. Each recognizer is scored on the
whole clip and on the hands-visible span (what the live segmenter hands over), top-1/3/5.

Run: uv run python -m training.eval_recognition [--per-gloss 3] [--wlasl-per-gloss 2] [--workers 10]
Writes data/eval/results.jsonl and data/eval/report.md.
"""

import argparse
import csv
import json
import re
import time
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from server.segmenter import CONTEXT_MS
from server.vocab import load_vocab
from training.landmarks import load_npz, save_npz, video_to_frames

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "data" / "eval"
CITIZEN_VIDEOS = ROOT / "data" / "asl_citizen" / "videos"
TEST_CSV = ROOT / "data" / "asl_citizen" / "splits" / "test.csv"


def _jpeg(bgr, q=12):
    import cv2
    return cv2.imdecode(cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def _lowres(bgr):
    import cv2
    h, w = bgr.shape[:2]
    s = 240 / h
    return cv2.resize(bgr, (max(1, int(w * s)), 240), interpolation=cv2.INTER_AREA) if h > 240 else bgr


def _blur(bgr):
    import cv2
    return cv2.GaussianBlur(bgr, (9, 9), 0)


def _noise(bgr):
    rng = np.random.default_rng(0)
    return np.clip(bgr.astype(np.int16) + rng.normal(0, 18, bgr.shape).astype(np.int16), 0, 255).astype(np.uint8)


def _dark(bgr):
    return (bgr.astype(np.float32) * 0.35).astype(np.uint8)


# name -> (per-frame transform, frame stride)
CONDITIONS = {"clean": (None, 1), "240p": (_lowres, 1), "jpeg_q12": (_jpeg, 1), "blur": (_blur, 1),
              "noise": (_noise, 1), "dark": (_dark, 1), "half_fps": (None, 2)}


def citizen_test_videos(vocab, per_gloss: int) -> list[dict]:
    """Up to per_gloss test videos (distinct signers) for every vocabulary sign ASL Citizen has, fetched on demand."""
    from server.citizen_classifier import ALIASES

    want = {s.asl_citizen.upper(): s.gloss for s in vocab.active}
    picked: dict[str, list] = defaultdict(list)
    with open(TEST_CSV, newline="", encoding="utf-8") as fh:
        for row in list(csv.reader(fh))[1:]:
            name = row[2].strip()
            base = re.sub(r"\d+$", "", name) or name
            g = want.get(ALIASES.get(base)) or want.get(name) or want.get(base)  # same order as to_app_gloss
            if g and len(picked[g]) < per_gloss and all(p["signer"] != row[0] for p in picked[g]):
                picked[g].append({"gloss": g, "signer": row[0], "file": row[1]})
    items = [it for v in picked.values() for it in v]
    missing = [it for it in items if not (CITIZEN_VIDEOS / it["file"]).exists()]
    if missing:
        from training.remote_zip import ASL_CITIZEN_ZIP, HttpRangeFile

        print(f"fetching {len(missing)} ASL Citizen test videos (range requests)...", flush=True)
        z = zipfile.ZipFile(HttpRangeFile(ASL_CITIZEN_ZIP))
        CITIZEN_VIDEOS.mkdir(parents=True, exist_ok=True)
        for n, it in enumerate(missing, 1):
            (CITIZEN_VIDEOS / it["file"]).write_bytes(z.read(f"ASL_Citizen/videos/{it['file']}"))
            if n % 50 == 0:
                print(f"  {n}/{len(missing)}", flush=True)
    return [{"source": "asl_citizen", "path": str(CITIZEN_VIDEOS / it["file"]), **it} for it in items]


def wlasl_videos(vocab, per_gloss: int) -> list[dict]:
    """WLASL whole-sign videos per vocabulary sign (one already fetched by clips_from_wlasl, more on demand)."""
    from training.clips_from_wlasl import INDEX, WLASL, candidates, download

    index = {e["gloss"]: e["instances"] for e in json.loads(INDEX.read_text(encoding="utf-8"))}
    out = []
    for sign in vocab.active:
        got = 0
        for inst in candidates(sign, index):
            if got >= per_gloss:
                break
            path = WLASL / "videos" / f"{sign.gloss}_{inst['video_id']}.mp4"
            if download(inst["url"], path):
                out.append({"source": "wlasl", "path": str(path), "gloss": sign.gloss, "signer": inst["source"],
                            "file": path.name})
                got += 1
    return out


def _cache(item: dict, cond: str) -> Path:
    return EVAL / "cache" / item["source"] / f"{Path(item['file']).stem}__{cond}.npz"


def _extract(job) -> tuple[str, str, bool]:
    item, cond = job
    dst = _cache(item, cond)
    if dst.exists():
        return item["file"], cond, True
    transform, stride = CONDITIONS[cond]
    try:
        frames = video_to_frames(item["path"], transform=transform, stride=stride)
    except Exception:
        frames = []
    if not frames:
        return item["file"], cond, False
    save_npz(frames, dst)
    return item["file"], cond, True


def hands_span(frames, margin_ms: float = CONTEXT_MS):
    """The hands-visible span plus the context the live segmenter adds around a sign (whole clip if no hands)."""
    idx = [i for i, f in enumerate(frames) if f.left is not None or f.right is not None]
    if not idx:
        return frames
    lo, hi = frames[idx[0]].t - margin_ms, frames[idx[-1]].t + margin_ms
    return [f for f in frames if lo <= f.t <= hi]


def recognizers(vocab) -> dict:
    out = {}
    from server.citizen_classifier import CitizenSignPredictor, DEFAULT_MODEL as CITIZEN
    from server.spike_classifier import DEFAULT_MODEL as SPIKE, SpikeSignPredictor

    if CITIZEN.exists():
        out["citizen_stgcn"] = CitizenSignPredictor(vocab)
        out["citizen_stgcn_vocab"] = CitizenSignPredictor(vocab, restrict_to_vocab=True)
    if SPIKE.exists():
        out["spike_143"] = SpikeSignPredictor(SPIKE)
    if "citizen_stgcn_vocab" in out and "spike_143" in out:
        from server.ensemble import EnsembleRecognizer

        for w in (1.0, 2.0):
            out[f"ensemble_{w:g}to1"] = EnsembleRecognizer([(out["citizen_stgcn_vocab"], w), (out["spike_143"], 1.0)], vocab)
    return out


def summarize(rows: list[dict]) -> str:
    agg: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r["in_classes"]:
            agg[(r["model"], r["span"], r["source"], r["condition"])].append(r)
    lines = ["| model | span | source | condition | n | top-1 | top-3 | top-5 | ms |", "|---|---|---|---|---|---|---|---|---|"]
    for (m, span, src, cond), rs in sorted(agg.items()):
        def acc(k):
            return np.mean([r["rank"] is not None and r["rank"] < k for r in rs])
        lines.append(f"| {m} | {span} | {src} | {cond} | {len(rs)} | {acc(1):.3f} | {acc(3):.3f} | {acc(5):.3f} | "
                     f"{np.mean([r['ms'] for r in rs]):.0f} |")
    overall = defaultdict(list)
    for r in rows:
        if r["in_classes"]:
            overall[(r["model"], r["span"])].append(r["rank"] is not None and r["rank"] < 1)
    lines += ["", "| model | span | runs | top-1 overall |", "|---|---|---|---|"]
    lines += [f"| {m} | {s} | {len(v)} | {np.mean(v):.3f} |" for (m, s), v in sorted(overall.items())]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gloss", type=int, default=3, help="ASL Citizen test videos per sign (distinct signers)")
    ap.add_argument("--wlasl-per-gloss", type=int, default=2)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--score-only", action="store_true", help="skip extraction; score cached landmarks")
    ap.add_argument("--wide", action="store_true", help="test videos for all ~2,900 signs, not just the curated 160")
    a = ap.parse_args()
    vocab = load_vocab()  # the recognizers choose among the app's whole vocabulary, as live
    tested = vocab if a.wide else load_vocab(extended=None)
    conds = [c for c in a.conditions.split(",") if c]
    items = citizen_test_videos(tested, a.per_gloss) + wlasl_videos(tested, a.wlasl_per_gloss)
    print(f"{len(items)} videos x {len(conds)} conditions = {len(items) * len(conds)} landmark runs", flush=True)
    if not a.score_only:
        jobs = [(it, c) for it in items for c in conds]
        t0, done, failed = time.time(), 0, 0
        with Pool(a.workers) as pool:
            for _, _, ok in pool.imap_unordered(_extract, jobs, chunksize=2):
                done += 1
                failed += not ok
                if done % 100 == 0:
                    print(f"  extracted {done}/{len(jobs)} ({failed} without a signer) {time.time() - t0:.0f}s", flush=True)
    models = recognizers(vocab)
    rows = []
    for it in items:
        for cond in conds:
            path = _cache(it, cond)
            if not path.exists():
                continue
            frames = load_npz(path)
            for span, fs in (("clip", frames), ("hands", hands_span(frames))):
                for name, m in models.items():
                    t = time.perf_counter()
                    top = [g for g, _ in m.topk(fs, 5)]
                    ms = (time.perf_counter() - t) * 1000
                    labels = set(m.labels)
                    rows.append({"model": name, "span": span, "source": it["source"], "condition": cond,
                                 "gloss": it["gloss"], "file": it["file"], "in_classes": it["gloss"] in labels,
                                 "rank": top.index(it["gloss"]) if it["gloss"] in top else None, "top": top, "ms": ms})
    EVAL.mkdir(parents=True, exist_ok=True)
    with open(EVAL / "results.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    report = summarize(rows)
    (EVAL / "report.md").write_text(f"# Recognition evaluation\n\n{len(rows)} scored predictions\n\n{report}\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
