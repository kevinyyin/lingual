"""Build the clip list from allowed sources and extract landmarks, as the browser sees them.

Each clip's landmarks come from training.landmarks.video_to_frames (the browser's MediaPipe models), twice: at the
video's own rate and thinned to ~15 frames/s (the rate the page sends). Saved as npz plus a small meta json
(fps, size, frame count). Resumable: done clips are skipped.

  uv run python -m training.teach_eval.ingest wlasl-local --from ../lingual   # the WLASL clips already on disk
  uv run python -m training.teach_eval.ingest extract [--workers 6]
"""
import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

from server.vocab import load_vocab
from training.clips_from_wlasl import candidates
from training.landmarks import save_npz, video_to_frames
from training.teach_eval.clips import CLIPS, DATA, Clip, allowed_sources, load_clips, save_clips

BROWSER_FPS = 15.0


def wlasl_local(old: Path) -> list[Clip]:
    """WLASL clips already downloaded by the old checkout, for the 160 curated signs, from WLASL sites the compliance
    review allowed (source id "wlasl:<site>"); never YouTube. Every take, not only the first."""
    allow = allowed_sources()
    index_path = old / "app" / "data" / "wlasl" / "WLASL_v0.3.json"
    index = {e["gloss"]: e["instances"] for e in json.loads(index_path.read_text(encoding="utf-8"))}
    dirs = [old / "app" / "data" / "wlasl" / "videos", old / "asl-spike" / "data" / "wlasl" / "raw"]
    clips = []
    for sign in load_vocab(extended=None).active:
        for inst in candidates(sign, index):  # whole-video instances from direct-file sites, best source first
            sid = f"wlasl:{inst['source']}"
            if sid not in allow or "youtu" in inst["url"]:
                continue
            path = next((p for p in (dirs[0] / f"{sign.gloss}_{inst['video_id']}.mp4", dirs[1] / f"{inst['video_id']}.mp4")
                         if p.exists() and p.stat().st_size > 10_000), None)
            if path is None:
                continue
            src = allow[sid]
            clips.append(Clip(id=f"{sid}:{inst['video_id']}", source=sid, gloss=sign.gloss, path=str(path),
                              signer=f"wlasl{inst.get('signer_id', '')}", url=inst["url"], licence=src.get("licence", "C-UDA"),
                              attribution=src.get("attribution", "WLASL (Li et al. 2020)"),
                              extra={"variation_id": inst.get("variation_id"), "wlasl_gloss": inst.get("gloss")}))
    return clips


def citizen_test(old: Path, per_sign: int, seed: int = 0) -> list[Clip]:
    """ASL Citizen's test split (Deaf signers the shipped model never trained on), for the model's signs: up to
    per_sign videos per sign, from as many different signers as possible, taken out of the local dataset archive."""
    import csv
    import random
    import re
    import zipfile

    from server.classifier import SignPredictor

    labels = set(SignPredictor().labels)
    vocab = load_vocab(extended=None)
    key = lambda g: re.sub(r"[^A-Z]", "", re.sub(r"\d+$", "", g.upper()))  # noqa: E731  (WANT1/WANT2 -> WANT)
    to_app = {key(s.asl_citizen): s.gloss for s in vocab.active if s.gloss in labels}
    rows = list(csv.reader(open(Path(__file__).resolve().parents[2] / "data" / "asl_citizen" / "splits" / "test.csv", encoding="utf-8")))[1:]
    by_sign: dict[str, list] = {}
    for r in rows:
        g = to_app.get(key(r[2]))
        if g:
            by_sign.setdefault(g, []).append(r)
    rng, out = random.Random(seed), []
    dst_dir = DATA / "videos" / "asl_citizen_test"
    dst_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(old / "asl-spike" / "data" / "asl-citizen" / "ASL_Citizen.zip") as z:
        for g, rs in sorted(by_sign.items()):
            rng.shuffle(rs)
            seen, picked = set(), []
            for r in rs:  # one per signer first, then more if the sign has few signers
                if r[0] not in seen:
                    seen.add(r[0])
                    picked.append(r)
            picked += [r for r in rs if r not in picked]
            for r in picked[:per_sign]:
                dst = dst_dir / r[1]
                if not dst.exists():
                    dst.write_bytes(z.read(f"ASL_Citizen/videos/{r[1]}"))
                out.append(Clip(id=f"asl_citizen_test:{Path(r[1]).stem}", source="asl_citizen_test", gloss=g, path=str(dst),
                                signer=f"citizen_{r[0]}", licence="ASL Citizen dataset licence (non-commercial research)",
                                attribution="ASL Citizen (Desai et al. 2023, Microsoft Research)", extra={"asl_citizen_gloss": r[2]}))
    return out


def _meta(path: Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(path))
    m = {"fps": cap.get(cv2.CAP_PROP_FPS) or 30.0, "w": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         "h": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "n": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return m


def _clip_range(frames, c: Clip):
    if c.start_ms is None and c.end_ms is None:
        return frames
    lo, hi = c.start_ms or 0.0, c.end_ms if c.end_ms is not None else float("inf")
    return [f for f in frames if lo <= f.t <= hi]


def extract_one(c: Clip) -> tuple[str, bool, str]:
    native, browser = c.frames_path("native"), c.frames_path("15fps")
    if native.exists() and browser.exists():
        return c.id, True, "cached"
    try:
        meta = _meta(Path(c.path))
        frames = _clip_range(video_to_frames(c.path), c)
        if not frames:
            return c.id, False, "no pose found"
        step = max(1, round(meta["fps"] / BROWSER_FPS))
        for p, fs in ((native, frames), (browser, frames[::step])):
            p.parent.mkdir(parents=True, exist_ok=True)
            save_npz(fs, p)
        native.with_suffix(".json").write_text(json.dumps({**meta, "frames": len(frames), "browser_step": step}))
        return c.id, True, "ok"
    except Exception as e:  # a broken file is reported, not fatal
        return c.id, False, f"{type(e).__name__}: {e}"[:200]


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wlasl-local")
    w.add_argument("--from", dest="old", default=str(Path(__file__).resolve().parents[4] / "lingual"))
    ct = sub.add_parser("citizen-test")
    ct.add_argument("--from", dest="old", default=str(Path(__file__).resolve().parents[4] / "lingual"))
    ct.add_argument("--per-sign", type=int, default=6)
    x = sub.add_parser("extract")
    x.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "wlasl-local":
        new = wlasl_local(Path(a.old))
        clips = load_clips() + new
        save_clips(clips)
        by = {}
        for c in new:
            by[c.source] = by.get(c.source, 0) + 1
        print(f"{len(new)} WLASL clips from allowed sites ({len({c.gloss for c in new})} signs): {by} -> {CLIPS}")
    elif a.cmd == "citizen-test":
        new = citizen_test(Path(a.old), a.per_sign)
        save_clips(load_clips() + new)
        print(f"{len(new)} ASL Citizen test clips, {len({c.gloss for c in new})} signs, {len({c.signer for c in new})} signers -> {CLIPS}")
    else:
        clips = load_clips()
        t0, done, bad = time.time(), 0, []
        with Pool(a.workers) as pool:
            for cid, ok, why in pool.imap_unordered(extract_one, clips, chunksize=2):
                done += 1
                if not ok:
                    bad.append((cid, why))
                if done % 50 == 0:
                    print(f"  {done}/{len(clips)} ({len(bad)} failed) {time.time() - t0:.0f}s", flush=True)
        (DATA / "extract_failures.json").write_text(json.dumps(bad, indent=1))
        print(f"extracted {done - len(bad)}/{len(clips)}; failures in {DATA / 'extract_failures.json'}")


if __name__ == "__main__":
    main()
