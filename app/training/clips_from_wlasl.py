"""Avatar sign clips from public WLASL dictionary videos (no ASL Citizen download needed).

WLASL (github.com/dxli94/WLASL, C-UDA licence: computational/research use) indexes ~21k sign videos from ASL
teaching sites. For each vocabulary sign this downloads up to TAKES whole-video instances from direct-file sources, runs
MediaPipe on each (video_to_frames; landmarks cached in data/wlasl/raw/), builds each take's
clip with clip_json and keeps the best take (best_take): hands tracked through the sign, then the take closest to the
others (the most typical production), else the steadiest. Writes web/public/signs/{GLOSS}.json. Resumable: existing
clips are skipped. Videos stay in data/wlasl/ (gitignored).

Run: uv run python -m training.clips_from_wlasl [--only HELLO,COFFEE] [--force]
"""

import argparse
import gzip
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from pathlib import Path

from server.vocab import load_vocab
import numpy as np

from server.protocol import Frame
from training.build_motion_library import PUBLIC, by_wrist, clip_json, trim_to_sign
from training.hand_cleanup import palm_frame
from training.landmarks import video_to_frames

ROOT = Path(__file__).resolve().parent.parent
WLASL = ROOT / "data" / "wlasl"
INDEX = WLASL / "WLASL_v0.3.json"
INDEX_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
# Direct-file sources that downloaded 100% in the asl-spike round 2 (REPORT-SPIKE-2.md); YouTube and the sites that
# refuse scripts (signingsavvy, handspeak, aslpro Flash, aslsearch, aslsignbank) are skipped.
SOURCES = ["signschool", "aslu", "spreadthesign", "asldeafined", "startasl", "aslbrick", "elementalasl", "scott",
           "valencia-asl", "northtexas", "asl5200"]
MIN_FRAMES = 8
RAW = WLASL / "raw"
TAKES = 4  # usable takes compared per sign
TRIES = 8  # instances tried to get them (dead links are common)


def candidates(sign, index: dict) -> list[dict]:
    """Whole-video instances (one sign per file) for a vocabulary sign, best source first."""
    keys = [sign.gloss.lower().replace("-", " "), sign.english.lower(), *sign.english.lower().split("/")]
    gloss = next((k for k in keys if k in index), None)
    rank = {s: i for i, s in enumerate(SOURCES)}
    out = [i for i in index.get(gloss, []) if i["source"] in rank and i["frame_start"] <= 1 and i["frame_end"] == -1]
    return sorted(out, key=lambda i: rank[i["source"]])


def is_video(head: bytes) -> bool:
    """MP4/MOV (ftyp box), WebM/MKV (EBML), FLV or AVI by their first bytes."""
    return head[4:8] == b"ftyp" or head[:4] == b"\x1a\x45\xdf\xa3" or head[:3] == b"FLV" or head[:4] == b"RIFF"


def download(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 10_000:
        with open(dst, "rb") as fh:
            if is_video(fh.read(16)):
                return True
        dst.unlink()  # an HTML error page saved by an earlier run
    dead = dst.with_suffix(".dead")  # a link that failed before is not waited on again
    if dead.exists():
        return False
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
    except Exception as e:  # dead links are common in WLASL: try the next instance
        print(f"    download failed ({type(e).__name__}): {url}")
        if not isinstance(e, TimeoutError):  # a slow answer may work another time
            dead.touch()
        return False
    if len(data) < 10_000 or not is_video(data):  # some sites answer dead links with an HTML page
        dead.touch()
        return False
    tmp = dst.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(dst)
    return True


def extract(video: Path) -> dict | None:
    """MediaPipe landmarks of one video, cached: {"fps", "frames": [Frame fields], "world", "hands_world"}."""
    import cv2

    cache = RAW / f"{video.stem}.json.gz"
    if cache.exists():
        return json.loads(gzip.decompress(cache.read_bytes()))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    world: list = []
    hands_world: list = []
    try:
        frames = video_to_frames(video, world_out=world, hands_world_out=hands_world)
    except Exception as e:
        print(f"    unreadable video ({type(e).__name__}): {video.name}")
        return None
    raw = {"fps": float(fps), "frames": [f.model_dump(include={"t", "left", "right", "pose"}) for f in frames],
           "world": world, "hands_world": hands_world}
    RAW.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(gzip.compress(json.dumps(raw).encode()))
    return raw


def _resample(x: np.ndarray, n: int = 16) -> np.ndarray:
    t = np.linspace(0, len(x) - 1, n)
    flat = x.reshape(len(x), -1)
    return np.stack([np.interp(t, np.arange(len(x)), flat[:, j]) for j in range(flat.shape[1])], 1)


def _shape(h: np.ndarray) -> np.ndarray:
    """Handshape: points in the palm's frame, palm lengths, wrist at the origin."""
    return (h - h[0]) @ palm_frame(h) / max(np.linalg.norm(h[9] - h[0]), 1e-9)


def take_summary(raw: dict, clip: dict) -> dict:
    """What best_take compares: which hand leads, how much of the sign it was tracked, its frame-to-frame handshape
    noise (palm lengths/s), and a 16-step path (wrist relative to the shoulders, in shoulder widths, + handshape)."""
    frames = [Frame(**f) for f in raw["frames"]]
    s, e = trim_to_sign(frames)
    sides = [by_wrist(f, h) for f, h in zip(frames[s:e], raw["hands_world"][s:e])]
    pose = np.asarray([f["pose"] for f in clip["frames"]])
    shoulders = (pose[:, 11] + pose[:, 12]) / 2
    width = np.linalg.norm(pose[:, 11] - pose[:, 12], axis=-1).mean()
    side = "left" if pose[:, 15, 1].mean() < pose[:, 16, 1].mean() - 0.05 else "right"  # the higher wrist leads
    seen = [np.asarray(w[side]) for _, w in sides if w[side] is not None]
    jitter = float("inf")
    if len(seen) > 2:
        sh = np.stack([_shape(h) for h in seen])
        jitter = float(np.median(np.linalg.norm(np.diff(sh, axis=0), axis=-1).mean(-1))) * raw["fps"]
    hand = [f[side + "W"] for f in clip["frames"]]
    if any(h is None for h in hand):
        return {"side": side, "coverage": 0.0, "jitter": jitter, "path": None}
    shape = np.stack([_shape(np.asarray(h)) for h in hand])
    wrist = (pose[:, 15 if side == "left" else 16] - shoulders) / width
    return {"side": side, "coverage": len(seen) / max(1, len(sides)), "jitter": jitter,
            "path": np.concatenate([_resample(wrist), 0.5 * _resample(shape[:, 1:])], 1)}


def best_take(summaries: list[dict]) -> int:
    """Index of the take to use: among takes tracked >= 80% as well as the best one (right-handed ones first), the one
    closest on average to the other takes when there are three or more, else the one with the least noise."""
    top = max(t["coverage"] for t in summaries)
    good = [i for i, t in enumerate(summaries) if t["coverage"] >= 0.8 * top and t["path"] is not None] or [0]
    good = [i for i in good if summaries[i]["side"] == "right"] or good
    if len(good) >= 3:
        dist = [np.mean([np.linalg.norm(summaries[i]["path"] - summaries[j]["path"], axis=-1).mean() for j in good if j != i])
                for i in good]
        return good[int(np.argmin(dist))]
    return min(good, key=lambda i: summaries[i]["jitter"])


def build_one(sign, instances: list[dict], out: Path, clean: bool = False) -> dict | None:
    takes = []  # (instance, raw landmarks, clip json)
    for inst in instances[:TRIES]:
        if len(takes) >= TAKES:
            break
        video = WLASL / "videos" / f"{sign.gloss}_{inst['video_id']}.mp4"
        if not download(inst["url"], video):
            continue
        raw = extract(video)
        if raw is None:
            continue
        frames = [Frame(**f) for f in raw["frames"]]
        if len(frames) < MIN_FRAMES or not any(f.left or f.right for f in frames):
            print(f"    no signer/hands found in {video.name}")
            continue
        takes.append((inst, raw, clip_json(frames, raw["world"], raw["fps"], raw["hands_world"], clean=clean)))
    if not takes:
        return None
    i = best_take([take_summary(raw, json.loads(c)) for _, raw, c in takes]) if len(takes) > 1 else 0
    inst, raw, clip = takes[i]
    (out / f"{sign.gloss}.json").write_text(clip, encoding="utf-8")
    return {"source": inst["source"], "url": inst["url"], "video_id": inst["video_id"], "frames": len(raw["frames"]),
            "takes": len(takes)}


def _build_job(job) -> tuple[str, dict | None]:
    sign, instances, out, clean = job
    return sign.gloss, build_one(sign, instances, out, clean)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated glosses")
    ap.add_argument("--force", action="store_true", help="rebuild clips that already exist")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(PUBLIC / "signs"), help="clip folder (experiments write elsewhere)")
    ap.add_argument("--clean", action="store_true", help="training.hand_cleanup on the hands (off: the avatar read worse)")
    a = ap.parse_args()
    if not INDEX.exists():
        INDEX.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(INDEX_URL, INDEX)
    index = {e["gloss"]: e["instances"] for e in json.loads(INDEX.read_text(encoding="utf-8"))}
    (WLASL / "videos").mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = WLASL / "clips.json" if out.resolve() == (PUBLIC / "signs").resolve() else out / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    only = {g.strip().upper() for g in a.only.split(",") if g.strip()}
    signs = [s for s in load_vocab().active if not only or s.gloss in only]
    todo = [(s, candidates(s, index), out, a.clean) for s in signs if a.force or not (out / f"{s.gloss}.json").exists()]
    todo = [t for t in todo if t[1]]  # signs WLASL has no direct-file video for are skipped up front
    print(f"{len(todo)} signs to build with {a.workers} workers", flush=True)
    # downloads first, many at once: dead links wait out a timeout, which would leave the MediaPipe workers idle
    jobs = [(i["url"], WLASL / "videos" / f"{s.gloss}_{i['video_id']}.mp4") for s, inst, _, _ in todo for i in inst[:TRIES]]
    with ThreadPoolExecutor(32) as ex:
        got = sum(ex.map(lambda j: download(*j), jobs))
    print(f"{got}/{len(jobs)} candidate videos on disk", flush=True)
    built, failed = 0, []
    with Pool(a.workers) as pool:
        for n, (gloss, got) in enumerate(pool.imap_unordered(_build_job, todo), 1):
            if got:
                manifest[gloss] = got
                manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
                built += 1
            else:
                failed.append(gloss)
            if n % 25 == 0:
                print(f"  {n}/{len(todo)} done, {built} built", flush=True)
    have = sum((out / f"{s.gloss}.json").exists() for s in signs)
    print(f"built {built} new clips; {have}/{len(signs)} vocabulary signs now have clips in {out}")
    if failed:
        print(f"no usable video ({len(failed)}): {' '.join(failed)}")


if __name__ == "__main__":
    main()
