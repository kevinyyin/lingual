"""ASL Citizen videos for the camera's signs -> data/landmarks/asl_citizen/{split}/{gloss}/{video}.npz.

Videos come from the unzipped dataset at data/asl_citizen/ASL_Citizen/videos/*.mp4, or with --remote straight out of
Microsoft's 42.8 GB zip, one video at a time (range requests; each is deleted once read). Split CSVs: the unzipped
dataset's, else data/asl_citizen/splits/ (training.remote_zip). Resumable: existing .npz files are skipped.
Run: uv run python -m training.extract_asl_citizen [--remote]
"""

import argparse
import csv
import re
import sys
import tempfile
import time
from collections import Counter
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from server.citizen_classifier import ALIASES
from server.vocab import load_vocab
from training.landmarks import save_npz, video_to_frames
from training.remote_zip import ASL_CITIZEN_ZIP, HttpRangeFile, http_fetch, member_bytes

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "asl_citizen" / "ASL_Citizen"
SPLITS = DATA / "splits" if (DATA / "splits").exists() else ROOT / "data" / "asl_citizen" / "splits"
OUT = ROOT / "data" / "landmarks" / "asl_citizen"
MIN_TRAIN = 15


def select_rows(csv_path: Path, wanted: dict[str, str]) -> list[tuple[str, str, str]]:
    """wanted maps dataset gloss (upper) -> our gloss. Returns (user, video file, our gloss). Numbered variants (WANT1,
    WANT2) are one sign, and ASL Citizen's other names (ME for I) count, as the released model reads them
    (citizen_classifier.to_app_gloss)."""
    out = []
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for user, fname, gloss, *_ in reader:
            name = gloss.strip().upper()
            base = re.sub(r"\d+$", "", name) or name
            ours = wanted.get(name) or wanted.get(base) or wanted.get(ALIASES.get(base, ""))
            if ours:
                out.append((user, fname, ours))
    return out


def _extract(job: tuple) -> str:
    src, dst, remote = job  # remote: (zip url, ZipInfo) when src is not on disk
    if dst.exists():
        return "skip"
    if not src.exists():
        if remote is None:
            return "missing"
        for attempt in range(3):
            try:
                data = member_bytes(http_fetch(remote[0]), remote[1])
                break
            except OSError:
                if attempt == 2:
                    return "fetch failed"
                time.sleep(2 ** attempt)
        with tempfile.NamedTemporaryFile(suffix=src.suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            frames = video_to_frames(tmp.name)
    else:
        frames = video_to_frames(src)
    if not frames:
        return "empty"
    save_npz(frames, dst)
    return "ok"


def all_glosses() -> set[str]:
    out = set()
    for split in ("train", "val", "test"):
        with open(SPLITS / f"{split}.csv", newline="") as fh:
            out |= {row[2].strip().upper() for row in list(csv.reader(fh))[1:]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", action="store_true", help="fetch each video from the remote zip instead of the disk")
    a = ap.parse_args()
    vocab = load_vocab(extended=None)  # the camera's signs (server/main.py CAMERA_VOCAB)
    index = {}
    if a.remote:
        index = {i.filename: i for i in zipfile.ZipFile(HttpRangeFile(ASL_CITIZEN_ZIP)).infolist()}
    wanted = {s.asl_citizen.upper(): s.gloss for s in vocab.active}
    for s in vocab.active:  # ALIASES name our glosses (EXCUSE -> EXCUSE-ME), not the vocab's asl_citizen spelling
        wanted.setdefault(s.gloss.upper(), s.gloss)
    missing = sorted(set(wanted) - all_glosses())
    if missing:
        print("vocab glosses not in ASL Citizen (fix vocab.yaml):", ", ".join(missing))
    jobs, counts = [], Counter()
    for split in ("train", "val", "test"):
        for user, fname, gloss in select_rows(SPLITS / f"{split}.csv", wanted):
            info = index.get(f"ASL_Citizen/videos/{fname}")
            jobs.append((DATA / "videos" / fname, OUT / split / gloss / f"{Path(fname).stem}.npz",
                         (ASL_CITIZEN_ZIP, info) if info else None))
            if split == "train":
                counts[gloss] += 1
    print(f"{len(jobs)} videos to process")
    with ProcessPoolExecutor() as pool:
        results = Counter(pool.map(_extract, jobs, chunksize=8))
    print("results:", dict(results))
    low = [g for g in wanted.values() if counts[g] < MIN_TRAIN]
    if low:
        print(f"glosses with < {MIN_TRAIN} train videos (swap in vocab.yaml):", ", ".join(low))
        sys.exit(1)


if __name__ == "__main__":
    main()
