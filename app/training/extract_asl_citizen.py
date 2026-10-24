"""ASL Citizen videos for active vocab glosses -> data/landmarks/asl_citizen/{split}/{gloss}/{video}.npz.

Expects the unzipped dataset at data/asl_citizen/ASL_Citizen/{splits/*.csv, videos/*.mp4}.
Resumable: existing .npz files are skipped. Run: uv run python -m training.extract_asl_citizen
"""

import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from server.vocab import load_vocab
from training.landmarks import save_npz, video_to_frames

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "asl_citizen" / "ASL_Citizen"
OUT = ROOT / "data" / "landmarks" / "asl_citizen"
MIN_TRAIN = 15


def select_rows(csv_path: Path, wanted: dict[str, str]) -> list[tuple[str, str, str]]:
    """wanted maps dataset gloss (upper) -> our gloss. Returns (user, video file, our gloss)."""
    out = []
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for user, fname, gloss, *_ in reader:
            ours = wanted.get(gloss.strip().upper())
            if ours:
                out.append((user, fname, ours))
    return out


def _extract(job: tuple[Path, Path]) -> str:
    src, dst = job
    if dst.exists():
        return "skip"
    frames = video_to_frames(src)
    if not frames:
        return "empty"
    save_npz(frames, dst)
    return "ok"


def all_glosses() -> set[str]:
    out = set()
    for split in ("train", "val", "test"):
        with open(DATA / "splits" / f"{split}.csv", newline="") as fh:
            out |= {row[2].strip().upper() for row in list(csv.reader(fh))[1:]}
    return out


def main() -> None:
    vocab = load_vocab()
    wanted = {s.asl_citizen.upper(): s.gloss for s in vocab.active}
    missing = sorted(set(wanted) - all_glosses())
    if missing:
        print("vocab glosses not in ASL Citizen (fix vocab.yaml):", ", ".join(missing))
    jobs, counts = [], Counter()
    for split in ("train", "val", "test"):
        for user, fname, gloss in select_rows(DATA / "splits" / f"{split}.csv", wanted):
            jobs.append((DATA / "videos" / fname, OUT / split / gloss / f"{Path(fname).stem}.npz"))
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
