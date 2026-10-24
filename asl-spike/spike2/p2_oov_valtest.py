"""P2 prep (continued): extract OOV clips for the validation and test splits only, capped at 4 videos per
gloss per split (the first pass extracted the training split; it was stopped to free CPU for R3)."""
import os, sys
from multiprocessing import Pool
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, ROOT
sys.path.insert(0, str(ROOT / "spike"))
import s2_extract as S
if __name__ == "__main__":
    sub = pd.read_csv(CACHE2 / "oov_subset.csv")
    sub = sub[sub.split.isin(["val", "test"])].groupby(["Gloss", "split"], group_keys=False).apply(lambda g: g.sample(min(4, len(g)), random_state=0))
    print("videos", len(sub), flush=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        log = list(pool.imap_unordered(S.one, list(sub["video"]), chunksize=2))
    print("done", sum(1 for x in log if x[1] in ("ok", "cached")), "of", len(log))
