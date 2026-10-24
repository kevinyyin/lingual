"""P2 prep: extract landmarks for 60 seeded random ASL Citizen glosses outside our vocabulary (all splits),
used as out-of-vocabulary (<unk>) signs in synthetic sentences. Same pipeline as round 1 (spike/s2_extract.one)."""
import json, os, sys, zipfile
from multiprocessing import Pool
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, ROOT, citizen_rows, vocab
sys.path.insert(0, str(ROOT / "spike"))
import s2_extract as S
def main(n=60, workers=3):
    used = {g for e in vocab() for g in e["asl_citizen"]}
    rows = citizen_rows()
    others = sorted(set(rows["Gloss"]) - used)
    pick = sorted(np.random.default_rng(0).choice(others, n, replace=False).tolist())
    sub = rows[rows["Gloss"].isin(pick)]
    z = zipfile.ZipFile(S.ZIP)
    members = {Path(m).name: m for m in z.namelist() if m.endswith(".mp4")}
    S.VID.mkdir(parents=True, exist_ok=True)
    for v in sub["video"]:
        if not (S.VID / v).exists():
            with z.open(members[v]) as src, open(S.VID / v, "wb") as dst:
                dst.write(src.read())
    sub.to_csv(CACHE2 / "oov_subset.csv", index=False)
    print("oov glosses", len(pick), "videos", len(sub), flush=True)
    with Pool(workers) as pool:
        log = list(pool.imap_unordered(S.one, list(sub["video"]), chunksize=4))
    json.dump({"glosses": pick, "log": log}, open(CACHE2 / "oov_extract_log.json", "w"))
    print("done", sum(1 for x in log if x[1] in ("ok", "cached")), "of", len(log))
if __name__ == "__main__":
    main()
