"""P1 follow-up: padding grid on validation signers at the chosen RISE/SPEED (rise 0.15, speed 0.3)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import jdump, load_model
from p1_validate import load_all, score
import trim as TR
model, classes = load_model()
items = load_all()
val = [it for it in items if it["split"] == "val"]
test = [it for it in items if it["split"] == "test"]
out = []
for pad in (2, 4, 6, 9, 12):
    tr = [TR.trim(it["d"]["lh"], it["d"]["rh"], it["d"]["pose"], it["fps"], it["aspect"], it["static"], rise=0.15, speed=0.3, pad=pad)[:2] for it in val]
    rem = float(np.mean([1 - (b - a) / len(it["d"]["lh"]) for (a, b), it in zip(tr, val)]))
    t1, t3 = score(model, classes, val, tr)
    out.append({"pad": pad, "removed_mean": round(rem, 3), "val_top1": round(t1, 4), "val_top3": round(t3, 4)})
    print(out[-1], flush=True)
base = score(model, classes, val)
ok = [o for o in out if o["val_top1"] >= base[0] - 0.01]
best = max(ok, key=lambda o: o["removed_mean"]) if ok else max(out, key=lambda o: o["val_top1"])
tt = [TR.trim(it["d"]["lh"], it["d"]["rh"], it["d"]["pose"], it["fps"], it["aspect"], it["static"], rise=0.15, speed=0.3, pad=best["pad"])[:2] for it in test]
t1, t3 = score(model, classes, test, tt)
rem = [1 - (b - a) / len(it["d"]["lh"]) for (a, b), it in zip(tt, test)]
res = {"grid": out, "val_untrimmed_top1": base[0], "chosen_pad": best["pad"], "test_trimmed_top1": t1, "test_trimmed_top3": t3,
       "test_removed_median": float(np.median(rem)), "test_seconds_after_median": float(np.median([(b - a) / it["fps"] for (a, b), it in zip(tt, test)]))}
jdump(res, "p1_pad.json"); print(res)
