"""Regenerate cache/s2r2/trim_citizen.json with the final trimmer defaults (trim.RISE/SPEED/PAD)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2
from p1_validate import load_all
import trim as TR
out = {}
for it in load_all():
    a, b, info = TR.trim(it["d"]["lh"], it["d"]["rh"], it["d"]["pose"], it["fps"], it["aspect"], it["static"])
    out[it["video"]] = {"a": a, "b": b, "T": len(it["d"]["lh"]), "fps": it["fps"], "static": it["static"], **info}
json.dump(out, open(CACHE2 / "trim_citizen.json", "w")); print(len(out))
