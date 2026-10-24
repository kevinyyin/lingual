"""R2: download WLASL instances of glosses that map to our 143 classes (excluding source == "asllex").

Same approach as the repo's start_kit/video_downloader.py: plain HTTP (with its user agent) for dictionary
sites, yt-dlp (python -m yt_dlp) for YouTube. aslpro.com files are Flash .swf and are recorded as an
unsupported format. One worker per host keeps the request rate polite; hosts run in parallel.
Every instance gets a row in results2/r2_downloads.json (ok / failed + reason).
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(__file__))
from common import ROOT, jdump, model_classes, vocab  # noqa: E402

RAW = ROOT / "data/wlasl/raw"
UA = "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7"


def wlasl_to_class() -> dict[str, str]:
    classes = set(model_classes())
    m = {}
    for e in vocab():
        if e["gloss"] in classes:
            for w in e["wlasl"]:
                m.setdefault(w, e["gloss"])
    return m


def instances():
    w2c = wlasl_to_class()
    out = []
    for entry in json.load(open(ROOT / "data/wlasl/WLASL_v0.3.json")):
        if entry["gloss"] not in w2c:
            continue
        for inst in entry["instances"]:
            out.append({**inst, "gloss": entry["gloss"], "class": w2c[entry["gloss"]]})
    return out


def is_yt(url):
    return "youtube" in url or "youtu.be" in url


def fetch_http(url, dst):
    req = urllib.request.Request(url, None, {"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=40) as r:
        data = r.read()
    if len(data) < 2000:
        raise RuntimeError(f"tiny response ({len(data)} bytes)")
    dst.write_bytes(data)
    time.sleep(random.uniform(0.5, 1.5))


def fetch_yt(url, dst):
    cmd = [sys.executable, "-m", "yt_dlp", "-q", "--no-warnings", "-f", "mp4[height<=480]/best[height<=480]/mp4/best",
           "-o", str(dst), "--socket-timeout", "30", url]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not dst.exists():
        raise RuntimeError((r.stderr or r.stdout or "yt-dlp failed").strip().splitlines()[-1][:200] if (r.stderr or r.stdout) else "yt-dlp failed")


def file_for(inst):
    if is_yt(inst["url"]):
        vid = inst["url"].split("v=")[-1].split("&")[0] if "v=" in inst["url"] else inst["url"].rstrip("/").split("/")[-1]
        return RAW / f"yt_{vid}.mp4"
    ext = ".swf" if "aslpro" in inst["url"] else ".mp4"
    return RAW / f"{inst['video_id']}{ext}"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    insts = instances()
    excluded = [i for i in insts if i["source"] == "asllex"]
    insts = [i for i in insts if i["source"] != "asllex"]
    # one download per file (YouTube videos are shared by several instances)
    by_file = defaultdict(list)
    for i in insts:
        by_file[str(file_for(i))].append(i)
    jobs_by_host = defaultdict(list)
    for f, lst in by_file.items():
        u = lst[0]["url"]
        host = "youtube" if is_yt(u) else urlparse(u).netloc
        jobs_by_host[host].append((f, u))
    status = {}

    def run_host(host):
        for f, u in jobs_by_host[host]:
            dst = Path(f)
            if dst.exists() and dst.stat().st_size > 2000:
                status[f] = ("ok", "cached")
                continue
            if dst.suffix == ".swf":
                status[f] = ("failed", "aslpro Flash .swf (unsupported video format)")
                continue
            try:
                (fetch_yt if host == "youtube" else fetch_http)(u, dst)
                status[f] = ("ok", "")
            except Exception as e:
                status[f] = ("failed", f"{type(e).__name__}: {e}"[:200])
        print("host done", host, sum(1 for f, _ in jobs_by_host[host] if status.get(f, ("",))[0] == "ok"), "/", len(jobs_by_host[host]), flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(run_host, sorted(jobs_by_host, key=lambda h: -len(jobs_by_host[h]))))
    rows = []
    for f, lst in by_file.items():
        st, why = status.get(f, ("failed", "not attempted"))
        for i in lst:
            rows.append({k: i[k] for k in ("video_id", "gloss", "class", "signer_id", "source", "variation_id", "frame_start",
                                          "frame_end", "fps", "bbox", "split", "url")} | {"file": os.path.basename(f), "status": st, "reason": why,
                                                                                          "youtube": is_yt(i["url"])})
    per_gloss = defaultdict(lambda: [0, 0])
    for r in rows:
        per_gloss[r["class"]][0] += r["status"] == "ok"
        per_gloss[r["class"]][1] += 1
    jdump({"n_instances": len(rows), "n_ok": sum(r["status"] == "ok" for r in rows), "excluded_asllex": len(excluded),
           "seconds": round(time.time() - t0), "per_class": {c: {"ok": a, "n": n} for c, (a, n) in sorted(per_gloss.items())},
           "rows": rows}, "r2_downloads.json")
    print("ok", sum(r["status"] == "ok" for r in rows), "of", len(rows), "instances;", len(excluded), "asllex excluded")


if __name__ == "__main__":
    main()
