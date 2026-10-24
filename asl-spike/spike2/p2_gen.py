"""P2: synthetic continuous sentences stitched from trimmed isolated ASL Citizen signs.

Coordinates: each clip is converted to body coordinates first (x made pixel-proportional relative to 4:3,
centered on the shoulder midpoint, scaled by shoulder width, dominant hand mirrored into the right slot),
so clips from different videos can be joined. Per frame: lh 21x2, rh 21x2, 8 pose, 6 face points (NaN when
missing) -> 112 coords + 2 hand-present flags = 114 features (NaN -> 0), the round-1 layout without the
32-frame resampling.

Sentence = optional lead-in rest + signs (each: P1 active segment, extra random trim 0-20% each end, speed
0.8-1.5x) joined by 80-250 ms minimum-jerk transitions of all landmarks + optional lead-out rest; ~15% of
positions get an inserted out-of-vocabulary ASL Citizen sign labeled <unk>; jitter (sd 0.01 shoulder widths)
and dropped hand frames (3%). Synthetic transitions are not real coarticulation; RWTH sentences are the check.

Signer separation: ASL Citizen official splits; sentences of split S use only split-S clips; the same signer
for the whole sentence where coverage allows (fallback: another signer of the same split; counted).
Content: 50% café script lines (30 gloss lines from S5/S1), 35% bigram samples (bigram from those lines),
15% uniform random; 2-7 signs.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, citizen_class_map, citizen_rows, model_classes, vocab  # noqa: E402
import trim as TR  # noqa: E402

POSE8 = [0, 11, 12, 13, 14, 15, 16, 23]
FACE6 = [8, 1, 2, 11, 14, 20]
MIRROR8 = [0, 2, 1, 4, 3, 6, 5, 7]
CIT_ASPECT = 4 / 3
FPS = 30.0
UNK = "<unk>"


def canon(lh, rh, pose8, face6, aspect_scale=1.0, swap=None):
    """-> (T, 56, 2) body coordinates with NaN, and whether hands were swapped."""
    lh, rh, pose8, face6 = (np.array(a[..., :2], np.float32, copy=True) for a in (lh, rh, pose8, face6))
    for a in (lh, rh, pose8, face6):
        a[..., 0] *= aspect_scale
    if swap is None:
        swap = TR._motion(lh) > TR._motion(rh)
    if swap:
        lh, rh = rh, lh
        for a in (lh, rh, pose8, face6):
            a[..., 0] = -a[..., 0]
        pose8 = pose8[:, MIRROR8]
    sh = pose8[:, 1:3]
    mid = np.nanmedian(sh.mean(1), axis=0)
    w = np.nanmedian(np.linalg.norm(sh[:, 0] - sh[:, 1], axis=-1))
    if not np.isfinite(w) or w < 1e-4 or not np.all(np.isfinite(mid)):
        allp = np.concatenate([lh, rh, pose8], 1)
        mid = np.nanmedian(allp.reshape(-1, 2), 0) if np.isfinite(allp).any() else np.zeros(2, np.float32)
        w = 0.3
    out = (np.concatenate([lh, rh, pose8, face6], 1) - mid) / w
    return out.astype(np.float32), bool(swap)


def to_feat(c):
    """(T,56,2) with NaN -> (T,114) features."""
    T = len(c)
    flags = np.stack([~np.isnan(c[:, :21]).all((1, 2)), ~np.isnan(c[:, 21:42]).all((1, 2))], 1).astype(np.float32)
    return np.concatenate([np.nan_to_num(c.reshape(T, -1), nan=0.0), flags], 1).astype(np.float32)


def resample(c, factor):
    """time-resample by speed factor (>1 faster), NaN-aware linear interpolation."""
    T = len(c)
    n = max(2, int(round(T / factor)))
    src = np.linspace(0, T - 1, n)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (src - i0)[:, None, None]
    return c[i0] * (1 - w) + c[i1] * w


def minjerk(a, b, n):
    """n frames from pose a to pose b (exclusive of both), minimum-jerk easing; NaN-safe."""
    if n <= 0:
        return np.zeros((0,) + a.shape, np.float32)
    t = np.arange(1, n + 1) / (n + 1)
    s = (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5)[:, None, None]
    a2 = np.where(np.isnan(a), b, a)
    b2 = np.where(np.isnan(b), a, b)
    return (a2 * (1 - s) + b2 * s).astype(np.float32)


def last_valid(c, pick_last=True):
    """frame (56,2) with each point's last (or first) non-NaN value."""
    rng = range(len(c) - 1, -1, -1) if pick_last else range(len(c))
    out = np.full(c.shape[1:], np.nan, np.float32)
    for i in rng:
        m = np.isnan(out) & ~np.isnan(c[i])
        out[m] = c[i][m]
        if not np.isnan(out).any():
            break
    return out


class Pools:
    """clip pools per split: class -> [(video, signer)], OOV list, cached canonical arrays and trims."""

    def __init__(self):
        self.classes = model_classes()
        cmap = citizen_class_map()
        rows = citizen_rows()
        self.trims = json.load(open(CACHE2 / "trim_citizen.json"))
        self.by = defaultdict(lambda: defaultdict(list))
        for r in rows[rows["Gloss"].isin(cmap)].itertuples():
            if r.video in self.trims:
                self.by[r.split][cmap[r.Gloss]].append((r.video, r.signer))
        self.oov = defaultdict(list)
        p = CACHE2 / "oov_subset.csv"
        if p.exists():
            for r in pd.read_csv(p).itertuples():
                if (ROOT / "cache/s2/citizen" / f"{os.path.splitext(r.video)[0]}.npz").exists():
                    self.oov[r.split].append((r.video, r.signer))
        self.cache = {}
        links = json.load(open(R2 / "asllex_links.json"))["classes"]
        self.static = {c: (links[c]["primary"] or {}).get("path_movement") == "None" for c in self.classes}

    def clip(self, video, static=False):
        """canonical array, active (a, b)."""
        if video not in self.cache:
            d = np.load(ROOT / "cache/s2/citizen" / f"{os.path.splitext(video)[0]}.npz")
            sc = (float(d["width"]) / float(d["height"])) / CIT_ASPECT
            c, _ = canon(d["lh"], d["rh"], d["pose"][:, POSE8], d["face"][:, FACE6], sc)
            if video in self.trims:
                a, b = self.trims[video]["a"], self.trims[video]["b"]
            else:
                a, b, _ = TR.trim(d["lh"], d["rh"], d["pose"], float(d["fps"]) or FPS, float(d["width"]) / float(d["height"]), static)
            fps = float(d["fps"]) or FPS
            if abs(fps - FPS) > 1:  # bring to 30 fps
                c = resample(c, fps / FPS)
                a, b = int(a * FPS / fps), max(int(a * FPS / fps) + 2, int(b * FPS / fps))
            self.cache[video] = (c, a, b)
        return self.cache[video]


def script_lines():
    """30 café gloss lines (S5 G2E clean items; first 14 are the scenario's customer turns) mapped to classes."""
    sys.path.insert(0, str(ROOT / "spike"))
    from s5_llm import G2E_ITEMS
    classes = set(model_classes())
    v = {e["gloss"]: e for e in vocab()}
    out = []
    for g, *_ in G2E_ITEMS[:30]:
        toks = []
        for t in g.split():
            if t in classes:
                toks.append(t)
            elif t in v and v[t].get("recognized_as") in classes:
                toks.append(v[t]["recognized_as"])
            else:
                toks.append(UNK)
        out.append(toks)
    return out


def bigram(lines, classes, add=0.05):
    idx = {c: i for i, c in enumerate(["<s>"] + classes + ["</s>"])}
    M = np.full((len(idx), len(idx)), add)
    for l in lines:
        seq = ["<s>"] + [t for t in l if t != UNK] + ["</s>"]
        for a, b in zip(seq, seq[1:]):
            M[idx[a], idx[b]] += 1
    M /= M.sum(1, keepdims=True)
    return idx, M


class Generator:
    def __init__(self, pools: Pools, split: str, seed: int):
        self.p, self.split = pools, split
        self.rng = np.random.default_rng(seed)
        self.lines = script_lines()
        self.cls = pools.classes
        self.bidx, self.B = bigram(self.lines, self.cls)
        self.inv = {i: c for c, i in self.bidx.items()}
        self.signers = sorted({s for c in pools.by[split].values() for _, s in c})
        self.mixed = 0
        self.total = 0

    def content(self):
        u = self.rng.random()
        if u < 0.5:
            toks = list(self.lines[self.rng.integers(len(self.lines))])
        elif u < 0.85:
            toks, cur = [], self.bidx["<s>"]
            n = self.rng.integers(2, 8)
            while len(toks) < n:
                nxt = self.rng.choice(len(self.B), p=self.B[cur])
                c = self.inv[nxt]
                if c in ("</s>", "<s>"):
                    if len(toks) >= 2:
                        break
                    continue
                toks.append(c)
                cur = nxt
        else:
            toks = list(self.rng.choice(self.cls, self.rng.integers(2, 8)))
        return toks

    def sentence(self, toks=None, oov_rate=0.15, signer=None):
        p, rng = self.p, self.rng
        toks = list(toks) if toks is not None else self.content()
        if oov_rate:
            out = []
            for t in toks:
                if rng.random() < oov_rate:
                    out.append(UNK + "+")
                out.append(t)
            toks = out
        if signer is None:
            # the signer who covers most of this sentence's signs (ties broken at random)
            need = [t for t in toks if not t.startswith(UNK)]
            cov = np.array([sum(any(s == x[1] for x in p.by[self.split][t]) for t in need) for s in self.signers], float)
            cov += rng.random(len(cov)) * 0.5
            signer = self.signers[int(np.argmax(cov))]
        self.last_signer = signer
        segs, labels, ends = [], [], []
        first_clip = None
        for t in toks:
            if t.startswith(UNK):
                pool = [x for x in p.oov[self.split] if x[1] == signer] or p.oov[self.split]
                if not pool:
                    continue
                video = pool[rng.integers(len(pool))][0]
                static = False
            else:
                pool_s = [x for x in p.by[self.split][t] if x[1] == signer]
                pool = pool_s or p.by[self.split][t]
                if not pool:
                    continue
                self.total += 1
                self.mixed += 0 if pool_s else 1
                video = pool[rng.integers(len(pool))][0]
                static = p.static.get(t, False)
            c, a, b = p.clip(video, static)
            first_clip = first_clip or (c, a, b)
            L = b - a
            cut0 = int(rng.uniform(0, 0.2) * L)
            cut1 = int(rng.uniform(0, 0.2) * L)
            seg = c[a + cut0: max(a + cut0 + 2, b - cut1)]
            seg = resample(seg, rng.uniform(0.8, 1.5))
            segs.append(seg)
            labels.append(UNK if t.startswith(UNK) else t)
        frames, sign_end = [], []
        if first_clip is not None and rng.random() < 0.6:
            c, a, b = first_clip
            frames.append(c[max(0, a - int(rng.uniform(0.2, 0.8) * FPS)):a])
        prev = frames[-1] if frames and len(frames[-1]) else None
        for i, seg in enumerate(segs):
            if prev is not None and len(prev):
                n = int(round(rng.uniform(0.08, 0.25) * FPS))
                frames.append(minjerk(last_valid(prev), last_valid(seg, False), n))
            frames.append(seg)
            sign_end.append(sum(len(f) for f in frames))
            prev = seg
        tail = int(rng.uniform(0.3, 0.8) * FPS)
        if segs:
            rest_pose = first_clip[0][0] if first_clip is not None else last_valid(segs[-1])
            frames.append(minjerk(last_valid(segs[-1]), rest_pose, tail // 2))
            frames.append(np.repeat(rest_pose[None], tail - tail // 2, 0))
        X = np.concatenate([f for f in frames if len(f)], 0)
        X = X + rng.normal(0, 0.01, X.shape).astype(np.float32)
        drop = rng.random(len(X)) < 0.03
        hand = rng.integers(0, 2, len(X))
        X[drop & (hand == 0), :21] = np.nan
        X[drop & (hand == 1), 21:42] = np.nan
        return to_feat(X), labels, sign_end


def build_fixed(pools, split, n, seed, script_only=False, oov_rate=0.15):
    g = Generator(pools, split, seed)
    data = []
    lines = g.lines[:14]
    for i in range(n):
        if script_only:
            li = i % len(lines)
            s = g.signers[(i // len(lines)) % len(g.signers)]
            X, lab, ends = g.sentence(lines[li], oov_rate=0.0, signer=s)
            data.append({"X": X.astype(np.float16), "labels": lab, "ends": ends, "line": li, "signer": s})
        else:
            X, lab, ends = g.sentence(oov_rate=oov_rate)
            data.append({"X": X.astype(np.float16), "labels": lab, "ends": ends, "signer": g.last_signer})
    return data, {"mixed_signer_signs": g.mixed, "signs": g.total}


def main():
    import pickle
    pools = Pools()
    meta = {}
    for name, split, n, seed, so in (("val", "val", 500, 11, False), ("test", "test", 2000, 12, False),
                                     ("val_script", "val", 140, 13, True), ("test_script", "test", 14 * 33, 14, True)):
        data, m = build_fixed(pools, split, n, seed, so)
        pickle.dump(data, open(CACHE2 / f"p2_{name}.pkl", "wb"))
        meta[name] = {"n": len(data), "split": split, **m, "mixed_rate": round(m["mixed_signer_signs"] / max(m["signs"], 1), 3),
                      "mean_frames": round(float(np.mean([len(d["X"]) for d in data])), 1),
                      "mean_signs": round(float(np.mean([len(d["labels"]) for d in data])), 2),
                      "unk_rate": round(float(np.mean([l == UNK for d in data for l in d["labels"]])), 3)}
        print(name, meta[name], flush=True)
    meta["script_lines"] = script_lines()
    json.dump(meta, open(R2 / "p2_sets.json", "w"), indent=1)


if __name__ == "__main__":
    main()
