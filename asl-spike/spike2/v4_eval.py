"""V4: avatar evaluation by automatic round trip.

Render every lexicon sign (single-sign sequence) in Microsoft Edge (Playwright, GPU) at 1280x720, 30 fps,
frontal upper-body framing, plain background, for configurations
  (a) round-1 avatar: whole clips, kalidokit fingers, direction-only wrist, 1.0x
  (b) + V1 trimming and direct transitions
  (c) + V2 handshape snapping, palm-orientation wrist, One-Euro smoothing
  (d) configuration (c) at 1.3x
Then run the round-1 MediaPipe Holistic pipeline + round-1 classifier on the renders: hand detection rate,
top-1/top-3 per configuration. Upper bound: the original human landmark sequences of the same lexicon clips
(training clips of ASL Citizen signer P11 and others - so the upper bound is optimistic).
Also: sentence-level round trip of the 3 round-1 avatar sentences and 5 barista lines through the P3
phrase model; contact sheets (human vs avatar key frames) and an mp4 capture of the 3 sentences (local only).
Usage: python v4_eval.py [render|score|sentences|sheets|all]
"""
from __future__ import annotations

import base64
import json
import os
import pickle
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, jdump  # noqa: E402

URL = "http://127.0.0.1:8766/index2.html?render=1"
CFGS = {"a": {"trim": False, "snap": False, "wrist": False, "speed": 1.0},
        "b": {"trim": True, "snap": False, "wrist": False, "speed": 1.0},
        "c": {"trim": True, "snap": True, "wrist": True, "speed": 1.0},
        "d": {"trim": True, "snap": True, "wrist": True, "speed": 1.3}}
OUT = CACHE2 / "v4"


def browser(p):
    b = p.chromium.launch(channel="msedge", headless=True, args=["--use-angle=d3d11", "--enable-gpu", "--ignore-gpu-blocklist"])
    pg = b.new_page(viewport={"width": 1280, "height": 720})
    pg.goto(URL)
    pg.wait_for_function("window.ready === true", timeout=180000)
    return b, pg


def decode(urls):
    return [cv2.imdecode(np.frombuffer(base64.b64decode(u.split(",")[1]), np.uint8), 1) for u in urls]


def render():
    from playwright.sync_api import sync_playwright
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b, pg = browser(p)
        keys = pg.evaluate("window.LEX_KEYS")
        for cfg, c in CFGS.items():
            for g in keys:
                dst = OUT / f"{cfg}_{g}.npz"
                if dst.exists():
                    continue
                r = pg.evaluate("([g, c]) => window.renderFrames([{gloss: g, nmm: {}}], c, 30, 0.85)", [g, c])
                fr = decode(r["frames"])
                jpgs = [cv2.imencode(".jpg", cv2.resize(f, (960, 540)), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes() for f in fr]
                np.savez(dst, jpgs=np.array(jpgs, dtype=object), dur=r["dur"])
            print("rendered", cfg, flush=True)
        b.close()


def landmarks_of(frames_bgr, fps=30.0):
    sys.path.insert(0, str(ROOT / "spike"))
    from extract import FACE_SUBSET, _arr, _make
    mp, lm = _make("holistic")
    T = len(frames_bgr)
    pose = np.full((T, 33, 4), np.nan, np.float32)
    lh = np.full((T, 21, 3), np.nan, np.float32)
    rh = np.full((T, 21, 3), np.nan, np.float32)
    face = np.full((T, len(FACE_SUBSET), 3), np.nan, np.float32)
    with lm:
        for i, f in enumerate(frames_bgr):
            r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))), int(i * 1000 / fps))
            if r.pose_landmarks:
                pose[i] = _arr(r.pose_landmarks, 33, 4)
            if r.left_hand_landmarks:
                lh[i] = _arr(r.left_hand_landmarks, 21)
            if r.right_hand_landmarks:
                rh[i] = _arr(r.right_hand_landmarks, 21)
            if r.face_landmarks:
                face[i] = _arr(r.face_landmarks, 478)[FACE_SUBSET]
    return {"pose": pose, "lh": lh, "rh": rh, "face": face}


def _extract_one(path):
    dst = str(path).replace(".npz", ".lm.npz")
    if os.path.exists(dst):
        return dst
    with np.load(path, allow_pickle=True) as z:
        fr = [cv2.imdecode(np.frombuffer(b, np.uint8), 1) for b in z["jpgs"]]
    d = landmarks_of(fr)
    np.savez_compressed(dst, **d, width=960, height=540)
    return dst


def score():
    sys.path.insert(0, str(ROOT / "spike"))
    import torch
    from features import from_holistic, normalize
    from common import load_model
    files = sorted(p for p in OUT.glob("*_*.npz") if not p.name.endswith(".lm.npz"))
    with Pool(5) as pool:
        list(pool.imap_unordered(_extract_one, files, chunksize=4))
    model, classes = load_model()
    asp = (960 / 540) / (4 / 3)
    res = {}
    for cfg in CFGS:
        X, y, hr = [], [], []
        for g in classes:
            p = OUT / f"{cfg}_{g}.lm.npz"
            if not p.exists():
                continue
            d = dict(np.load(p))
            hr.append(float(np.mean(~np.isnan(d["lh"][:, 0, 0]) | ~np.isnan(d["rh"][:, 0, 0]))))
            lh, rh, pose, face = from_holistic(d)
            for a in (lh, rh, pose, face):
                a[..., 0] *= asp
            X.append(normalize(lh, rh, pose, face))
            y.append(classes.index(g))
        with torch.no_grad():
            top = model(torch.tensor(np.stack(X))).topk(3, 1).indices.numpy()
        y = np.array(y)
        res[cfg] = {"config": CFGS[cfg], "n_signs": len(y), "hand_detection_rate_median": round(float(np.median(hr)), 3),
                    "signs_with_hands_under_50pct": int(sum(h < 0.5 for h in hr)),
                    "top1": round(float((top[:, 0] == y).mean()), 4), "top3": round(float((top == y[:, None]).any(1).mean()), 4),
                    "per_sign_top1": {classes[c]: bool(t) for c, t in zip(y, top[:, 0] == y)}}
    # upper bound: the human lexicon clips themselves
    lex = json.load(open(CACHE2 / "lexicon_v1.json"))
    X, y = [], []
    for g, e in lex.items():
        d = dict(np.load(ROOT / "cache/s2/citizen" / f"{os.path.splitext(e['video'])[0]}.npz"))
        X.append(normalize(*from_holistic(d)))
        y.append(classes.index(g))
    with torch.no_grad():
        top = model(torch.tensor(np.stack(X))).topk(3, 1).indices.numpy()
    y = np.array(y)
    res["human_upper_bound"] = {"n_signs": len(y), "top1": round(float((top[:, 0] == y).mean()), 4),
                                "top3": round(float((top == y[:, None]).any(1).mean()), 4),
                                "note": "lexicon clips are ASL Citizen training clips - optimistic"}
    jdump(res, "v4_roundtrip.json")
    print({k: (v.get("top1"), v.get("top3"), v.get("hand_detection_rate_median")) for k, v in res.items()})


def sentences():
    """render 3 round-1 sentences + 5 barista lines (config c at 1.3x and a), run the P3 phrase model."""
    from playwright.sync_api import sync_playwright
    import torch
    import p2_gen as G
    import p4_decode as D
    from p3_model import PhraseNet
    from p5_eval import script_lm_and_lines
    from common import model_classes
    classes = model_classes()
    sents = [(p["label"], p["sentence"]["signs"]) for p in json.load(open(ROOT / "avatar/sentences.json"))]
    for r in list(map(json.loads, open(ROOT / "results/s5_e2g_v2.jsonl")))[:5]:
        sents.append(("s5: " + r["english"], (r.get("output") or {}).get("signs", [])))
    out = []
    cap_frames = []
    with sync_playwright() as p:
        b, pg = browser(p)
        for cfg in ("c13", "a"):
            c = dict(CFGS["c"], speed=1.3) if cfg == "c13" else CFGS["a"]
            for label, signs in sents:
                r = pg.evaluate("([s, c]) => window.renderFrames(s, c, 30, 0.85)", [signs, c])
                fr = decode(r["frames"])
                if cfg == "c13" and len(cap_frames) < 3 * 1000 and label in [s[0] for s in sents[:3]]:
                    cap_frames += fr
                out.append({"cfg": cfg, "label": label, "glosses": [s["gloss"] for s in signs], "missing": r["missing"],
                            "seconds": round(r["dur"] / 1000, 2), "frames": [cv2.resize(f, (640, 360)) for f in fr]})
        b.close()
    # capture of the 3 round-1 sentences (config c, 1.3x)
    vw = cv2.VideoWriter(str(ROOT / "avatar/capture2.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))
    for f in cap_frames:
        vw.write(f)
    vw.release()
    net = PhraseNet("bi")
    net.load_state_dict(torch.load(CACHE2 / "p3_a_bi2.pt")["state"])
    net.eval()
    lm, lines = script_lm_and_lines(classes)
    res = []
    for s in out:
        d = landmarks_of(s["frames"])
        sc = (960 / 540) / G.CIT_ASPECT
        c, _ = G.canon(d["lh"], d["rh"], d["pose"][:, G.POSE8], d["face"][:, G.FACE6] if d["face"].shape[1] == 24 else d["face"], sc)
        X = G.to_feat(c)
        with torch.no_grad():
            lp = net(torch.tensor(X)[None]).log_softmax(-1)[0]
        hyp = D.beam_search(lp.numpy(), lm)[0][0]
        hyp_g = [classes[h] if h < 143 else "<unk>" for h in hyp]
        ref = [g if g in classes else "<unk>" for g in s["glosses"]]
        tp = D.lcs([x for x in ref if x != "<unk>"], [x for x in hyp_g if x != "<unk>"])
        res.append({"cfg": s["cfg"], "label": s["label"], "ref": ref, "hyp": hyp_g, "seconds": s["seconds"], "missing": s["missing"],
                    "vocab_recall": round(tp / max(1, sum(x != "<unk>" for x in ref)), 3)})
        print(s["cfg"], s["label"][:40], ref, "->", hyp_g, flush=True)
    jdump({"sentences": res, "capture": "avatar/capture2.mp4 (local, git-ignored)"}, "v4_sentences.json")


def sheets():
    """contact sheet per sign: human key frames (ASL Citizen clip) vs avatar (config c) key frames."""
    lex = json.load(open(CACHE2 / "lexicon_v1.json"))
    trims = json.load(open(CACHE2 / "trim_citizen.json"))
    dst = R2 / "v4_contact"
    dst.mkdir(exist_ok=True)
    for g, e in lex.items():
        cap = cv2.VideoCapture(str(ROOT / "data/asl-citizen/videos_vocab" / e["video"]))
        a, b = trims[e["video"]]["a"], trims[e["video"]]["b"]
        want = [a, (2 * a + b) // 3, (a + 2 * b) // 3, max(a, b - 1)]
        hum, i = {}, 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i in want:
                hum[i] = cv2.resize(f, (320, 240))
            i += 1
        p = OUT / f"c_{g}.npz"
        if not p.exists():
            continue
        av = [cv2.imdecode(np.frombuffer(b, np.uint8), 1) for b in np.load(p, allow_pickle=True)["jpgs"]]
        idx = np.linspace(0, len(av) - 1, 6)[1:5].astype(int)
        top = np.hstack([hum.get(j, np.zeros((240, 320, 3), np.uint8)) for j in want])
        bot = np.hstack([cv2.resize(av[j][:, 120:840], (320, 240)) for j in idx])
        sheet = np.vstack([top, bot])
        cv2.putText(sheet, f"{g}: top = human (ASL Citizen {e['signer']}), bottom = avatar (V2)", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
        cv2.imwrite(str(dst / f"{g}.jpg"), sheet)
    print("sheets", len(list(dst.glob("*.jpg"))))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("render", "all"):
        render()
    if what in ("score", "all"):
        score()
    if what in ("sheets", "all"):
        sheets()
    if what in ("sentences", "all"):
        sentences()
