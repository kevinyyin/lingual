"""V4 fallback: MediaPipe rarely detects the avatar's hands, so compare finger angles of rendered frames where a hand
WAS detected against the sign's canonical handshape (V2 library), per configuration; also report pose detection."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, jdump
from v2_handshapes import ANGLE_NAMES, hand_angles
links = json.load(open(R2 / "asllex_links.json"))["classes"]
lib = json.load(open(R2 / "handshapes.json"))["library"]
FING = [n for n in ANGLE_NAMES if n.split("_")[0] in ("index", "middle", "ring", "pinky") and n.endswith(("mcp", "pip"))]
out = {}
for cfg in "abcd":
    errs, pose_rate, hand_frames, signs_with = [], [], 0, 0
    for p in sorted((CACHE2 / "v4").glob(f"{cfg}_*.lm.npz")):
        g = p.name[2:-7]
        d = dict(np.load(p))
        pose_rate.append(float(np.mean(~np.isnan(d["pose"][:, 0, 0]))))
        hs = (links.get(g, {}).get("primary") or {}).get("handshape")
        if hs not in lib:
            continue
        canon = np.array([lib[hs]["median"][n] for n in FING])
        got = []
        for key, mirror in (("rh", False), ("lh", True)):  # avatar right hand (dominant) appears as MediaPipe 'left' or 'right' depending on view
            h = d[key].copy()
            h[..., 0] *= 960 / 540
            h[..., 2] *= 960 / 540
            if mirror:
                h[..., 0] = -h[..., 0]
            ok = ~np.isnan(h[:, 0, 0])
            if ok.any():
                A = hand_angles(h[ok])
                got.append(A[:, [ANGLE_NAMES.index(n) for n in FING]])
        if got:
            A = np.concatenate(got)
            hand_frames += len(A)
            signs_with += 1
            errs.append(float(np.median(np.abs(A - canon))))
    out[cfg] = {"pose_detection_rate_median": round(float(np.median(pose_rate)), 3), "signs_with_any_hand_frame": signs_with,
                "hand_frames": hand_frames, "finger_angle_abs_error_deg_median": round(float(np.median(errs)), 1) if errs else None}
    print(cfg, out[cfg])
jdump(out, "v4_fallback_angles.json")
