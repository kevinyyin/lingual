"""R2: landmark extraction for downloaded WLASL instances (same MediaPipe Holistic pipeline as round 1).

Each instance is cut to frame_start..frame_end on WLASL's 25 fps timeline (1-indexed; frame_end = -1 means to
the end): native frame k (time k/fps) is kept when (frame_start-1)/25 <= k/fps <= (frame_end-1)/25.
Full frames are used (no bbox crop). Output: cache/s2r2/wlasl/<video_id>.npz with width/height/fps so the
aspect correction can be applied later. Log: results2/r2_extract_log.json.
"""
from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import CACHE2, R2, ROOT, jdump  # noqa: E402
from extract import FACE_SUBSET, _arr, _make  # noqa: E402

OUT = CACHE2 / "wlasl"


def read_cut(path, fs, fe):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not (5 < fps < 121):
        fps = 25.0
    t0 = (fs - 1) / 25.0
    t1 = (fe - 1) / 25.0 if fe and fe > 0 else 1e9
    frames, k = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = k / fps
        k += 1
        if t < t0 - 1e-6:
            continue
        if t > t1 + 1e-6:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def one(job):
    vid, path, fs, fe = job
    dst = OUT / f"{vid}.npz"
    if dst.exists():
        return vid, "cached", 0
    try:
        frames, fps = read_cut(path, fs, fe)
        if len(frames) < 4:
            return vid, f"failed: only {len(frames)} frames decoded", 0
        mp, lm = _make("holistic")
        T = len(frames)
        pose = np.full((T, 33, 4), np.nan, np.float32)
        pw = np.full((T, 33, 3), np.nan, np.float32)
        lh = np.full((T, 21, 3), np.nan, np.float32)
        rh = np.full((T, 21, 3), np.nan, np.float32)
        face = np.full((T, len(FACE_SUBSET), 3), np.nan, np.float32)
        blend = np.full((T, 52), np.nan, np.float32)
        t0 = time.perf_counter_ns()
        with lm:
            for i, fr in enumerate(frames):
                r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr)), int(i * 1000 / fps))
                if r.pose_landmarks:
                    pose[i] = _arr(r.pose_landmarks, 33, 4)
                if getattr(r, "pose_world_landmarks", None):
                    pw[i] = _arr(r.pose_world_landmarks, 33)
                if r.left_hand_landmarks:
                    lh[i] = _arr(r.left_hand_landmarks, 21)
                if r.right_hand_landmarks:
                    rh[i] = _arr(r.right_hand_landmarks, 21)
                if r.face_landmarks:
                    face[i] = _arr(r.face_landmarks, 478)[FACE_SUBSET]
                if r.face_blendshapes:
                    blend[i] = [c.score for c in r.face_blendshapes]
        h, w = frames[0].shape[:2]
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, pose=pose, pose_world=pw, lh=lh, rh=rh, face=face, blend=blend, fps=fps, n_frames=T,
                            width=w, height=h, extract_ms=(time.perf_counter_ns() - t0) / 1e6)
        os.replace(tmp, dst)
        return vid, "ok", T
    except Exception as e:
        return vid, f"failed: {type(e).__name__}: {e}"[:200], 0


def main(workers=7):
    OUT.mkdir(parents=True, exist_ok=True)
    dl = json.load(open(R2 / "r2_downloads.json"))["rows"]
    jobs = [(r["video_id"], str(ROOT / "data/wlasl/raw" / r["file"]), r["frame_start"], r["frame_end"]) for r in dl if r["status"] == "ok"]
    log = []
    with Pool(workers) as pool:
        for i, (vid, st, T) in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            log.append({"video_id": vid, "status": st, "frames": T})
            if (i + 1) % 200 == 0:
                print(i + 1, "of", len(jobs), flush=True)
    jdump(log, "r2_extract_log.json")
    print("done", sum(1 for x in log if x["status"] in ("ok", "cached")), "of", len(log))


if __name__ == "__main__":
    main()
