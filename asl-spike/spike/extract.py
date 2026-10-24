"""MediaPipe extraction shared by S2/S3: one Holistic Landmarker pass per video (CPU).

Per frame we keep: pose 33x(x,y,z,vis), left/right hand 21x(x,y,z) (NaN when not detected),
face 478x(x,y,z) (only a subset is cached for S2; S3 keeps what it needs), face blendshapes (52),
and timestamps. Coordinates are MediaPipe's normalized image coordinates (same convention as the
Kaggle asl-signs landmark files, which also came from MediaPipe Holistic).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parent.parent / "cache" / "models"
# face points kept for S2 features: lips outline, nose tip, brows, eyes (subset of the 468 mesh)
FACE_SUBSET = [0, 13, 14, 17, 61, 291, 78, 308, 1, 4, 70, 105, 107, 336, 334, 300, 33, 133, 362, 263, 152, 10, 234, 454]


def _make(kind: str = "holistic", running_mode: str = "VIDEO"):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    rm = getattr(vision.RunningMode, running_mode)
    if kind == "holistic":
        opts = vision.HolisticLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "holistic_landmarker.task")), running_mode=rm,
            output_face_blendshapes=True, min_face_detection_confidence=0.3, min_pose_detection_confidence=0.3,
            min_hand_landmarks_confidence=0.3)
        return mp, vision.HolisticLandmarker.create_from_options(opts)
    if kind == "face":
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "face_landmarker.task")), running_mode=rm,
            output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1,
            min_face_detection_confidence=0.3)
        return mp, vision.FaceLandmarker.create_from_options(opts)
    raise ValueError(kind)


def read_frames(path: str, max_frames: int | None = None, crop=None, upscale: float = 1.0):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if crop is not None:
            x0, y0, x1, y1 = crop
            f = f[y0:y1, x0:x1]
        if upscale != 1.0:
            f = cv2.resize(f, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


def _arr(lms, n, dims=3):
    if not lms:
        return np.full((n, dims), np.nan, np.float32)
    out = np.array([[p.x, p.y, p.z] + ([p.visibility or 0.0] if dims == 4 else []) for p in lms], np.float32)
    return out[:n] if len(out) >= n else np.vstack([out, np.full((n - len(out), dims), np.nan, np.float32)])


def holistic_video(path: str, crop=None, upscale: float = 1.0, keep_full_face: bool = False) -> dict:
    """Run Holistic Landmarker over every frame. Returns arrays plus timing (wall ms per frame)."""
    frames, fps = read_frames(path, crop=crop, upscale=upscale)
    mp, lm = _make("holistic")
    T = len(frames)
    pose = np.full((T, 33, 4), np.nan, np.float32)
    lh = np.full((T, 21, 3), np.nan, np.float32)
    rh = np.full((T, 21, 3), np.nan, np.float32)
    face = np.full((T, 478 if keep_full_face else len(FACE_SUBSET), 3), np.nan, np.float32)
    blend = np.full((T, 52), np.nan, np.float32)
    pose_world = np.full((T, 33, 3), np.nan, np.float32)
    names = None
    t0 = time.perf_counter_ns()
    with lm:
        for i, fr in enumerate(frames):
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(fr))
            r = lm.detect_for_video(img, int(i * 1000 / fps))
            if r.pose_landmarks:
                pose[i] = _arr(r.pose_landmarks, 33, 4)
            if getattr(r, "pose_world_landmarks", None):
                pose_world[i] = _arr(r.pose_world_landmarks, 33)
            if r.left_hand_landmarks:
                lh[i] = _arr(r.left_hand_landmarks, 21)
            if r.right_hand_landmarks:
                rh[i] = _arr(r.right_hand_landmarks, 21)
            if r.face_landmarks:
                f = _arr(r.face_landmarks, 478)
                face[i] = f if keep_full_face else f[FACE_SUBSET]
            if r.face_blendshapes:
                cats = r.face_blendshapes
                if names is None:
                    names = [c.category_name for c in cats]
                blend[i] = [c.score for c in cats]
    ms = (time.perf_counter_ns() - t0) / 1e6
    h, w = (frames[0].shape[:2] if frames else (0, 0))
    return {"pose": pose, "pose_world": pose_world, "lh": lh, "rh": rh, "face": face, "blend": blend, "blend_names": names or [],
            "fps": fps, "n_frames": T, "width": w, "height": h, "extract_ms": ms}
