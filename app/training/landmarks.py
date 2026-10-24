"""Video file -> list[Frame] with the same MediaPipe Tasks models the browser uses; .npz storage."""

from pathlib import Path

import numpy as np

from server.protocol import Frame

MODELS = Path(__file__).resolve().parent.parent / "web" / "public" / "models"


def save_npz(frames: list[Frame], path: Path | str) -> None:
    n = len(frames)
    names = sorted({k for f in frames if f.blend for k in f.blend})
    arr = {
        "t": np.array([f.t for f in frames]),
        "pose": np.array([f.pose for f in frames]).reshape(n, 33, 4),
        "blend_names": np.array(names, dtype=str),
        "blend": np.array([[f.blend.get(k, 0.0) for k in names] if f.blend else [0.0] * len(names) for f in frames]).reshape(n, len(names)),
        "blend_ok": np.array([f.blend is not None for f in frames]),
        "head": np.array([f.head or (0.0, 0.0, 0.0) for f in frames]).reshape(n, 3),
        "head_ok": np.array([f.head is not None for f in frames]),
    }
    for side in ("left", "right"):
        arr[side] = np.array([getattr(f, side) or [[0.0] * 3] * 21 for f in frames]).reshape(n, 21, 3)
        arr[f"{side}_ok"] = np.array([getattr(f, side) is not None for f in frames])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arr)


def load_npz(path: Path | str) -> list[Frame]:
    z = np.load(path)
    names = list(z["blend_names"])
    return [
        Frame(
            t=float(z["t"][i]),
            left=z["left"][i].tolist() if z["left_ok"][i] else None,
            right=z["right"][i].tolist() if z["right_ok"][i] else None,
            pose=z["pose"][i].tolist(),
            blend=dict(zip(names, z["blend"][i].tolist())) if z["blend_ok"][i] else None,
            head=tuple(z["head"][i].tolist()) if z["head_ok"][i] else None,
        )
        for i in range(len(z["t"]))
    ]


def _ypr(m: np.ndarray) -> tuple[float, float, float]:
    # Same formula as web/src/camera.ts matToYPR. Only yaw oscillation is used downstream,
    # which is insensitive to a transposed (row- vs column-major) matrix.
    deg = 180 / np.pi
    return (float(np.arctan2(m[0, 2], m[2, 2]) * deg),
            float(np.arcsin(np.clip(-m[1, 2], -1, 1)) * deg),
            float(np.arctan2(m[1, 0], m[1, 1]) * deg))


_detectors: dict = {}  # pose model file -> [hands, pose, face, next timestamp ms]


def _open(pose_model: Path):
    """The three video-mode detectors, opened once per process and reused: closing one takes ~42 s in mediapipe 1.1.1
    on Windows (a thread-join timeout), ~25x the detection time of a short sign video."""
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision as v

    if pose_model not in _detectors:
        def opts(p):
            return BaseOptions(model_asset_path=str(p))

        video = v.RunningMode.VIDEO
        _detectors[pose_model] = [
            v.HandLandmarker.create_from_options(v.HandLandmarkerOptions(base_options=opts(MODELS / "hand_landmarker.task"), running_mode=video, num_hands=2)),
            v.PoseLandmarker.create_from_options(v.PoseLandmarkerOptions(base_options=opts(pose_model), running_mode=video)),
            v.FaceLandmarker.create_from_options(v.FaceLandmarkerOptions(
                base_options=opts(MODELS / "face_landmarker.task"), running_mode=video,
                output_face_blendshapes=True, output_facial_transformation_matrixes=True)),
            0,
        ]
    return _detectors[pose_model]


def video_to_frames(path: Path | str, world_out: list | None = None, transform=None, stride: int = 1,
                    hands_world_out: list | None = None, pose_model: Path | None = None) -> list[Frame]:
    """If world_out is given, pose world landmarks (metres, hip-centred) are appended to it per returned frame;
    hands_world_out likewise gets {"left", "right"} hand world landmarks (metres) or None per hand.
    transform(bgr) -> bgr degrades each frame before detection and stride keeps every n-th frame (evaluation).
    pose_model: another pose .task file (default the browser's lite). The heavy model is not used for avatar clips:
    its world landmarks put the hands ~0.5 shoulder widths further back, and the avatar read worse (round trip 0.03 top-1
    against 0.09)."""
    import cv2
    import mediapipe as mp

    det = _open(Path(pose_model or MODELS / "pose_landmarker_lite.task"))
    hands, pose, face = det[:3]
    # Video timestamps must keep increasing across videos; one blank frame first drops the previous video's tracking.
    base = det[3]
    blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((64, 64, 3), np.uint8))
    for d in (hands, pose, face):
        d.detect_for_video(blank, base)
    base += 1000
    det[3] = base  # even if this video yields no frames
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    frames: list[Frame] = []
    i = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = i * 1000.0 / fps
            i += 1
            if (i - 1) % stride:
                continue
            if transform is not None:
                bgr = transform(bgr)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))  # never mirrored
            ts = base + int(t)
            det[3] = ts + 1000
            p = pose.detect_for_video(img, ts)
            if not p.pose_landmarks:
                continue
            f = Frame(t=t, pose=[[l.x, l.y, l.z, l.visibility or 0.0] for l in p.pose_landmarks[0]])
            if world_out is not None:
                world_out.append([[l.x, l.y, l.z, l.visibility or 0.0] for l in p.pose_world_landmarks[0]])
            h = hands.detect_for_video(img, ts)
            hw: dict = {"left": None, "right": None}
            for lm, wl, hd in zip(h.hand_landmarks, h.hand_world_landmarks, h.handedness):
                pts = [[l.x, l.y, l.z] for l in lm]
                side = "left" if hd[0].category_name == "Left" else "right"
                setattr(f, side, pts)
                hw[side] = [[l.x, l.y, l.z] for l in wl]  # metres, true-to-scale: finger bends read correctly
            if hands_world_out is not None:
                hands_world_out.append(hw)
            fc = face.detect_for_video(img, ts)
            if fc.face_blendshapes:
                f.blend = {c.category_name: c.score for c in fc.face_blendshapes[0]}
            if fc.facial_transformation_matrixes:
                f.head = _ypr(np.asarray(fc.facial_transformation_matrixes[0]))
            frames.append(f)
    finally:
        cap.release()
    return frames
