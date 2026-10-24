// Webcam -> MediaPipe Tasks landmarks -> Frame. Only landmarks leave the page, never video.
import { FaceLandmarker, FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";
import type { Frame, P3 } from "./ws";

/** Face-mesh points the sign classifier reads (nose tip, upper/lower lip, brows, chin); order matters. */
export const FACE_POINTS = [1, 13, 14, 105, 334, 152];

/** Row-major rotation from MediaPipe's column-major 4x4 -> [yaw, pitch, roll] degrees. Same formula as training/landmarks.py. */
export function matToYPR(m: ArrayLike<number>): [number, number, number] {
  const r = (row: number, col: number) => m[col * 4 + row];
  const deg = 180 / Math.PI;
  return [
    Math.atan2(r(0, 2), r(2, 2)) * deg,
    Math.asin(Math.max(-1, Math.min(1, -r(1, 2)))) * deg,
    Math.atan2(r(1, 0), r(1, 1)) * deg,
  ];
}

let trackers: ReturnType<typeof createTrackers> | null = null;
/** The three MediaPipe trackers (~30 MB of wasm and models), created once. Teaching mode calls this early so the
 *  first camera exercise does not wait for the download; a failed load is retried on the next call. */
export const loadTrackers = () => (trackers ??= createTrackers().catch((e) => { trackers = null; throw e; }));

async function createTrackers() {
  const fileset = await FilesetResolver.forVisionTasks("/wasm");
  const opts = (path: string) => ({ baseOptions: { modelAssetPath: path, delegate: "GPU" as const }, runningMode: "VIDEO" as const });
  return Promise.all([
    HandLandmarker.createFromOptions(fileset, { ...opts("/models/hand_landmarker.task"), numHands: 2 }),
    PoseLandmarker.createFromOptions(fileset, opts("/models/pose_landmarker_lite.task")),
    FaceLandmarker.createFromOptions(fileset, {
      ...opts("/models/face_landmarker.task"),
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    }),
  ]);
}

/** 5 decimals: far finer than the trackers' noise, and a frame half the size on the wire (7.3 -> 3.6 KB). */
const r5 = (v: number) => Math.round(v * 1e5) / 1e5;
const r3 = (v: number) => Math.round(v * 1e3) / 1e3; // the face mesh: 9.5 KB a frame instead of 12.3, letters read the same

/** `paused()`: skip detection (the avatar is playing). The three models run on the main thread and starved the
 *  avatar's animation to ~1 frame/s; the backend ignores frames during avatar output anyway. */
export async function startCamera(video: HTMLVideoElement, onFrame: (f: Frame) => void, paused: () => boolean = () => false): Promise<void> {
  const [hands, pose, face] = await loadTrackers();

  // Feed MediaPipe the unmirrored stream; the preview is mirrored with CSS only.
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 }, audio: false });
  video.srcObject = stream;
  video.style.transform = "scaleX(-1)";
  await video.play();

  let last = -1;
  const tick = () => {
    if (video.srcObject !== stream) return; // a fresh feed (startCamera again on this video) has its own loop
    requestAnimationFrame(tick); // first: a detector that throws on one frame must not end tracking for good
    const t = performance.now();
    if (!paused() && video.currentTime !== last) {
      last = video.currentTime;
      const h = hands.detectForVideo(video, t);
      const p = pose.detectForVideo(video, t);
      const fc = face.detectForVideo(video, t);
      if (p.landmarks.length) {
        const frame: Frame = { t, left: null, right: null, pose: p.landmarks[0].map((l) => [r5(l.x), r5(l.y), r5(l.z), r5(l.visibility ?? 0)]), blend: null, head: null };
        h.landmarks.forEach((lm, i) => {
          const pts = lm.map((l) => [r5(l.x), r5(l.y), r5(l.z)] as P3);
          if (h.handedness[i][0].categoryName === "Left") frame.left = pts;
          else frame.right = pts;
        });
        if (fc.faceBlendshapes?.length) {
          frame.blend = Object.fromEntries(fc.faceBlendshapes[0].categories.map((c) => [c.categoryName, r5(c.score)]));
        }
        if (fc.facialTransformationMatrixes?.length) frame.head = matToYPR(fc.facialTransformationMatrixes[0].data);
        if (fc.faceLandmarks?.length) frame.face = FACE_POINTS.map((i) => [r5(fc.faceLandmarks[0][i].x), r5(fc.faceLandmarks[0][i].y)]);
        // the fingerspelling reader standardises over the whole face (without it, Kaggle letters: 37% wrong vs 17%)
        if (fc.faceLandmarks?.length && (frame.left || frame.right)) frame.mesh = fc.faceLandmarks[0].slice(0, 468).map((l) => [r3(l.x), r3(l.y), r3(l.z)] as P3);
        onFrame(frame);
      }
    }
  };
  requestAnimationFrame(tick);
}
