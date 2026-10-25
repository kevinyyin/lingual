// MediaPipe Tasks trackers and their results -> Frame. Shared by the page (camera.ts) and the tracker workers.
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

export type Part = "hands" | "pose" | "face";
export const PARTS: Part[] = ["hands", "pose", "face"];
export interface Trackers { hands?: HandLandmarker; pose?: PoseLandmarker; face?: FaceLandmarker }
export type TrackerMs = Partial<Record<Part, number>>;
/** Metric landmarks for the avatar mirror (mirror.ts), never sent to the backend: pose world points (metres, hip-centred,
 *  with depth) and each hand's world points (metres, around the hand's centre), keyed like Frame.left/right. */
export interface World { pose: number[][] | null; left: number[][] | null; right: number[][] | null }
/** What some of the trackers saw in one picture; `mesh` is kept only when a hand is up, `world` never (toFrame). */
export type Parts = Partial<Omit<Frame, "t">> & { world?: Partial<World> };

/** Parts from trackers that ran apart (in different workers): later ones win, `world` merged key by key. */
export function mergeParts(list: Parts[]): Parts {
  const world = Object.assign({}, ...list.map((p) => p.world ?? {}));
  return Object.assign({}, ...list, { world });
}

export const toWorld = (parts: Parts): World => ({ pose: parts.world?.pose ?? null, left: parts.world?.left ?? null, right: parts.world?.right ?? null });

/** `useModule`: the wasm as an ES module, which a module worker can load (a classic <script> can't be added there). */
/** `cpu`: trackers run on the CPU (XNNPACK) instead of the GPU, which a laptop's integrated GPU runs one at a time. */
export async function createTrackers(useModule = false, parts: Part[] = PARTS, cpu: Part[] = []): Promise<Trackers> {
  // absolute: Vite's dev server adds ?import to a dynamic import("/wasm/…") and then refuses it as a public file
  const fileset = await FilesetResolver.forVisionTasks(new URL("/wasm", self.location.href).href, useModule);
  const opts = (path: string, p: Part) => ({ baseOptions: { modelAssetPath: path, delegate: cpu.includes(p) ? "CPU" as const : "GPU" as const }, runningMode: "VIDEO" as const });
  const make = {
    hands: () => HandLandmarker.createFromOptions(fileset, { ...opts("/models/hand_landmarker.task", "hands"), numHands: 2 }),
    pose: () => PoseLandmarker.createFromOptions(fileset, opts("/models/pose_landmarker_lite.task", "pose")),
    face: () => FaceLandmarker.createFromOptions(fileset, {
      ...opts("/models/face_landmarker.task", "face"),
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    }),
  };
  return Object.fromEntries(await Promise.all(parts.map(async (p) => [p, await make[p]()])));
}

/** 5 decimals: far finer than the trackers' noise, and a frame half the size on the wire (7.3 -> 3.6 KB). */
const r5 = (v: number) => Math.round(v * 1e5) / 1e5;
const r3 = (v: number) => Math.round(v * 1e3) / 1e3; // the face mesh: 9.5 KB a frame instead of 12.3, letters read the same

/** One picture through the given trackers. `ms` gets each one's time. */
export function detect(trackers: Trackers, image: HTMLVideoElement | ImageBitmap, t: number, ms: TrackerMs): Parts {
  const out: Parts = {};
  const timed = <T>(k: Part, run: () => T): T => { const t0 = performance.now(), r = run(); ms[k] = performance.now() - t0; return r; };
  if (trackers.hands) {
    const h = timed("hands", () => trackers.hands!.detectForVideo(image, t));
    out.left = out.right = null;
    const world: Partial<World> = (out.world ??= {});
    world.left = world.right = null;
    h.landmarks.forEach((lm, i) => {
      const pts = lm.map((l) => [r5(l.x), r5(l.y), r5(l.z)] as P3);
      const w = h.worldLandmarks?.[i]?.map((l) => [r5(l.x), r5(l.y), r5(l.z)]) ?? null;
      if (h.handedness[i][0].categoryName === "Left") [out.left, world.left] = [pts, w];
      else [out.right, world.right] = [pts, w];
    });
  }
  if (trackers.pose) {
    const p = timed("pose", () => trackers.pose!.detectForVideo(image, t));
    if (p.landmarks.length) out.pose = p.landmarks[0].map((l) => [r5(l.x), r5(l.y), r5(l.z), r5(l.visibility ?? 0)]);
    (out.world ??= {}).pose = p.worldLandmarks?.[0]?.map((l) => [r5(l.x), r5(l.y), r5(l.z)]) ?? null;
  }
  if (trackers.face) {
    const fc = timed("face", () => trackers.face!.detectForVideo(image, t));
    out.blend = out.head = null;
    if (fc.faceBlendshapes?.length) {
      out.blend = Object.fromEntries(fc.faceBlendshapes[0].categories.map((c) => [c.categoryName, r5(c.score)]));
    }
    if (fc.facialTransformationMatrixes?.length) out.head = matToYPR(fc.facialTransformationMatrixes[0].data);
    if (fc.faceLandmarks?.length) {
      const lm = fc.faceLandmarks[0];
      out.face = FACE_POINTS.map((i) => [r5(lm[i].x), r5(lm[i].y)]);
      out.mesh = lm.slice(0, 468).map((l) => [r3(l.x), r3(l.y), r3(l.z)] as P3);
    }
  }
  return out;
}

/** All trackers' parts -> the Frame sent to the backend; null without a body in the picture. */
export function toFrame(parts: Parts, t: number): Frame | null {
  if (!parts.pose) return null;
  const { mesh, world: _world, ...rest } = parts; // world points stay on the page (the mirror): frames don't grow
  const frame: Frame = { t, left: null, right: null, blend: null, head: null, ...rest, pose: parts.pose };
  // the fingerspelling reader standardises over the whole face (without it, Kaggle letters: 37% wrong vs 17%)
  if (mesh && (frame.left || frame.right)) frame.mesh = mesh;
  return frame;
}
