// MediaPipe world landmarks' x and depth disagree with the image for some clips (GRANDMOTHER: thumb at the chin in the
// image, wrist in front of the shoulder in the world pose). Near the face the image is the better guide to where the hand
// is across and up; depth stays from the world pose. Mirrors training/audit_placement.py.
import * as THREE from "three";
import type { ClipFrame } from "./player";

const mid = (p: number[][], a: number, b: number) => [(p[a][0] + p[b][0]) / 2, (p[a][1] + p[b][1]) / 2];

/** Image W/H from one frame: (|dy2d|/|dyw|) / (|dx2d|/|dxw|), dx the shoulder width, dy shoulders to hips. NaN if degenerate. */
export function frameAspect(pose: number[][], pose2d: number[][]): number {
  const dxw = Math.abs(pose[11][0] - pose[12][0]), dx2 = Math.abs(pose2d[11][0] - pose2d[12][0]);
  const dyw = Math.abs(mid(pose, 11, 12)[1] - mid(pose, 23, 24)[1]), dy2 = Math.abs(mid(pose2d, 11, 12)[1] - mid(pose2d, 23, 24)[1]);
  return dy2 / dyw / (dx2 / dxw);
}

export const medianAspect = (xs: number[]): number | undefined => {
  const v = xs.filter((x) => Number.isFinite(x) && x > 0).sort((a, b) => a - b);
  return v.length ? (v[(v.length - 1) >> 1] + v[v.length >> 1]) / 2 : undefined;
};

/** Median aspect over a clip's frames (undefined if none is usable: no correction then). */
export const estimateAspect = (frames: ClipFrame[]): number | undefined => medianAspect(frames.map((f) => frameAspect(f.pose, f.pose2d)));

/** Wrist in world-pose coordinates: across/up from the aspect-corrected image, as an offset from the nose scaled by world
 *  over image shoulder width; depth from the world landmark. */
export function wristFrom2d(pose: number[][], pose2d: number[][], wrist: 15 | 16, aspect: number): THREE.Vector3 {
  const sw2 = Math.abs(pose2d[11][0] - pose2d[12][0]) * aspect; // image height units
  if (!(sw2 > 1e-6)) return new THREE.Vector3(pose[wrist][0], pose[wrist][1], pose[wrist][2]); // no usable image shoulders: the world wrist
  const k = Math.abs(pose[11][0] - pose[12][0]) / sw2;
  const [n, w] = [pose2d[0], pose2d[wrist]];
  return new THREE.Vector3(pose[0][0] + (w[0] - n[0]) * aspect * k, pose[0][1] + (w[1] - n[1]) * k, pose[wrist][2]);
}
