// Motion quality per frame, for before/after comparison of the arm solver (bench.ts). Definitions: spike = a bone turning
// over 25 degrees between consecutive 60 fps frames; backward elbow = flexion below -5; twist out of range = beyond 90.
import * as THREE from "three";
import type { ArmState } from "./rocketbox";

export type Arms = Record<"L" | "R", ArmState>;
export const SPIKE_DEG = 25, BACKWARD_DEG = -5, TWIST_MAX_DEG = 90;

/** Closest distance between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection 5.1.9). */
function segmentDistance(p1: THREE.Vector3, q1: THREE.Vector3, p2: THREE.Vector3, q2: THREE.Vector3): number {
  const d1 = q1.clone().sub(p1), d2 = q2.clone().sub(p2), r = p1.clone().sub(p2);
  const a = d1.lengthSq(), e = d2.lengthSq(), f = d2.dot(r), EPS = 1e-12;
  let s = 0, t = 0;
  if (a <= EPS && e <= EPS) return r.length();
  if (a <= EPS) t = Math.min(Math.max(f / e, 0), 1);
  else {
    const c = d1.dot(r);
    if (e <= EPS) s = Math.min(Math.max(-c / a, 0), 1);
    else {
      const b = d1.dot(d2), den = a * e - b * b;
      s = den > EPS ? Math.min(Math.max((b * f - c * e) / den, 0), 1) : 0; // parallel: any s, pick the start
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = Math.min(Math.max(-c / a, 0), 1); }
      else if (t > 1) { t = 1; s = Math.min(Math.max((b - c) / a, 0), 1); }
    }
  }
  return p1.clone().addScaledVector(d1, s).distanceTo(p2.clone().addScaledVector(d2, t));
}

export function frameMetrics(prev: Arms | null, cur: Arms) {
  let maxStepDeg = 0;
  if (prev) for (const s of ["L", "R"] as const) for (const b of ["upper", "fore", "hand"] as const) {
    maxStepDeg = Math.max(maxStepDeg, (prev[s][b].angleTo(cur[s][b]) * 180) / Math.PI);
  }
  return {
    maxStepDeg,
    backwardElbow: cur.L.elbowDeg < BACKWARD_DEG || cur.R.elbowDeg < BACKWARD_DEG,
    twistOut: Math.abs(cur.L.twistDeg) > TWIST_MAX_DEG || Math.abs(cur.R.twistDeg) > TWIST_MAX_DEG,
    armGap: segmentDistance(cur.L.elbow, cur.L.wrist, cur.R.elbow, cur.R.wrist),
  };
}

const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b), m = s.length >> 1;
  return !s.length ? 0 : s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};

/** Median wrist speed and jerkiness (mm per frame) within `windowMs` of a sign edge and elsewhere: speed is half the
 *  distance between the frames either side, jerkiness the length of P[i+1] - 2 P[i] + P[i-1]. */
export function edgeMotion(pts: THREE.Vector3[], times: number[], edges: number[], windowMs = 40) {
  const edge: number[][] = [[], []], inner: number[][] = [[], []]; // [speeds, jerks]
  for (let i = 1; i < pts.length - 1; i++) {
    const set = edges.some((e) => Math.abs(times[i] - e) < windowMs) ? edge : inner;
    set[0].push((pts[i + 1].distanceTo(pts[i - 1]) / 2) * 1000);
    set[1].push(pts[i + 1].clone().sub(pts[i]).sub(pts[i]).add(pts[i - 1]).length() * 1000);
  }
  return { edgeSpeed: median(edge[0]), edgeJerk: median(edge[1]), innerSpeed: median(inner[0]), innerJerk: median(inner[1]) };
}

/** The elbow's hinge axis in the world bind pose: normal of the plane the bent bind arm lies in, signed so a positive turn
 *  folds the forearm toward the front (+z, the way a human elbow bends). Collinear (straight) bind arms fold toward +z too. */
export function hingeAxis(upperDir: THREE.Vector3, foreDir: THREE.Vector3): THREE.Vector3 {
  const du = upperDir.clone().normalize();
  let h = du.clone().cross(foreDir.clone().normalize());
  if (h.length() < 1e-3) h = du.clone().cross(new THREE.Vector3(0, 0, 1));
  h.normalize();
  return h.clone().cross(du).z < 0 ? h.negate() : h; // a positive turn moves du toward +z
}

/** Elbow flexion in degrees: the full angle between the upper-arm and forearm directions, negative when the forearm lies
 *  on the backward side of the hinge. */
export function elbowFlexion(upperDir: THREE.Vector3, foreDir: THREE.Vector3, hinge: THREE.Vector3): number {
  const c = upperDir.clone().cross(foreDir);
  const deg = (Math.atan2(c.length(), upperDir.dot(foreDir)) * 180) / Math.PI;
  return c.dot(hinge) < 0 ? -deg : deg;
}
