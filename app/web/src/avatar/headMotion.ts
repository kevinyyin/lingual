import * as THREE from "three";
import { QuatFilter } from "./rotationFilter";

const RAD = Math.PI / 180, DEG = 180 / Math.PI;
const TURN_YAW = 4, TURN_PITCH = 3; // degrees toward the working hand
const DRIFT: [number, number, number][] = [[0.11, 0.5, 0], [0.17, 0.5, 1.7], [0.23, 0.5, 3.1]]; // Hz, degrees, phase: sums to 1.5 per axis
const AXIS_PHASE = [0, 2.1, 4.2];
const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

/** Head motion of a signer at rest: a slow drift (<= 1.5 deg, <= 0.23 Hz) plus a turn of up to 4 deg yaw and 3 deg
 *  pitch toward the more active (higher) wrist, eased by its own filter. Far below the grammar nod (8) and shake (12). */
export function createHeadMotion() {
  const filter = new QuatFilter(0.8, 0, 1);
  const q = new THREE.Quaternion(), e = new THREE.Euler(0, 0, 0, "YXZ");
  let last: number | null = null, clock = 0; // the filter's clock: an idle gap (> 200 ms) counts as one 60 fps frame, so it eases, not snaps
  return (tMs: number, wristL: THREE.Vector3 | null, wristR: THREE.Vector3 | null, nose: THREE.Vector3) => {
    const t = tMs / 1000;
    const drift = AXIS_PHASE.map((ph) => DRIFT.reduce((s, [hz, amp, p]) => s + amp * Math.sin(2 * Math.PI * hz * t + p + ph), 0));
    let yaw = 0, pitch = 0, best = 0;
    for (const w of [wristL, wristR]) {
      if (!w || !Number.isFinite(w.x + w.y + w.z)) continue;
      const a = clamp((w.y - nose.y + 0.5) / 0.5, 0, 1); // 0 with the hand hanging, 1 at the nose
      if (a <= best) continue;
      best = a;
      yaw = TURN_YAW * a * clamp((w.x - nose.x) / 0.3, -1, 1);
      pitch = TURN_PITCH * a * clamp((nose.y - w.y) / 0.3, 0, 1); // looks down at a hand below the face
    }
    e.set((drift[0] + pitch) * RAD, (drift[1] + yaw) * RAD, drift[2] * RAD);
    clock = last === null ? tMs : clock + (tMs - last > 200 ? 1000 / 60 : tMs - last);
    last = tMs;
    const out = filter.filter(q.setFromEuler(e), clock);
    e.setFromQuaternion(out, "YXZ");
    return { pitchDeg: e.x * DEG, yawDeg: e.y * DEG, rollDeg: e.z * DEG };
  };
}
