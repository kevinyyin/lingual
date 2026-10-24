// The base hand of a two-handed sign keeps one handshape. In signs like NICE, COFFEE or SCHOOL one hand acts on the
// other, which stays put in a single shape (ASL's dominance condition); but in the source videos the acting hand covers
// the base hand, and the tracker then guesses its fingers (NICE: 1-4 fingers bent backward from 1.4 to 2.1 s, after
// the dominant hand slid over it). So when a clip loads, for the base hand training/touch_from_asllex marked (`base`: ASL-LEX
// sign type asymmetrical; wrist travel could not tell it: tracker noise made NICE's still hand 0.77 of the other), the most
// typical clean handshape (the medoid of frames with no finger bent backward and not covered) is used for the whole sign,
// each frame keeping its own palm turn unless that frame's hand is mirrored.
import * as THREE from "three";
import { backwardFingers, coveredHand, palm } from "./handModel";
import type { Clip } from "./player";
import { toThree } from "./rig";

const fromThree = (v: THREE.Vector3) => [v.x, -v.y, -v.z]; // back to MediaPipe's axes (rig.toThree's inverse)
const MIN_CLEAN = 3; // clean frames needed to pick a shape

type Side = { world: "leftW" | "rightW"; img: "left" | "right"; wrist: 15 | 16; right: boolean };
const SIDES: Record<"L" | "R", Side> = {
  L: { world: "leftW", img: "left", wrist: 15, right: false },
  R: { world: "rightW", img: "right", wrist: 16, right: true },
};

/** The hand's landmarks in its own palm frame (wrist at the origin, scaled by the wrist -> middle knuckle length). */
function local(pts: THREE.Vector3[], right: boolean) {
  const P = palm(pts[9].clone().sub(pts[0]), pts[5].clone().sub(pts[17]), right);
  const s = Math.max(pts[9].distanceTo(pts[0]), 1e-6);
  return { P, shape: pts.map((p) => { const d = p.clone().sub(pts[0]).divideScalar(s); return [d.dot(P.F), d.dot(P.S), d.dot(P.N)]; }) };
}

const shapeDistance = (a: number[][], b: number[][]) => a.reduce((t, p, i) => t + Math.hypot(p[0] - b[i][0], p[1] - b[i][1], p[2] - b[i][2]), 0) / a.length;

/** The clip with its base hand (if it has one) held in one clean handshape; other clips are returned unchanged. */
export function steadyBaseHand(clip: Clip): Clip {
  const F = clip.frames;
  if (clip.base !== "L" && clip.base !== "R") return clip; // ASL-LEX gives the sign no stationary base hand
  const [base, act] = clip.base === "L" ? [SIDES.L, SIDES.R] : [SIDES.R, SIDES.L];

  // clean frames of the base hand: no finger bent backward, and not covered by the acting hand
  const clean: { i: number; P: ReturnType<typeof palm>; shape: number[][] }[] = [];
  F.forEach((f, i) => {
    const w = f[base.world];
    if (!w) return;
    const pts = w.map(toThree);
    const mine = f[base.img] as number[][] | null, theirs = f[act.img] as number[][] | null;
    if (backwardFingers(pts, base.right) > 0) return;
    if (mine && theirs && coveredHand(mine, theirs, f.pose[base.wrist][2], f.pose[act.wrist][2])) return;
    clean.push({ i, ...local(pts, base.right) });
  });
  if (clean.length < MIN_CLEAN) return clip;
  let best = clean[0], bestSum = Infinity;
  for (const a of clean) {
    const sum = clean.reduce((t, b) => t + shapeDistance(a.shape, b.shape), 0);
    if (sum < bestSum) [best, bestSum] = [a, sum];
  }

  // the palm turn: each frame's own, which the tracker keeps even when it loses the fingers of a covered hand; only a
  // mirrored frame (two or more fingers bent backward: the palm itself flipped) takes the nearest unmirrored frame's
  const turns = F.map((f) => {
    const w = f[base.world];
    if (!w) return null;
    const pts = w.map(toThree);
    return backwardFingers(pts, base.right) >= 2 ? null : local(pts, base.right).P;
  });
  const sound = turns.flatMap((P, i) => (P ? [{ i, P }] : []));
  const frames = F.map((f, i) => {
    const w = f[base.world];
    if (!w || !sound.length) return f;
    const pts = w.map(toThree);
    const P = turns[i] ?? sound.reduce((n, c) => (Math.abs(c.i - i) < Math.abs(n.i - i) ? c : n)).P;
    const s = pts[9].distanceTo(pts[0]) || 0.08;
    const held = best.shape.map(([x, y, z]) => fromThree(pts[0].clone()
      .addScaledVector(P.F, x * s).addScaledVector(P.S, y * s).addScaledVector(P.N, z * s)));
    return { ...f, [base.world]: held };
  });
  return { ...clip, frames };
}
