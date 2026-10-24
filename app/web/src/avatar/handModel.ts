// Anatomical hand: every finger joint gets a frame (bend toward the palm, sideways) and a human range of motion, and
// each finger bone's target direction is clamped into that range before the bone is aimed. MediaPipe's single-camera
// hand depth is a guess: across the 156 sign clips, middle joints were asked to bend backward past 45 deg in 6% of hand
// frames (a mirrored or garbage hand) and hinge joints to bend sideways 30-60 deg; aiming bones at those directions bent
// fingers the way no finger bends. Frames: three.js world space (rig.toThree), angles in degrees.
import * as THREE from "three";

/** bend: [backward (negative), toward the palm]; side: +- sideways (abduction; hinge joints only tolerate noise). */
export interface JointLimit { bend: [number, number]; side: number }

// How a finger moves: only the knuckle where it joins the palm (MCP) has two axes, curling toward the palm and spreading
// sideways; the middle (PIP) and end (DIP) joints are hinges that only curl, in the plane the knuckle sets, and no joint
// twists on its own. So PIP and DIP get no sideways range (MediaPipe's ~10 deg of depth noise there was bending fingers
// sideways). Bending back stays in the finger's plane: the knuckle and end joint give a little (a flat hand pressed
// flat), the middle joint none. Ranges: published
// active motion (MCP -30-90, PIP 0-105, DIP -10-85); the spread narrows as the knuckle curls (a fist's fingers lie
// together), and the end joint follows the middle one (DIP_COUPLE).
export const FINGER: Record<"mcp" | "pip" | "dip", JointLimit> = {
  mcp: { bend: [-30, 90], side: 22 },
  pip: { bend: [0, 105], side: 0 },
  dip: { bend: [-10, 85], side: 0 },
};
/** One tendon curls both outer joints: the end joint stays within these shares of the middle joint's bend (+-DIP_SLACK). */
export const DIP_COUPLE: [number, number] = [0.5, 1.0];
const DIP_SLACK = 8, MCP_SPREAD_CURLED = 0.35; // share of the knuckle's spread left at full curl
// The thumb: the metacarpal swings in a cone (never behind the palm plane, which read as a thumb bent back), its knuckle
// is nearly a hinge, its end joint a hinge; neither bends backward.
export const THUMB = {
  cmc: { across: [-10, 65] as [number, number], palmar: [-5, 55] as [number, number] }, // metacarpal: in the palm plane from along the fingers toward the thumb side / out of it toward the palm
  mcp: { bend: [0, 60], side: 6 } as JointLimit,
  ip: { bend: [0, 80], side: 0 } as JointLimit,
};
const FINGER_JOINTS = [FINGER.mcp, FINGER.pip, FINGER.dip];
const THUMB_JOINTS = [null, THUMB.mcp, THUMB.ip];
const RAD = Math.PI / 180;

/** Palm frame: F wrist -> middle knuckle, S little -> index knuckle (across the palm toward the thumb, made orthogonal
 *  to F), N out of the palm (S x F for a right hand, F x S for a left one; measured: fingers bend toward +N). */
export interface Palm { F: THREE.Vector3; S: THREE.Vector3; N: THREE.Vector3; right: boolean }

export function palm(F: THREE.Vector3, S: THREE.Vector3, right: boolean): Palm {
  const f = F.clone().normalize();
  const s = S.clone().sub(f.clone().multiplyScalar(S.dot(f))).normalize();
  const n = right ? s.clone().cross(f) : f.clone().cross(s);
  return { F: f, S: s, N: n, right };
}

const ortho = (v: THREE.Vector3, r: THREE.Vector3) => v.clone().sub(r.clone().multiplyScalar(v.dot(r))).normalize();

/** Bend and sideways angles of direction d at a joint whose parent points along r, bending toward p (unit, orthogonal to
 *  r), sideways along h. */
export function angles(d: THREE.Vector3, r: THREE.Vector3, p: THREE.Vector3, h: THREE.Vector3): { bend: number; side: number } {
  const x = d.dot(r), y = d.dot(p), z = d.dot(h);
  return { bend: Math.atan2(y, x) / RAD, side: Math.atan2(z, Math.hypot(x, y)) / RAD };
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** d clamped into the joint's range (unit vector) and whether it moved. */
export function limitDir(d: THREE.Vector3, r: THREE.Vector3, p: THREE.Vector3, h: THREE.Vector3, lim: JointLimit) {
  const a = angles(d.clone().normalize(), r, p, h);
  const bend = clamp(a.bend, lim.bend[0], lim.bend[1]), side = clamp(a.side, -lim.side, lim.side);
  if (bend === a.bend && side === a.side) return { dir: d.clone().normalize(), clamped: false, ...a };
  const t = bend * RAD, s = side * RAD;
  const dir = r.clone().multiplyScalar(Math.cos(t)).add(p.clone().multiplyScalar(Math.sin(t))).multiplyScalar(Math.cos(s))
    .add(h.clone().multiplyScalar(Math.sin(s))).normalize();
  return { dir, clamped: true, ...a };
}

/** The thumb metacarpal's direction clamped into its cone in the palm frame. */
export function limitThumbBase(d: THREE.Vector3, P: Palm) {
  const u = d.clone().normalize();
  const across = Math.atan2(u.dot(P.S), u.dot(P.F)) / RAD, palmar = Math.asin(clamp(u.dot(P.N), -1, 1)) / RAD;
  const a = clamp(across, ...THUMB.cmc.across), b = clamp(palmar, ...THUMB.cmc.palmar);
  if (a === across && b === palmar) return { dir: u, clamped: false, across, palmar };
  const dir = P.F.clone().multiplyScalar(Math.cos(a * RAD)).add(P.S.clone().multiplyScalar(Math.sin(a * RAD)))
    .multiplyScalar(Math.cos(b * RAD)).add(P.N.clone().multiplyScalar(Math.sin(b * RAD))).normalize();
  return { dir, clamped: true, across, palmar };
}

/** Joint frame for bone k of a chain: fingers bend toward the palm about a hinge carried down the finger (it swings with
 *  the knuckle's sideways turn, so a spread finger still bends in its own plane); the thumb bends across the palm toward
 *  the little finger (measured on the clips: its mean bend direction was -S with a little -N). */
export function jointFrame(P: Palm, r: THREE.Vector3, hinge: THREE.Vector3 | null, thumb: boolean) {
  if (thumb) {
    const p = ortho(P.S.clone().negate().addScaledVector(P.N, -0.3), r);
    return { p, h: r.clone().cross(p).normalize() };
  }
  const h = ortho(hinge ?? P.S, r);
  const p = h.clone().cross(r).multiplyScalar(P.right ? 1 : -1).normalize(); // at r = F: p = N
  return { p, h };
}

/** The rotation carrying the hinge from parent direction a to child direction b (the swing between them). */
export const carry = (hinge: THREE.Vector3, a: THREE.Vector3, b: THREE.Vector3) =>
  hinge.clone().applyQuaternion(new THREE.Quaternion().setFromUnitVectors(a.clone().normalize(), b.clone().normalize()));

/** Limits for bone k (0 = from the knuckle) of chain c (0 = thumb). */
export const limitFor = (c: number, k: number): JointLimit | null => (c === 0 ? THUMB_JOINTS[k] : FINGER_JOINTS[k]);

/** Chain c's bone k: the direction it may take, given where its parent bone points (r) and the carried hinge.
 *  `parentBend`: the middle joint's bend (degrees), which the end joint of a finger then follows (DIP_COUPLE).
 *  Returns the clamped direction, its bend, and the hinge to carry to the next bone. */
export function constrainBone(P: Palm, c: number, k: number, r: THREE.Vector3, hinge: THREE.Vector3 | null, want: THREE.Vector3, parentBend?: number) {
  if (c === 0 && k === 0) {
    const out = limitThumbBase(want, P);
    return { ...out, bendOut: 0, hinge: null, outBy: Math.max(0, out.across - THUMB.cmc.across[1], THUMB.cmc.across[0] - out.across, out.palmar - THUMB.cmc.palmar[1], THUMB.cmc.palmar[0] - out.palmar) };
  }
  let lim = limitFor(c, k)!;
  const { p, h } = jointFrame(P, r, hinge, c === 0);
  if (c > 0 && k === 0) { // the spread narrows as the knuckle curls
    const curl = clamp(angles(want.clone().normalize(), r, p, h).bend / lim.bend[1], 0, 1);
    lim = { ...lim, side: lim.side * (1 - (1 - MCP_SPREAD_CURLED) * curl) };
  }
  if (c > 0 && k === 2 && parentBend !== undefined) {
    const lo = Math.max(lim.bend[0], DIP_COUPLE[0] * parentBend - DIP_SLACK), hi = Math.min(lim.bend[1], DIP_COUPLE[1] * parentBend + DIP_SLACK);
    lim = { ...lim, bend: [Math.min(lo, hi), hi] };
  }
  const out = limitDir(want, r, p, h, lim);
  const outBy = Math.max(0, out.bend - lim.bend[1], lim.bend[0] - out.bend, Math.abs(out.side) - lim.side);
  return { ...out, bendOut: angles(out.dir, r, p, h).bend, hinge: c === 0 ? null : carry(h, r, out.dir), outBy };
}

/** Relaxed hand at rest (degrees): thumb [across, out of the palm plane], then each joint's curl, knuckle out. Fingers
 *  curl a little more toward the little finger, as a hanging hand's do. */
export const RELAXED = { thumb: [25, 30, 12, 12], fingers: [[15, 22, 12], [20, 26, 15], [24, 30, 17], [28, 32, 19]] };

/** Chain c's bone k in the relaxed hand: its direction given its parent's (r) and the carried hinge (as constrainBone). */
export function relaxedDir(P: Palm, c: number, k: number, r: THREE.Vector3, hinge: THREE.Vector3 | null) {
  if (c === 0 && k === 0) {
    const [a, b] = RELAXED.thumb.map((x) => x * RAD);
    const dir = P.F.clone().multiplyScalar(Math.cos(a)).add(P.S.clone().multiplyScalar(Math.sin(a))).multiplyScalar(Math.cos(b))
      .add(P.N.clone().multiplyScalar(Math.sin(b))).normalize();
    return { dir, hinge: null };
  }
  const b = c === 0 ? RELAXED.thumb[k + 1] : RELAXED.fingers[c - 1][k];
  const { p, h } = jointFrame(P, r, hinge, c === 0);
  const dir = limitDir(r, r, p, h, { bend: [b, b], side: 0 }).dir;
  return { dir, hinge: c === 0 ? null : carry(h, r, dir) };
}

/** A hand the tracker got wrong: two or more middle finger joints bent backward past 45 deg (a mirrored or garbage
 *  hand; 6.2% of the clips' hand frames). Its rotation and finger shape are not used: the last good ones are held. */
export function implausibleHand(pts: THREE.Vector3[], right: boolean): boolean {
  return backwardFingers(pts, right) >= 2;
}

/** How many fingers' middle joints the landmarks bend backward past 45 deg (no finger can: tracker error). */
export function backwardFingers(pts: THREE.Vector3[], right: boolean): number {
  const P = palm(pts[9].clone().sub(pts[0]), pts[5].clone().sub(pts[17]), right);
  let back = 0;
  for (const [m, p, d] of [[5, 6, 7], [9, 10, 11], [13, 14, 15], [17, 18, 19]]) {
    const r = pts[p].clone().sub(pts[m]).normalize(), dd = pts[d].clone().sub(pts[p]).normalize();
    const { p: pd, h } = jointFrame(P, r, null, false);
    if (angles(dd, r, pd, h).bend < -45) back++;
  }
  return back;
}

/** Whether a hand is covered by the other one in the image: their 2D boxes overlap by more than 15% of the smaller
 *  box and it is the farther of the two (pose wrist z: MediaPipe's z grows away from the camera). In NICE the dominant
 *  hand slides over the base hand, whose fingers the tracker then guesses (8.6% of the clips' hand frames). */
export function coveredHand(mine: number[][], theirs: number[][], myWristZ: number, theirWristZ: number): boolean {
  if (myWristZ <= theirWristZ) return false;
  const box = (h: number[][]) => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of h) [x0, y0, x1, y1] = [Math.min(x0, x), Math.min(y0, y), Math.max(x1, x), Math.max(y1, y)];
    return [x0, y0, x1, y1];
  };
  const a = box(mine), b = box(theirs);
  const inter = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const small = Math.min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]));
  return small > 0 && inter / small > 0.15;
}

/** Closest points between segments p1-q1 and p2-q2 and where they lie along each (0-1) (Ericson, Real-Time Collision
 *  Detection 5.1.9). */
export function closestPoints(p1: THREE.Vector3, q1: THREE.Vector3, p2: THREE.Vector3, q2: THREE.Vector3) {
  const d1 = q1.clone().sub(p1), d2 = q2.clone().sub(p2), r = p1.clone().sub(p2);
  const a = d1.lengthSq(), e = d2.lengthSq(), f = d2.dot(r), EPS = 1e-12;
  let s = 0, t = 0;
  if (a <= EPS && e > EPS) t = Math.min(Math.max(f / e, 0), 1);
  else if (a > EPS) {
    const c = d1.dot(r);
    if (e <= EPS) s = Math.min(Math.max(-c / a, 0), 1);
    else {
      const b = d1.dot(d2), den = a * e - b * b;
      s = den > EPS ? Math.min(Math.max((b * f - c * e) / den, 0), 1) : 0;
      t = (b * s + f) / e;
      if (t < 0) { t = 0; s = Math.min(Math.max(-c / a, 0), 1); }
      else if (t > 1) { t = 1; s = Math.min(Math.max((b - c) / a, 0), 1); }
    }
  }
  return { c1: p1.clone().addScaledVector(d1, s), c2: p2.clone().addScaledVector(d2, t), s, t };
}

export function segmentDistance(p1: THREE.Vector3, q1: THREE.Vector3, p2: THREE.Vector3, q2: THREE.Vector3): number {
  const { c1, c2 } = closestPoints(p1, q1, p2, q2);
  return c1.distanceTo(c2);
}

/** Overlap of two capsules: depth (metres, > 0 when they intersect, less `slack`), the unit direction pushing x out of
 *  y, and where along x it happens. Coincident axes push along `fallback`. */
export function capsuleContact(x: Capsule, y: Capsule, slack = 0.003, fallback = new THREE.Vector3(0, 0, 1)) {
  const { c1, c2, s } = closestPoints(x.a, x.b, y.a, y.b);
  const sep = c1.clone().sub(c2), d = sep.length();
  return { depth: x.r + y.r - slack - d, dir: d > 1e-6 ? sep.divideScalar(d) : fallback.clone().normalize(), s };
}

/** A finger bone as a capsule (segment + radius, metres); `bone` = its index in the finger (0 = from the knuckle). */
export interface Capsule { a: THREE.Vector3; b: THREE.Vector3; r: number; finger: number; bone?: number }

/** Deepest overlap between capsules of different fingers of one hand (metres; 0 = none). Proximal bones start 40% along
 *  (at the knuckles neighbouring fingers touch by design); `slack` absorbs soft-tissue contact. */
export function selfOverlap(caps: Capsule[], slack = 0.003): number {
  let worst = 0;
  for (let i = 0; i < caps.length; i++) for (let j = i + 1; j < caps.length; j++) {
    if (caps[i].finger === caps[j].finger) continue;
    const d = segmentDistance(caps[i].a, caps[i].b, caps[j].a, caps[j].b);
    worst = Math.max(worst, caps[i].r + caps[j].r - slack - d);
  }
  return worst;
}

/** Deepest overlap between two hands' capsules (metres; 0 = none). */
export function crossOverlap(a: Capsule[], b: Capsule[], slack = 0.003): number {
  let worst = 0;
  for (const x of a) for (const y of b) worst = Math.max(worst, x.r + y.r - slack - segmentDistance(x.a, x.b, y.a, y.b));
  return worst;
}
