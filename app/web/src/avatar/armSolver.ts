// Per-frame arm solver with human joint limits and frame-to-frame continuity. Pure: it reads only the upper arm's parent
// world rotation and returns local rotations; the caller copies them onto the bones.
//   Hinge convention: ArmRig.hinge is the elbow's bend axis in the upper arm's bone-local frame, from metrics.hingeAxis on
//   the bind pose, so a positive turn about it folds the forearm forward; flexion is the signed angle about it (0 = straight).
//   Twist convention: forearm twist = signed angle of twistAbout(fore.bind^-1 * fore, fore.axis), as rocketbox armState measures it.
import * as THREE from "three";
import { hingeAxis } from "./metrics";
import { frameRotation, type Joint, parentWorld, twistAbout, twoBoneElbow } from "./rig";
import { QuatFilter } from "./rotationFilter";

/** `palm` (hand-local): wrist -> middle knuckle, little -> index knuckle, and the side; with it the wrist gets its
 *  anatomical ranges (ArmLimits.wrist), without it a cone of wristMaxDeg. */
export interface ArmRig { upper: Joint; fore: Joint; hand: Joint; l1: number; l2: number; hinge: THREE.Vector3; palm?: { f: THREE.Vector3; s: THREE.Vector3; right: boolean } }
/** wrist (degrees): bend toward the palm (flex) and back (ext), tilt toward the thumb (radial) and the little finger
 *  (ulnar). A wrist tilts far less than it bends, and further toward the little finger than the thumb. */
export interface ArmLimits { elbowMaxDeg: number; twistMaxDeg: number; wristMaxDeg: number; wrist?: { flex: number; ext: number; radial: number; ulnar: number } }
// wrist: published active ranges (flexion 80, extension 70, radial 20, ulnar 35); 10 less bend back put THANK-YOU's
// fingertips 44 mm into the chin
export const DEFAULT_LIMITS: ArmLimits = { elbowMaxDeg: 150, twistMaxDeg: 90, wristMaxDeg: 70, wrist: { flex: 80, ext: 70, radial: 20, ulnar: 35 } };
export interface ArmTarget { shoulder: THREE.Vector3; elbowHint: THREE.Vector3; wrist: THREE.Vector3; handRot: THREE.Quaternion | null }
/** `twistDemand`: the forearm twist the hand asked for (radians), unclamped and unwrapped across frames; the next frame
 *  unwraps against it, so a demand held past the limit never wraps to the other side. */
export interface ArmPose { upper: THREE.Quaternion; fore: THREE.Quaternion; hand: THREE.Quaternion; pole: THREE.Vector3; twistDemand: number;
  tMs?: number } // tMs: when stepArm produced it (its turn-rate cap)

/** Reach beyond (1 - SOFT) of full length eases asymptotically into extension ("soft IK"), so the elbow never snaps
 *  straight: hard two-bone IK turns the elbow at an unbounded rate as the arm straightens. */
const SOFT = 0.1;
const RAD = Math.PI / 180;
const POLE_FIRM = 0.05, POLE_MIN = 0.01, POLE_STEP = 10 * RAD; // hint off-line distances (m) and max turn per call
const DOWN = new THREE.Vector3(0, -1, 0), BACK = new THREE.Vector3(0, 0, -1);

/** Build the solver's rig from three joints; call while the skeleton is in its bind pose. */
export function armRig(upper: Joint, fore: Joint, hand: Joint, palm?: ArmRig["palm"]): ArmRig {
  const pos = (o: THREE.Object3D) => (o.updateWorldMatrix(true, false), o.getWorldPosition(new THREE.Vector3()));
  const s = pos(upper.bone), e = pos(fore.bone), w = pos(hand.bone);
  const hinge = hingeAxis(e.clone().sub(s), w.clone().sub(e)).applyQuaternion(upper.bone.getWorldQuaternion(new THREE.Quaternion()).invert());
  return { upper, fore, hand, l1: s.distanceTo(e), l2: e.distanceTo(w), hinge, palm };
}

/** The wrist's swing D (a rotation in the forearm's frame applied before the hand's bind rotation) kept in range: with
 *  the palm frame, bend and tilt of the fingers' direction are clamped separately (flexion toward the palm normal, tilt
 *  toward the thumb side); the correction turns the clamped direction straight back, keeping D's roll. */
function clampWrist(D: THREE.Quaternion, rig: ArmRig, lim: ArmLimits): THREE.Quaternion {
  if (D.w < 0) D.set(-D.x, -D.y, -D.z, -D.w);
  if (!rig.palm || !lim.wrist) {
    const swing = 2 * Math.acos(Math.min(D.w, 1)), sMax = lim.wristMaxDeg * RAD;
    return swing > sMax ? new THREE.Quaternion().slerp(D, sMax / swing) : D;
  }
  const F0 = rig.palm.f.clone().applyQuaternion(rig.hand.bind).normalize();
  const S0 = rig.palm.s.clone().applyQuaternion(rig.hand.bind);
  S0.addScaledVector(F0, -S0.dot(F0)).normalize();
  const N0 = rig.palm.right ? S0.clone().cross(F0) : F0.clone().cross(S0); // toward the palm (handModel.palm)
  const F1 = F0.clone().applyQuaternion(D);
  const flex = Math.atan2(F1.dot(N0), F1.dot(F0)), tilt = Math.asin(Math.min(1, Math.max(-1, F1.dot(S0))));
  const w = lim.wrist;
  const f = Math.min(Math.max(flex, -w.ext * RAD), w.flex * RAD), t = Math.min(Math.max(tilt, -w.ulnar * RAD), w.radial * RAD);
  if (f === flex && t === tilt) return D;
  const F2 = F0.clone().multiplyScalar(Math.cos(f)).addScaledVector(N0, Math.sin(f)).multiplyScalar(Math.cos(t)).addScaledVector(S0, Math.sin(t)).normalize();
  return new THREE.Quaternion().setFromUnitVectors(F1, F2).multiply(D);
}

/** Signed angle (-pi, pi] of a rotation about unit `ax` (q assumed to be a twist about ax). */
function angleAbout(q: THREE.Quaternion, ax: THREE.Vector3): number {
  const a = 2 * Math.atan2(q.x * ax.x + q.y * ax.y + q.z * ax.z, q.w);
  return Math.atan2(Math.sin(a), Math.cos(a));
}
const wrapPi = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
const offAxis = (p: THREE.Vector3, u: THREE.Vector3) => p.addScaledVector(u, -p.dot(u));
const sameHemisphere = (q: THREE.Quaternion, ref: THREE.Quaternion | undefined) =>
  ref && q.dot(ref) < 0 ? q.set(-q.x, -q.y, -q.z, -q.w) : q;

export function solveArmPose(rig: ArmRig, t: ArmTarget, prev: ArmPose | null, lim: ArmLimits = DEFAULT_LIMITS): ArmPose {
  const { l1, l2 } = rig, reach = l1 + l2, s = t.shoulder;
  const st = t.wrist.clone().sub(s), d = st.length();
  const u = d > 1e-9 ? st.divideScalar(d) : DOWN.clone();

  // 1. bend-plane pole from the elbow hint's offset off the shoulder-wrist line: >= 5 cm is trusted outright (even a flip),
  //    1-5 cm turns last frame's pole toward it by at most 10 degrees, under 1 cm keeps last frame's
  const pole = offAxis(t.elbowHint.clone().sub(s), u), off = pole.length();
  const carried = prev ? offAxis(prev.pole.clone(), u) : null;
  const hasCarried = !!carried && carried.lengthSq() > 1e-12;
  if (off < POLE_MIN) pole.copy(hasCarried ? carried! : offAxis(DOWN.clone(), u));
  else if (off < POLE_FIRM && hasCarried) {
    const c = carried!.normalize(), h = pole.divideScalar(off);
    const a = Math.atan2(c.clone().cross(h).dot(u), c.dot(h)); // signed angle about u; exactly opposite turns +
    pole.copy(c).applyAxisAngle(u, Math.min(Math.max(a, -POLE_STEP), POLE_STEP));
  }
  if (pole.lengthSq() < 1e-12) offAxis(pole.copy(BACK), u); // arm along the vertical: bend backward
  pole.normalize();

  // 2. elbow, with the wrist kept within the elbow limit (closest) and eased into full extension (farthest)
  const dMin = Math.sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * Math.cos(lim.elbowMaxDeg * RAD));
  const ds = (1 - SOFT) * reach, k = SOFT * reach;
  const dr = d < dMin ? dMin : d > ds ? ds + k * (1 - Math.exp(-(d - ds) / k)) : d;
  const wr = s.clone().addScaledVector(u, dr);
  const elbow = twoBoneElbow(s, pole.clone().add(s), wr, l1, l2);
  const du = elbow.clone().sub(s).normalize(), df = wr.sub(elbow).normalize();

  // 3. upper arm: bone axis onto the elbow direction, hinge onto the bend-plane normal (fixes the roll)
  const n = pole.clone().cross(u); // unit: pole is a unit vector perpendicular to u; positive turns fold toward -pole
  const Wu = frameRotation(rig.upper.axis, rig.hinge, du, n);
  const upper = parentWorld(rig.upper.bone).invert().multiply(Wu);

  // 4. forearm: turn about the hinge from its bind flexion to the clamped flexion
  const flex = Math.min(Math.max(Math.atan2(du.clone().cross(df).dot(n), du.dot(df)), 0), lim.elbowMaxDeg * RAD);
  const f0 = rig.fore.axis.clone().applyQuaternion(rig.fore.bind);
  const flex0 = Math.atan2(rig.upper.axis.clone().cross(f0).dot(rig.hinge), rig.upper.axis.dot(f0));
  const fore0 = new THREE.Quaternion().setFromAxisAngle(rig.hinge, flex - flex0).multiply(rig.fore.bind);

  // 5. hand: wrist swing clamped; its twist goes to the forearm, unwrapped against last frame's demand, then clamped
  const ax = rig.fore.axis, tMax = lim.twistMaxDeg * RAD;
  let demand = prev ? prev.twistDemand : 0;
  if (t.handRot && Math.abs(demand) > Math.PI + tMax) demand = wrapPi(demand); // tracked and wound past any forearm's reach: unwind (stepArm's turn cap spreads it)
  const clampT = (x: number) => Math.min(Math.max(x, -tMax), tMax);
  let tau = clampT(demand);
  if (!t.handRot) demand = tau; // untracked: re-anchor on the twist the forearm holds, so it does not move
  let hand = rig.hand.bind.clone();
  if (t.handRot) {
    const M = Wu.multiply(fore0).invert().multiply(t.handRot).multiply(rig.hand.bind.clone().invert()); // in forearm frame
    const T = twistAbout(M, ax), S = M.multiply(T.clone().invert());
    const raw = angleAbout(T, ax);
    const step = wrapPi(raw - demand);
    demand = Math.abs(step) < Math.PI / 2 ? demand + step : raw; // a jump (a palm flip) is no turn to follow: no ratchet
    tau = clampT(demand);
    const Tc = new THREE.Quaternion().setFromAxisAngle(ax, tau);
    hand = clampWrist(Tc.clone().invert().multiply(S).multiply(Tc), rig, lim).multiply(rig.hand.bind);
  }
  const fore = fore0.multiply(new THREE.Quaternion().setFromAxisAngle(ax, tau));

  // 6. stay in the previous frame's hemisphere so later slerps and filters never take the long way
  return {
    upper: sameHemisphere(upper.normalize(), prev?.upper),
    fore: sameHemisphere(fore.normalize(), prev?.fore),
    hand: sameHemisphere(hand.normalize(), prev?.hand),
    pole,
    twistDemand: demand,
  };
}

/** A hand's ease from frame to frame: last frame's k, and the share of its turn made (`w`). */
export interface HandEase { k: number; w: number }
/** Within this of half a turn, which way a hand's twist goes from the forearm's is a toss: HELLO with its old trim appeared
 *  at 180 degrees (the clip's raised hand on an arm still at rest), then turned on to -70. Wider waited on today's HELLO (165). */
const TOSS = 10 * RAD;

/** A hand appearing in the video (`rising`) or leaving it turns in or out with the arm by k (0 the relaxed hand at the
 *  forearm's twist .. 1 the tracked one, `seen`), the short way round. A hand appearing half a turn from relaxed made
 *  that way a toss: noise picked it, and the losing side wound the forearm past its reach until it snapped back (HELLO).
 *  So the turn's share advances only as fast as the twist the hand asks of the forearm is clearly not half a turn: it
 *  waits out a toss, and anything else eases by k as before. `relaxed`: this frame's solve with no hand target. Returns
 *  the hand rotation to solve for and the ease to carry. */
export function easeHand(rig: ArmRig, relaxed: ArmPose, seen: THREE.Quaternion, k: number, rising: boolean, ease: HandEase | null): { rot: THREE.Quaternion; ease: HandEase } {
  const W = parentWorld(rig.upper.bone).multiply(relaxed.upper).multiply(relaxed.fore), ax = rig.fore.axis; // the forearm, at its twist
  let w = ease ? (ease.w * k) / ease.k : k; // leaving: the share falls with k
  if (rising) {
    const was = ease ?? { k: 0, w: 0 }, turn = W.clone().invert().multiply(seen).multiply(rig.hand.bind.clone().invert());
    const c = Math.min(Math.max((Math.PI - Math.abs(angleAbout(twistAbout(turn, ax), ax))) / TOSS, 0), 1);
    w = was.w + (c * (k - was.k) * (1 - was.w)) / (1 - was.k); // w = k while c = 1
  }
  return { rot: W.multiply(relaxed.hand).slerp(seen, w), ease: { k, w } };
}

/** One-euro settings for the arm bones, tuned on the motion bench at 60 fps (spot-checked at 30) together with
 *  TURN_MAX_DEG_S: with the cap stopping jumps, (1.5, 0.5) and (3, 0.5) both gave 0 spikes and contact 0.144 m; the
 *  heavier one cost round-trip accuracy (top-1 .067-.080 vs .073), so the lighter keeps signs crisper. */
export const FILTER_HZ = 3, FILTER_BETA = 0.5;
/** Cap on how fast any arm bone turns (degrees/s, world, i.e. under the shoulder's parent). The one-euro filter alone
 *  either let 60-110 degree target jumps (a hand appearing, a contact correction kicking in, MediaPipe flipping the
 *  palm) through as spikes or, tuned heavy enough to stop them, lagged contacts by 5+ cm; the cap stops the jumps and
 *  leaves the filter light. 1200/s = 20 degrees per 60 fps frame, under the 25-degree spike gate (1500/s). */
export const TURN_MAX_DEG_S = 1200;
export interface ArmFilters { upper: QuatFilter; fore: QuatFilter; hand: QuatFilter }
export const armFilters = (): ArmFilters => ({ upper: new QuatFilter(FILTER_HZ, FILTER_BETA), fore: new QuatFilter(FILTER_HZ, FILTER_BETA), hand: new QuatFilter(FILTER_HZ, FILTER_BETA) });

const finiteQ = (q: THREE.Quaternion) => Number.isFinite(q.x + q.y + q.z + q.w);

/** Forearm local rotation as (bend about the hinge from bind, twist about its own axis): the solver's two degrees of
 *  freedom. Off-hinge parts of q (a slerp of two poses leaves some) are dropped. */
function foreParams(rig: ArmRig, q: THREE.Quaternion): [number, number] {
  const b = rig.fore.axis.clone().applyQuaternion(rig.fore.bind), d = rig.fore.axis.clone().applyQuaternion(q);
  const a = Math.atan2(b.clone().cross(d).dot(rig.hinge), b.dot(d));
  const T = new THREE.Quaternion().setFromAxisAngle(rig.hinge, a).multiply(rig.fore.bind).invert().multiply(q);
  return [a, angleAbout(twistAbout(T, rig.fore.axis), rig.fore.axis)];
}
const foreFrom = (rig: ArmRig, a: number, tau: number) => new THREE.Quaternion().setFromAxisAngle(rig.hinge, a)
  .multiply(rig.fore.bind).multiply(new THREE.Quaternion().setFromAxisAngle(rig.fore.axis, tau));

/** Back onto the joint model and inside the limits: forearm = hinge bend (flexion 0..max) + twist (clamped), hand = a
 *  pure swing (its twist is the forearm's) within the wrist limit. A no-op on solveArmPose output. */
function project(rig: ArmRig, p: ArmPose, lim: ArmLimits) {
  const f0 = rig.fore.axis.clone().applyQuaternion(rig.fore.bind);
  const flex0 = Math.atan2(rig.upper.axis.clone().cross(f0).dot(rig.hinge), rig.upper.axis.dot(f0));
  const [a, tau] = foreParams(rig, p.fore), tMax = lim.twistMaxDeg * RAD;
  p.fore.copy(foreFrom(rig, Math.min(Math.max(a, -flex0), lim.elbowMaxDeg * RAD - flex0), Math.min(Math.max(tau, -tMax), tMax)));
  const delta = p.hand.clone().multiply(rig.hand.bind.clone().invert());
  const S = delta.multiply(twistAbout(delta, rig.fore.axis).invert());
  p.hand.copy(clampWrist(S, rig, lim).multiply(rig.hand.bind));
}

/** Limit the step from `prev` so no bone turns more than maxRad in world terms (the chain upper, upper*fore,
 *  upper*fore*hand under the shoulder's parent): every joint moves the same fraction of its way, in joint space
 *  (upper and hand by slerp, forearm by bend and twist), so the capped pose stays on the hinge. */
function capTurn(rig: ArmRig, p: ArmPose, prev: ArmPose, maxRad: number) {
  const chain = (x: ArmPose) => {
    const u = x.upper.clone(), f = u.clone().multiply(x.fore);
    return [u, f, f.clone().multiply(x.hand)];
  };
  const was = chain(prev), turn = (x: ArmPose) => Math.max(...chain(x).map((q, i) => was[i].angleTo(q)));
  const worst = turn(p);
  if (worst <= maxRad) return;
  const [a0, t0] = foreParams(rig, prev.fore), [a1, t1] = foreParams(rig, p.fore), u = p.upper.clone(), h = p.hand.clone();
  const at = (k: number) => {
    p.upper.copy(prev.upper).slerp(u, k);
    p.fore.copy(foreFrom(rig, a0 + (a1 - a0) * k, t0 + (t1 - t0) * k));
    p.hand.copy(prev.hand).slerp(h, k);
  };
  let k = maxRad / worst;
  for (let i = 0; i < 4; i++) { // world turn isn't linear in k: shrink until it fits
    at(k);
    const w = turn(p);
    if (w <= maxRad) break;
    k *= maxRad / w;
  }
}

/** The turn-rate cap against `prev` (by the time since it), then back onto the joint model and limits; stamps tMs.
 *  Every pose that reaches the bones goes through here. */
export function settlePose(rig: ArmRig, p: ArmPose, prev: ArmPose | null, tMs: number, lim: ArmLimits = DEFAULT_LIMITS): ArmPose {
  const dt = prev?.tMs === undefined ? 0 : (tMs - prev.tMs) / 1000;
  if (prev && dt > 0) capTurn(rig, p, prev, TURN_MAX_DEG_S * RAD * dt);
  project(rig, p, lim);
  for (const b of ["upper", "fore", "hand"] as const) sameHemisphere(p[b].normalize(), prev?.[b]);
  p.tMs = tMs;
  return p;
}

/** solveArmPose, then each bone's local rotation through its filter at time tMs, then settlePose (turn cap, limits).
 *  Returns the pose to carry as `prev` and copy onto the bones. Anything non-finite (a NaN landmark) keeps the previous
 *  value: a NaN would poison the filter, and a NaN twist demand or pole would be carried forever. */
export function stepArm(rig: ArmRig, filters: ArmFilters, t: ArmTarget, prev: ArmPose | null, tMs: number, lim: ArmLimits = DEFAULT_LIMITS): ArmPose {
  const p = solveArmPose(rig, t, prev, lim);
  if (!Number.isFinite(p.twistDemand)) p.twistDemand = prev?.twistDemand ?? 0;
  if (!Number.isFinite(p.pole.x + p.pole.y + p.pole.z)) p.pole.copy(prev?.pole ?? DOWN);
  for (const b of ["upper", "fore", "hand"] as const) {
    if (!finiteQ(p[b])) p[b].copy(prev ? prev[b] : rig[b].bind);
    p[b].copy(filters[b].filter(p[b], tMs));
  }
  return settlePose(rig, p, prev, tMs, lim);
}
