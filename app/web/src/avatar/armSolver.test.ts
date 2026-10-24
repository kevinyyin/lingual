import * as THREE from "three";
import { describe, expect, test } from "vitest";
import { armFilters, armRig, DEFAULT_LIMITS, easeHand, settlePose, solveArmPose, stepArm, type ArmPose, type ArmRig, type ArmTarget, type HandEase } from "./armSolver";
import { elbowFlexion } from "./metrics";
import { joint, parentWorld, twistAbout, twoBoneElbow } from "./rig";

const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
const DEG = 180 / Math.PI;

/** shoulder -> elbow -> wrist -> fingertip chain along +x, under a rotated parent (copied from rig.test.ts, plus a hand child) */
function arm() {
  const root = new THREE.Object3D();
  root.quaternion.setFromEuler(new THREE.Euler(0.3, -0.7, 0.2));
  const upper = new THREE.Object3D(), fore = new THREE.Object3D(), hand = new THREE.Object3D(), tip = new THREE.Object3D();
  upper.quaternion.setFromEuler(new THREE.Euler(0.1, 0.4, -0.2));
  fore.position.set(0.3, 0, 0);
  hand.position.set(0.25, 0, 0);
  tip.position.set(0.08, 0, 0);
  root.add(upper);
  upper.add(fore);
  fore.add(hand);
  hand.add(tip);
  root.updateMatrixWorld(true);
  const rig = armRig(joint(upper, fore), joint(fore, hand), joint(hand, tip));
  return { root, upper, fore, hand, rig };
}
const world = (o: THREE.Object3D) => (o.updateWorldMatrix(true, false), o.getWorldPosition(new THREE.Vector3()));

/** mulberry32: a seeded PRNG so failures reproduce */
function rng(seed: number) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const unit = (r: () => number) => {
  for (;;) {
    const p = v(r() * 2 - 1, r() * 2 - 1, r() * 2 - 1);
    if (p.lengthSq() > 0.01 && p.lengthSq() <= 1) return p.normalize();
  }
};
const randQuat = (r: () => number) => new THREE.Quaternion().setFromAxisAngle(unit(r), r() * 2 * Math.PI);

function apply(a: ReturnType<typeof arm>, p: ArmPose) {
  a.upper.quaternion.copy(p.upper);
  a.fore.quaternion.copy(p.fore);
  a.hand.quaternion.copy(p.hand);
  a.root.updateMatrixWorld(true);
}
const flexDeg = (rig: ArmRig, p: ArmPose) => elbowFlexion(rig.upper.axis, rig.fore.axis.clone().applyQuaternion(p.fore), rig.hinge);
function twistDeg(rig: ArmRig, p: ArmPose) {
  const ax = rig.fore.axis, t = twistAbout(rig.fore.bind.clone().invert().multiply(p.fore), ax);
  const a = 2 * Math.atan2(t.x * ax.x + t.y * ax.y + t.z * ax.z, t.w);
  return Math.atan2(Math.sin(a), Math.cos(a)) * DEG;
}
const offAxisOf = (p: THREE.Vector3, ax: THREE.Vector3) => p.addScaledVector(ax, -p.dot(ax)).normalize();
const finite =(q: THREE.Quaternion) => [q.x, q.y, q.z, q.w].every(Number.isFinite) && Math.abs(q.length() - 1) < 1e-6;
/** largest world-space turn of any bone between two poses (what the 25-degree spike gate measures) */
function maxTurn(a: ReturnType<typeof arm>, p: ArmPose, q: ArmPose) {
  const worlds = (x: ArmPose) => (apply(a, x), [a.upper, a.fore, a.hand].map((o) => o.getWorldQuaternion(new THREE.Quaternion())));
  const wp = worlds(p), wq = worlds(q);
  return Math.max(...wp.map((w, i) => w.angleTo(wq[i]))) * DEG;
}

/** random targets the arm can reach within its limits (radius 0.2 m .. 0.9 (l1 + l2)) */
function targets(a: ReturnType<typeof arm>, n: number, seed: number, withHand = false): ArmTarget[] {
  const r = rng(seed), s = world(a.upper), reach = a.rig.l1 + a.rig.l2;
  return Array.from({ length: n }, () => ({
    shoulder: s.clone(),
    wrist: s.clone().addScaledVector(unit(r), 0.2 + r() * (0.9 * reach - 0.2)),
    elbowHint: s.clone().addScaledVector(unit(r), 0.3),
    handRot: withHand ? randQuat(r) : null,
  }));
}

describe("armSolver", () => {
  test("reaches a target within reach", () => {
    const a = arm();
    for (const t of targets(a, 50, 1)) {
      apply(a, solveArmPose(a.rig, t, null));
      expect(world(a.hand).distanceTo(t.wrist)).toBeLessThan(1e-3);
    }
  });

  test("never bends the elbow backward", () => {
    const a = arm();
    for (const t of targets(a, 50, 1)) {
      const f = flexDeg(a.rig, solveArmPose(a.rig, t, null));
      expect(f).toBeGreaterThanOrEqual(-1e-6);
      expect(f).toBeLessThanOrEqual(DEFAULT_LIMITS.elbowMaxDeg + 1e-6);
    }
  });

  test("forearm twist stays within ±90° and wrist swing within 70°", () => {
    const a = arm();
    let prev: ArmPose | null = null;
    for (const t of targets(a, 50, 2, true)) {
      const p = solveArmPose(a.rig, t, prev);
      expect(Math.abs(twistDeg(a.rig, p))).toBeLessThanOrEqual(DEFAULT_LIMITS.twistMaxDeg + 1e-6);
      expect(p.hand.angleTo(a.rig.hand.bind) * DEG).toBeLessThanOrEqual(DEFAULT_LIMITS.wristMaxDeg + 1e-4);
      prev = p;
    }
  });

  test("continuous through a straight arm", () => {
    const a = arm(), s = world(a.upper), reach = a.rig.l1 + a.rig.l2;
    const dirs = [v(0, -1, 0.3).normalize(), v(1, 0.2, 0.4).normalize()];
    for (const d of dirs) {
      const side = v(0, 0, 1).cross(d).normalize();
      // radial: from bent to past full extension; lateral: a chord crossing the reach sphere
      const lines = [
        (k: number) => s.clone().addScaledVector(d, reach * (0.5 + k)),
        (k: number) => s.clone().addScaledVector(side, reach * 0.6).addScaledVector(d, reach * (1.4 - 1.6 * k)),
      ];
      for (const line of lines) {
        let prev: ArmPose | null = null;
        for (let i = 0; i <= 100; i++) {
          const p = solveArmPose(a.rig, { shoulder: s, wrist: line(i / 100), elbowHint: s.clone().add(v(0.05, -0.2, -0.1)), handRot: null }, prev);
          if (prev) expect(maxTurn(a, prev, p)).toBeLessThan(5);
          prev = p;
        }
      }
    }
  });

  test("no NaN for degenerate targets", () => {
    const a = arm(), s = world(a.upper), reach = a.rig.l1 + a.rig.l2;
    const cases: ArmTarget[] = [
      { shoulder: s, wrist: s.clone(), elbowHint: s.clone().add(v(0, -0.2, 0)), handRot: null },
      { shoulder: s, wrist: s.clone().add(v(3 * reach, 0, 0)), elbowHint: s.clone().add(v(0, -0.2, 0)), handRot: null },
      { shoulder: s, wrist: s.clone().add(v(0.2, -0.3, 0.1)), elbowHint: s.clone(), handRot: null },
      { shoulder: s, wrist: s.clone().add(v(0, -0.4, 0)), elbowHint: s.clone().add(v(0, -0.3, 0)), handRot: new THREE.Quaternion() },
    ];
    for (const t of cases) for (const prev of [null, solveArmPose(a.rig, cases[2], null)]) {
      const p = solveArmPose(a.rig, t, prev);
      for (const q of [p.upper, p.fore, p.hand]) expect(finite(q)).toBe(true);
      expect([p.pole.x, p.pole.y, p.pole.z].every(Number.isFinite)).toBe(true);
    }
  });

  test("small target change, small pose change", () => {
    const a = arm(), r = rng(3);
    for (const t of targets(a, 50, 4)) {
      // a real signer's elbow hint is an actual elbow (not a point on the shoulder-wrist line), and their hand is within
      // the wrist's range: build it from the solved forearm plus a random twist (<= 80) and swing (<= 60)
      t.elbowHint = twoBoneElbow(t.shoulder, t.elbowHint, t.wrist, a.rig.l1, a.rig.l2);
      const base = solveArmPose(a.rig, t, null);
      apply(a, base);
      a.fore.quaternion.multiply(new THREE.Quaternion().setFromAxisAngle(a.rig.fore.axis, (r() * 2 - 1) * 80 / DEG));
      const sw = unit(r);
      offAxisOf(sw, a.rig.fore.axis);
      a.hand.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(sw, (r() * 60) / DEG));
      a.root.updateMatrixWorld(true);
      t.handRot = a.hand.getWorldQuaternion(new THREE.Quaternion());
      const p0 = solveArmPose(a.rig, t, null);
      apply(a, p0);
      expect(a.hand.getWorldQuaternion(new THREE.Quaternion()).angleTo(t.handRot) * DEG).toBeLessThan(0.1); // within limits: exact
      const t1 = { ...t, wrist: t.wrist.clone().addScaledVector(unit(r), 1e-3) };
      expect(maxTurn(a, p0, solveArmPose(a.rig, t1, p0))).toBeLessThan(1);
    }
  });

  test("outputs stay in the previous quaternion's hemisphere", () => {
    const a = arm();
    let prev: ArmPose | null = null;
    for (const t of targets(a, 50, 5, true)) {
      const p = solveArmPose(a.rig, t, prev);
      if (prev) for (const b of ["upper", "fore", "hand"] as const) expect(p[b].dot(prev[b])).toBeGreaterThanOrEqual(0);
      prev = p;
    }
  });
  test("recovers when the elbow genuinely crosses over", () => {
    const a = arm(), s = world(a.upper), wrist = s.clone().add(v(0.15, -0.3, 0.1));
    const u = wrist.clone().sub(s).normalize(), side = v(0, 0, 1).cross(u).normalize();
    const mid = s.clone().addScaledVector(u, 0.2);
    const hint = (k: number) => ({ shoulder: s, wrist, elbowHint: mid.clone().addScaledVector(side, k), handRot: null });
    let prev: ArmPose | null = null;
    for (let i = 0; i < 5; i++) prev = solveArmPose(a.rig, hint(0.1), prev);
    const p = solveArmPose(a.rig, hint(-0.06), prev); // held on the far side, 6 cm off the line
    expect(p.pole.dot(side)).toBeLessThan(-0.99);
    apply(a, p);
    expect(world(a.fore).sub(mid).dot(side)).toBeLessThan(0);
  });

  test("marginal hint turns the pole gradually", () => {
    const a = arm(), s = world(a.upper), wrist = s.clone().add(v(0.15, -0.3, 0.1));
    const u = wrist.clone().sub(s).normalize(), side = v(0, 0, 1).cross(u).normalize();
    const mid = s.clone().addScaledVector(u, 0.2);
    const hint = (k: number) => ({ shoulder: s, wrist, elbowHint: mid.clone().addScaledVector(side, k), handRot: null });
    let prev = solveArmPose(a.rig, hint(0.1), null);
    for (let i = 0; i < 25; i++) {
      const p = solveArmPose(a.rig, hint(-0.03), prev); // 3 cm off the line, opposite side
      expect(p.pole.angleTo(prev.pole) * DEG).toBeLessThanOrEqual(10 + 1e-6);
      prev = p;
    }
    expect(prev.pole.angleTo(side.clone().negate()) * DEG).toBeLessThan(1e-3); // 180 degrees in 18 calls, then held
  });

  test("twist stays at the limit when the demand wraps past 180°", () => {
    const a = arm(), s = world(a.upper);
    const t: ArmTarget = { shoulder: s, wrist: s.clone().add(v(0.15, -0.3, 0.1)), elbowHint: s.clone().add(v(0.1, -0.2, -0.15)), handRot: null };
    apply(a, solveArmPose(a.rig, t, null)); // forearm with zero twist: the hand's demand is measured from here
    const fore = a.fore.getWorldQuaternion(new THREE.Quaternion());
    let prev: ArmPose | null = null, last = -Infinity;
    for (let deg = 80; deg <= 280; deg += 10) { // 80 .. 170, -170 .. -80 (the old clamped-prev unwrap flipped at -80)
      t.handRot = fore.clone().multiply(new THREE.Quaternion().setFromAxisAngle(a.rig.fore.axis, deg / DEG)).multiply(a.rig.hand.bind);
      const p = solveArmPose(a.rig, t, prev), tw = twistDeg(a.rig, p);
      expect(tw).toBeCloseTo(Math.min(deg, 90), 3);
      expect(tw).toBeGreaterThanOrEqual(last - 1e-6);
      if (prev) expect(maxTurn(a, prev, p)).toBeLessThan(11);
      last = tw;
      prev = p;
    }
  });

  /** the arm of the wrap test, and the world hand rotation that twists the forearm by deg */
  function twistRig() {
    const a = arm(), s = world(a.upper);
    const t: ArmTarget = { shoulder: s, wrist: s.clone().add(v(0.15, -0.3, 0.1)), elbowHint: s.clone().add(v(0.1, -0.2, -0.15)), handRot: null };
    apply(a, solveArmPose(a.rig, t, null));
    const fore = a.fore.getWorldQuaternion(new THREE.Quaternion());
    const at = (deg: number) => fore.clone().multiply(new THREE.Quaternion().setFromAxisAngle(a.rig.fore.axis, deg / DEG)).multiply(a.rig.hand.bind);
    return { a, t, at, fore };
  }

  /** A hand in the video from frame 1 on (`seen(n)`), its turn easing in with the arm as rocketbox's idle ease does
   *  (60 fps, 150 ms: k = 1 - 0.895^n), then leaving from frame `leave` on; unfiltered solves, as rocketbox's trial ones. */
  function ease(a: ReturnType<typeof arm>, t: ArmTarget, seen: (n: number) => THREE.Quaternion, frames: number, leave = Infinity) {
    let prev = solveArmPose(a.rig, t, null), state: HandEase | null = null, k = 0;
    const twists: number[] = [], demands: number[] = [];
    for (let n = 1; n <= frames; n++) {
      const rising = n < leave;
      k = rising ? 1 - (1 - k) * 0.895 : k * 0.895;
      const r = easeHand(a.rig, solveArmPose(a.rig, t, prev), seen(n), k, rising, state);
      state = r.ease;
      prev = solveArmPose(a.rig, { ...t, handRot: r.rot }, prev);
      twists.push(twistDeg(a.rig, prev));
      demands.push(prev.twistDemand * DEG);
    }
    return { twists, demands };
  }
  const maxStep = (xs: number[]) => Math.max(...xs.slice(1).map((x, i) => Math.abs(x - xs[i])));

  test("a hand appearing half a turn from the relaxed hand turns the same way whichever side noise starts it on (HELLO)", () => {
    // HELLO's hand with its old trim, as measured: the clip's raised hand on an arm still at rest (wrist bent 146 degrees),
    // twisted 180 degrees as it appears, then turning on to 290 (-70); noise (the clip's cleaning, the breathing) put
    // it either side of 180, and the side picked the way the forearm turned
    const { a, t, at, fore } = twistRig();
    const bent = fore.clone().multiply(new THREE.Quaternion().setFromAxisAngle(v(0, 0, 1), 146 / DEG)).multiply(fore.clone().invert());
    const run = (noise: number) => ease(a, t, (n) => bent.clone().multiply(at(180 + noise + 5 * Math.min(n - 1, 22))), 60);
    const p = run(0.5), m = run(-0.5);
    p.twists.forEach((x, i) => expect(Math.abs(x - m.twists[i])).toBeLessThan(5)); // a degree of noise moves it a few degrees, not to the other side
    for (const r of [p, m]) {
      expect(Math.max(...r.twists)).toBeLessThan(2); // never sets off the wrong way round
      expect(Math.max(...r.demands.map(Math.abs))).toBeLessThanOrEqual(180); // nor winds past the forearm's reach
      expect(maxStep(r.twists)).toBeLessThan(20); // under the turn cap: no snap
      expect(Math.abs(r.twists[59] + 70)).toBeLessThan(2);
    }
  });

  test("a palm-up hand needing more twist than the forearm has keeps the side it appeared on, through its ease and as it leaves (COOK)", () => {
    // COOK's base hand: wrist bent back 150 degrees, twisted -155; it turns on to -240 near its ease's end (-180 at
    // k 0.89, where the first fix's untwisted base wound it to the far side and snapped it back), back to -55, then leaves
    const { a, t, at, fore } = twistRig();
    const bent = fore.clone().multiply(new THREE.Quaternion().setFromAxisAngle(v(0, 0, 1), 150 / DEG)).multiply(fore.clone().invert());
    const deg = (n: number) => (n <= 15 ? -155 : n <= 32 ? -155 - 5 * (n - 15) : Math.min(-240 + 6 * (n - 32), -55));
    for (const noise of [0.5, -0.5]) {
      const r = ease(a, t, (n) => bent.clone().multiply(at(deg(n) + noise)), 160, 100);
      expect(Math.max(...r.twists)).toBeLessThan(0); // the forearm never flips to the far side
      expect(Math.max(...r.demands.map(Math.abs))).toBeLessThan(270); // so nothing unwinds
      expect(maxStep(r.twists)).toBeLessThan(20);
    }
  });

  test("a hand clear of half a turn eases in and out exactly as the plain slerp did", () => {
    const { a, t, at } = twistRig();
    let prev = solveArmPose(a.rig, t, null), state: HandEase | null = null, k = 0;
    for (let n = 1; n <= 40; n++) {
      const rising = n < 25;
      k = rising ? 1 - (1 - k) * 0.895 : k * 0.895;
      const relaxed = solveArmPose(a.rig, t, prev), seen = at(120 - 2 * n);
      const r = easeHand(a.rig, relaxed, seen, k, rising, state);
      const plain = parentWorld(a.upper).multiply(relaxed.upper).multiply(relaxed.fore).multiply(relaxed.hand).slerp(seen, k);
      expect(r.rot.angleTo(plain)).toBeLessThan(1e-6);
      state = r.ease;
      prev = solveArmPose(a.rig, { ...t, handRot: r.rot }, prev);
    }
  });

  test("flip-and-back does not ratchet", () => {
    const { a, t, at } = twistRig();
    let prev: ArmPose | null = null;
    for (const true_ of [20, -20, 60]) for (let k = 0; k < 6; k++) {
      prev = solveArmPose(a.rig, { ...t, handRot: at(true_ + 180) }, prev); // MediaPipe flips the palm
      prev = solveArmPose(a.rig, { ...t, handRot: at(true_) }, prev); // and back
      expect(Math.abs(twistDeg(a.rig, prev) - true_)).toBeLessThan(5);
    }
  });

  test("untracked hand re-anchors the demand", () => {
    const { a, t, at } = twistRig();
    let prev: ArmPose | null = null;
    for (let deg = 80; deg <= 260; deg += 10) prev = solveArmPose(a.rig, { ...t, handRot: at(deg) }, prev); // held at the clamp
    const tw = twistDeg(a.rig, prev!);
    prev = solveArmPose(a.rig, t, prev); // the hand leaves
    expect(Math.abs(prev.twistDemand)).toBeLessThanOrEqual(Math.PI);
    expect(twistDeg(a.rig, prev)).toBeCloseTo(tw, 3); // the forearm does not move
    prev = solveArmPose(a.rig, { ...t, handRot: at(20) }, prev); // it comes back
    expect(twistDeg(a.rig, prev)).toBeCloseTo(20, 3);
  });

  test("a demand wound past the reach of any forearm unwinds within the spike bound", () => {
    const { a, t, at } = twistRig(), f = armFilters();
    let prev: ArmPose | null = null;
    for (let i = 0, deg = 0; deg <= 720; i++, deg += 10) { // two full turns at 600 deg/s
      const p = stepArm(a.rig, f, { ...t, handRot: at(deg) }, prev, (i * 1000) / 60);
      if (prev) expect(maxTurn(a, prev, p)).toBeLessThan(25);
      expect(Math.abs(p.twistDemand)).toBeLessThanOrEqual(Math.PI + Math.PI / 2 + 0.2);
      prev = p;
    }
  });

  test("stepArm eases a hand appearing mid-plan", () => {
    const a = arm(), s = world(a.upper), f = armFilters();
    const hint = s.clone().add(v(0.05, -0.25, -0.1));
    const rest: ArmTarget = { shoulder: s, wrist: s.clone().add(v(0.02, -0.47, 0.08)), elbowHint: hint, handRot: null };
    const raised: ArmTarget = { ...rest, wrist: s.clone().add(v(-0.1, 0.15, 0.3)) };
    let prev: ArmPose | null = null;
    for (let i = 0; i < 10; i++) prev = stepArm(a.rig, f, rest, prev, (i * 1000) / 60);
    for (let i = 10; i <= 10 + 24; i++) { // 400 ms after the switch at frame 10
      const p = stepArm(a.rig, f, raised, prev, (i * 1000) / 60);
      expect(maxTurn(a, prev!, p)).toBeLessThan(25);
      prev = p;
    }
    apply(a, prev!);
    expect(world(a.hand).distanceTo(raised.wrist)).toBeLessThan(0.01);
  });

  test("stepArm first frame is unfiltered", () => {
    const a = arm(), [t] = targets(a, 1, 6, true);
    const p = stepArm(a.rig, armFilters(), t, null, 1234), q = solveArmPose(a.rig, t, null);
    for (const b of ["upper", "fore", "hand"] as const) expect(p[b].angleTo(q[b])).toBeLessThan(1e-6);
  });

  test("stepArm recovers from a NaN hand rotation", () => {
    const a = arm(), f = armFilters();
    const [t] = targets(a, 1, 7, true);
    const nan = { ...t, handRot: new THREE.Quaternion(NaN, 0, 0, 1) };
    let p: ArmPose | null = null;
    for (let i = 0; i < 5; i++) p = stepArm(a.rig, f, t, p, (i * 1000) / 60);
    p = stepArm(a.rig, f, nan, p, (5 * 1000) / 60);
    for (let i = 6; i < 30; i++) p = stepArm(a.rig, f, t, p, (i * 1000) / 60);
    for (const b of ["upper", "fore", "hand"] as const) expect(finite(p![b])).toBe(true);
    expect(Number.isFinite(p!.twistDemand)).toBe(true);
    expect([p!.pole.x, p!.pole.y, p!.pole.z].every(Number.isFinite)).toBe(true);
    const q = solveArmPose(a.rig, t, null);
    for (const b of ["upper", "fore", "hand"] as const) expect(p![b].angleTo(q[b]) * DEG).toBeLessThan(1);
  });

  test("a turn-capped step stays a valid pose", () => {
    const a = arm();
    const ts = targets(a, 40, 9, true);
    for (let i = 0; i + 1 < ts.length; i++) {
      const p0 = solveArmPose(a.rig, ts[i], null);
      p0.tMs = 0;
      const p1 = settlePose(a.rig, solveArmPose(a.rig, ts[i + 1], p0), p0, 1000 / 60); // random far targets: saturates
      expect(maxTurn(a, p0, p1)).toBeLessThan(21);
      const fl = flexDeg(a.rig, p1);
      expect(fl).toBeGreaterThanOrEqual(-1e-3);
      expect(fl).toBeLessThanOrEqual(DEFAULT_LIMITS.elbowMaxDeg + 1e-3);
      expect(Math.abs(twistDeg(a.rig, p1))).toBeLessThanOrEqual(DEFAULT_LIMITS.twistMaxDeg + 1e-3);
      const d = a.rig.fore.axis.clone().applyQuaternion(p1.fore); // forearm direction in the upper arm's frame
      expect(Math.abs(90 - d.angleTo(a.rig.hinge) * DEG)).toBeLessThan(1); // on the hinge plane: no off-hinge elbow
      expect(p1.hand.angleTo(a.rig.hand.bind) * DEG).toBeLessThanOrEqual(DEFAULT_LIMITS.wristMaxDeg + 1e-3);
    }
  });
});

describe("anatomical wrist", () => {
  /** the test arm with a palm: fingers along the hand's +x, index side +z (a right hand) */
  function palmArm() {
    const a = arm();
    a.rig.palm = { f: v(1, 0, 0), s: v(0, 0, 1), right: true };
    return a;
  }
  /** bend (+ toward the palm) and tilt (+ toward the thumb) of the fingers' direction from the bind pose, degrees */
  function wristAngles(rig: ArmRig, hand: THREE.Quaternion) {
    const F0 = rig.palm!.f.clone().applyQuaternion(rig.hand.bind), S0 = rig.palm!.s.clone().applyQuaternion(rig.hand.bind);
    const N0 = S0.clone().cross(F0), F1 = rig.palm!.f.clone().applyQuaternion(hand);
    return { flex: Math.atan2(F1.dot(N0), F1.dot(F0)) * DEG, tilt: Math.asin(Math.min(1, Math.max(-1, F1.dot(S0)))) * DEG };
  }
  const w = DEFAULT_LIMITS.wrist!;

  test("bends toward the palm and back further than it tilts, and tilts further toward the little finger", () => {
    const a = palmArm();
    let prev: ArmPose | null = null;
    for (const t of targets(a, 200, 7, true)) {
      const { flex, tilt } = wristAngles(a.rig, (prev = solveArmPose(a.rig, t, prev)).hand);
      expect(flex).toBeLessThanOrEqual(w.flex + 0.5);
      expect(flex).toBeGreaterThanOrEqual(-w.ext - 0.5);
      expect(tilt).toBeLessThanOrEqual(w.radial + 0.5);
      expect(tilt).toBeGreaterThanOrEqual(-w.ulnar - 0.5);
    }
  });

  test("the filtered path (stepArm) keeps the wrist in range too", () => {
    const a = palmArm(), f = armFilters();
    let prev: ArmPose | null = null, ms = 0;
    for (const t of targets(a, 120, 8, true)) {
      prev = stepArm(a.rig, f, t, prev, (ms += 1000 / 60));
      const { flex, tilt } = wristAngles(a.rig, prev.hand);
      expect(flex).toBeLessThanOrEqual(w.flex + 0.5);
      expect(flex).toBeGreaterThanOrEqual(-w.ext - 0.5);
      expect(tilt).toBeLessThanOrEqual(w.radial + 0.5);
      expect(tilt).toBeGreaterThanOrEqual(-w.ulnar - 0.5);
    }
  });

  test("a hand within range is left as asked", () => {
    const a = palmArm(), s = world(a.upper);
    const t: ArmTarget = { shoulder: s, wrist: s.clone().add(v(0.2, -0.3, 0.1)), elbowHint: s.clone().add(v(0, -0.2, -0.1)), handRot: null };
    const p0 = solveArmPose(a.rig, t, null);
    apply(a, p0);
    // the hand bent 30 degrees toward the palm (about the palm's side axis, which for this right hand curls toward +N)
    const bent = a.hand.getWorldQuaternion(new THREE.Quaternion()).multiply(new THREE.Quaternion().setFromAxisAngle(v(0, 0, 1), -30 / DEG));
    const { flex, tilt } = wristAngles(a.rig, solveArmPose(a.rig, { ...t, handRot: bent }, p0).hand);
    expect(Math.abs(flex)).toBeCloseTo(30, 0);
    expect(Math.abs(tilt)).toBeLessThan(1);
  });
});
