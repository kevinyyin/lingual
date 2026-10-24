import * as THREE from "three";
import { describe, expect, test } from "vitest";
import { RotationSpring } from "./motionSmoother";

const about = (deg: number, axis = new THREE.Vector3(0, 0, 1)) => new THREE.Quaternion().setFromAxisAngle(axis, (deg * Math.PI) / 180);
const angleOf = (q: THREE.Quaternion) => (2 * Math.acos(Math.min(1, Math.abs(q.w))) * 180) / Math.PI;
const DT = 1 / 60;

describe("rotation spring (critically damped PD)", () => {
  test("the first call and a reset snap to the target", () => {
    const s = new RotationSpring({ hz: 6, maxAccel: 1e9 });
    expect(angleOf(s.step(about(40), DT))).toBeCloseTo(40);
    s.reset();
    expect(angleOf(s.step(about(10), DT))).toBeCloseTo(10);
  });

  test("a step of the target is followed without overshoot (no rubberbanding)", () => {
    const s = new RotationSpring({ hz: 6, maxAccel: 1e9 });
    s.step(about(0), DT);
    let peak = 0, last = 0;
    for (let i = 0; i < 120; i++) {
      last = angleOf(s.step(about(60), DT));
      peak = Math.max(peak, last);
    }
    expect(peak).toBeLessThanOrEqual(60 + 1e-6);
    expect(last).toBeGreaterThan(59.9); // and it gets there
  });

  test("it settles within ~0.15 s at 6 Hz (follows signing, removes faster back-and-forth)", () => {
    const s = new RotationSpring({ hz: 6, maxAccel: 1e9 });
    s.step(about(0), DT);
    let t = 0, a = 0;
    while (a < 54 && t < 1) {
      a = angleOf(s.step(about(60), DT));
      t += DT;
    }
    expect(t).toBeLessThan(0.15);
  });

  test("a 10 Hz wobble of the target is mostly removed, a 1 Hz movement mostly kept", () => {
    const amp = (hz: number) => {
      const s = new RotationSpring({ hz: 6, maxAccel: 1e9 });
      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < 240; i++) {
        const target = 20 * Math.sin(2 * Math.PI * hz * i * DT);
        const q = s.step(about(target), DT);
        const signed = (2 * Math.atan2(q.z, q.w) * 180) / Math.PI;
        if (i > 120) [lo, hi] = [Math.min(lo, signed), Math.max(hi, signed)];
      }
      return (hi - lo) / 2;
    };
    expect(amp(1)).toBeGreaterThan(18); // > 90% of a 20 degree swing kept
    expect(amp(10)).toBeLessThan(8); // the fast wobble cut by well over half
  });

  test("the acceleration cap turns a sudden jump into a gradual move", () => {
    const capped = new RotationSpring({ hz: 6, maxAccel: 50 }), free = new RotationSpring({ hz: 6, maxAccel: 1e9 });
    capped.step(about(0), DT);
    free.step(about(0), DT);
    const a = angleOf(capped.step(about(90), DT)), b = angleOf(free.step(about(90), DT));
    expect(a).toBeLessThan(b);
    expect(a).toBeLessThan(50 * DT * DT * (180 / Math.PI)); // under a*t^2 in the first frame (a*t^2/2, plus the sub-step's share)
  });

  test("it takes the short way round and ignores a non-finite target", () => {
    const s = new RotationSpring({ hz: 6, maxAccel: 1e9 });
    s.step(about(170), DT);
    for (let i = 0; i < 60; i++) s.step(about(-170), DT); // 20 degrees away through 180, not 340 the long way
    expect(angleOf(s.step(about(-170), DT))).toBeCloseTo(170, 0);
    const held = s.step(new THREE.Quaternion(NaN, 0, 0, 1), DT);
    expect(Number.isFinite(held.x + held.y + held.z + held.w)).toBe(true);
  });
});
