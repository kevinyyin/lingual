import { describe, expect, test } from "vitest";
import * as THREE from "three";
import { createHeadMotion } from "./headMotion";

const nose = new THREE.Vector3(0, 1.6, 0.1);
const run = (wl: THREE.Vector3 | null, wr: THREE.Vector3 | null, seconds: number) => {
  const h = createHeadMotion(), out = [];
  for (let t = 0; t < seconds * 1000; t += 1000 / 60) out.push(h(t, wl, wr, nose));
  return out;
};

describe("headMotion", () => {
  test("idle drift stays under 2 degrees and 0.3 Hz", () => {
    const s = run(null, null, 20);
    for (const a of s) for (const v of [a.yawDeg, a.pitchDeg, a.rollDeg]) expect(Math.abs(v)).toBeLessThanOrEqual(2);
    for (const k of ["yawDeg", "pitchDeg", "rollDeg"] as const) {
      const m = s.reduce((x, a) => x + a[k], 0) / s.length;
      let c = 0;
      for (let i = 1; i < s.length; i++) if (Math.sign(s[i][k] - m) !== Math.sign(s[i - 1][k] - m)) c++;
      expect(c / 2 / 20).toBeLessThanOrEqual(0.3);
    }
  });

  test("turns toward the working hand", () => {
    const s = run(new THREE.Vector3(0.2, 0.9, 0.1), new THREE.Vector3(-0.2, 1.6, 0.4), 6); // avatar's right is -x
    const last = s[s.length - 1];
    expect(last.yawDeg).toBeLessThan(-1);
    expect(Math.abs(last.yawDeg)).toBeLessThanOrEqual(4 + 1.5);
  });

  test("never looks like a nod or shake", () => {
    const s = run(null, new THREE.Vector3(0.2, 1.4, 0.4), 20);
    for (let i = 0; i + 60 <= s.length; i++) {
      const w = s.slice(i, i + 60).map((a) => a.pitchDeg);
      expect(Math.max(...w) - Math.min(...w)).toBeLessThan(4);
    }
  });

  test("an idle gap does not snap the head", () => {
    const h = createHeadMotion(), w = new THREE.Vector3(-0.2, 1.6, 0.4);
    let last = { yawDeg: 0, pitchDeg: 0, rollDeg: 0 };
    for (let t = 0; t < 3000; t += 1000 / 60) last = h(t, null, w, nose);
    const next = h(5000, new THREE.Vector3(0.2, 1.6, 0.4), null, nose); // 2 s later, the other hand up
    for (const k of ["yawDeg", "pitchDeg", "rollDeg"] as const) expect(Math.abs(next[k] - last[k])).toBeLessThan(1);
  });
});
