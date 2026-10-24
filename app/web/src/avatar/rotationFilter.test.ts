import { describe, expect, it } from "vitest";
import { Quaternion, Vector3 } from "three";
import { QuatFilter } from "./rotationFilter";

const Y = new Vector3(0, 1, 0);
const rotY = (deg: number) => new Quaternion().setFromAxisAngle(Y, (deg * Math.PI) / 180);
const deg = (rad: number) => (rad * 180) / Math.PI;

function mulberry32(a: number) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe("QuatFilter", () => {
  it("first sample passes through", () => {
    const q = rotY(40);
    const out = new QuatFilter().filter(q, 1234);
    expect(out.angleTo(q)).toBeLessThan(1e-9);
  });

  it("holds a constant input", () => {
    const f = new QuatFilter();
    const q = rotY(33);
    let out = q;
    for (let i = 0; i < 100; i++) out = f.filter(q, i * 16.67);
    expect(out.angleTo(q)).toBeLessThan(1e-6);
  });

  it("frame-rate independent", () => {
    const run = (fps: number) => {
      const f = new QuatFilter();
      const dt = 1000 / fps;
      f.filter(rotY(0), 0);
      let out = rotY(0);
      for (let t = dt; t <= 200 + 1e-6; t += dt) out = f.filter(rotY(60), t);
      return out.clone();
    };
    expect(deg(run(30).angleTo(run(120)))).toBeLessThan(3);
  });

  it("no overshoot on a step", () => {
    const f = new QuatFilter();
    const target = rotY(60);
    f.filter(rotY(0), 0);
    let prev = Infinity;
    for (let i = 1; i <= 120; i++) {
      const a = f.filter(target, i * 16.67).angleTo(target);
      expect(a).toBeLessThanOrEqual(prev + 1e-6);
      prev = a;
    }
  });

  it("fast motion has little lag", () => {
    const f = new QuatFilter();
    let out = rotY(0);
    let input = rotY(0);
    for (let i = 0; i <= 30; i++) {
      const t = i * (1000 / 60);
      input = rotY((360 * t) / 1000);
      out = f.filter(input, t);
    }
    expect(deg(out.angleTo(input))).toBeLessThan(15);
  });

  it("jitter is reduced", () => {
    const rnd = mulberry32(7);
    const f = new QuatFilter();
    const base = rotY(20);
    let inSq = 0, outSq = 0, n = 0;
    for (let i = 0; i < 300; i++) {
      const noisy = base.clone().multiply(
        new Quaternion().setFromAxisAngle(new Vector3(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5).normalize(), ((rnd() * 4 - 2) * Math.PI) / 180),
      );
      const out = f.filter(noisy, i * (1000 / 60));
      if (i >= 30) {
        inSq += deg(noisy.angleTo(base)) ** 2;
        outSq += deg(out.angleTo(base)) ** 2;
        n++;
      }
    }
    expect(Math.sqrt(outSq / n)).toBeLessThan(0.5 * Math.sqrt(inSq / n));
  });

  it("reset restarts at the next input; dt<=0 returns previous output; stays in hemisphere", () => {
    const f = new QuatFilter();
    f.filter(rotY(10), 0);
    const prev = f.filter(rotY(20), 16).clone();
    expect(f.filter(rotY(90), 16).angleTo(prev)).toBe(0);
    f.reset();
    expect(f.filter(rotY(90), 5).angleTo(rotY(90))).toBeLessThan(1e-9);
    const q = rotY(10);
    const neg = new Quaternion(-q.x, -q.y, -q.z, -q.w);
    f.reset();
    f.filter(q, 0);
    const out = f.filter(neg, 16);
    expect(out.dot(q)).toBeGreaterThan(0);
  });
});
