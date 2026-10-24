import { describe, expect, test } from "vitest";
import { estimateAspect, medianAspect, wristFrom2d } from "./placement";
import type { ClipFrame } from "./player";

// world pose (metres, MediaPipe: y down), then an orthographic 16:9 image of it: x' = 0.5 + x/1.6, y' = 0.5 + y/0.9
const world = (): number[][] => {
  const p = Array.from({ length: 33 }, () => [0, 0, 0]);
  p[11] = [0.2, 0, 0]; p[12] = [-0.2, 0, 0]; p[23] = [0.15, 0.5, 0]; p[24] = [-0.15, 0.5, 0];
  p[0] = [0, -0.25, -0.05]; p[16] = [-0.05, -0.55, -0.3]; p[15] = [0.3, 0.3, -0.1];
  return p;
};
const image = (p: number[][]) => p.map(([x, y]) => [0.5 + x / 1.6, 0.5 + y / 0.9]);
const clip = (): ClipFrame => ({ pose: world(), pose2d: image(world()), left: null, right: null });

describe("placement", () => {
  test("wristFrom2d matches world when consistent", () => {
    const f = clip();
    for (const w of [15, 16] as const) {
      const r = wristFrom2d(f.pose, f.pose2d, w, 16 / 9), t = f.pose[w];
      expect(Math.hypot(r.x - t[0], r.y - t[1], r.z - t[2])).toBeLessThan(1e-3);
    }
  });
  test("estimateAspect recovers 16:9", () => {
    expect(Math.abs(estimateAspect([clip(), clip()])! - 16 / 9)).toBeLessThan(0.01);
  });
  test("wristFrom2d takes across/up from the image, depth from the world", () => {
    const f = clip();
    f.pose[16] = [0.3, 0.3, -0.3]; // world wrist far from where the image has it
    const r = wristFrom2d(f.pose, f.pose2d, 16, 16 / 9);
    expect(r.x).toBeCloseTo(-0.05, 3); expect(r.y).toBeCloseTo(-0.55, 3); expect(r.z).toBeCloseTo(-0.3, 6);
  });
  test("an unusable aspect is undefined, and zero image shoulder width falls back to the world wrist", () => {
    expect(medianAspect([NaN, 0, -1])).toBeUndefined();
    const f = clip();
    f.pose2d[12] = [...f.pose2d[11]]; // shoulders at one image x
    expect(estimateAspect([{ ...f, pose: f.pose }])).toBeUndefined();
    const r = wristFrom2d(f.pose, f.pose2d, 16, 16 / 9);
    expect(Number.isFinite(r.x) && r.x === f.pose[16][0] && r.y === f.pose[16][1]).toBe(true);
  });
});
