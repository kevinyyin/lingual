import { describe, expect, test } from "vitest";
import { smoothClip } from "./clipSmooth";
import type { Clip, ClipFrame } from "./player";

/** n frames whose right wrist (pose 16) follows `wrist(t)`; every other point at the origin; touchL 0.5 throughout. */
const clipOf = (n: number, wrist: (t: number) => number[]): Clip => ({
  fps: 30,
  frames: Array.from({ length: n }, (_, t): ClipFrame => {
    const pose = Array.from({ length: 33 }, () => [0, 0, 0]);
    pose[16] = wrist(t);
    return { pose, pose2d: pose.map(([x, y]) => [x, y]), left: null, right: null, touchL: 0.5 };
  }),
});
const wobbly = (t: number) => [0.3 + (t % 2 ? 0.004 : -0.004), -0.4, 0];

describe("smoothClip", () => {
  test("removes a frame-to-frame wobble", () => {
    const out = smoothClip(clipOf(9, wobbly));
    for (let t = 1; t < 8; t++) expect(out.frames[t].pose[16][0]).toBeCloseTo(0.3, 9);
  });
  test("keeps a straight-line motion, the clip's length and its end frames", () => {
    const out = smoothClip(clipOf(9, (t) => [0.01 * t, -0.4, 0]));
    expect(out.frames).toHaveLength(9);
    out.frames.forEach((f, t) => expect(f.pose[16][0]).toBeCloseTo(0.01 * t, 9));
  });
  test("a missing hand stays missing, and the frames beside the gap are left as they are", () => {
    const src = clipOf(9, wobbly);
    src.frames.forEach((f, t) => { f.rightW = t === 4 ? null : [[t % 2 ? 1 : -1, 0, 0]]; });
    const out = smoothClip(src);
    expect(out.frames[4].rightW).toBeNull();
    expect(out.frames[3].rightW).toEqual([[1, 0, 0]]);
    expect(out.frames[5].rightW).toEqual([[1, 0, 0]]);
    expect(out.frames[2].rightW![0][0]).toBeCloseTo(0, 9); // ¼·1 + ½·(−1) + ¼·1
  });
  test("leaves contact data and the source clip as they are", () => {
    const src = clipOf(9, wobbly);
    const out = smoothClip(src);
    expect(out.frames.every((f) => f.touchL === 0.5)).toBe(true);
    expect(src.frames[1].pose[16][0]).toBeCloseTo(0.304, 9);
  });
});
