import { describe, expect, it } from "vitest";
import { distanceBySize, fingerCurls, liveFrame, reach, signLabel, wristDepthCm } from "./mirror";
import type { Frame } from "./ws";

/** Image pose with the signer's left wrist (15) on the image's right (x 0.7) and right wrist (16) on its left (x 0.3). */
const pose = () => Array.from({ length: 33 }, (_, i) => [i === 15 ? 0.7 : i === 16 ? 0.3 : 0.5, 0.5, 0, 1] as [number, number, number, number]);
const handAt = (x: number) => Array.from({ length: 21 }, () => [x, 0.5, 0] as [number, number, number]);
const worldPose = () => Array.from({ length: 33 }, () => [0, 0, 0]);
const frame = (left: number | null, right: number | null): Frame =>
  ({ t: 0, pose: pose(), left: left === null ? null : handAt(left), right: right === null ? null : handAt(right), blend: null, head: null });

describe("liveFrame", () => {
  it("keys each hand by the wrist it sits on, not MediaPipe's label, with its world hand", () => {
    const wl = [[1, 1, 1]], wr = [[2, 2, 2]];
    // labelled "left" but at the signer's right wrist (x 0.3), and the other way round
    const f = liveFrame(frame(0.31, 0.69), { pose: worldPose(), left: wl, right: wr })!;
    expect(f.right![0]).toEqual([0.31, 0.5, 0]);
    expect(f.rightW).toBe(wl);
    expect(f.left![0]).toEqual([0.69, 0.5, 0]);
    expect(f.leftW).toBe(wr);
    expect(f.aspect).toBeCloseTo(4 / 3);
    expect(f.pose2d[15]).toEqual([0.7, 0.5]);
  });

  it("gives crossed hands both sides, and is nothing without a body", () => {
    const f = liveFrame(frame(0.32, 0.33), { pose: worldPose(), left: [[1, 1, 1]], right: [[2, 2, 2]] })!;
    expect([f.right![0][0], f.left![0][0]]).toEqual([0.32, 0.33]);
    expect(liveFrame(frame(null, null), { pose: null, left: null, right: null })).toBeNull();
  });

  it("leaves an unseen hand empty so the avatar rests it", () => {
    const f = liveFrame(frame(null, 0.7), { pose: worldPose(), left: null, right: [[3, 3, 3]] })!;
    expect([f.right, f.rightW]).toEqual([null, null]);
    expect(f.leftW).toEqual([[3, 3, 3]]);
  });
});

const FOCAL = 0.5 / Math.tan((65 * Math.PI) / 360); // image heights, at aspect 1 (mirror.ts HFOV_DEG)

describe("distanceBySize", () => {
  const square = (r: number, c = 0.5) => [[c - r, c - r], [c + r, c - r], [c + r, c + r], [c - r, c + r]];
  const wld = square(0.05, 0).map(([x, y]) => [x, y, 0]);
  it("is one focal length off at true size, and half as far when the picture doubles", () => {
    expect(distanceBySize(square(0.05), wld, 1)).toBeCloseTo(FOCAL, 6);
    expect(distanceBySize(square(0.1), wld, 1)! / distanceBySize(square(0.05), wld, 1)!).toBeCloseTo(0.5, 6);
  });
  it("is nothing for points with no spread", () => {
    expect(distanceBySize([[0.5, 0.5], [0.5, 0.5]], [[0, 0, 0], [0, 0, 0]], 1)).toBeNull();
  });
});

describe("liveFrame depth", () => {
  it("puts a seen hand in front of the shoulders by perspective, its elbow half as far, and leaves an unseen one", () => {
    // body (nose, ears, shoulders) seen at 0.5 image heights per metre; the hand at 0.6: 1.2x nearer. Its metric palm
    // (wrist to middle knuckle, points 0 and 9 of the ring) must be a hand's size to be believed: radius 0.03 / 0.6 * ~2
    const BODY: Record<number, [number, number]> = { 0: [0.5, 0.3], 7: [0.45, 0.3], 8: [0.55, 0.3], 11: [0.4, 0.45], 12: [0.6, 0.45] };
    const f = frame(null, null);
    const w = { pose: worldPose(), left: null as number[][] | null, right: null };
    for (const [i, [x, y]] of Object.entries(BODY)) {
      f.pose[+i] = [x, y, 0, 1];
      w.pose[+i] = [(x - 0.5) / 0.5, (y - 0.5) / 0.5, 0];
    }
    w.pose[13] = [-0.25, 0.2, 0]; // a left arm hanging, then bent up: 0.51 m long, reaching 0.43 m forward at most
    w.pose[15] = [-0.2, 0, 0];
    const ring = Array.from({ length: 21 }, (_, k) => [0.69 + 0.03 * Math.cos(k), 0.5 + 0.03 * Math.sin(k), 0] as [number, number, number]);
    f.left = ring; // labelled left and on the signer's left wrist (x 0.7)
    w.left = ring.map(([x, y]) => [(x - 0.69) / 0.6, (y - 0.5) / 0.6, 0]);
    const out = liveFrame(f, w, 1)!;
    const front = FOCAL / 0.5 - FOCAL / 0.6; // body distance - hand distance: 0.26 m
    expect(-out.pose[15][2]).toBeCloseTo(front, 3); // world z is smaller toward the camera; shoulders at 0
    expect(-out.pose[19][2]).toBeCloseTo(front, 3); // the hand's pose points come along
    expect(-out.pose[13][2]).toBeCloseTo(front / 2, 3);
    expect(out.pose[16][2]).toBe(0); // the unseen right hand keeps the pose model's depth
    expect(w.pose[15][2]).toBe(0); // the world pose given is not changed
    let eased = 0;
    liveFrame(f, w, 1, (_, v) => (eased = v) / 2);
    expect(eased).toBeCloseTo(front, 3);
  });
});

describe("liveFrame wrist anchoring", () => {
  it("puts the wrist where the hand tracker sees it, in the picture and (by the body's scale) in the world", () => {
    const BODY: Record<number, [number, number]> = { 0: [0.5, 0.3], 7: [0.45, 0.3], 8: [0.55, 0.3], 11: [0.4, 0.45], 12: [0.6, 0.45] };
    const f = frame(null, null);
    const w = { pose: worldPose(), left: null as number[][] | null, right: null };
    for (const [i, [x, y]] of Object.entries(BODY)) {
      f.pose[+i] = [x, y, 0, 1];
      w.pose[+i] = [(x - 0.5) / 0.5, (y - 0.5) / 0.5, 0]; // 0.5 image heights per metre
    }
    f.left = handAt(0.66); // the pose model's left wrist is at x 0.7; the hand tracker's 0.04 to its left
    const out = liveFrame(f, w, 1)!;
    expect(out.pose2d[15]).toEqual([0.66, 0.5]);
    expect(out.pose[15][0]).toBeCloseTo(-0.04 / 0.5, 6); // 0.04 image heights at 0.5 per metre: 8 cm
    expect(out.pose[19][0]).toBeCloseTo(-0.08, 6); // the hand's pose points come along
    expect(w.pose[15][0]).toBe(0);
  });
});

describe("reach", () => {
  it("is how far forward the wrist can go given how far across and up it already is, the elbow kept bent", () => {
    const p = worldPose();
    p[11] = [0, 0, 0]; p[13] = [0, 0.3, 0]; p[15] = [0, 0.3, 0]; // a 0.3 m arm whose wrist hangs 0.3 m below: no reach left
    expect(reach(p, "left")).toBeCloseTo(0, 6);
    p[15] = [0, 0.6, 0]; // 0.6 m arm hanging straight down: no forward reach
    expect(reach(p, "left")).toBeCloseTo(0, 6);
    p[13] = [0, 0.15, 0.26]; p[15] = [0, 0.3, 0]; // 0.6 m arm (0.3 + 0.3), wrist 0.3 m below: sqrt(0.36 - 0.09) forward
    expect(reach(p, "left")).toBeCloseTo(0.85 * Math.sqrt(0.27), 2);
  });
});

describe("liveFrame hands together", () => {
  it("marks hands whose nearest points touch in the picture as meeting, and gives them one depth", () => {
    const BODY: Record<number, [number, number]> = { 0: [0.5, 0.3], 7: [0.45, 0.3], 8: [0.55, 0.3], 11: [0.4, 0.45], 12: [0.6, 0.45] };
    const f = frame(null, null);
    const w = { pose: worldPose(), left: null as number[][] | null, right: null as number[][] | null };
    for (const [i, [x, y]] of Object.entries(BODY)) {
      f.pose[+i] = [x, y, 0, 1];
      w.pose[+i] = [(x - 0.5) / 0.5, (y - 0.5) / 0.5, 0];
    }
    // two rings almost touching at x 0.5, sized as a little nearer (left) and nearer still (right) than the body
    const ring = (cx: number, r: number) => Array.from({ length: 21 }, (_, k) => [cx + r * Math.cos(k), 0.5 + r * Math.sin(k), 0] as [number, number, number]);
    f.left = ring(0.535, 0.03);
    f.right = ring(0.465, 0.033);
    w.left = f.left.map(([x, y]) => [(x - 0.535) / 0.6, (y - 0.5) / 0.6, 0]);
    w.right = f.right.map(([x, y]) => [(x - 0.465) / 0.6, (y - 0.5) / 0.6, 0]);
    const out = liveFrame(f, w, 1)!;
    expect(out.meetL).toBe(1);
    expect(out.meetR).toBe(1);
    expect(out.pose[15][2]).toBeCloseTo(out.pose[16][2], 6);
    f.left = ring(0.9, 0.03); // far apart: no meeting
    w.left = f.left.map(([x, y]) => [(x - 0.9) / 0.6, (y - 0.5) / 0.6, 0]);
    expect(liveFrame(f, w, 1)!.meetL).toBeUndefined();
  });
});

describe("wristDepthCm", () => {
  it("is how far the wrist is in front of the shoulders (world z is smaller toward the camera)", () => {
    const p = worldPose();
    p[11][2] = p[12][2] = -0.1;
    p[15][2] = -0.4; // 30 cm nearer the camera than the shoulders
    p[16][2] = 0;
    expect(wristDepthCm(p, "left")).toBe(30);
    expect(wristDepthCm(p, "right")).toBe(-10);
  });
});

describe("fingerCurls", () => {
  const straight = () => {
    const h = Array.from({ length: 21 }, () => [0, 0, 0]);
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]].forEach((chain, c) =>
      chain.forEach((j, k) => (h[j] = [c * 0.02 - 0.04, -(k + 1) * 0.02, 0])));
    return h;
  };

  it("reads an open hand as straight fingers", () => {
    expect(fingerCurls(straight()).every((c) => c < 0.05)).toBe(true);
  });

  it("reads each finger on its own: a bent index, the rest straight", () => {
    const h = straight();
    h[7] = [h[6][0], h[6][1], -0.02]; // the middle joint bends 90 degrees toward the palm
    h[8] = [h[6][0], h[6][1] + 0.02, -0.02]; // and the last one 90 more
    const c = fingerCurls(h);
    expect(c[1]).toBeGreaterThan(0.8);
    expect([c[0], c[2], c[3], c[4]].every((v) => v < 0.05)).toBe(true);
  });
});

describe("signLabel", () => {
  it("shows the top guess as a word and percent, and the next two as alternatives", () => {
    const l = signLabel([{ gloss: "COFFEE", english: "coffee", p: 0.62 }, { gloss: "MAKE", english: "make", p: 0.2 },
      { gloss: "TEA", english: "tea", p: 0.1 }, { gloss: "X", english: "x", p: 0.01 }]);
    expect(l).toEqual({ word: "coffee", pct: 62, alts: ["make", "tea"] });
  });
});
