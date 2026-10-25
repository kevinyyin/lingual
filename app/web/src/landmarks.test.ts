import { describe, expect, it } from "vitest";
import { detect, toFrame, type Trackers } from "./landmarks";

const pt = (v: number) => ({ x: v, y: v, z: v, visibility: 0.9 });
const hand = (v: number) => Array.from({ length: 21 }, () => pt(v));
const fake = (opts: { hands?: string[]; pose?: boolean; face?: boolean }): Trackers => ({
  hands: { detectForVideo: () => ({ landmarks: (opts.hands ?? []).map((_, i) => hand(i)), handedness: (opts.hands ?? []).map((s) => [{ categoryName: s }]) }) } as any,
  pose: { detectForVideo: () => ({ landmarks: opts.pose === false ? [] : [Array.from({ length: 33 }, () => pt(0.5))] }) } as any,
  face: {
    detectForVideo: () => opts.face === false ? { faceLandmarks: [], faceBlendshapes: [], facialTransformationMatrixes: [] } : {
      faceLandmarks: [Array.from({ length: 478 }, () => pt(0.123456))],
      faceBlendshapes: [{ categories: [{ categoryName: "browInnerUp", score: 0.1234567 }] }],
      facialTransformationMatrixes: [{ data: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1] }],
    },
  } as any,
});
const run = (t: Trackers) => toFrame(detect(t, {} as ImageBitmap, 10, {}), 10);

describe("landmarks", () => {
  it("builds the frame the backend reads, with the face mesh only while a hand is up", () => {
    const f = run(fake({ hands: ["Left"] }))!;
    expect(f.t).toBe(10);
    expect(f.left).toHaveLength(21);
    expect(f.right).toBeNull();
    expect(f.pose).toHaveLength(33);
    expect(f.pose[0]).toEqual([0.5, 0.5, 0.5, 0.9]);
    expect(f.blend).toEqual({ browInnerUp: 0.12346 });
    expect(f.head).toEqual([0, -0, 0]);
    expect(f.face).toHaveLength(6);
    expect(f.mesh).toHaveLength(468);
    expect(f.mesh![0]).toEqual([0.123, 0.123, 0.123]);
    expect(run(fake({}))!.mesh).toBeUndefined();
  });

  it("is no frame without a body, and a frame with nulls without hands or a face", () => {
    expect(run(fake({ pose: false }))).toBeNull();
    const f = run(fake({ face: false }))!;
    expect([f.left, f.right, f.blend, f.head, f.face, f.mesh]).toEqual([null, null, null, null, undefined, undefined]);
  });

  it("merges trackers that ran apart (hands and pose in one worker, the face in another)", () => {
    const all = fake({ hands: ["Right"] });
    const split = toFrame({ ...detect({ face: all.face }, {} as ImageBitmap, 10, {}), ...detect({ hands: all.hands, pose: all.pose }, {} as ImageBitmap, 10, {}) }, 10);
    expect(split).toEqual(run(all));
  });
});
