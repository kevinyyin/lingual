import { describe, expect, test } from "vitest";
import type { Tone } from "../ws";
import { RAMP_MS, faceTrack } from "./face";

describe("faceTrack", () => {
  test("yn holds browUp through middle of sentence", () => {
    const f = faceTrack("yn", "neutral", 2000);
    for (const t of [400, 1000, 1600]) expect(f(t).browUp).toBeGreaterThan(0.7);
    expect(f(1000).browDown).toBe(0);
  });

  test("wh lowers brows", () => {
    expect(faceTrack("wh", "neutral", 2000)(1000).browDown).toBeGreaterThan(0.7);
  });

  test("neg yaw crosses zero at least 4 times over 2 s", () => {
    const f = faceTrack("neg", "neutral", 2000);
    let crossings = 0;
    let prev = f(RAMP_MS).headYawDeg;
    for (let t = RAMP_MS + 10; t < 2000 - RAMP_MS; t += 10) {
      const y = f(t).headYawDeg;
      if (Math.sign(y) !== Math.sign(prev) && y !== 0) crossings++;
      prev = y;
    }
    expect(crossings).toBeGreaterThanOrEqual(4);
  });

  test("tone never sets brow values", () => {
    for (const tone of ["warm", "apologetic", "sad", "angry", "urgent", "neutral"] as const) {
      const f = faceTrack("statement", tone, 1500);
      for (let t = 0; t <= 1500; t += 50) {
        expect(f(t).browUp).toBe(0);
        expect(f(t).browDown).toBe(0);
        expect(Object.keys(f(t).expressions).some((k) => k.startsWith("BRW"))).toBe(false);
      }
    }
  });

  test("ramps start and end at zero", () => {
    const f = faceTrack("yn", "warm", 1500);
    for (const t of [0, 1500]) {
      expect(f(t).browUp).toBeCloseTo(0);
      expect(Object.values(f(t).expressions).every((v) => Math.abs(v as number) < 1e-9)).toBe(true);
    }
  });

  test("apologetic tone lowers the gaze, warm tone smiles", () => {
    expect(faceTrack("statement", "apologetic", 1500)(750).headPitchDeg).toBeGreaterThan(0);
    expect(faceTrack("statement", "warm", 1500)(750).expressions.MTH_Joy).toBeGreaterThan(0.3);
  });

  test("sad tone looks down further than apologetic; angry keeps the head level", () => {
    const pitch = (tone: Tone) => faceTrack("statement", tone, 1500)(750).headPitchDeg;
    expect(pitch("sad")).toBeGreaterThan(pitch("apologetic"));
    expect(pitch("angry")).toBe(0);
  });
});

test("tone shows at full strength mid-sentence, on the mouth and eyes only", () => {
  const at = (tone: Tone) => faceTrack("statement", tone, 1500)(750).expressions;
  expect(at("warm")).toEqual({ MTH_Joy: 1, EYE_Joy: 0.7 });
  expect(at("apologetic")).toEqual({ MTH_Sorrow: 0.9, EYE_Sorrow: 0.6 });
  expect(at("sad")).toEqual({ MTH_Sorrow: 1, EYE_Sorrow: 1 });
  expect(at("angry")).toEqual({ MTH_Angry: 1, EYE_Angry: 1 });
  expect(at("urgent")).toEqual({ MTH_Surprised: 0.6, EYE_Surprised: 0.8 });
  expect(at("neutral")).toEqual({});
});
