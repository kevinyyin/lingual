import { expect, test } from "vitest";
import { PositionWatch, positionHint } from "./position";
import type { Frame } from "./ws";

/** A seated signer: shoulders `width` apart around `mid` (image x), hands resting low unless `handsUp`. */
function sitter({ width = 0.4, mid = 0.5, face = 1, handsUp = false } = {}): Frame {
  const pose = Array.from({ length: 33 }, () => [0.5, 0.5, 0, 1] as [number, number, number, number]);
  pose[0] = [mid, 0.3, 0, face]; // nose
  pose[11] = [mid + width / 2, 0.55, 0, 1]; // shoulders (the camera's left is the signer's right)
  pose[12] = [mid - width / 2, 0.55, 0, 1];
  pose[15] = pose[16] = [mid, handsUp ? 0.5 : 1.05, 0, 0.4]; // wrists: in the lap is below the picture
  return { t: 0, left: null, right: null, pose, blend: null, head: null };
}

test("a centred signer at a good distance with resting hands is in position", () => {
  expect(positionHint(sitter(), true)).toBeNull();
});

test("it says how to get into position", () => {
  expect(positionHint(sitter({ width: 0.7 }), true)).toBe("Move back a little");
  expect(positionHint(sitter({ width: 0.15 }), true)).toBe("Come a little closer");
  expect(positionHint(sitter({ mid: 0.2 }), true)).toBe("Scoot to the middle of the picture");
  expect(positionHint(sitter({ face: 0.2 }), true)).toBe("Get your face and shoulders in the picture");
});

test("raised hands matter only while the camera learns the resting pose", () => {
  expect(positionHint(sitter({ handsUp: true }), true)).toBe("Rest your hands in your lap");
  expect(positionHint(sitter({ handsUp: true }), false)).toBeNull(); // signing
});

test("moving clearly elsewhere for a moment asks for a new calibration; a lean while signing doesn't", () => {
  const w = new PositionWatch();
  expect(w.moved(sitter(), 0)).toBe(false); // the reference
  expect(w.moved(sitter({ width: 0.6 }), 100)).toBe(false); // leaning in
  expect(w.moved(sitter(), 400)).toBe(false); // and back
  expect(w.moved(sitter({ mid: 0.75 }), 1000)).toBe(false);
  expect(w.moved(sitter({ mid: 0.75 }), 2600)).toBe(true); // scooted over and stayed
  w.reset();
  expect(w.moved(sitter({ mid: 0.75 }), 2700)).toBe(false); // the new place is the reference
});

test("having moved, it asks to calibrate once, not on every frame until the backend answers", () => {
  const w = new PositionWatch();
  w.check(sitter(), "ready", 0);
  w.check(sitter({ mid: 0.75 }), "ready", 100);
  expect(w.check(sitter({ mid: 0.75 }), "ready", 1700).recalibrate).toBe(true);
  expect(w.check(sitter({ mid: 0.75 }), "ready", 1730).recalibrate).toBe(false); // the backend still says ready
});
