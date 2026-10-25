import { expect, test } from "vitest";
import { CaptureCoach, faceWidthPx, meanLuma, type TipKind } from "./capture";
import type { Frame } from "./ws";

/** A seated signer like position.test.ts: ears `ear` apart (image x), head turned `yaw` degrees. */
function sitter({ ear = 0.2, yaw = 0 } = {}): Frame {
  const pose = Array.from({ length: 33 }, () => [0.5, 0.5, 0, 1] as [number, number, number, number]);
  pose[7] = [0.5 + ear / 2, 0.3, 0, 1];
  pose[8] = [0.5 - ear / 2, 0.3, 0, 1];
  return { t: 0, left: null, right: null, pose, blend: null, head: [yaw, 0, 0] };
}
/** Feeds ~30 fps from `from` for `secs`; returns the last change (a tip, null to hide) or undefined for none. */
const run = (c: CaptureCoach, f: Frame, from: number, secs: number, luma = 120, quiet = false) => {
  let out: TipKind | null | undefined;
  for (let t = from; t < from + secs * 1000; t += 33) { const r = c.feed(f, t, 640, () => luma, quiet); if (r !== undefined) out = r; }
  return out;
};

test("face width is measured in px of the video frame, like the eval", () => {
  expect(faceWidthPx(sitter({ ear: 0.1 }), 640)).toBeCloseTo(64);
  expect(faceWidthPx(sitter({ ear: 0.1 }), 1280)).toBeCloseTo(128);
  expect(faceWidthPx(sitter({ ear: 0.1, yaw: 40 }), 640)).toBeNull(); // a turned head says nothing about distance
});

test("a well-lit signer at a good distance gets no tip", () => {
  expect(run(new CaptureCoach(), sitter({ ear: 0.2 }), 0, 10)).toBeUndefined(); // 128 px
});

test("a face under 80 px asks to sit closer only after 2 checks (~2 s), not on the first frame", () => {
  const c = new CaptureCoach();
  expect(run(c, sitter({ ear: 0.1 }), 0, 1.9)).toBeUndefined(); // 64 px, one check so far
  expect(run(c, sitter({ ear: 0.1 }), 1900, 0.5)).toBe("small");
});

test("a turned head never asks to sit closer", () => {
  expect(run(new CaptureCoach(), sitter({ ear: 0.1, yaw: 40 }), 0, 5)).toBeUndefined();
});

test("a pause (the avatar signing) starts the hold over", () => {
  const c = new CaptureCoach();
  expect(run(c, sitter({ ear: 0.1 }), 0, 1.5)).toBeUndefined(); // one failing check
  expect(run(c, sitter({ ear: 0.1 }), 5000, 1.5)).toBeUndefined(); // back after 3.5 s: no check yet since the gap
  expect(run(c, sitter({ ear: 0.1 }), 6500, 1)).toBe("small");
});

test("dark wins over small", () => {
  expect(run(new CaptureCoach(), sitter({ ear: 0.1 }), 0, 3, 30)).toBe("dark");
});

test("nothing shows while quiet, and a shown tip hides the moment it goes quiet", () => {
  expect(run(new CaptureCoach(), sitter(), 0, 5, 30, true)).toBeUndefined();
  const c = new CaptureCoach();
  expect(run(c, sitter(), 0, 3, 30)).toBe("dark");
  expect(c.feed(sitter(), 3000, 640, () => 30, true)).toBeNull(); // signing started, or a position hint
});

test("a fixed problem hides the tip, the same tip repeats at most every 30 s, and tips are 10 s apart", () => {
  const c = new CaptureCoach();
  expect(run(c, sitter(), 0, 3, 30)).toBe("dark");
  expect(run(c, sitter(), 3000, 1.5, 120)).toBeNull(); // light turned on
  expect(run(c, sitter(), 4500, 20, 30)).toBeUndefined(); // dark again inside 30 s
  expect(run(c, sitter(), 24_500, 10, 30)).toBe("dark");

  const d = new CaptureCoach();
  expect(run(d, sitter({ ear: 0.1 }), 0, 3, 30)).toBe("dark");
  expect(run(d, sitter({ ear: 0.1 }), 3000, 1.5, 120)).toBeNull(); // light on; still too far, but within 10 s
  expect(run(d, sitter({ ear: 0.1 }), 4500, 5, 120)).toBeUndefined();
  expect(run(d, sitter({ ear: 0.1 }), 9500, 3, 120)).toBe("small");
});

test("a tip shown expires after 6 s", () => {
  const c = new CaptureCoach();
  expect(run(c, sitter(), 0, 3, 30)).toBe("dark");
  expect(run(c, sitter(), 3000, 7, 30)).toBeNull();
});

test("mean luma matches cv2 gray weights", () => {
  expect(meanLuma(new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 255]))).toBeCloseTo((0.299 + 0.587) * 255 / 2);
});
