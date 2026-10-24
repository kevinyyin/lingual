// @vitest-environment jsdom
import { expect, test, vi } from "vitest";
vi.mock("../ws", () => ({ connect: () => ({ send() {}, buffered: () => 0 }) }));
vi.mock("../camera", () => ({ startCamera: () => new Promise(() => {}) }));
vi.mock("../avatar", () => ({ loadAvatar: () => new Promise(() => {}) }));
vi.mock("../avatar/stage", () => ({ createStage: vi.fn() }));

test("mountSigner moves both canvases into the new slot", async () => {
  const { avatarCanvas, mountSigner, trailCanvas } = await import("./signer");
  const a = document.createElement("div"), b = document.createElement("div");
  mountSigner(a);
  mountSigner(b);
  expect([...b.children].slice(0, 2)).toEqual([avatarCanvas, trailCanvas]);
  expect(a.children).toHaveLength(0);
});

test("the learner's palm lands where the mirrored, cropped preview shows it, and a hand in the lap draws nothing", async () => {
  const { cameraPalm } = await import("./signer");
  const pose = Array.from({ length: 33 }, () => [0.5, 0.5, 0, 1]);
  for (const i of [23, 24]) pose[i] = [0.5, 1.1, 0, 0.5]; // hips below the picture, as on a laptop
  const hand = (wristY: number) => Array.from({ length: 21 }, (_, i) => (i === 9 ? [0.25, 0.5, 0] : [0.25, wristY, 0])) as [number, number, number][];
  expect(cameraPalm(hand(0.6), pose, 400, 300, 640, 480)).toEqual([300, 150]); // same shape: mirrored only
  expect(cameraPalm(hand(0.6), pose, 300, 300, 640, 480)).toEqual([250, 150]); // a square slot crops 50 px off each side
  expect(cameraPalm(hand(0.95), pose, 400, 300, 640, 480)).toBeNull(); // wrist past 70% of shoulders -> hips
  expect(cameraPalm(null, pose, 400, 300, 640, 480)).toBeNull();
});

test("the face reader tells a smile and a frown from a resting face, as the server's emotion reader does", async () => {
  const { faceLabel, faceScores } = await import("./signer");
  expect(faceLabel(faceScores({ mouthSmileLeft: 0.7, mouthSmileRight: 0.6, cheekSquintLeft: 0.3, cheekSquintRight: 0.3 }))).toBe("happy");
  expect(faceLabel(faceScores({ mouthFrownLeft: 0.5, mouthFrownRight: 0.5 }))).toBe("sad");
  expect(faceLabel(faceScores({ noseSneerLeft: 0.6, noseSneerRight: 0.6, mouthPressLeft: 0.4, mouthPressRight: 0.4 }))).toBe("angry");
  expect(faceLabel(faceScores({ mouthSmileLeft: 0.05, browInnerUp: 0.9 }))).toBe("neutral"); // brows are grammar, not feeling
  expect(faceLabel(faceScores(null))).toBe("neutral");
});
