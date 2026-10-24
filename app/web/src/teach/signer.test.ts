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
