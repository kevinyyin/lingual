// @vitest-environment jsdom
import { afterEach, expect, test, vi } from "vitest";

const play = vi.fn((_plan: unknown) => Promise.resolve());
vi.mock("./index", () => ({ loadAvatar: vi.fn(() => Promise.resolve({})) }));
vi.mock("./stage", () => ({ createStage: vi.fn(() => Promise.resolve({ play })) }));
HTMLCanvasElement.prototype.getContext = (() => null) as never; // jsdom has no WebGL

const flush = () => new Promise((r) => setTimeout(r, 0));
afterEach(() => { vi.useRealTimers(); play.mockClear(); });

test("pops up in its host, captioned, and signs what was read", async () => {
  const { showMini } = await import("./mini");
  const host = document.createElement("div");
  showMini(host, [{ kind: "sign", gloss: "COFFEE" }], "I saw: coffee");
  const box = host.querySelector<HTMLButtonElement>(".mini-buddy")!;
  expect(box.textContent).toBe("I saw: coffee");
  expect(box.getAttribute("aria-label")).toBe("Play again: I saw: coffee");
  await flush();
  expect(play.mock.calls[0][0]).toMatchObject({ type: "sign_plan", tokens: [{ kind: "sign", gloss: "COFFEE" }] });
});

test("fades away a while after playing; a tap before then plays it again", async () => {
  const { showMini, LINGER_MS, FADE_MS } = await import("./mini");
  const host = document.createElement("div");
  vi.useFakeTimers();
  showMini(host, [{ kind: "spell", word: "ANA" }], "I saw: ANA");
  await vi.advanceTimersByTimeAsync(LINGER_MS - 100);
  host.querySelector<HTMLButtonElement>(".mini-buddy")!.click();
  await vi.advanceTimersByTimeAsync(0);
  expect(play).toHaveBeenCalledTimes(2);
  await vi.advanceTimersByTimeAsync(LINGER_MS + FADE_MS);
  expect(host.querySelector(".mini-buddy")).toBeNull();
});

test("after a new buddy is picked, the mini one is the new buddy", async () => {
  const { loadAvatar } = await import("./index");
  const { showMini } = await import("./mini");
  const host = document.createElement("div");
  showMini(host, [{ kind: "sign", gloss: "HELLO" }], "I saw: hello");
  const before = vi.mocked(loadAvatar).mock.calls.length;
  localStorage.setItem("lingual.buddy", "lexi");
  showMini(host, [{ kind: "sign", gloss: "HELLO" }], "I saw: hello");
  expect(vi.mocked(loadAvatar).mock.calls.length).toBe(before + 1);
  localStorage.removeItem("lingual.buddy");
});
