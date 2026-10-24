// @vitest-environment jsdom
import { expect, test, vi } from "vitest";
vi.mock("./signer", () => ({
  mountSigner: vi.fn(),
  loadStage: () => new Promise(() => {}),
  playSigns: vi.fn(() => Promise.resolve()),
}));
import { ALL_LESSONS } from "./curriculum";
import { coursePath, renderHome, TAB_IDS, type HomeCtx } from "./home";
import { fresh } from "./progress";
import { mountSigner, playSigns } from "./signer";

const EMOJI = /\p{Extended_Pictographic}/u;
const ctx = (done: string[] = []): HomeCtx => {
  const p = fresh();
  for (const id of done) p.done[id] = 1;
  return { p, settings: { sound: true, camera: true }, english: (g) => g.toLowerCase(), update() {}, updateSettings() {},
    start: vi.fn(), startPractice() {}, words: [], showMe() {}, reset() {}, pickBuddy: vi.fn() };
};

test("the course line leads in from the top of the unit", () => {
  expect(coursePath([{ x: 50, y: 64 }, { x: 70, y: 192 }], 1)).toMatch(/^M50 0 L50 64 C/);
});

test("only the active lesson carries the arrowhead", () => {
  renderHome(document.body, "learn", ctx());
  const heads = document.querySelectorAll(".stop .arrowhead");
  expect(heads).toHaveLength(1);
  expect(heads[0].closest(".stop")!.classList).toContain("active");
});

test("a finished course has no arrowhead and no play button", () => {
  renderHome(document.body, "learn", ctx(ALL_LESSONS.map((l) => l.id)));
  expect(document.querySelector(".arrowhead")).toBeNull();
  expect(document.querySelector(".hero-play")).toBeNull();
  expect(document.querySelector(".hero")!.textContent).toContain("You finished the course!");
});

test("the hero's play button signs the next lesson's first sign", () => {
  renderHome(document.body, "learn", ctx());
  document.querySelector<HTMLButtonElement>(".hero-play")!.click();
  expect(playSigns).toHaveBeenCalledWith(["HELLO"]);
});

test("closing a unit's notes gives the signer back to the hero", () => {
  renderHome(document.body, "learn", ctx());
  document.querySelector<HTMLButtonElement>(".ub-notes")!.click();
  document.querySelector<HTMLButtonElement>(".modal-actions .btn")!.click();
  expect(vi.mocked(mountSigner).mock.lastCall![0]).toBe(document.querySelector(".hero-avatar"));
});

test("the course has no greeting and no emoji, and units have no banners", () => {
  renderHome(document.body, "learn", ctx());
  expect(document.body.textContent).not.toMatch(/Good (morning|afternoon|evening)/);
  expect(document.body.textContent).not.toMatch(EMOJI);
  expect(document.querySelector(".unit-banner")).toBeNull();
  expect(document.querySelector(".unit-head p")!.textContent).toBe("Unit 1 · 0/4 done · Say hello, thank someone, introduce yourself");
});

test.each(TAB_IDS)("no emoji on the %s tab", (tab) => {
  renderHome(document.body, tab, ctx());
  expect(document.body.textContent).not.toMatch(EMOJI);
});

test("the XP chart is drawn at its own width, so its 15px labels stay 15px on a phone", async () => {
  const width = vi.spyOn(HTMLElement.prototype, "clientWidth", "get").mockReturnValue(327);
  renderHome(document.body, "profile", ctx());
  await new Promise((r) => requestAnimationFrame(r));
  expect(document.querySelector(".chart svg")!.getAttribute("viewBox")).toBe("0 0 327 200");
  width.mockRestore();
});

test("the profile shows the mark", () => {
  renderHome(document.body, "profile", ctx());
  expect(document.querySelector(".big-av svg")).not.toBeNull();
});

test("the profile picks a signing buddy: six headshots, the saved one pressed", () => {
  localStorage.setItem("lingual.buddy", "lexi");
  const c = ctx();
  renderHome(document.body, "profile", c);
  const buttons = [...document.querySelectorAll<HTMLButtonElement>(".buddies button")];
  expect(buttons.map((b) => b.getAttribute("aria-label"))).toEqual(["Leo", "Benji", "Zeke", "Lexi", "Avery", "Daisy"]);
  expect(buttons.filter((b) => b.getAttribute("aria-pressed") === "true").map((b) => b.getAttribute("aria-label"))).toEqual(["Lexi"]);
  buttons[5].click();
  expect(c.pickBuddy).toHaveBeenCalledWith("daisy");
});

test("a unit's notes put the heading first, the unit line under it", () => {
  renderHome(document.body, "learn", ctx());
  document.querySelector<HTMLButtonElement>(".ub-notes")!.click();
  expect(document.querySelector(".modal")!.firstElementChild!.tagName).toBe("H2");
  expect(document.querySelector(".modal h2")!.nextElementSibling!.textContent).toBe("Unit 1 notes");
  document.querySelector<HTMLButtonElement>(".modal-actions .btn")!.click();
});

test("the hero sizes the next lesson's title by its longest word", () => {
  renderHome(document.body, "learn", ctx(["u1-1", "u1-2", "u1-3", "u1-review", "u2-1", "u2-2"]));
  expect(document.querySelector<HTMLElement>(".hero h1 span")!.style.getPropertyValue("--chars")).toBe("9");
});
