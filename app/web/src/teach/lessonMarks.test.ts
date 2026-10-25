// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from "vitest";
import { getByRole } from "@testing-library/dom";
vi.mock("../ws", () => ({ connect: () => ({ send() {}, buffered: () => 0 }) }));
vi.mock("../camera", () => ({ loadTrackers: async () => {}, startCamera: () => new Promise(() => {}) }));
vi.mock("../avatar", () => ({ loadAvatar: () => new Promise(() => {}) }));
vi.mock("../avatar/stage", () => ({ createStage: vi.fn() }));
import { runLesson, type LessonHost } from "./lessonView";

const host: LessonHost = { english: (g) => g.toLowerCase(), sound: () => false, say() {},
  cameraOn: () => false, disableCamera() {}, quit() {}, save() {}, finish: () => { throw new Error("not reached"); } };
const opt = (word: string) => getByRole(document.body, "button", { name: new RegExp(`^\\d${word}$`) });
const answer = (word: string) => { opt(word).click(); getByRole(document.body, "button", { name: "Check" }).click(); };

beforeEach(() => runLesson(document.body, [{ kind: "watch", gloss: "HELLO", options: ["MEET", "GOODBYE", "NICE", "HELLO"] }], host));

test("a right answer gets a check and nothing is crossed", () => {
  answer("hello");
  expect(opt("hello").classList.contains("right")).toBe(true);
  expect(opt("hello").querySelector(".tile-mark")).not.toBeNull();
  expect(document.querySelector(".opt.wrong")).toBeNull();
  expect(document.querySelector(".verdict h3")!.textContent).toMatch(/^(Spot on\.|That's it!|Nailed it\.|Exactly right\.|Got it\.)$/);
});

test("a wrong answer crosses the choice and checks the right one", () => {
  answer("meet");
  expect(opt("meet").classList.contains("wrong")).toBe(true);
  expect(opt("meet").querySelector(".tile-mark")).not.toBeNull();
  expect(opt("hello").classList.contains("right")).toBe(true);
  expect(document.querySelector(".verdict")!.textContent).toContain("Not quite.");
  expect(document.querySelector(".verdict")!.textContent).toContain("Answer: hello");
});

test("Continue is the plain ink button, not a colored one", () => {
  answer("meet");
  expect(document.querySelector(".lesson-foot .btn.red, .lesson-foot .btn.green")).toBeNull();
});

test("a new sign's tag sits under the prompt, not above it", () => {
  runLesson(document.body, [{ kind: "intro", gloss: "HELLO" }], host);
  expect(document.querySelector(".prompt")!.previousElementSibling).toBeNull();
  expect(document.querySelector(".prompt + .ex-tag")?.textContent).toBe("New sign");
});

test("a retry's tag sits under the prompt too", () => {
  answer("meet");
  getByRole(document.body, "button", { name: /Continue/ }).click();
  expect(document.querySelector(".prompt")!.previousElementSibling).toBeNull();
  expect(document.querySelector(".prompt + .ex-tag.retry")?.textContent).toBe("Try this one again");
});
