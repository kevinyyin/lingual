/// <reference types="node" />
// (Node's types come with Vite. A ?raw import won't do here: Vitest hands CSS files back empty.)
import { readFileSync } from "node:fs";
import { expect, test } from "vitest";

const OLD = ["#1cc6d6", "#10a5b4", "#8de4ec", "#0b98a6", "#dcf8fb", "#1cb0f6", "#ddf4ff", "#1899d6", "#ce82ff", "#f4e4ff",
  "#a560d4", "#ff4b4b", "#ea2b2b", "#ffdfe0", "#ffc800", "#ff9600", "#fff4cc", "#cd7900", "#4b4b4b", "#777777",
  "#afafaf", "#e5e5e5", "#58cc02"];
const FILES = ["teach/teach.css", "style.css", "../index.html", "../teach.html", "../shell/index.html"];
const HERE = import.meta.url; // not inline: Vite rewrites new URL(..., import.meta.url) into an asset lookup
const read = (f: string) => readFileSync(new URL(`./${f}`, HERE), "utf8");

test.each(FILES)("%s uses none of the old colors or fonts", (f) => {
  const s = read(f).toLowerCase();
  for (const hex of OLD) expect(s).not.toContain(hex);
  expect(s).not.toMatch(/bricolage|dm sans/);
});

test.each(FILES)("%s has no ledge shadows", (f) => {
  expect(read(f)).not.toMatch(/box-shadow:[^;]*\b0(px)?\s+\d+(\.\d+)?px\s+0(px)?\s/);
});

test("teach.css styles answer marks as svg.mark, not every .mark (the logo has one too)", () => {
  expect(read("teach/teach.css")).not.toMatch(/^\s*\.mark\b/m);
});

test.each(["teach/main.ts", "teach/home.ts", "teach/lessonView.ts"])("%s puts no emoji on screen", (f) => {
  expect(read(f)).not.toMatch(/\p{Emoji_Presentation}|️/u); // text arrows like ↔ are fine; emoji-style glyphs aren't
});

test("the hand trail draws over the sign's label and under the play controls", () => {
  const css = read("teach/teach.css");
  const z = (sel: string) => Number(new RegExp(`${sel}\\s*\\{[^}]*z-index:\\s*(\\d+)`).exec(css)?.[1] ?? 0);
  expect(z("\\.trail-canvas")).toBeGreaterThan(z("\\.avatar-slot \\.sign-label"));
  expect(z("\\.avatar-slot \\.controls")).toBeGreaterThan(z("\\.trail-canvas"));
  expect(z("\\.lesson-foot")).toBeGreaterThan(z("\\.avatar-slot \\.controls")); // the sticky foot stays on top while the lesson scrolls
});

test("a wrong match pair flashes rose, so a kid sees the miss", () => {
  expect(read("teach/teach.css")).toMatch(/\.match \.opt\.wrong\s*\{[^}]*var\(--rose\)/);
});

test("the translator page has no emoji", () => {
  expect(read("../index.html")).not.toMatch(/\p{Extended_Pictographic}/u);
});

test.each(FILES)("%s has no bounce easing", (f) => {
  for (const [, args] of read(f).matchAll(/cubic-bezier\(([^)]*)\)/g)) {
    const [, y1, , y2] = args.split(",").map(Number);
    for (const y of [y1, y2]) { expect(y).toBeGreaterThanOrEqual(0); expect(y).toBeLessThanOrEqual(1); }
  }
});
