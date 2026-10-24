import { expect, test } from "vitest";
import { ICON } from "./icons";

const OLD = ["#ff9600", "#e07f00", "#ff4b4b", "#ea2b2b", "#ffd36b", "#e08a00", "#b36b00"];

test("no icon carries one of the old colors", () => {
  const all = Object.values(ICON).join("").toLowerCase();
  for (const hex of OLD) expect(all).not.toContain(hex);
});

test("the mark is the motion arrow: a start dot and a curved arrow in go", () => {
  expect(ICON.mark).toMatch(/<circle[^>]*fill="#18943F"/);
  expect(ICON.mark).toMatch(/<path[^>]*stroke="#18943F"/);
});
