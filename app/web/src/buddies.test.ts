import { expect, test } from "vitest";
import { savedBuddy, saveBuddy } from "./buddies";

test("Leo until a buddy is picked; the pick is remembered", () => {
  localStorage.clear();
  expect(savedBuddy().id).toBe("leo");
  saveBuddy("daisy");
  expect(savedBuddy().id).toBe("daisy");
});

test("a saved name that is no buddy falls back to Leo", () => {
  localStorage.setItem("lingual.buddy", "bob");
  expect(savedBuddy().id).toBe("leo");
});
