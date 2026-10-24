import { expect, test } from "vitest";
import { lookupSign } from "./showMe";

const V = [
  { gloss: "THANK-YOU", english: "thank you", avatar_ok: true },
  { gloss: "DOG", english: "dog", avatar_ok: true },
];

test("matches english or gloss", () => {
  expect(lookupSign(" Thank You ", V)).toEqual({ kind: "sign", gloss: "THANK-YOU" });
  expect(lookupSign("dog", V)).toEqual({ kind: "sign", gloss: "DOG" });
  expect(lookupSign("thank-you", V)).toEqual({ kind: "sign", gloss: "THANK-YOU" });
});

test("unknown words are fingerspelled, junk is nothing", () => {
  expect(lookupSign("Kev1n!", V)).toEqual({ kind: "spell", word: "KEVN" });
  expect(lookupSign("   ", V)).toBeNull();
  expect(lookupSign("123", V)).toBeNull();
});

test("prefers a sign the avatar can show", () => {
  const v = [{ gloss: "CAT", english: "cat", avatar_ok: false }, { gloss: "CAT-2", english: "cat", avatar_ok: true }];
  expect(lookupSign("cat", v)).toEqual({ kind: "sign", gloss: "CAT-2" });
});
