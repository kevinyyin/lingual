import { describe, expect, test } from "vitest";
import { COURSE, faceFor } from "./curriculum";

describe("faceFor", () => {
  test("a feeling sign brings its face", () => {
    expect(faceFor(["HAPPY"])).toBe("warm");
    expect(faceFor(["SORRY"])).toBe("apologetic");
  });
  test("other signs, a spelled word and mixed feelings stay neutral", () => {
    expect(faceFor(["HELLO"])).toBe("neutral");
    expect(faceFor([])).toBe("neutral"); // a fingerspelled word has no sign glosses
    expect(faceFor(["HAPPY", "SAD", "HELLO"])).toBe("neutral");
  });
  test("agreeing feeling signs share their face, and other signs don't count", () => {
    expect(faceFor(["HAPPY", "GOOD", "HELLO"])).toBe("warm");
    expect(faceFor(["TIRED", "TIRED"])).toBe("apologetic");
  });
});

test("unit 1 is one short lesson, HELLO, GOODBYE and MEET, with no review that would only repeat it", () => {
  expect(COURSE[0].lessons.map((l) => l.signs)).toEqual([["HELLO", "GOODBYE", "MEET"]]);
  expect(COURSE[1].lessons.at(-1)!.id).toBe("u2-review"); // longer units keep theirs
});
