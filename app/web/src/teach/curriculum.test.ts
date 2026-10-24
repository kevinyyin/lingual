import { describe, expect, test } from "vitest";
import { faceFor } from "./curriculum";

describe("faceFor", () => {
  test("a feeling sign brings its face", () => {
    expect(faceFor(["HAPPY"])).toBe("warm");
    expect(faceFor(["SORRY"])).toBe("apologetic");
    expect(faceFor(["SAD"])).toBe("sad");
    expect(faceFor(["ANGRY"])).toBe("angry");
  });
  test("other signs, a spelled word and mixed feelings stay neutral", () => {
    expect(faceFor(["HELLO"])).toBe("neutral");
    expect(faceFor([])).toBe("neutral"); // a fingerspelled word has no sign glosses
    expect(faceFor(["HAPPY", "SAD", "HELLO"])).toBe("neutral");
  });
  test("agreeing feeling signs share their face, and other signs don't count", () => {
    expect(faceFor(["HAPPY", "GOOD", "HELLO"])).toBe("warm");
    expect(faceFor(["TIRED", "SICK"])).toBe("sad");
  });
});
