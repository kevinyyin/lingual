// @vitest-environment jsdom
import { expect, test, vi } from "vitest";

/** A Web Audio context as iOS makes one: suspended until resumed (which only works during a tap). */
class FakeContext {
  static made: FakeContext[] = [];
  state = "suspended";
  currentTime = 0;
  destination = {};
  played = 0;
  constructor() { FakeContext.made.push(this); }
  resume() { this.state = "running"; return Promise.resolve(); }
  createOscillator() {
    return { type: "", frequency: { value: 0 }, connect: (x: unknown) => x, start: () => { this.played++; }, stop() {} };
  }
  createGain() { return { gain: { setValueAtTime() {}, exponentialRampToValueAtTime() {} }, connect: (x: unknown) => x }; }
}

test("a right answer's chime plays through the output a tap unlocked, even when a camera result triggers it", async () => {
  vi.stubGlobal("AudioContext", FakeContext);
  const { unlockAudio } = await import("../phone");
  const { chime } = await import("./dom");
  unlockAudio(); // the learner's first tap
  chime("right"); // later, from a camera-graded answer: no tap to start an output of its own
  expect(FakeContext.made).toHaveLength(1); // one output, the unlocked one (a chime's own stayed suspended: silent)
  expect(FakeContext.made[0].state).toBe("running");
  expect(FakeContext.made[0].played).toBe(3); // all three notes
});
