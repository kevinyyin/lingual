import { describe, expect, test } from "vitest";
import { buildLesson, LessonRun, signScore, type Exercise } from "./lesson";

const seeded = (s = 1) => () => ((s = (s * 16807) % 2147483647) / 2147483647);
const VOCAB = ["HELLO", "GOODBYE", "NICE", "MEET", "THANK-YOU", "PLEASE", "SORRY", "WELCOME", "YES", "NO"];
const opts = (over = {}) => ({
  signs: ["HELLO", "GOODBYE", "NICE", "MEET"], known: ["HELLO", "GOODBYE", "NICE", "MEET"], vocab: VOCAB,
  review: false, perform: true, rng: seeded(), ...over,
});

describe("buildLesson", () => {
  test("introduces each new sign before any other exercise uses it", () => {
    const ex = buildLesson(opts());
    for (const g of ["HELLO", "GOODBYE", "NICE", "MEET"]) {
      const intro = ex.findIndex((e) => e.kind === "intro" && e.gloss === g);
      const firstUse = ex.findIndex((e) => e.kind !== "intro" && ("gloss" in e ? e.gloss === g : e.glosses.includes(g)));
      expect(intro).toBeGreaterThanOrEqual(0);
      expect(intro).toBeLessThan(firstUse);
    }
  });

  test("each new sign is tried on camera right after its intro, before any other exercise", () => {
    const ex = buildLesson(opts());
    expect(ex.slice(0, 8).map((e) => e.kind)).toEqual(Array(4).fill(["intro", "perform"]).flat());
    for (let i = 0; i < 8; i += 2) expect((ex[i + 1] as { gloss: string }).gloss).toBe((ex[i] as { gloss: string }).gloss);
    const off = buildLesson(opts({ perform: false }));
    expect(off.slice(0, 4).every((e) => e.kind === "intro")).toBe(true);
  });

  test("choices hold the answer, no duplicates, and fill from the vocabulary early on", () => {
    const ex = buildLesson(opts({ signs: ["HELLO", "GOODBYE"], known: ["HELLO", "GOODBYE"] }));
    for (const e of ex) {
      if (e.kind !== "watch" && e.kind !== "pick-sign") continue;
      expect(e.options).toContain(e.gloss);
      expect(new Set(e.options).size).toBe(e.options.length);
      expect(e.options.length).toBe(e.kind === "watch" ? 4 : 3);
    }
  });

  test("the whole lesson: each sign shown then signed, then each recognized, and nothing more", () => {
    const kinds = (o = {}) => buildLesson(opts(o)).map((e) => e.kind).join(" ");
    expect(kinds()).toBe("intro perform ".repeat(4) + "watch watch watch watch");
    expect(kinds({ perform: false })).toBe("intro ".repeat(4) + "watch watch watch watch");
    expect(kinds({ review: true, signs: VOCAB })).toBe("intro perform ".repeat(8) + Array(8).fill("watch").join(" ")); // 8 of a review's signs
  });
});

describe("LessonRun", () => {
  const watch = (g: string): Exercise => ({ kind: "watch", gloss: g, options: [g] });

  test("a miss goes to the back of the queue and the lesson ends once it is answered right", () => {
    const run = new LessonRun([watch("A"), watch("B")]);
    run.answer(false);
    expect(run.current!.ex).toEqual(watch("B"));
    run.answer(true);
    expect(run.current).toEqual({ ex: watch("A"), retry: true });
    expect(run.fraction).toBe(0.5);
    run.answer(true);
    expect(run.finished).toBe(true);
    expect(run.mistakes).toBe(1);
    expect(run.accuracy).toBeCloseTo(2 / 3);
  });

  test("combo resets on a miss; skipping moves on without a mistake", () => {
    const run = new LessonRun([watch("A"), watch("B"), watch("C"), { kind: "perform", gloss: "D" }]);
    run.answer(true); run.answer(true); run.answer(false); run.skip();
    expect(run.bestCombo).toBe(2);
    expect(run.combo).toBe(0);
    expect(run.mistakes).toBe(1);
    expect(run.current!.ex).toEqual(watch("C"));
  });

  test("back undoes the last answer, and any skips after it", () => {
    const run = new LessonRun([watch("A"), { kind: "perform", gloss: "B" }, watch("C")]);
    expect(run.canGoBack).toBe(false);
    run.answer(false); // A goes to the back of the queue
    run.skip();
    expect(run.current!.ex).toEqual(watch("C"));
    expect(run.back()).toBe(true);
    expect(run.current).toEqual({ ex: watch("A"), retry: false });
    expect([run.done, run.mistakes, run.answers.length, run.canGoBack]).toEqual([0, 0, 0, false]);
    expect(run.back()).toBe(false);
  });

  test("a saved run picks up where it was, history and all", () => {
    const run = new LessonRun([watch("A"), watch("B"), watch("C")]);
    run.answer(true);
    run.answer(false);
    const again = new LessonRun([watch("X")], JSON.parse(JSON.stringify(run.save()))); // through storage, as store.ts does
    expect([again.current!.ex, again.done, again.total, again.mistakes]).toEqual([watch("C"), 1, 3, 1]);
    again.back();
    expect(again.current!.ex).toEqual(watch("B"));
  });
});

describe("signScore", () => {
  const o = (gloss: string, p: number) => ({ gloss, english: gloss.toLowerCase(), p });

  test("the top guess scores 100, a lower one relative to it, a missing one 0", () => {
    expect(signScore([[o("HELLO", 0.4), o("NICE", 0.2)]], "HELLO")).toEqual({ score: 100, rank: 1 });
    expect(signScore([[o("HELLO", 0.4), o("NICE", 0.2)]], "NICE")).toEqual({ score: 50, rank: 2 });
    expect(signScore([[o("HELLO", 0.4)]], "MEET")).toEqual({ score: 0, rank: null });
  });

  test("the best segment counts", () => {
    expect(signScore([[o("NICE", 0.5), o("HELLO", 0.1)], [o("HELLO", 0.3), o("NICE", 0.1)]], "HELLO").score).toBe(100);
  });
});
