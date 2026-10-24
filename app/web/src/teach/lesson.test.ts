import { describe, expect, test } from "vitest";
import { buildLesson, LessonRun, signScore, type Exercise } from "./lesson";
import { fresh } from "./progress";

const seeded = (s = 1) => () => ((s = (s * 16807) % 2147483647) / 2147483647);
const VOCAB = ["HELLO", "GOODBYE", "NICE", "MEET", "THANK-YOU", "PLEASE", "SORRY", "WELCOME", "YES", "NO"];
const opts = (over = {}) => ({
  signs: ["HELLO", "GOODBYE", "NICE", "MEET"], known: ["HELLO", "GOODBYE", "NICE", "MEET"], vocab: VOCAB,
  review: false, perform: true, progress: fresh(), today: "2026-10-21", rng: seeded(), ...over,
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

  test("camera exercises only when allowed, and review lessons have no intros", () => {
    expect(buildLesson(opts({ perform: false })).some((e) => e.kind === "perform")).toBe(false);
    expect(buildLesson(opts()).filter((e) => e.kind === "perform")).toHaveLength(4); // one per new sign
    expect(buildLesson(opts({ review: true })).some((e) => e.kind === "intro")).toBe(false);
  });

  test("signs already learned are not introduced again", () => {
    const progress = { ...fresh(), signs: { HELLO: { correct: 3, wrong: 0, strength: 3, lastSeen: "2026-10-21" } } };
    expect(buildLesson(opts({ progress })).some((e) => e.kind === "intro" && e.gloss === "HELLO")).toBe(false);
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
