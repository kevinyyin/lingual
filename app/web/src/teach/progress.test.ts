import { describe, expect, test } from "vitest";
import { catchUp, finishLesson, fresh, quests, signStrength } from "./progress";

const at = (d: string, hh = 12) => new Date(`${d}T${String(hh).padStart(2, "0")}:00`);
const lesson = (mistakes = 0, answers = [{ gloss: "HELLO", correct: true }]) => ({ lessonId: "u1-1", answers, mistakes, bestCombo: 5, ms: 60_000 });

describe("streak", () => {
  test("first lesson starts it, the next day extends it, a second lesson the same day doesn't", () => {
    let p = finishLesson(fresh(), lesson(), at("2026-10-20")).progress;
    expect(p.streak).toBe(1);
    const again = finishLesson(p, lesson(), at("2026-10-20", 18));
    expect(again.progress.streak).toBe(1);
    expect(again.earned.streakExtended).toBe(false);
    p = finishLesson(p, lesson(), at("2026-10-21")).progress;
    expect(p.streak).toBe(2);
  });

  test("a missed day breaks it; yesterday's lesson keeps it alive", () => {
    const p = finishLesson(fresh(), lesson(), at("2026-10-19")).progress;
    expect(catchUp(p, at("2026-10-21")).streak).toBe(0);
    expect(catchUp(p, at("2026-10-20")).streak).toBe(1);
  });
});

describe("xp and goals", () => {
  test("perfect lessons earn the bonus and the daily goal is reported once", () => {
    const a = finishLesson(fresh(), lesson(0), at("2026-10-21"));
    expect(a.earned.xp).toBe(15);
    expect(a.earned.goalReached).toBe(false);
    const b = finishLesson(a.progress, lesson(1), at("2026-10-21"));
    expect(b.earned.xp).toBe(10);
    expect(b.earned.goalReached).toBe(true); // 25 >= 20
    expect(finishLesson(b.progress, lesson(), at("2026-10-21")).earned.goalReached).toBe(false);
  });
});

describe("sign strength", () => {
  test("rises with right answers, falls with mistakes and time away", () => {
    let p = finishLesson(fresh(), lesson(0, [{ gloss: "TEA", correct: true }, { gloss: "TEA", correct: true }]), at("2026-10-01")).progress;
    expect(signStrength(p.signs.TEA, "2026-10-01")).toBe(2);
    expect(signStrength(p.signs.TEA, "2026-10-07")).toBe(0);
    p = finishLesson(p, lesson(1, [{ gloss: "TEA", correct: false }]), at("2026-10-01")).progress;
    expect(p.signs.TEA.strength).toBe(1);
  });
});

describe("daily challenges", () => {
  test("track today's lessons and cap at the goal", () => {
    const now = at("2026-10-21");
    let p = fresh();
    for (let i = 0; i < 3; i++) p = finishLesson(p, lesson(), now).progress;
    expect(quests(p, now).find((x) => x.id === "lessons")).toMatchObject({ value: 2, goal: 2 });
    expect(quests(p, at("2026-10-22")).find((x) => x.id === "lessons")!.value).toBe(0);
  });
});
