// The learner's progress: XP, streak, daily challenges and per-sign strength. Pure functions over a plain
// object (testable with any date); the page saves it to localStorage (store.ts).

export const LESSON_XP = 10;
export const PERFECT_BONUS_XP = 5;

export interface SignStat { correct: number; wrong: number; strength: number; lastSeen: string }
export interface Progress {
  xp: number;
  streak: number;
  longestStreak: number;
  /** Last day (YYYY-MM-DD, local) with a finished lesson. */
  lastDay: string | null;
  dailyGoal: number;
  xpByDay: Record<string, number>;
  lessonsByDay: Record<string, number>;
  /** Longest run of right answers in one lesson, per day (the combo quest). */
  comboByDay: Record<string, number>;
  /** Lesson id -> times completed. */
  done: Record<string, number>;
  signs: Record<string, SignStat>;
  perfectLessons: number;
  bestCombo: number;
}

export const fresh = (): Progress => ({
  xp: 0, streak: 0, longestStreak: 0, lastDay: null,
  dailyGoal: 20, xpByDay: {}, lessonsByDay: {}, comboByDay: {}, done: {}, signs: {}, perfectLessons: 0, bestCombo: 0,
});

export const dayKey = (d: Date): string =>
  `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
const addDays = (d: Date, n: number) => new Date(d.getFullYear(), d.getMonth(), d.getDate() + n);
const daysBetween = (a: string, b: string) =>
  Math.round((new Date(`${b}T00:00`).getTime() - new Date(`${a}T00:00`).getTime()) / 86_400_000);

/** Brings a loaded progress up to `now`: breaks the streak after a missed day. */
export function catchUp(p: Progress, now: Date): Progress {
  const q = structuredClone(p);
  if (q.lastDay && q.streak > 0 && daysBetween(q.lastDay, dayKey(now)) > 1) q.streak = 0;
  return q;
}

export interface Answer { gloss: string; correct: boolean }
export interface LessonResult { lessonId: string; answers: Answer[]; mistakes: number; bestCombo: number; ms: number }
export interface Earned { xp: number; streakExtended: boolean; goalReached: boolean; perfect: boolean }

/** Strength 0-4 per sign, Duolingo-style: right answers build it, mistakes and time away wear it down. */
export function signStrength(s: SignStat | undefined, today: string): number {
  if (!s) return 0;
  const idle = daysBetween(s.lastSeen, today);
  return Math.max(0, s.strength - Math.floor(idle / 3));
}

export function finishLesson(p: Progress, r: LessonResult, now: Date): { progress: Progress; earned: Earned } {
  const q = structuredClone(p);
  const today = dayKey(now);
  const perfect = r.mistakes === 0;
  const xp = LESSON_XP + (perfect ? PERFECT_BONUS_XP : 0);
  const before = q.xpByDay[today] ?? 0;
  q.xp += xp;
  q.xpByDay[today] = before + xp;
  q.lessonsByDay[today] = (q.lessonsByDay[today] ?? 0) + 1;
  q.done[r.lessonId] = (q.done[r.lessonId] ?? 0) + 1;
  if (perfect) q.perfectLessons++;
  q.bestCombo = Math.max(q.bestCombo, r.bestCombo);
  q.comboByDay[today] = Math.max(q.comboByDay[today] ?? 0, r.bestCombo);
  for (const a of r.answers) {
    const s = q.signs[a.gloss] ?? { correct: 0, wrong: 0, strength: 0, lastSeen: today };
    const base = signStrength(s, today);
    q.signs[a.gloss] = a.correct
      ? { ...s, correct: s.correct + 1, strength: Math.min(4, base + 1), lastSeen: today }
      : { ...s, wrong: s.wrong + 1, strength: Math.max(0, base - 1), lastSeen: today };
  }
  const streakExtended = q.lastDay !== today;
  if (streakExtended) {
    q.streak = q.lastDay && daysBetween(q.lastDay, today) === 1 ? q.streak + 1 : 1;
    q.lastDay = today;
    q.longestStreak = Math.max(q.longestStreak, q.streak);
  }
  return { progress: q, earned: { xp, streakExtended, goalReached: before < q.dailyGoal && q.xpByDay[today] >= q.dailyGoal, perfect } };
}

/** The last `n` days' XP, oldest first (the profile's weekly chart). */
export function xpHistory(p: Progress, now: Date, n = 7): { day: string; label: string; xp: number }[] {
  return Array.from({ length: n }, (_, i) => {
    const d = addDays(now, i - n + 1);
    return { day: dayKey(d), label: d.toLocaleDateString(undefined, { weekday: "short" }), xp: p.xpByDay[dayKey(d)] ?? 0 };
  });
}

export interface Quest { id: string; title: string; goal: number; value: number }
export function quests(p: Progress, now: Date): Quest[] {
  const today = dayKey(now);
  return [
    { id: "xp", title: `Earn ${Math.max(30, p.dailyGoal)} XP`, goal: Math.max(30, p.dailyGoal), value: p.xpByDay[today] ?? 0 },
    { id: "lessons", title: "Finish 2 lessons", goal: 2, value: p.lessonsByDay[today] ?? 0 },
    { id: "combo", title: "Answer 5 in a row right", goal: 5, value: p.comboByDay[today] ?? 0 },
  ].map((q) => ({ ...q, value: Math.min(q.value, q.goal) }));
}

export interface Achievement { id: string; title: string; detail: string; value: number; goal: number }
export function achievements(p: Progress): Achievement[] {
  const learned = Object.values(p.signs).filter((s) => s.correct > 0).length;
  return [
    { id: "streak", title: "Week of signing", detail: "Reach a 7 day streak", value: p.longestStreak, goal: 7 },
    { id: "xp", title: "Half a thousand", detail: "Earn 500 XP", value: p.xp, goal: 500 },
    { id: "signs", title: "First 25 signs", detail: "Learn 25 signs", value: learned, goal: 25 },
    { id: "perfect", title: "Flawless five", detail: "Finish 5 lessons with no mistakes", value: p.perfectLessons, goal: 5 },
  ].map((a) => ({ ...a, value: Math.min(a.value, a.goal) }));
}
