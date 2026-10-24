// Builds a lesson's exercises and runs its queue: each new sign is introduced and tried on camera, then recognized;
// a missed exercise goes back to the end of the queue and the lesson ends once every exercise is answered right.
import { signStrength, type Progress } from "./progress";
import type { PickOption } from "../ws";

export type Exercise =
  | { kind: "intro"; gloss: string }
  /** The avatar signs; pick the English meaning. */
  | { kind: "watch"; gloss: string; options: string[] }
  /** Read the English; pick which of the avatar's signs it is. */
  | { kind: "pick-sign"; gloss: string; options: string[] }
  | { kind: "match"; glosses: string[] }
  /** The avatar signs a short sequence; put the English words in its order. */
  | { kind: "order"; glosses: string[]; bank: string[] }
  /** Sign it on camera; graded by the backend recognizer (practice mode). */
  | { kind: "perform"; gloss: string };

export type Rng = () => number;
export const shuffle = <T>(xs: T[], rng: Rng): T[] => {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};
const sample = <T>(xs: T[], n: number, rng: Rng) => shuffle(xs, rng).slice(0, n);

/** `n` choices including `gloss`, the others from `pool` (falling back to `extra` when the course is young). */
function choices(gloss: string, pool: string[], extra: string[], n: number, rng: Rng): string[] {
  const others = sample(pool.filter((g) => g !== gloss), n - 1, rng);
  if (others.length < n - 1) others.push(...sample(extra.filter((g) => g !== gloss && !others.includes(g)), n - 1 - others.length, rng));
  return shuffle([gloss, ...others], rng);
}

export interface BuildOpts {
  /** Signs this lesson teaches. */
  signs: string[];
  /** Signs taught so far, this lesson's included (distractors and review). */
  known: string[];
  /** Whole vocabulary, for distractors before enough signs are known. */
  vocab: string[];
  /** A review lesson: nothing new, every sign drilled. */
  review: boolean;
  /** Include webcam exercises (off when there's no camera or the learner said "can't sign now"). */
  perform: boolean;
  rng: Rng;
}

/** Each sign is shown and then signed on camera; once all are, each is recognized from the avatar. That's the lesson
 *  (pick-sign, match and order are off for now). */
export function buildLesson(o: BuildOpts): Exercise[] {
  const { known, vocab, rng } = o;
  const signs = o.review ? shuffle(o.signs, rng).slice(0, 8) : o.signs;
  return [
    ...signs.flatMap((g): Exercise[] => [{ kind: "intro", gloss: g }, ...(o.perform ? [{ kind: "perform", gloss: g } as Exercise] : [])]),
    ...signs.map((g): Exercise => ({ kind: "watch", gloss: g, options: choices(g, known, vocab, 4, rng) })),
  ];
}

/** How close a camera attempt came to `gloss`, 0-100: its probability relative to the recognizer's top guess in the
 *  segment where it scored best (100 = the top guess), 0 when no segment lists it. `rank` is 1-based, null if absent. */
export function signScore(segments: PickOption[][], gloss: string): { score: number; rank: number | null } {
  let best = { score: 0, rank: null as number | null };
  for (const seg of segments) {
    const i = seg.findIndex((o) => o.gloss === gloss);
    if (i < 0 || !seg[0].p) continue;
    const score = Math.round((100 * seg[i].p) / seg[0].p);
    if (score > best.score) best = { score, rank: i + 1 };
  }
  return best;
}

/** Practice from the hub: the learner's weakest known signs, no new ones. */
export function practiceSigns(p: Progress, today: string, n = 6): string[] {
  return Object.keys(p.signs).sort((a, b) => signStrength(p.signs[a], today) - signStrength(p.signs[b], today)).slice(0, n);
}

interface RunState {
  queue: { ex: Exercise; retry: boolean }[];
  done: number;
  mistakes: number;
  combo: number;
  bestCombo: number;
  answers: { gloss: string; correct: boolean }[];
}
/** A run part way through, as saved (store.ts) so a lesson left early picks up where it was. */
export interface SavedRun extends RunState { total: number; past: RunState[]; ms: number }

/** The lesson's run: answers move it forward, misses requeue. Progress counts first-try and fixed exercises alike. */
export class LessonRun {
  private queue: { ex: Exercise; retry: boolean }[];
  readonly total: number;
  done = 0;
  mistakes = 0;
  combo = 0;
  bestCombo = 0;
  answers: { gloss: string; correct: boolean }[] = [];
  started = performance.now();
  private past: RunState[] = []; // the run before each answer, newest last: back() returns to it

  constructor(exercises: Exercise[], saved?: SavedRun) {
    this.queue = exercises.map((ex) => ({ ex, retry: false }));
    this.total = saved?.total ?? exercises.length;
    if (saved) {
      this.restore(saved);
      this.past = saved.past;
      this.started -= saved.ms;
    }
  }

  get current(): { ex: Exercise; retry: boolean } | undefined { return this.queue[0]; }
  get finished(): boolean { return this.queue.length === 0; }
  get fraction(): number { return this.done / this.total; }
  get canGoBack(): boolean { return this.past.length > 0; }

  private state(): RunState {
    const { queue, done, mistakes, combo, bestCombo, answers } = this;
    return structuredClone({ queue, done, mistakes, combo, bestCombo, answers });
  }
  private restore(s: RunState): void {
    const { queue, done, mistakes, combo, bestCombo, answers } = structuredClone(s);
    Object.assign(this, { queue, done, mistakes, combo, bestCombo, answers });
  }

  /** Back to the exercise before the last answer, as it was then (that answer undone). Skips after it are undone too,
   *  and redone on the way forward. */
  back(): boolean {
    const s = this.past.pop();
    if (s) this.restore(s);
    return !!s;
  }

  save(): SavedRun {
    return { ...this.state(), total: this.total, past: structuredClone(this.past), ms: performance.now() - this.started };
  }

  /** Intro cards are always "right"; `glosses` lets match/order exercises credit every sign they used. */
  answer(correct: boolean): void {
    if (!this.queue.length) return;
    this.past.push(this.state());
    const item = this.queue.shift()!;
    const { ex } = item;
    if (ex.kind !== "intro") {
      for (const g of ex.kind === "match" || ex.kind === "order" ? ex.glosses : [ex.gloss]) this.answers.push({ gloss: g, correct });
    }
    if (correct) {
      this.done++;
      if (ex.kind !== "intro") this.bestCombo = Math.max(this.bestCombo, ++this.combo);
    } else {
      this.mistakes++;
      this.combo = 0;
      this.queue.push({ ex, retry: true });
    }
  }

  /** "Can't sign now": drops the exercise without a mistake (the camera may be off or busy). */
  skip(): void {
    if (this.queue.shift()) this.done++;
  }

  get accuracy(): number {
    const graded = this.answers.length;
    return graded ? this.answers.filter((a) => a.correct).length / graded : 1;
  }
}
