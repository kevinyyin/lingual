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
  progress: Progress;
  today: string;
  rng: Rng;
}

export function buildLesson(o: BuildOpts): Exercise[] {
  const { signs, known, vocab, rng } = o;
  const fresh = o.review ? [] : signs.filter((g) => !o.progress.signs[g]);
  const weakest = (gs: string[], n: number) =>
    [...gs].sort((a, b) => signStrength(o.progress.signs[a], o.today) - signStrength(o.progress.signs[b], o.today) || rng() - 0.5).slice(0, n);
  const earlier = known.filter((g) => !signs.includes(g));
  const review = weakest(earlier, o.review ? 0 : 3);
  const watch = (g: string): Exercise => ({ kind: "watch", gloss: g, options: choices(g, known, vocab, 4, rng) });
  const pick = (g: string): Exercise => ({ kind: "pick-sign", gloss: g, options: choices(g, known, vocab, 3, rng) });

  const out: Exercise[] = [];
  // see each new sign and try it on camera right away; recognizing them comes after every one is introduced
  for (const g of fresh) out.push({ kind: "intro", gloss: g }, ...(o.perform ? [{ kind: "perform", gloss: g } as Exercise] : []));
  out.push(...fresh.map(watch));
  const drill = o.review ? shuffle(signs, rng).slice(0, 8) : signs;
  out.push(...shuffle([...drill.filter((g) => !fresh.includes(g)).map(watch), ...drill.map(pick), ...review.map(watch)], rng));
  if (signs.length >= 4) out.push({ kind: "match", glosses: sample(signs, 4, rng) });
  if (known.length >= 3) {
    const seq = sample(signs.length >= 3 ? signs : known, 3, rng);
    out.push({ kind: "order", glosses: seq, bank: shuffle([...seq, ...sample(known.filter((g) => !seq.includes(g)), 2, rng)], rng) });
  }
  const tried = signs.filter((g) => !fresh.includes(g)); // new signs were already signed after their intro
  if (o.perform) out.push(...weakest(tried, o.review ? 3 : 2).map((g): Exercise => ({ kind: "perform", gloss: g })));
  return out;
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

/** The lesson's run: answers move it forward, misses requeue. Progress counts first-try and fixed exercises alike. */
export class LessonRun {
  private queue: { ex: Exercise; retry: boolean }[];
  readonly total: number;
  done = 0;
  mistakes = 0;
  combo = 0;
  bestCombo = 0;
  answers: { gloss: string; correct: boolean }[] = [];
  readonly started = performance.now();

  constructor(exercises: Exercise[]) {
    this.queue = exercises.map((ex) => ({ ex, retry: false }));
    this.total = exercises.length;
  }

  get current(): { ex: Exercise; retry: boolean } | undefined { return this.queue[0]; }
  get finished(): boolean { return this.queue.length === 0; }
  get fraction(): number { return this.done / this.total; }

  /** Intro cards are always "right"; `glosses` lets match/order exercises credit every sign they used. */
  answer(correct: boolean): void {
    const item = this.queue.shift();
    if (!item) return;
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
