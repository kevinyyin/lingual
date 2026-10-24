// The course: units of short lessons, each teaching a few signs from the app's vocabulary (vocab.yaml, avatar_ok).
import type { Tone } from "../ws";

/** Feeling signs carry their face in ASL (the Feelings unit's tip): a smile, a frown or a glare, given to the avatar as a
 *  tone. */
export const FEELING_FACE: Record<string, Tone> = {
  HAPPY: "warm", EXCITED: "warm", GOOD: "warm", FINE: "warm",
  SAD: "sad", TIRED: "sad", SICK: "sad", ANGRY: "angry", SORRY: "apologetic",
};

/** The face for playing `glosses`: their feeling signs' shared face, neutral when there are none or they disagree. */
export function faceFor(glosses: string[]): Tone {
  const faces = new Set(glosses.flatMap((g) => (FEELING_FACE[g] ? [FEELING_FACE[g]] : [])));
  return faces.size === 1 ? [...faces][0] : "neutral";
}

export interface Lesson { id: string; title: string; signs: string[] }
/** `tip`: the guidebook's note for the unit (facial grammar, culture). */
export interface Unit { id: string; title: string; blurb: string; tip: string; lessons: Lesson[] }

/** Every unit ends with a review lesson over all its signs (the path's trophy node). */
const unit = (id: string, title: string, blurb: string, tip: string, groups: [string, string[]][]): Unit => ({
  id, title, blurb, tip,
  lessons: [
    ...groups.map(([t, signs], i) => ({ id: `${id}-${i + 1}`, title: t, signs })),
    { id: `${id}-review`, title: "Unit review", signs: groups.flatMap(([, s]) => s) },
  ],
});

export const COURSE: Unit[] = [
  unit("u1", "Greetings", "Say hello, thank someone, introduce yourself",
    "Keep eye contact while you sign. In ASL your face and eyes are part of the conversation.", [
    ["Hello", ["HELLO", "GOODBYE", "MORNING", "MEET"]],
    ["Manners", ["THANK-YOU", "PLEASE", "SORRY", "WELCOME"]],
    ["Yes and no", ["YES", "NO", "OK", "NAME"]],
  ]),
  unit("u2", "People", "Talk about yourself, family and friends",
    "Pronouns point: I, YOU, HE-SHE and THEY are signed by pointing at the person, or at a spot you set up for them.", [
    ["Me and you", ["I", "YOU", "MY", "YOUR"]],
    ["Family", ["FAMILY", "MOTHER", "FATHER", "BABY"]],
    ["Community", ["FRIEND", "PERSON", "DEAF", "HEARING"]],
  ]),
  unit("u3", "Questions", "Ask who, what, where, when and why",
    "WH-questions (who, what, where, when, why, how) come with lowered, furrowed brows. Yes/no questions raise them.", [
    ["Who and what", ["WHO", "WHAT", "WHICH"]],
    ["Where and when", ["WHERE", "WHEN", "WHY", "HOW"]],
  ]),
  unit("u4", "Food & drink", "Order a coffee, say you're hungry",
    "Many food and drink signs happen near the mouth, where eating and drinking happen.", [
    ["Drinks", ["COFFEE", "TEA", "WATER", "MILK"]],
    ["Meals", ["EAT", "DRINK", "HUNGRY", "THIRSTY"]],
    ["Snacks", ["PIZZA", "APPLE", "CAKE", "COOKIE"]],
  ]),
  unit("u5", "Feelings", "Share how you feel",
    "Your face carries the feeling: sign HAPPY with a happy face. A blank face can change what you mean.", [
    ["Good days", ["HAPPY", "GOOD", "FINE", "EXCITED"]],
    ["Bad days", ["SAD", "ANGRY", "TIRED", "SICK"]],
  ]),
];

export const ALL_LESSONS: Lesson[] = COURSE.flatMap((u) => u.lessons);

/** Signs taught up to and including `lessonId`, in course order (distractors and review come from these). */
export function signsTaughtThrough(lessonId: string): string[] {
  const out: string[] = [];
  for (const l of ALL_LESSONS) {
    for (const s of l.signs) if (!out.includes(s)) out.push(s);
    if (l.id === lessonId) break;
  }
  return out;
}

/** Drops signs the backend doesn't serve (inactive or no avatar clip) so a lesson never asks for a missing sign. */
export function restrictCourse(available: Set<string>): void {
  for (const u of COURSE) for (const l of u.lessons) l.signs = l.signs.filter((s) => available.has(s));
}
