// Avatar face over a signed sentence: grammar on brows and head, the speaker's tone on mouth and eyes only.
//
// Tone keys keep VRoid's names (rocketbox.ts maps them to ARKit shapes): MTH_* (mouth) and EYE_* (eyes) only, so tone
// can never fake or hide question grammar.
import type { Grammar, Tone } from "../ws";
import type { FacePose } from "./rocketbox";

export const RAMP_MS = 200;
const SHAKE_DEG = 12;
const SHAKE_HZ = 2.5;

// full strength: at a quarter to 40% the toon's mouth corners moved under 4 mm, too faint to read, above all on the phone
const TONE: Record<Tone, { expressions: Record<string, number>; pitch: number }> = {
  warm: { expressions: { MTH_Joy: 1, EYE_Joy: 0.7 }, pitch: 0 },
  apologetic: { expressions: { MTH_Sorrow: 0.9, EYE_Sorrow: 0.6 }, pitch: 6 }, // slightly lowered gaze
  sad: { expressions: { MTH_Sorrow: 1, EYE_Sorrow: 1 }, pitch: 10 }, // a full frown, eyes and head down
  angry: { expressions: { MTH_Angry: 1, EYE_Angry: 1 }, pitch: 0 }, // a tight frown and narrowed eyes, head level
  urgent: { expressions: { MTH_Surprised: 0.6, EYE_Surprised: 0.8 }, pitch: 0 },
  neutral: { expressions: {}, pitch: 0 },
};

/** Returns the face pose at time t (ms) for a sentence lasting durationMs. */
export function faceTrack(grammar: Grammar, tone: Tone, durationMs: number): (t: number) => FacePose {
  return (t) => {
    const ramp = Math.max(0, Math.min(1, t / RAMP_MS, (durationMs - t) / RAMP_MS));
    const toneSpec = TONE[tone];
    return {
      browUp: grammar === "yn" ? 0.8 * ramp : 0,
      browDown: grammar === "wh" ? 0.8 * ramp : 0,
      headYawDeg: grammar === "neg" ? SHAKE_DEG * ramp * Math.sin((2 * Math.PI * SHAKE_HZ * t) / 1000) : 0,
      headPitchDeg: (grammar === "yn" ? 8 : 0) * ramp + toneSpec.pitch * ramp,
      expressions: Object.fromEntries(Object.entries(toneSpec.expressions).map(([k, v]) => [k, v * ramp])),
    };
  };
}
