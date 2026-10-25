// Capture tips: soft, non-blocking advice about the camera itself (too far, too dark).
// Unlike position.ts it never withholds frames or touches calibration; it only reads the Frame camera.ts already
// built, plus one 32x24 pixel sample a second. Thresholds: reports/teaching_videos.md section 3.
import type { Frame } from "./ws";

export type TipKind = "dark" | "small";
/** Each tip in full, and short for the phone's small corner camera (style.css). */
export const TIPS: Record<TipKind, { text: string; short: string }> = {
  dark: { text: "Too dark: turn on a light in front of you", short: "Too dark" },
  small: { text: "Sit a little closer: your hands read better bigger", short: "Sit closer" },
};
const ORDER: TipKind[] = ["dark", "small"]; // light fixes the most, so it goes first

export const FACE_MIN_PX = 80; // under 60 px clips are right 82% vs 87%; the report's fix says ~80 px
export const DARK_LUMA = 50; // qc.py's "dark" flag (mean gray < 50), same luma weights as cv2 gray
const MAX_YAW = 25; // degrees: a turned head foreshortens the ears, so its width says nothing about distance
const EVAL_MS = 1000, HOLD = 2, SHOW_MS = 6000, REPEAT_MS = 30_000, GAP_MS = 10_000;
export { SHOW_MS as TIP_MS };

/** Ear-to-ear width in px of the video frame: the same measure as training/teach_eval/features.py (pose 7, 8).
 *  null when the ears aren't seen or the head is turned. */
export function faceWidthPx(f: Frame, videoWidth: number): number | null {
  const [l, r] = [f.pose[7], f.pose[8]];
  if (!l || !r || l[3] < 0.5 || r[3] < 0.5) return null;
  if (f.head && Math.abs(f.head[0]) > MAX_YAW) return null;
  return Math.abs(l[0] - r[0]) * videoWidth;
}

/** Mean luma 0-255 of an RGBA buffer (a 32x24 downscale: 768 pixels). */
export function meanLuma(px: Uint8ClampedArray): number {
  let s = 0;
  for (let i = 0; i < px.length; i += 4) s += 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
  return s / (px.length / 4);
}

const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s[s.length >> 1]; };

/** Collects one second of face widths; once a second decides which tip (if any) to show. */
export class CaptureCoach {
  private faces: number[] = [];
  private lastEval: number | null = null; // the first check is a second after the first frame, not on it
  private lastFrame = -Infinity;
  private streak: Record<TipKind, number> = { dark: 0, small: 0 };
  private lastShown: Record<TipKind, number> = { dark: -Infinity, small: -Infinity };
  private lastAny = -Infinity;
  showing: { kind: TipKind; until: number } | null = null;

  /** Every frame (cheap). `luma` is called at most once per EVAL_MS; `quiet`: don't show now (not ready, or the
   *  badge has a position hint). Returns the tip to show, null to hide, or undefined for no change. */
  feed(f: Frame, now: number, videoWidth: number, luma: () => number | null, quiet: boolean): TipKind | null | undefined {
    if (now - this.lastFrame > 2 * EVAL_MS) { // first frame, or back after the avatar paused detection: start over
      this.faces = [];
      this.streak = { dark: 0, small: 0 };
      this.lastEval = now;
    }
    this.lastFrame = now;
    const w = faceWidthPx(f, videoWidth);
    if (w !== null) this.faces.push(w);
    if (quiet && this.showing) { this.showing = null; return null; } // never over a hint, a turn or calibration
    if (now - this.lastEval! < EVAL_MS) return this.expire(now);
    this.lastEval = now;

    const y = luma();
    const bad: Record<TipKind, boolean> = {
      dark: y !== null && y < DARK_LUMA,
      small: this.faces.length > 0 && median(this.faces) < FACE_MIN_PX,
    };
    this.faces = [];
    for (const k of ORDER) this.streak[k] = bad[k] ? this.streak[k] + 1 : 0;

    if (this.showing && !bad[this.showing.kind]) { this.showing = null; return null; } // fixed: hide right away
    if (this.showing || quiet || now - this.lastAny < GAP_MS) return this.expire(now);
    const kind = ORDER.find((k) => this.streak[k] >= HOLD && now - this.lastShown[k] >= REPEAT_MS);
    if (!kind) return undefined;
    this.lastShown[kind] = this.lastAny = now;
    this.showing = { kind, until: now + SHOW_MS };
    return kind;
  }

  private expire(now: number): null | undefined {
    if (this.showing && now >= this.showing.until) { this.showing = null; return null; }
    return undefined;
  }
}

/** Samples the video's mean brightness on a 32x24 canvas (~768 px read back; well under 0.1 ms). */
export function lumaSampler(video: HTMLVideoElement): () => number | null {
  const c = document.createElement("canvas");
  c.width = 32; c.height = 24;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  return () => {
    if (!ctx || !video.videoWidth) return null;
    ctx.drawImage(video, 0, 0, 32, 24);
    return meanLuma(ctx.getImageData(0, 0, 32, 24).data);
  };
}
