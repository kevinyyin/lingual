// Sign plan -> timeline of motion clips with blends; sampled every animation frame.
// Human signing is fluid: signs flow into each other (coarticulation), each hand move follows a minimum-jerk speed
// profile, and a longer reach takes longer. So every clip drops its lead-in/lead-out (the recorded signer's move from and
// back to rest: kept, it plays after the blend from rest and reads as part of the sign), and blends ease with minimum
// jerk over a time that grows with distance.
import type { P3, SignPlan } from "../ws";
import { estimateAspect } from "./placement";

export const BLEND_MS = 150; // shortest sign-to-sign transition
export const REST_EASE_MS = 300; // shortest move from or back to rest
export const MAX_BLEND_MS = 450;
export const BLEND_MS_PER_M = 600; // extra transition time per metre the wrists travel
export const COART_TRIM = 0.12; // fraction of a clip dropped at each end (the median lead-in across the clips is 12%)
export const CAPTION_ONLY_MS = 400;
/** A hand whose wrist never rises this far above the hips is not signing: across the clips, idle hands peak at 0.2 m
 * and signing hands start at 0.25 m. Its recorded pose is the signer's resting hand, often half out of the video, whose
 * landmarks point it anywhere; the avatar's rest arm and relaxed hand replace it. */
export const IDLE_RISE_M = 0.2;
/** A wrist this far above the hips is signing (see IDLE_RISE_M): signSpan looks for the sign's holds from when it first
 *  rises this far until it last drops below IDLE_RISE_M. */
export const UP_RISE_M = 0.25;
export const HOLD_MS = 100; // a found sign holds its first and last pose this long, so its start and end read
/** A wrist slower than this (m/s) has stopped: under UP_RISE_M only such a stop ends a sign (THANK-YOU ends at the chest,
 *  and the drop to rest wobbles there). */
export const STILL_M_S = 0.1;
const MIN_SIGN_FRAMES = 8, MAX_TRIM = 0.4; // signSpan's limits: shortest sign kept, largest share cut from either end
/** ?oldmotion: the motion before 2026-10-21 (fixed trim, no holds, no load-time cleaning, the old live smoothing), for
 *  side-by-side checks and as a demo escape hatch. */
export const OLD_MOTION = new URLSearchParams(location.search).has("oldmotion");
export interface ClipFrame {
  pose: number[][]; // 33 world landmarks [x, y, z]
  pose2d: number[][]; // 33 image landmarks [x, y]
  left: P3[] | number[][] | null;
  right: P3[] | number[][] | null;
  leftW?: number[][] | null; // hand world landmarks (metres), preferred by the finger solver
  rightW?: number[][] | null;
  // 0-1 contact weights from ASL-LEX (training/touch_from_asllex): hand on the body or head / on the other hand
  touchL?: number; touchR?: number; meetL?: number; meetR?: number;
  touchAtL?: number[]; touchAtR?: number[]; // the place touched (pose world coordinates), from ASL-LEX
  aspect?: number; // image W/H of the clip this frame came from (placement.estimateAspect); unset for rest
}

const TOUCH_KEYS = ["touchL", "touchR", "meetL", "meetR"] as const;

const wristTravel = (a: ClipFrame, b: ClipFrame) =>
  Math.max(...[15, 16].map((i) => Math.hypot(...[0, 1, 2].map((k) => (b.pose[i]?.[k] ?? 0) - (a.pose[i]?.[k] ?? 0)))));

/** Transition time between two poses: at least `min`, longer for longer reaches (capped). */
export function blendMs(a: ClipFrame, b: ClipFrame, min = BLEND_MS): number {
  return Math.min(Math.max(min, min + BLEND_MS_PER_M * wristTravel(a, b)), Math.max(min, MAX_BLEND_MS));
}

/** Minimum-jerk position profile (Flash & Hogan): zero velocity and acceleration at both ends. */
export const minJerk = (w: number) => w * w * w * (10 - 15 * w + 6 * w * w);
/** base: the stationary base hand of a two-handed sign (training/touch_from_asllex; baseHand.ts holds its shape). */
export interface Clip { fps: number; frames: ClipFrame[]; base?: "L" | "R" }

export type TimelineItem =
  | { kind: "clip"; gloss: string; clip: Clip; durationMs: number; letter?: boolean }
  | { kind: "caption-only"; gloss: string; durationMs: number }
  | { kind: "blend"; from: ClipFrame; to: ClipFrame; durationMs: number };

type Content = Exclude<TimelineItem, { kind: "blend" }>;

const SIDES = [
  { wrist: 15, shoulder: 11, arm: [13, 15, 17, 19, 21], hand: "left", handW: "leftW", touch: ["touchL", "meetL", "touchAtL"] },
  { wrist: 16, shoulder: 12, arm: [14, 16, 18, 20, 22], hand: "right", handW: "rightW", touch: ["touchR", "meetR", "touchAtR"] },
] as const;

/** The clip with each idle hand (see IDLE_RISE_M) replaced by the rest arm, hung from the clip's own shoulder. */
export function restIdleHands(frames: ClipFrame[], rest: ClipFrame): ClipFrame[] {
  if (frames.some((f) => (f.meetL ?? 0) > 0 || (f.meetR ?? 0) > 0)) return frames; // the hands touch: a low base hand (COOK) is in use
  const idle = SIDES.filter((s) => frames.every((f) => (f.pose[23][1] + f.pose[24][1]) / 2 - f.pose[s.wrist][1] < IDLE_RISE_M));
  if (!idle.length) return frames;
  return frames.map((f) => {
    const out: ClipFrame = { ...f, pose: f.pose.map((p) => [...p]), pose2d: f.pose2d.map((p) => [...p]) };
    for (const s of idle) {
      for (const [pts, restPts] of [[out.pose, rest.pose], [out.pose2d, rest.pose2d]] as const) {
        const d = pts[s.shoulder].map((v, k) => v - restPts[s.shoulder][k]);
        for (const k of s.arm) pts[k] = restPts[k].map((v, j) => v + d[j]);
      }
      out[s.hand] = null;
      out[s.handW] = null;
      out[s.touch[0]] = 0;
      out[s.touch[1]] = 0;
      delete out[s.touch[2]];
    }
    return out;
  });
}

/** The sign itself, frames [start, end): from the first to the last moment the most raised wrist slows down while it is
 *  up (the sign's first and last holds), or null when the clip shows no such sign and the fixed trim applies. */
export function signSpan(frames: ClipFrame[], fps: number): [number, number] | null {
  const n = frames.length;
  const rise = (w: number) => frames.map((f) => (f.pose[23][1] + f.pose[24][1]) / 2 - f.pose[w][1]);
  const [rL, rR] = [rise(15), rise(16)];
  const [w, r] = Math.max(...rR) > Math.max(...rL) ? [16, rR] : [15, rL];
  const up = r.flatMap((v, t) => (v > UP_RISE_M ? [t] : []));
  if (!up.length || n < 3) return null;
  const [u0, high] = [up[0], up[up.length - 1]];
  let u1 = high;
  while (u1 + 1 < n && r[u1 + 1] > IDLE_RISE_M) u1++;
  const speed = frames.map((f, t) => (t ? Math.hypot(...f.pose[w].map((v, k) => v - frames[t - 1].pose[w][k])) : 0));
  speed[0] = speed[1];
  const s = speed.map((_, t) => { const near = speed.slice(Math.max(0, t - 1), t + 2); return near.reduce((a, b) => a + b) / near.length; });
  // a slowdown: no faster than either neighbour and slower than one of them (a stretch of steady speed doesn't count)
  const slows = (t: number) => s[t] <= s[t - 1] && s[t] <= s[t + 1] && (s[t] < s[t - 1] || s[t] < s[t + 1]);
  const still = (t: number) => s[t] * fps < STILL_M_S;
  let start = -1, end = -1;
  for (let t = Math.max(u0, 1); t <= n - 2 && start < 0; t++) if (slows(t)) start = t;
  for (let t = Math.min(u1, n - 2); t >= 1 && end < 0; t--) if (slows(t) && (t <= high || still(t))) end = t;
  const cap = MAX_TRIM * (u1 - u0 + 1); // of the frames the hand is up: its rise, drop and rest are no part of the sign
  if (start < 0 || end <= start || end + 1 - start < MIN_SIGN_FRAMES || start - u0 > cap || u1 - end > cap) return null;
  return [start, end + 1];
}

const first = (c: Content, rest: ClipFrame) => (c.kind === "clip" ? c.clip.frames[0] : rest);
const last = (c: Content, rest: ClipFrame) => (c.kind === "clip" ? c.clip.frames[c.clip.frames.length - 1] : rest);

export function buildTimeline(
  plan: SignPlan,
  clips: Map<string, Clip>,
  letters: Map<string, Clip>,
  vocab: { gloss: string; avatar_ok: boolean }[],
  rest: ClipFrame,
  old = OLD_MOTION,
): TimelineItem[] {
  const ok = new Map(vocab.map((v) => [v.gloss, v.avatar_ok]));
  const content: Content[] = [];
  const add = (gloss: string, clip: Clip | undefined, allowed: boolean, letter = false) => {
    if (clip?.frames.length) { const aspect = estimateAspect(clip.frames); clip = { fps: clip.fps, frames: clip.frames.map((f) => ({ ...f, aspect })) }; }
    content.push(clip && allowed && clip.frames.length
      ? { kind: "clip", gloss, clip, durationMs: (clip.frames.length / clip.fps) * 1000, letter }
      : { kind: "caption-only", gloss, durationMs: CAPTION_ONLY_MS });
  };
  for (const t of plan.tokens) {
    if (t.kind === "sign" && t.gloss) {
      const c = clips.get(t.gloss);
      add(t.gloss, c && { fps: c.fps, frames: restIdleHands(c.frames, rest), base: c.base }, ok.get(t.gloss) ?? false);
    }
    else if (t.kind === "spell" && t.word) for (const ch of t.word) add(ch, letters.get(ch), true, true);
  }
  if (!content.length) return [];
  // each clip loses the recorded signer's travel from/to rest; the blends (from rest, between signs) replace it. A sign
  // is cut to itself and holds its first and last pose; letters (built as held handshapes) and ?oldmotion keep the
  // fixed trim, without holds: holding a frame where the hand still moves would add a stop-and-go
  content.forEach((c) => {
    if (c.kind !== "clip") return;
    const f = c.clip.frames, n = f.length, cut = Math.floor(n * COART_TRIM);
    const span = c.letter || old ? null : signSpan(f, c.clip.fps);
    const [a, b] = span ?? (n - 2 * cut >= 2 && cut > 0 ? [cut, n - cut] : [0, n]);
    const k = span ? Math.round((HOLD_MS * c.clip.fps) / 1000) : 0;
    c.clip = { fps: c.clip.fps, frames: [...Array<ClipFrame>(k).fill(f[a]), ...f.slice(a, b), ...Array<ClipFrame>(k).fill(f[b - 1])] };
    c.durationMs = (c.clip.frames.length / c.clip.fps) * 1000;
  });
  const f0 = first(content[0], rest), fn = last(content[content.length - 1], rest);
  const out: TimelineItem[] = [{ kind: "blend", from: rest, to: f0, durationMs: blendMs(rest, f0, REST_EASE_MS) }];
  content.forEach((c, i) => {
    if (i > 0) {
      const from = last(content[i - 1], rest), to = first(c, rest);
      out.push({ kind: "blend", from, to, durationMs: blendMs(from, to) });
    }
    out.push(c);
  });
  out.push({ kind: "blend", from: fn, to: rest, durationMs: blendMs(fn, rest, REST_EASE_MS) });
  return out;
}

/** Where sign clips start and end on the timeline (ms): the start of every item that is a clip or follows one. */
export function signEdges(tl: TimelineItem[]): number[] {
  const out: number[] = [];
  let at = 0;
  tl.forEach((it, i) => {
    if (i > 0 && (it.kind === "clip" || tl[i - 1].kind === "clip")) out.push(at);
    at += it.durationMs;
  });
  return out;
}

const lerpPts = (a: number[][] | null | undefined, b: number[][] | null | undefined, w: number): number[][] | null =>
  !a || !b ? (w < 0.5 ? a : b) ?? a ?? b ?? null : a.map((p, i) => p.map((v, k) => v + (b[i][k] - v) * w));

export const ARC_K = 0.25;

/** Push the wrist (pose idx) and its world hand points toward the viewer (-z) so the blend arcs instead of cutting straight. */
function bow(out: ClipFrame, a: ClipFrame, b: ClipFrame, e: number): void {
  if (e <= 0 || e >= 1) return; // Math.sin(Math.PI) is 1.2e-16, not 0: keep the end frames exact
  const s = Math.sin(Math.PI * e);
  for (const [wi, key] of [[15, "leftW"], [16, "rightW"]] as const) {
    const p = a.pose[wi], q = b.pose[wi];
    if (!p || !q) continue;
    const dz = -ARC_K * Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) * s;
    if (!dz) continue;
    out.pose[wi][2] += dz;
    // lerpPts can hand back a source array when one side is null, so copy rather than mutate
    if (out[key]) out[key] = out[key]!.map((pt) => [pt[0], pt[1], pt[2] + dz]);
  }
}

/** `arc` is for blends between signs only; frame-to-frame interpolation inside a clip must stay straight. */
export function lerpFrame(a: ClipFrame, b: ClipFrame, w: number, ease: (w: number) => number = minJerk, arc = true): ClipFrame {
  const e = ease(w);
  const out: ClipFrame = {
    pose: lerpPts(a.pose, b.pose, e)!, pose2d: lerpPts(a.pose2d, b.pose2d, e)!,
    left: lerpPts(a.left, b.left, e), right: lerpPts(a.right, b.right, e),
    leftW: lerpPts(a.leftW, b.leftW, e), rightW: lerpPts(a.rightW, b.rightW, e),
    ...Object.fromEntries(TOUCH_KEYS.map((k) => [k, (a[k] ?? 0) + ((b[k] ?? 0) - (a[k] ?? 0)) * e])),
    touchAtL: lerpPts(a.touchAtL && [a.touchAtL], b.touchAtL && [b.touchAtL], e)?.[0],
    touchAtR: lerpPts(a.touchAtR && [a.touchAtR], b.touchAtR && [b.touchAtR], e)?.[0],
    aspect: a.aspect ?? b.aspect,
  };
  if (arc) bow(out, a, b, e);
  return out;
}

/** Frame to show at time t (ms since start); `index` is the timeline item, `done` once past the end. */
export function sample(tl: TimelineItem[], t: number, rest?: ClipFrame): { frame: ClipFrame; index: number; done: boolean } {
  let start = 0;
  for (let i = 0; i < tl.length; i++) {
    const it = tl[i];
    if (t < start + it.durationMs) {
      const local = t - start;
      if (it.kind === "blend") return { frame: lerpFrame(it.from, it.to, local / it.durationMs), index: i, done: false };
      if (it.kind === "clip") {
        // between source frames: clips are ~30 fps, the screen 60-120 Hz; holding each frame made motion step
        const x = (local / 1000) * it.clip.fps, n = it.clip.frames.length;
        const f = Math.min(n - 1, Math.floor(x));
        const frame = f + 1 < n ? lerpFrame(it.clip.frames[f], it.clip.frames[f + 1], x - f, (w) => w, false) : it.clip.frames[f];
        return { frame, index: i, done: false };
      }
      const prev = tl[i - 1];
      return { frame: prev && prev.kind === "blend" ? prev.to : (rest ?? (tl[0] as { from: ClipFrame }).from), index: i, done: false };
    }
    start += it.durationMs;
  }
  const end = tl[tl.length - 1];
  return { frame: end && end.kind === "blend" ? end.to : (rest as ClipFrame), index: tl.length - 1, done: true };
}
