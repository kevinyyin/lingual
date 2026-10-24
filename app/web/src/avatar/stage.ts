// Wires sign plans to the avatar: fetch clips, build the timeline, play it, report when done.
import type { SignPlan } from "../ws";
import { faceTrack } from "./face";
import { steadyBaseHand } from "./baseHand";
import { smoothClip } from "./clipSmooth";
import { buildTimeline, OLD_MOTION, sample, type Clip, type ClipFrame } from "./player";
import { DEFAULT_REST } from "./rest";
import type { Avatar } from "./rocketbox";

const cache = new Map<string, Promise<Clip | null>>();

async function fetchClip(path: string): Promise<Clip | null> {
  if (!cache.has(path)) { // the request is cached, so tokens sharing a clip fetch it once; a failed request is retried
    cache.set(path, fetch(path).then((r) => (r.ok && r.headers.get("content-type")?.includes("json") ? r.json() : null))
      .then((c) => (c?.frames?.length && !OLD_MOTION ? smoothClip(c) : c))
      .catch((e) => { cache.delete(path); throw e; }));
  }
  const clip = await cache.get(path)!;
  if (!clip || !clip.frames?.length || !baseHandOn) return clip;
  if (!steadied.has(clip)) steadied.set(clip, steadyBaseHand(clip)); // a two-handed sign's base hand in one clean shape
  return steadied.get(clip)!;
}

const steadied = new WeakMap<Clip, Clip>();
let baseHandOn = true;
/** Hold a two-handed sign's base hand in one clean shape (baseHand.ts; on by default; off for bench before/after). */
export const setBaseHand = (on: boolean) => { baseHandOn = on; };

/** `speed` < 1 plays slower (the learn page's slow replay). */
export interface Stage { play(plan: SignPlan, onItem: (gloss: string | null) => void, speed?: number): Promise<void> }

/** One played frame, for the hand trail: the clock, whether a sign's own clip is playing (not a blend to or from
 *  rest, not a fingerspelled letter), and the playback speed. */
export interface StageFrame { now: number; signing: boolean; speed: number }

export interface VocabSign { gloss: string; english: string; avatar_ok: boolean }
let vocabReq: Promise<VocabSign[]> | null = null;
/** The backend's vocabulary, fetched once per page; a failed fetch is retried on the next call. */
export const fetchVocab = (): Promise<VocabSign[]> => (vocabReq ??= fetch("/vocab")
  .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`/vocab ${r.status}`))))
  .catch((e) => { vocabReq = null; throw e; }));
const loadRest = async (): Promise<ClipFrame> => (await fetchClip("/signs/_REST.json"))?.frames[0] ?? DEFAULT_REST;

/** Everything a plan needs to be played: its clips fetched, the timeline built, the face track and the rest pose. */
export async function prepare(plan: SignPlan, knownVocab?: VocabSign[], knownRest?: ClipFrame) {
  const [vocab, rest] = await Promise.all([knownVocab ?? fetchVocab(), knownRest ?? loadRest()]);
  const clips = new Map<string, Clip>();
  const letters = new Map<string, Clip>();
  await Promise.all(plan.tokens.map(async (t) => {
    if (t.kind === "sign" && t.gloss) {
      const c = await fetchClip(`/signs/${encodeURIComponent(t.gloss)}.json`);
      if (c) clips.set(t.gloss, c);
    } else if (t.word) {
      await Promise.all([...new Set(t.word)].map(async (ch) => {
        const c = await fetchClip(`/letters/${ch}.json`);
        if (c) letters.set(ch, c);
      }));
    }
  }));
  const timeline = buildTimeline(plan, clips, letters, vocab, rest);
  const durationMs = timeline.reduce((s, i) => s + i.durationMs, 0);
  return { timeline, face: faceTrack(plan.grammar, plan.tone, durationMs), rest, durationMs };
}

const REST_RISE_M = 0.14; // a wrist this low (m above the hips) is back at rest: the recorded signers rest 6-13 cm up
const FALL_M_S = 0.45; // the travel back to rest falls faster than this; a sign's own downward strokes run slower

/** Where a clip's sign ends for the hand trail (frames): before the recorded signer's travel back to rest, which some
 *  clips keep at their end. That travel is a fast fall, often settling into rest; a sign ending in a hold, even low
 *  (PERSON), or in a slow downward stroke (HUNGRY) keeps all of it.
 *  shortcut: thresholds measured on ten course signs' clips; measure again if the motion library changes */
export function trailEnd(clip: Clip): number {
  const f = clip.frames, n = f.length;
  const rise = (t: number, w: number) => (f[t].pose[23][1] + f[t].pose[24][1]) / 2 - f[t].pose[w][1];
  const w = Math.max(...f.map((_, t) => rise(t, 16))) >= Math.max(...f.map((_, t) => rise(t, 15))) ? 16 : 15;
  let t = n - 1;
  while (t > 0 && rise(t, w) < REST_RISE_M) t--; // the hand already back at rest
  if (t === 0) return n; // a sign made down at rest height: nothing to cut
  while (t > 0 && (rise(t - 1, w) - rise(t, w)) * clip.fps >= FALL_M_S) t--; // its fast fall there
  return t + 1;
}

export async function createStage(avatar: Avatar, opts: { onFrame?: (f: StageFrame) => void } = {}): Promise<Stage> {
  const [vocab, rest] = await Promise.all([fetchVocab(), loadRest()]);
  avatar.applyFrame(rest); // the first frame snaps every filter and spring: the avatar starts at rest
  return {
    async play(plan, onItem, speed = 1) {
      const { timeline: tl, face } = await prepare(plan, vocab, rest);
      // when each item's signing ends on the timeline (ms): a clip's at its trailEnd; blends, letters and captions never sign
      let at = 0;
      const signEnd = tl.map((it) => (at += it.durationMs, it.kind === "clip" && !it.letter ? at - it.durationMs + (trailEnd(it.clip) / it.clip.fps) * 1000 : -1));
      const start = performance.now();
      let shown: string | null | undefined; // onItem runs when the item changes, not every frame
      await new Promise<void>((resolve, reject) => {
        const tick = () => {
          try { // a throw in a later frame's callback would leave play() pending, and the server waiting on avatar_done
            const t = (performance.now() - start) * speed;
            const s = sample(tl, t, rest);
            avatar.applyFrame(s.frame);
            avatar.applyFace(face(t));
            const it = tl[s.index];
            opts.onFrame?.({ now: performance.now(), signing: t < (signEnd[s.index] ?? -1), speed });
            const g = it && it.kind !== "blend" ? it.gloss : null;
            if (g !== shown) onItem((shown = g));
            if (s.done) return resolve();
            requestAnimationFrame(tick);
          } catch (e) {
            reject(e);
          }
        };
        tick();
      });
      onItem(null);
      opts.onFrame?.({ now: performance.now(), signing: false, speed });
    },
  };
}
