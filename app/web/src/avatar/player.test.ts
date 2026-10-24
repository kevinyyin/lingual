import { describe, expect, test } from "vitest";
import type { SignPlan } from "../ws";
import { ARC_K, BLEND_MS, CAPTION_ONLY_MS, COART_TRIM, MAX_BLEND_MS, REST_EASE_MS, blendMs, buildTimeline, lerpFrame, minJerk, sample, signEdges, signSpan, type Clip, type ClipFrame } from "./player";

const frame = (x: number): ClipFrame => ({
  pose: Array.from({ length: 33 }, () => [x, 0, 0]),
  pose2d: Array.from({ length: 33 }, () => [x, 0]),
  left: null,
  right: Array.from({ length: 21 }, () => [x, x, 0]),
});
const clip = (x: number, n = 30): Clip => ({ fps: 30, frames: Array.from({ length: n }, () => frame(x)) });
const REST = frame(0);
const plan = (tokens: SignPlan["tokens"]): SignPlan => ({ type: "sign_plan", tokens, grammar: "statement", tone: "neutral", english: "" });
const vocab = [
  { gloss: "COFFEE", avatar_ok: true },
  { gloss: "WANT", avatar_ok: true },
  { gloss: "BAD", avatar_ok: false },
];
const clips = new Map([["COFFEE", clip(1)], ["WANT", clip(2)], ["BAD", clip(3)]]);
const letters = new Map([["A", clip(4, 6)], ["N", clip(5, 6)]]);

const content = (items: ReturnType<typeof buildTimeline>) => items.filter((i) => i.kind !== "blend");

describe("buildTimeline", () => {
  test("maps sign tokens to clips in order", () => {
    const tl = buildTimeline(plan([{ kind: "sign", gloss: "COFFEE" }, { kind: "sign", gloss: "WANT" }]), clips, letters, vocab, REST);
    expect(content(tl).map((i) => i.kind === "clip" && i.gloss)).toEqual(["COFFEE", "WANT"]);
  });

  test("missing clip becomes caption-only", () => {
    const tl = buildTimeline(plan([{ kind: "sign", gloss: "TEA" }]), clips, letters, [...vocab, { gloss: "TEA", avatar_ok: true }], REST);
    expect(content(tl)).toMatchObject([{ kind: "caption-only", gloss: "TEA", durationMs: CAPTION_ONLY_MS }]);
  });

  test("avatar_ok false becomes caption-only", () => {
    const tl = buildTimeline(plan([{ kind: "sign", gloss: "BAD" }]), clips, letters, vocab, REST);
    expect(content(tl)[0].kind).toBe("caption-only");
  });

  test("spell token expands to one item per letter", () => {
    const tl = buildTimeline(plan([{ kind: "spell", word: "ANA" }]), clips, letters, vocab, REST);
    expect(content(tl).map((i) => ("gloss" in i ? i.gloss : ""))).toEqual(["A", "N", "A"]);
  });

  test("total duration includes blends and rest easing", () => {
    const tl = buildTimeline(plan([{ kind: "sign", gloss: "COFFEE" }, { kind: "sign", gloss: "WANT" }]), clips, letters, vocab, REST);
    const total = tl.reduce((s, i) => s + i.durationMs, 0);
    const kept = 2 * (30 - 2 * Math.floor(30 * COART_TRIM)); // each clip loses its lead-in and lead-out
    const blends = blendMs(REST, frame(1), REST_EASE_MS) + blendMs(frame(1), frame(2)) + blendMs(frame(2), REST, REST_EASE_MS);
    expect(total).toBeCloseTo(blends + (kept / 30) * 1000, 5);
  });

  test("every clip loses the recorded signer's lead-in and lead-out: the blends are the travel", () => {
    const cut = Math.floor(30 * COART_TRIM);
    const one = content(buildTimeline(plan([{ kind: "sign", gloss: "COFFEE" }]), clips, letters, vocab, REST));
    expect(one.map((c) => c.kind === "clip" && c.clip.frames.length)).toEqual([30 - 2 * cut]); // alone: from rest and back
    const two = content(buildTimeline(plan([{ kind: "sign", gloss: "COFFEE" }, { kind: "sign", gloss: "WANT" }]), clips, letters, vocab, REST));
    expect(two.map((c) => c.kind === "clip" && c.clip.frames.length)).toEqual([30 - 2 * cut, 30 - 2 * cut]);
    expect(clips.get("COFFEE")!.frames.length).toBe(30); // the shared clip cache is not mutated
  });

  test("longer reaches take longer, within bounds", () => {
    expect(blendMs(frame(0), frame(0))).toBe(BLEND_MS);
    expect(blendMs(frame(0), frame(0.2))).toBeGreaterThan(blendMs(frame(0), frame(0.1)));
    expect(blendMs(frame(0), frame(5))).toBe(MAX_BLEND_MS);
  });

  test("empty plan gives empty timeline", () => {
    expect(buildTimeline(plan([]), clips, letters, vocab, REST)).toEqual([]);
  });
});

describe("sample", () => {
  const tl = buildTimeline(plan([{ kind: "sign", gloss: "COFFEE" }]), clips, letters, vocab, REST);

  test("starts at rest and eases into the clip", () => {
    const ease = tl[0].durationMs;
    expect(ease).toBeGreaterThanOrEqual(REST_EASE_MS);
    expect(sample(tl, 0).frame.pose[0][0]).toBeCloseTo(0);
    expect(sample(tl, ease / 2).frame.pose[0][0]).toBeCloseTo(0.5);
    expect(sample(tl, ease + 10).frame.pose[0][0]).toBeCloseTo(1);
  });

  test("minimum-jerk easing: still at both ends, symmetric", () => {
    expect(minJerk(0)).toBe(0);
    expect(minJerk(1)).toBeCloseTo(1);
    expect(minJerk(0.5)).toBeCloseTo(0.5);
    const v = (w: number) => (minJerk(w + 1e-4) - minJerk(w)) / 1e-4;
    expect(v(0)).toBeCloseTo(0, 2);
    expect(v(1 - 1e-4)).toBeCloseTo(0, 2);
  });

  test("reports done after the end", () => {
    expect(sample(tl, 10_000).done).toBe(true);
    expect(sample(tl, 10).done).toBe(false);
  });
});

describe("arcs between signs", () => {
  const hand = (x: number): number[][] => Array.from({ length: 21 }, () => [x, 0, 0]);
  const arm = (x: number): ClipFrame => {
    const f = frame(0);
    f.pose[16] = [x, 0, 0];
    f.rightW = hand(x);
    return f;
  };
  test("blend endpoints unchanged", () => {
    const a = arm(0), b = arm(0.4);
    for (const [w, ref] of [[0, a], [1, b]] as const) {
      const f = lerpFrame(a, b, w);
      expect(f.pose).toEqual(ref.pose);
      expect(f.rightW).toEqual(ref.rightW);
    }
  });
  test("blend midpoint bows forward", () => {
    const f = lerpFrame(arm(0), arm(0.4), 0.5);
    expect(f.pose[16][0]).toBeCloseTo(0.2, 6);
    expect(f.pose[16][2]).toBeCloseTo(-ARC_K * 0.4, 6);
    expect(f.rightW![5][2]).toBeCloseTo(-ARC_K * 0.4, 6);
  });
  test("no bow when the hand doesn't move", () => {
    expect(lerpFrame(arm(0.1), arm(0.1), 0.5).pose[16][2]).toBe(0);
  });
  test("clip-internal lerp stays straight and sources are not mutated", () => {
    const a = arm(0), b = arm(0.4);
    b.rightW = null;
    expect(lerpFrame(a, b, 0.5, (w) => w, false).pose[16][2]).toBe(0);
    lerpFrame(a, b, 0.5);
    expect(a.rightW![0][2]).toBe(0);
  });
});

describe("idle hand", () => {
  // hip-centred metres, y down: shoulders 0.5 m above the hips; wrists `riseL`/`riseR` above them
  const body = (riseL: number, riseR: number): ClipFrame => {
    const pose = Array.from({ length: 33 }, () => [0, 0, 0]);
    pose[11] = [0.17, -0.5, 0]; pose[12] = [-0.17, -0.5, 0]; pose[23] = [0.1, 0, 0]; pose[24] = [-0.1, 0, 0];
    for (const k of [13, 15, 17, 19, 21]) pose[k] = [0.3, -riseL, 0.2]; // the clip's (garbage) left arm
    for (const k of [14, 16, 18, 20, 22]) pose[k] = [-0.2, -riseR, -0.2];
    const hand = Array.from({ length: 21 }, () => [1, 1, 1]);
    return { pose, pose2d: pose.map(([x, y]) => [x, y]), left: hand, leftW: hand, right: hand, rightW: hand, touchL: 1 };
  };
  const rest: ClipFrame = { ...body(0, 0), left: null, right: null, leftW: null, rightW: null, touchL: 0 };
  rest.pose = rest.pose.map((p, i) => ([13, 15, 17, 19, 21].includes(i) ? [0.21, 0.03, -0.1] : p));
  rest.pose2d = rest.pose.map(([x, y]) => [x, y]);

  const played = (clip: Clip) => {
    const tl = buildTimeline(plan([{ kind: "sign", gloss: "X" }]), new Map([["X", clip]]), letters, [{ gloss: "X", avatar_ok: true }], rest);
    return (content(tl)[0] as { clip: Clip }).clip.frames;
  };

  test("a hand that never leaves the hips rests: rest arm from the clip's shoulder, no hand data, no contact", () => {
    const frames = played({ fps: 30, frames: Array.from({ length: 30 }, () => body(0.08, 0.4)) });
    for (const f of frames) {
      expect(f.pose[15]).toEqual([0.21, 0.03, -0.1]); // same shoulder as rest, so the rest arm unchanged
      expect(f.left).toBeNull();
      expect(f.leftW).toBeNull();
      expect(f.touchL).toBe(0);
      expect(f.pose[16]).toEqual([-0.2, -0.4, -0.2]); // the signing hand is the clip's
      expect(f.right).not.toBeNull();
    }
  });

  test("a low base hand of a two-handed sign keeps the clip's motion (COOK: the other hand meets it)", () => {
    const frames = played({ fps: 30, frames: Array.from({ length: 30 }, (_, i) => ({ ...body(0.14, 0.24), meetR: i === 15 ? 1 : 0 })) });
    expect(frames.every((f) => f.left !== null)).toBe(true);
  });

  test("a hand that rises into signing space keeps the clip's motion", () => {
    const frames = played({ fps: 30, frames: Array.from({ length: 30 }, (_, i) => body(i === 15 ? 0.3 : 0.05, 0.4)) });
    expect(frames.some((f) => f.left !== null)).toBe(true);
  });
});

describe("signEdges", () => {
  test("marks where each clip starts and ends on the timeline", () => {
    const tl = buildTimeline(plan([{ kind: "sign", gloss: "COFFEE" }, { kind: "sign", gloss: "WANT" }]), clips, letters, vocab, REST);
    const starts = tl.map((_, i) => tl.slice(0, i).reduce((s, it) => s + it.durationMs, 0));
    expect(signEdges(tl)).toEqual([starts[1], starts[2], starts[3], starts[4]]); // blend, clip, blend, clip, blend
  });
});

describe("sign span and holds", () => {
  const U = 1 / 16, X = 1 / 64; // exact in binary, so equal speeds really are equal
  type Pt = [number, number]; // a right wrist's [x, rise above the hips] in metres
  const still = (n: number, x: number, rise: number) => Array.from({ length: n }, (): Pt => [x, rise]);
  /** The points after each step ([dx, drise] per frame) from `from`. */
  const walk = (from: Pt, steps: Pt[]) => steps.map(([dx, dr]): Pt => (from = [from[0] + dx, from[1] + dr]));
  const rep = (n: number, step: Pt) => Array.from({ length: n }, (): Pt => step);
  /** World landmarks (y down, hips at y 0) for a right wrist along `path`; the left hand `left(t)` above the hips. */
  const along = (path: Pt[], left = (_t: number) => 0): ClipFrame[] => path.map(([x, rise], t) => {
    const pose = Array.from({ length: 33 }, () => [0, 0, 0]);
    pose[16] = [x, -rise, 0];
    pose[15] = [0.25, -left(t), 0];
    return { pose, pose2d: pose.map(([px, py]) => [px, py]), left: null, right: null };
  });
  /** A right hand that rises (`top` frames of U), holds 2 frames, signs across and back (`sign` frames of X), holds 2,
   *  drops, and rests: `lead` frames before, `rest` after. */
  const signingPath = ({ lead = 6, top = 6, sign = 12, rest = 6 } = {}): Pt[] => [
    ...still(lead, 0, 0),
    ...walk([0, 0], rep(top, [0, U])),
    ...still(2, 0, top * U),
    ...Array.from({ length: sign }, (_, k): Pt => [(k < sign / 2 ? k + 1 : sign - 1 - k) * X, top * U]),
    ...still(2, 0, top * U),
    ...walk([0, top * U], rep(top, [0, -U])),
    ...still(rest, 0, 0),
  ];
  /** `base`: the left hand rises higher and stays still. */
  const signing = ({ base = false, ...o }: Parameters<typeof signingPath>[0] & { base?: boolean } = {}) =>
    along(signingPath(o), (t) => (!base ? 0 : t === 6 || t === 33 ? 3 * U : t === 7 || t === 32 ? 6 * U : t >= 8 && t <= 31 ? 9 * U : 0));
  const one = (frames: ClipFrame[], spell = false, old?: boolean) => {
    const clip: Clip = { fps: 30, frames };
    const p = spell ? plan([{ kind: "spell", word: "Q" }]) : plan([{ kind: "sign", gloss: "Q" }]);
    const tl = buildTimeline(p, new Map([["Q", clip]]), new Map([["Q", clip]]), [{ gloss: "Q", avatar_ok: true }], REST, old);
    return content(tl)[0] as { clip: Clip; durationMs: number };
  };

  test("finds the sign between its first and last hold", () => {
    expect(signSpan(signing(), 30)).toEqual([13, 27]);
  });
  test("a low sign is found the same way", () => {
    expect(signSpan(signing({ top: 5 }), 30)).toEqual([12, 26]); // the hand rises 0.3125 m, just clear of 0.25
  });
  test("a still raised base hand keeps the moving hand's whole sign", () => {
    expect(signSpan(signing({ base: true }), 30)).toEqual([10, 31]); // the right hand signs over frames 14-25
  });
  test("the 40% cap counts only the frames the hand is up: a long rest before or after the sign is no cut (HELLO)", () => {
    expect(signSpan(signing({ lead: 18 }), 30)).toEqual([25, 39]);
    expect(signSpan(signing({ rest: 30 }), 30)).toEqual([13, 27]);
  });
  test("falls back when the hand never rises or the sign is too short", () => {
    expect(signSpan(signing({ top: 3 }), 30)).toBeNull(); // 0.1875 m: never up
    expect(signSpan(signing({ sign: 4 }), 30)).toBeNull(); // 6 frames between the holds
  });
  test("falls back when either end would cut more than 40% of the frames the hand is up", () => {
    const drift = (from: Pt) => walk(from, rep(16, [U, 0])); // still up, as fast as it rises and drops: no slowdown
    const sign = signingPath().slice(12, 28).map(([x, r]): Pt => [x + 16 * U, r]); // first hold to last, after the drift
    const late = [...still(6, 0, 0), ...walk([0, 0], rep(6, [0, U])), ...drift([0, 6 * U]), ...sign, ...walk([16 * U, 6 * U], rep(6, [0, -U])), ...still(6, 16 * U, 0)];
    expect(signSpan(along(late), 30)).toBeNull(); // the first hold 19 of 36 up frames in
    const early = [...signingPath().slice(0, 28), ...drift([0, 6 * U]), ...walk([16 * U, 6 * U], rep(6, [0, -U])), ...still(6, 16 * U, 0)];
    expect(signSpan(along(early), 30)).toBeNull(); // the last hold 19 of 36 up frames before the end
  });
  test("falls back when the hand never slows while it is up, or slows only once", () => {
    const steady = [...still(6, 0, 0), ...walk([0, 0], rep(10, [0, U])), ...walk([0, 10 * U], rep(10, [U, 0]))];
    expect(signSpan(along(steady), 30)).toBeNull();
    const bounce = [...still(6, 0, 0), ...walk([0, 0], [...rep(6, [0, U]), [0, U / 2], [0, U / 4], [0, U / 8], [0, -U / 4], [0, -U / 2], ...rep(6, [0, -U])])];
    expect(signSpan(along(bounce), 30)).toBeNull(); // one slowdown, at the top: the sign would end where it starts
  });
  test("a sign whose last hold is lower than where signing starts ends at that hold (THANK-YOU ends at the chest)", () => {
    const thanks = [...still(6, 0, 0), ...walk([0, 0], rep(6, [0, U])), ...still(4, 0, 6 * U), ...walk([0, 6 * U], rep(5, [0, -U / 2])),
      ...still(8, 0, 3.5 * U), ...walk([0, 3.5 * U], rep(7, [0, -U / 2])), ...still(6, 0, 0)];
    expect(signSpan(along(thanks), 30)).toEqual([13, 28]); // the last hold at 0.22 m, under UP_RISE_M
  });
  test("a slowdown in the drop to rest does not end the sign: under UP_RISE_M only a real stop does", () => {
    const drop = walk([0, 6 * U], [[0, -U], [0, -U], [0, -U / 4], [0, -U / 4], [0, -U], [0, -U], [0, -U], [0, -U / 2]]);
    expect(signSpan(along([...signingPath().slice(0, 28), ...drop, ...still(6, 0, 0)]), 30)).toEqual([13, 27]); // slows to 0.94 m/s at 0.22 m
  });
  test("a found sign is held 100 ms at both ends", () => {
    const c = one(signing());
    expect(c.clip.frames).toHaveLength(14 + 2 * 3);
    for (const f of c.clip.frames.slice(0, 4)) expect(f.pose).toEqual(c.clip.frames[0].pose);
    for (const f of c.clip.frames.slice(-4)) expect(f.pose).toEqual(c.clip.frames[19].pose);
    expect(c.durationMs).toBeCloseTo((20 / 30) * 1000, 6);
  });
  test("letters and old motion keep the fixed trim and no holds", () => {
    const fixed = 40 - 2 * Math.floor(40 * COART_TRIM);
    expect(one(signing(), true).clip.frames).toHaveLength(fixed);
    expect(one(signing(), false, true).clip.frames).toHaveLength(fixed);
  });
});
