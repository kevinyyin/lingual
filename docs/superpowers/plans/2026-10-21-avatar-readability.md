# Avatar Readability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the avatar's signs read clearly for the demo: a still hand at clean sign edges, steadier motion that isn't mushy, and a face that shows the tone (translator) and the feeling (teaching mode).

**Architecture:** Everything is in the browser. The timeline builder (`player.ts`) cuts each sign clip to the sign itself and holds its two ends; each clip is cleaned once as it loads (new `clipSmooth.ts`, called from `stage.ts`); the live filters are retuned by measurement; the tone strengths go up (`face.ts`) and teaching plays carry a feeling face (`curriculum.ts` → `signer.ts`). The motion bench gains sign-edge and speed-limit metrics first, so every before/after number comes from one tool.

**Tech Stack:** TypeScript, three.js, Vite, Vitest (jsdom); Python 3 with Playwright for the bench and round-trip scripts; the FastAPI backend serves `/vocab`.

**Spec:** `docs/superpowers/specs/2026-10-21-avatar-readability-design.md`

## Global Constraints

- Branch `avatar-readability`; the default body is the toon. Paths below are from the repo root.
- `?oldmotion` in the page address restores today's motion: the 12% trim (`COART_TRIM`), no holds, no load-time cleaning, today's live smoothing (arm `FILTER_HZ` 3, fingers `FINGER_HZ` 8). The face changes are not behind it.
- Nothing new runs per frame: the cleaning runs once per clip as it loads.
- Not changing: `TURN_MAX_DEG_S` 1200, the springs (`motionSmoother.ts`), the arm solver, the body and contact corrections, hand placement.
- Tone stays on the mouth and eyes; the brows stay for grammar.
- After every task: `cd app/web && npx tsc --noEmit -p . && npm test` passes. After Tasks 1 and 7 also `cd app && uv run pytest -q`.
- Measurements need the backend (:8000) and Vite (:5173): from the repo root run `uv --directory app run asl-server` and `npm --prefix app/web run dev -- --port 5173 --strictPort` (in the background). First time only: `cd app && uv run --with playwright playwright install chromium`. The round-trip renders 150 signs and takes a while.
- Live-smoothing decision rule (spec Part 2): measure each candidate with Part 1 and the cleaning in place; keep the lightest that (1) has no more frames at the speed limit than today's code, (2) has edge and inner wrist jerkiness no higher than candidate (a)'s, and (3) scores a round-trip top-3 no more than 2 points below today's code. If neither (b) nor (c) passes, keep (a).
- Commits: one line, `type(scope): what changed`, like the repo's history. Never commit `.claude/launch.json`, `app/web/ios/`, `.impeccable/` or the untracked `app/reports/motion/{checkpoint_after,final_capture,final_capture2}.*` (the user's own work).

## Review Focus

1. Low signs whose hand barely clears 0.25 m (about 380 of the 1,982 clips) must be found the same way and never cut past the caps. → Task 2, test "a low sign is found the same way".
2. A two-handed sign whose still base hand is raised higher than the moving hand must not lose any of the moving hand's sign. → Task 2, test "a still raised base hand keeps the moving hand's whole sign".
3. A hand missing from part of a clip stays missing, and the frames beside the gap are left as they are. → Task 3, test "a missing hand stays missing…".
4. Teaching plays of a fingerspelled word, of mixed feelings, or of a feeling sign among other signs get the right face. → Task 5 tests.
5. The `?oldmotion` escape hatch restores today's timeline exactly, for the side-by-side and the demo. → Task 2, test "letters and old motion keep the fixed trim and no holds".

## Files

| File | Change |
|---|---|
| `app/web/src/avatar/player.ts` | `OLD_MOTION`, `signSpan`, holds, letters marked, `signEdges` |
| `app/web/src/avatar/clipSmooth.ts` | new: `smoothClip` |
| `app/web/src/avatar/stage.ts` | clean each clip as it loads |
| `app/web/src/avatar/metrics.ts` | `edgeMotion` |
| `app/web/src/bench.ts`, `app/scripts/motion_bench.py` | report `atLimit` and the edge metrics |
| `app/web/src/avatar/face.ts` | tone strengths |
| `app/web/src/teach/curriculum.ts`, `app/web/src/teach/signer.ts` | `FEELING_FACE`, `faceFor`, used by `playTokens` |
| `app/web/src/avatar/armSolver.ts`, `app/web/src/avatar/rocketbox.ts` | only if Task 6 picks a lighter setting |
| tests | `player.test.ts`, `metrics.test.ts`, `clipSmooth.test.ts` (new), `stage.test.ts`, `face.test.ts`, `teach/curriculum.test.ts` (new), maybe `armSolver.test.ts` |
| `app/reports/motion/` | `readability_*.{json,md}`, `roundtrip_readability_*.json`, a README section |

---

### Task 1: The bench reports sign edges and the speed limit; measure today's baseline

**Files:**
- Modify: `app/web/src/avatar/player.ts`, `app/web/src/avatar/metrics.ts`, `app/web/src/bench.ts`, `app/scripts/motion_bench.py`
- Test: `app/web/src/avatar/player.test.ts`, `app/web/src/avatar/metrics.test.ts`
- Reports: `app/reports/motion/readability_before.{json,md}`, `app/reports/motion/roundtrip_readability_before.json`

**Interfaces:**
- Produces: `signEdges(tl: TimelineItem[]): number[]` in `player.ts`: the start time (ms) of every item at index ≥ 1 that is a clip or follows a clip. `edgeMotion(pts: THREE.Vector3[], times: number[], edges: number[], windowMs = 40): { edgeSpeed: number; edgeJerk: number; innerSpeed: number; innerJerk: number }` in `metrics.ts`: for frames 1..n−2, speed = |P[i+1] − P[i−1]| / 2 and jerkiness = |P[i+1] − 2·P[i] + P[i−1]|, both in millimetres; a frame is an edge frame when |times[i] − e| < windowMs for some edge e; each value is the median of its set (the middle value, the mean of the middle two for an even count, 0 for an empty set). `BenchResult` gains `atLimit`, `edgeSpeed`, `edgeJerk`, `innerSpeed`, `innerJerk`.

- [ ] **Step 1: Write the failing tests**

In `player.test.ts` (add `signEdges` to the `./player` import):

```ts
describe("signEdges", () => {
  test("marks where each clip starts and ends on the timeline", () => {
    const tl = buildTimeline(plan([{ kind: "sign", gloss: "COFFEE" }, { kind: "sign", gloss: "WANT" }]), clips, letters, vocab, REST);
    const starts = tl.map((_, i) => tl.slice(0, i).reduce((s, it) => s + it.durationMs, 0));
    expect(signEdges(tl)).toEqual([starts[1], starts[2], starts[3], starts[4]]); // blend, clip, blend, clip, blend
  });
});
```

In `metrics.test.ts` (add `edgeMotion` to the `./metrics` import):

```ts
describe("edgeMotion", () => {
  it("reports speed and jerkiness in millimetres per frame, split at sign edges", () => {
    // x = 0.0005·i² m: the speed at frame i is i mm and the jerkiness 1 mm everywhere
    const pts = Array.from({ length: 60 }, (_, i) => new THREE.Vector3(0.0005 * i * i, 0, 0));
    const m = edgeMotion(pts, pts.map((_, i) => (i * 1000) / 60), [500]);
    expect(m.edgeSpeed).toBeCloseTo(30, 6); // frames 28-32 lie within 40 ms of 500 ms
    expect(m.innerSpeed).toBeCloseTo(27, 6); // frames 1-27 and 33-58
    expect(m.edgeJerk).toBeCloseTo(1, 6);
    expect(m.innerJerk).toBeCloseTo(1, 6);
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd app/web && npx vitest run src/avatar/player.test.ts src/avatar/metrics.test.ts`
Expected: FAIL, `signEdges` and `edgeMotion` are not exported.

- [ ] **Step 3: Implement `signEdges` in `player.ts` and `edgeMotion` in `metrics.ts`** with the signatures above.

- [ ] **Step 4: Run them to verify they pass**

Run: `cd app/web && npx vitest run src/avatar/player.test.ts src/avatar/metrics.test.ts`
Expected: PASS.

- [ ] **Step 5: Report the new metrics from the bench**

In `bench.ts` `run()`: each frame, after `const cur = avatar.armState!()`, keep the higher wrist (`cur.L.wrist.y > cur.R.wrist.y ? cur.L.wrist : cur.R.wrist`, cloned) and its time `start + t`; per play, add `signEdges(timeline)` shifted by that play's `start`. Return `atLimit: steps.filter((d) => d > 19.5).length` and the four fields of `edgeMotion(...)`. In `motion_bench.py`, append `"atLimit", "edgeSpeed", "edgeJerk", "innerSpeed", "innerJerk"` to `COLUMNS`; in `totals()`, `atLimit` is the sum over items and the other four are the mean over items.

- [ ] **Step 6: Measure today's baseline**

With both servers running (Global Constraints), from `app/`:
Run: `uv run --with playwright python scripts/motion_bench.py --label readability_before`
Run: `uv run --with playwright python scripts/roundtrip.py --label readability_before --n 150`
Expected: `reports/motion/readability_before.md` has the five new columns with non-zero `edgeSpeed`/`innerSpeed`, and `reports/motion/roundtrip_readability_before.json` has `top1`, `top3`, `top5`.

- [ ] **Step 7: Run all tests, then commit**

Run: `cd app/web && npx tsc --noEmit -p . && npm test`, then `cd app && uv run pytest -q`. Expected: all pass.

```bash
git add app/web/src/avatar/player.ts app/web/src/avatar/metrics.ts app/web/src/avatar/player.test.ts app/web/src/avatar/metrics.test.ts app/web/src/bench.ts app/scripts/motion_bench.py app/reports/motion/readability_before.json app/reports/motion/readability_before.md app/reports/motion/roundtrip_readability_before.json
git commit -m "feat(bench): sign-edge motion and frames at the speed limit; today's toon baseline"
```

---

### Task 2: Clean edges on every sign

**Files:**
- Modify: `app/web/src/avatar/player.ts`
- Test: `app/web/src/avatar/player.test.ts`

**Interfaces:**
- Consumes: nothing new.
- Produces: `export const OLD_MOTION: boolean` (`new URLSearchParams(location.search).has("oldmotion")`), `export const UP_RISE_M = 0.25`, `export const HOLD_MS = 100`, `export function signSpan(frames: ClipFrame[]): [number, number] | null` (the sign's frames `[start, end)`, or `null` for the fallback), `buildTimeline(plan, clips, letters, vocab, rest, old = OLD_MOTION)`. The `"clip"` variant of `TimelineItem` gains `letter?: boolean`.

- [ ] **Step 1: Write the failing tests** (add `signSpan` to the `./player` import)

```ts
describe("sign span and holds", () => {
  const U = 1 / 16, X = 1 / 64; // exact in binary, so equal speeds really are equal
  /** A right hand that rises (`top` frames of U), holds 2 frames, signs across and back (`sign` frames of X), holds 2,
   *  drops, and rests 6: world landmarks, y down, hips at y 0. `base`: the left hand rises higher and stays still. */
  const signing = ({ lead = 6, top = 6, sign = 12, base = false } = {}): ClipFrame[] => {
    const right: [number, number][] = [
      ...Array.from({ length: lead }, (): [number, number] => [0, 0]),
      ...Array.from({ length: top }, (_, k): [number, number] => [0, (k + 1) * U]),
      ...Array.from({ length: 2 }, (): [number, number] => [0, top * U]),
      ...Array.from({ length: sign }, (_, k): [number, number] => [(k < sign / 2 ? k + 1 : sign - 1 - k) * X, top * U]),
      ...Array.from({ length: 2 }, (): [number, number] => [0, top * U]),
      ...Array.from({ length: top }, (_, k): [number, number] => [0, (top - k - 1) * U]),
      ...Array.from({ length: 6 }, (): [number, number] => [0, 0]),
    ];
    return right.map(([x, rise], t) => {
      const pose = Array.from({ length: 33 }, () => [0, 0, 0]);
      pose[16] = [x, -rise, 0];
      const left = !base ? 0 : t === 6 || t === 33 ? 3 * U : t === 7 || t === 32 ? 6 * U : t >= 8 && t <= 31 ? 9 * U : 0;
      pose[15] = [0.25, -left, 0];
      return { pose, pose2d: pose.map(([px, py]) => [px, py]), left: null, right: null };
    });
  };
  const one = (frames: ClipFrame[], spell = false, old?: boolean) => {
    const clip: Clip = { fps: 30, frames };
    const p = spell ? plan([{ kind: "spell", word: "Q" }]) : plan([{ kind: "sign", gloss: "Q" }]);
    const tl = buildTimeline(p, new Map([["Q", clip]]), new Map([["Q", clip]]), [{ gloss: "Q", avatar_ok: true }], REST, old);
    return content(tl)[0] as { clip: Clip; durationMs: number };
  };

  test("finds the sign between its first and last hold", () => {
    expect(signSpan(signing())).toEqual([13, 27]);
  });
  test("a low sign is found the same way", () => {
    expect(signSpan(signing({ top: 5 }))).toEqual([12, 26]); // the hand rises 0.3125 m, just clear of 0.25
  });
  test("a still raised base hand keeps the moving hand's whole sign", () => {
    expect(signSpan(signing({ base: true }))).toEqual([10, 31]); // the right hand signs over frames 14-25
  });
  test("falls back when the hand never rises, the sign is too short, or a cut would pass 40%", () => {
    expect(signSpan(signing({ top: 3 }))).toBeNull(); // 0.1875 m: never up
    expect(signSpan(signing({ sign: 4 }))).toBeNull(); // 6 frames between the holds
    expect(signSpan(signing({ lead: 18 }))).toBeNull(); // the first hold is 48% in
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd app/web && npx vitest run src/avatar/player.test.ts`
Expected: FAIL, `signSpan` is not exported.

- [ ] **Step 3: Implement `signSpan` and the holds in `player.ts`**

`signSpan(frames)`, exactly as the spec's Part 1:
- the active wrist is 15 or 16, whichever has the larger maximum over the clip of rise = (hip midpoint y, from pose 23 and 24) − wrist y;
- up frames: rise > `UP_RISE_M`;
- speed[t] = distance between the active wrist's world points at t−1 and t for t ≥ 1, speed[0] = speed[1]; s[t] = mean of speed over t−1..t+1 (only indices that exist);
- t in 1..n−2 is a minimum when s[t] ≤ s[t−1], s[t] ≤ s[t+1], and s[t] < s[t−1] or s[t] < s[t+1];
- start = the first minimum ≥ max(first up, 1); end = the last minimum ≤ min(last up, n−2);
- return `null` when there is no up frame, start or end is missing, start ≥ end, end + 1 − start < 8, start > 0.4·n, or n − 1 − end > 0.4·n; otherwise `[start, end + 1]`.

`buildTimeline`: `add()` marks letter clips with `letter: true`. The trim loop: `span = c.letter || old ? null : signSpan(frames)`; `[a, b] = span ?? today's fixed trim` (cut = ⌊n·`COART_TRIM`⌋, applied only when n − 2·cut ≥ 2 and cut > 0, else all frames); when `span` was found, k = round(`HOLD_MS`·fps/1000) copies of frame a go before `frames.slice(a, b)` and k copies of frame b−1 after it, otherwise none; `durationMs` = frames.length / fps · 1000.

- [ ] **Step 4: Run them to verify they pass, with the old tests unchanged**

Run: `cd app/web && npx vitest run src/avatar/player.test.ts`
Expected: PASS, including the existing trim and idle-hand tests (their flat clips never rise, so they take the fallback).

- [ ] **Step 5: Commit**

```bash
git add app/web/src/avatar/player.ts app/web/src/avatar/player.test.ts
git commit -m "feat(avatar): cut each sign clip to the sign and hold its first and last pose; ?oldmotion keeps the old trim"
```

---

### Task 3: Clean each clip as it loads

**Files:**
- Create: `app/web/src/avatar/clipSmooth.ts`
- Modify: `app/web/src/avatar/stage.ts` (`fetchClip`)
- Test: `app/web/src/avatar/clipSmooth.test.ts` (new), `app/web/src/avatar/stage.test.ts`

**Interfaces:**
- Consumes: `OLD_MOTION` from `player.ts`.
- Produces: `export function smoothClip(clip: Clip): Clip`: a new clip with the same `fps`, `base` and frame count; for each of `pose`, `pose2d`, `left`, `right`, `leftW`, `rightW`, every number of frame t becomes ¼·(t−1) + ½·t + ¼·(t+1) when frames t−1, t and t+1 all have that array with the same number of points; otherwise frame t keeps its value. Every other field is copied; the input is not mutated.

- [ ] **Step 1: Write the failing tests**

`clipSmooth.test.ts`:

```ts
import { describe, expect, test } from "vitest";
import { smoothClip } from "./clipSmooth";
import type { Clip, ClipFrame } from "./player";

/** n frames whose right wrist (pose 16) follows `wrist(t)`; every other point at the origin; touchL 0.5 throughout. */
const clipOf = (n: number, wrist: (t: number) => number[]): Clip => ({
  fps: 30,
  frames: Array.from({ length: n }, (_, t): ClipFrame => {
    const pose = Array.from({ length: 33 }, () => [0, 0, 0]);
    pose[16] = wrist(t);
    return { pose, pose2d: pose.map(([x, y]) => [x, y]), left: null, right: null, touchL: 0.5 };
  }),
});
const wobbly = (t: number) => [0.3 + (t % 2 ? 0.004 : -0.004), -0.4, 0];

describe("smoothClip", () => {
  test("removes a frame-to-frame wobble", () => {
    const out = smoothClip(clipOf(9, wobbly));
    for (let t = 1; t < 8; t++) expect(out.frames[t].pose[16][0]).toBeCloseTo(0.3, 9);
  });
  test("keeps a straight-line motion, the clip's length and its end frames", () => {
    const out = smoothClip(clipOf(9, (t) => [0.01 * t, -0.4, 0]));
    expect(out.frames).toHaveLength(9);
    out.frames.forEach((f, t) => expect(f.pose[16][0]).toBeCloseTo(0.01 * t, 9));
  });
  test("a missing hand stays missing, and the frames beside the gap are left as they are", () => {
    const src = clipOf(9, wobbly);
    src.frames.forEach((f, t) => { f.rightW = t === 4 ? null : [[t % 2 ? 1 : -1, 0, 0]]; });
    const out = smoothClip(src);
    expect(out.frames[4].rightW).toBeNull();
    expect(out.frames[3].rightW).toEqual([[1, 0, 0]]);
    expect(out.frames[5].rightW).toEqual([[1, 0, 0]]);
    expect(out.frames[2].rightW![0][0]).toBeCloseTo(0, 9); // ¼·1 + ½·(−1) + ¼·1
  });
  test("leaves contact data and the source clip as they are", () => {
    const src = clipOf(9, wobbly);
    const out = smoothClip(src);
    expect(out.frames.every((f) => f.touchL === 0.5)).toBe(true);
    expect(src.frames[1].pose[16][0]).toBeCloseTo(0.304, 9);
  });
});
```

In `stage.test.ts` (add `prepare` to the `./stage` import; import `DEFAULT_REST` from `./rest` and `type Clip` from `./player`):

```ts
test("each clip is cleaned once as it loads", async () => {
  const frames = Array.from({ length: 20 }, (_, t) => {
    const pose = Array.from({ length: 33 }, () => [0, 0, 0]);
    pose[16] = [0.3 + (t % 2 ? 0.004 : -0.004), -0.4, 0]; // raised, wobbling across
    return { pose, pose2d: pose.map(([x, y]) => [x, y]), left: null, right: null };
  });
  vi.stubGlobal("fetch", async (url: string) => url === "/signs/WOBBLE.json"
    ? new Response(JSON.stringify({ fps: 30, frames }), { headers: { "content-type": "application/json" } })
    : new Response("", { status: 404 }));
  const { timeline } = await prepare({ ...plan, tokens: [{ kind: "sign", gloss: "WOBBLE" }] },
    [{ gloss: "WOBBLE", english: "wobble", avatar_ok: true }], DEFAULT_REST);
  const item = timeline.find((i) => i.kind === "clip") as { clip: Clip };
  for (const f of item.clip.frames) expect(f.pose[16][0]).toBeCloseTo(0.3, 9);
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd app/web && npx vitest run src/avatar/clipSmooth.test.ts src/avatar/stage.test.ts`
Expected: FAIL, `./clipSmooth` does not exist, and the stage test sees the wobble.

- [ ] **Step 3: Implement `smoothClip` in `clipSmooth.ts`, and call it in `stage.ts`**

In `fetchClip`, the cached request smooths what it fetched before anything else sees it: add `.then((c) => (c?.frames?.length && !OLD_MOTION ? smoothClip(c) : c))` after the JSON step, before the `.catch`. `steadyBaseHand` then runs on the smoothed clip as before.

- [ ] **Step 4: Run them to verify they pass**

Run: `cd app/web && npx vitest run src/avatar/clipSmooth.test.ts src/avatar/stage.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add app/web/src/avatar/clipSmooth.ts app/web/src/avatar/clipSmooth.test.ts app/web/src/avatar/stage.ts app/web/src/avatar/stage.test.ts
git commit -m "feat(avatar): clean each clip once as it loads (1-2-1 over time), not under ?oldmotion"
```

---

### Task 4: Tone at full strength

**Files:**
- Modify: `app/web/src/avatar/face.ts` (`TONE`)
- Test: `app/web/src/avatar/face.test.ts`

**Interfaces:** `faceTrack` is unchanged; only the `TONE` expression values change.

- [ ] **Step 1: Write the failing test** (import `type Tone` from `../ws`)

```ts
test("tone shows at full strength mid-sentence, on the mouth and eyes only", () => {
  const at = (tone: Tone) => faceTrack("statement", tone, 1500)(750).expressions;
  expect(at("warm")).toEqual({ MTH_Joy: 1, EYE_Joy: 0.7 });
  expect(at("apologetic")).toEqual({ MTH_Sorrow: 0.9, EYE_Sorrow: 0.6 });
  expect(at("urgent")).toEqual({ MTH_Surprised: 0.6, EYE_Surprised: 0.8 });
  expect(at("neutral")).toEqual({});
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd app/web && npx vitest run src/avatar/face.test.ts`
Expected: FAIL, warm is `{ MTH_Joy: 0.4, EYE_Joy: 0.3 }`.

- [ ] **Step 3: Set those values in `TONE`** (apologetic keeps `pitch: 6`).

- [ ] **Step 4: Run it to verify it passes**

Run: `cd app/web && npx vitest run src/avatar/face.test.ts`
Expected: PASS, with the existing face tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add app/web/src/avatar/face.ts app/web/src/avatar/face.test.ts
git commit -m "feat(avatar): show the speaker's tone at full strength"
```

---

### Task 5: Feeling signs bring their face in teaching mode

**Files:**
- Modify: `app/web/src/teach/curriculum.ts`, `app/web/src/teach/signer.ts` (`playTokens`)
- Test: `app/web/src/teach/curriculum.test.ts` (new)

**Interfaces:**
- Produces: `export const FEELING_FACE: Record<string, Tone>` = HAPPY, EXCITED, GOOD, FINE → `"warm"`; SAD, ANGRY, TIRED, SICK, SORRY → `"apologetic"`. `export function faceFor(glosses: string[]): Tone`: the shared face when `glosses` has at least one feeling sign and they all map to the same face, otherwise `"neutral"`; other signs don't count. (`Tone` is a type import from `../ws`.)

- [ ] **Step 1: Write the failing tests**

```ts
import { describe, expect, test } from "vitest";
import { faceFor } from "./curriculum";

describe("faceFor", () => {
  test("a feeling sign brings its face", () => {
    expect(faceFor(["HAPPY"])).toBe("warm");
    expect(faceFor(["SORRY"])).toBe("apologetic");
  });
  test("other signs, a spelled word and mixed feelings stay neutral", () => {
    expect(faceFor(["HELLO"])).toBe("neutral");
    expect(faceFor([])).toBe("neutral"); // a fingerspelled word has no sign glosses
    expect(faceFor(["HAPPY", "SAD", "HELLO"])).toBe("neutral");
  });
  test("agreeing feeling signs share their face, and other signs don't count", () => {
    expect(faceFor(["HAPPY", "GOOD", "HELLO"])).toBe("warm");
    expect(faceFor(["TIRED", "TIRED"])).toBe("apologetic");
  });
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd app/web && npx vitest run src/teach/curriculum.test.ts`
Expected: FAIL, `faceFor` is not exported.

- [ ] **Step 3: Implement `FEELING_FACE` and `faceFor` in `curriculum.ts`; in `playTokens`, the plan's `tone` is `faceFor` of the tokens' sign glosses** (`kind === "sign"` with a gloss) instead of `"neutral"`.

- [ ] **Step 4: Run them to verify they pass**

Run: `cd app/web && npx vitest run src/teach/curriculum.test.ts && npx tsc --noEmit -p .`
Expected: PASS, no type errors.

- [ ] **Step 5: Commit**

```bash
git add app/web/src/teach/curriculum.ts app/web/src/teach/curriculum.test.ts app/web/src/teach/signer.ts
git commit -m "feat(teach): feeling signs show their face (smile or frown)"
```

---

### Task 6: Lighter live smoothing, chosen by measurement

**Files:**
- Modify (only if (b) or (c) wins): `app/web/src/avatar/armSolver.ts` (`armFilters`), `app/web/src/avatar/rocketbox.ts` (`FINGER_HZ` and `makeArm`)
- Test: `app/web/src/avatar/armSolver.test.ts` (only if the change stays)
- Reports: `app/reports/motion/readability_{a,b,c}.{json,md}`, `app/reports/motion/roundtrip_readability_{a,b,c}.json`

**Interfaces:**
- Consumes: `OLD_MOTION` (`player.ts`); `FILTER_HZ` (`armSolver.ts`).
- Produces (only if it stays): `armFilters(hz = FILTER_HZ): ArmFilters`; in `rocketbox.ts`, `const ARM_HZ = OLD_MOTION ? FILTER_HZ : 6` for (b) or `: Infinity` for (c), passed as `armFilters(ARM_HZ)`, and `const FINGER_HZ = OLD_MOTION ? 8 : 12`.

- [ ] **Step 1: Measure candidate (a): today's settings with Tasks 2-3 in place**

With both servers running, from `app/`:
Run: `uv run --with playwright python scripts/motion_bench.py --label readability_a`, then `uv run --with playwright python scripts/roundtrip.py --label readability_a --n 150`
Expected: both report files written.

- [ ] **Step 2: Make the live smoothing switchable and measure candidate (b)**

Add the `hz` parameter to `armFilters`. In `rocketbox.ts`, import `OLD_MOTION` from `./player`, add `ARM_HZ = OLD_MOTION ? FILTER_HZ : 6` used by `makeArm`, and set `FINGER_HZ = OLD_MOTION ? 8 : 12`. Restart Vite if it doesn't reload, then run both scripts with `--label readability_b`.

- [ ] **Step 3: Measure candidate (c)**

Set the new `ARM_HZ` to `Infinity` (the one-euro filter then passes each solve straight through; the springs and the speed limit remain), keep fingers at 12, and run both scripts with `--label readability_c`.

- [ ] **Step 4: Decide by the rule in Global Constraints**

Compare `readability_b` and `readability_c` with `readability_before` (rules 1 and 3) and `readability_a` (rule 2), using the TOTAL rows and `top3`. Prefer (c) over (b) when both pass. Set `ARM_HZ`/`FINGER_HZ` to the winner, or, if (a) wins, revert the code from Step 2.

- [ ] **Step 5: If a lighter setting won, pin the parameter with a test**

In `armSolver.test.ts`:

```ts
test("armFilters(Infinity) passes each rotation straight through", () => {
  const f = armFilters(Infinity), q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), 1);
  f.upper.filter(new THREE.Quaternion(), 0);
  expect(f.upper.filter(q, 16).angleTo(q)).toBeLessThan(1e-9);
});
```

Run: `cd app/web && npx vitest run src/avatar/armSolver.test.ts`
Expected: PASS.

- [ ] **Step 6: Run all tests, then commit the outcome and the three report sets**

Run: `cd app/web && npx tsc --noEmit -p . && npm test`
Expected: all pass.

```bash
git add app/reports/motion/readability_a.* app/reports/motion/readability_b.* app/reports/motion/readability_c.* app/reports/motion/roundtrip_readability_a.json app/reports/motion/roundtrip_readability_b.json app/reports/motion/roundtrip_readability_c.json
git add app/web/src/avatar/armSolver.ts app/web/src/avatar/rocketbox.ts app/web/src/avatar/armSolver.test.ts  # only if changed
git commit -m "perf(avatar): lighter live smoothing, arm 6 Hz / fingers 12 Hz (b), measured against a and c"
```

Use the message that matches the outcome: for (c), `"perf(avatar): arm one-euro filter off, fingers 12 Hz (c), measured against a and b"`; if (a) won, `"docs(motion): live smoothing candidates measured; today's settings kept (a)"`.

---

### Task 7: Check it all, and the user signs off

**Files:**
- Modify: `app/reports/motion/README.md`; `app/web/src/avatar/face.ts` and `face.test.ts` if the face strengths are tuned

- [ ] **Step 1: Run everything and build the phone's bundle**

Run: `cd app/web && npx tsc --noEmit -p . && npm test && npm run build`, then `cd app && uv run pytest -q`
Expected: all pass; `app/web/dist/` rebuilt (the backend serves it to the phone).

- [ ] **Step 2: Check the done criteria against the numbers**

The Task 6 winner's reports are the "after": nothing later changes motion. From the TOTAL rows: edgeSpeed/innerSpeed and edgeJerk/innerJerk at most 1.5 (criterion 1); `atLimit` no higher than before (2); `backwardElbow`, `twistOut`, `maxPenetration`, `maxArmPenetration` no higher and `p95FrameMs` no worse (3); round-trip `top3` no lower than before minus 0.02 (4). If any fails, stop and report the numbers to the user rather than continuing.

- [ ] **Step 3: Write the results into `app/reports/motion/README.md`**

A "Readability (2026-10-21)" section: a before/after table of those metrics plus round-trip top-1/3/5, the chosen live smoothing and why.

- [ ] **Step 4: The user's side-by-side (checkpoint: wait for the user)**

With the backend and Vite running, show the user, each with and without `oldmotion` in the address:
- the translator at `http://localhost:8000/?dev=1&camera=0` (add `&oldmotion` for before), typing "Thank you so much, it was lovely to meet you!" (warm), "I'm so sorry, I can't come tomorrow." (apologetic) and "Where is the train station?" (a wh-question);
- teaching mode at `http://localhost:8000/teach.html#/practice` (`teach.html?oldmotion#/practice` for before), "Show me a sign": HELLO, THANK-YOU, GRANDMOTHER, HAPPY, SAD;
- the same on the phone through the tunnel.

Tune the `TONE` values in `face.ts` (and the Task 4 test) until the user says the faces read on both screens without looking cartoonish.

- [ ] **Step 5: Commit**

```bash
git add app/reports/motion/README.md app/web/src/avatar/face.ts app/web/src/avatar/face.test.ts
git commit -m "docs(motion): readability before/after; face strengths signed off"
```
