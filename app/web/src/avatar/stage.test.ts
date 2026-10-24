import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { SignPlan } from "../ws";
import { createStage, prepare, trailEnd, type StageFrame } from "./stage";
import { DEFAULT_REST } from "./rest";
import type { Clip } from "./player";

// /vocab answers and every clip 404s, so HELLO plays caption-only: blend from rest, the sign, blend back (1 s)
const plan: SignPlan = { type: "sign_plan", tokens: [{ kind: "sign", gloss: "HELLO" }], grammar: "statement", tone: "neutral", english: "hello" };

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["requestAnimationFrame", "performance"] });
  vi.stubGlobal("fetch", async (url: string) => url === "/vocab"
    ? new Response(JSON.stringify([{ gloss: "HELLO", english: "hello", avatar_ok: true }]), { headers: { "content-type": "application/json" } })
    : new Response("", { status: 404 }));
});
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

test("reports each item once, not every frame", async () => {
  const stage = await createStage({ applyFrame() {}, applyFace() {} });
  const items: (string | null)[] = [];
  const done = stage.play(plan, (g) => items.push(g));
  await vi.advanceTimersByTimeAsync(2000);
  await done;
  expect(items).toEqual([null, "HELLO", null, null]);
});

test("rejects when a later frame throws, so the page can still report the avatar done", async () => {
  let frames = 0;
  const stage = await createStage({ applyFrame() { if (++frames === 5) throw new Error("bad frame"); }, applyFace() {} });
  const err = stage.play(plan, () => {}).catch((e: Error) => e.message);
  await vi.advanceTimersByTimeAsync(2000);
  expect(await err).toBe("bad frame");
});

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

describe("onFrame (the hand trail's feed)", () => {
  const clip = {
    fps: 30,
    frames: Array.from({ length: 20 }, (_, t) => {
      const pose = Array.from({ length: 33 }, () => [0, 0, 0]);
      pose[16] = [0.3 + t * 0.01, -0.4, 0];
      return { pose, pose2d: pose.map(([x, y]) => [x, y]), left: null, right: null };
    }),
  };
  const json = (body: unknown) => new Response(JSON.stringify(body), { headers: { "content-type": "application/json" } });
  beforeEach(() => {
    vi.resetModules(); // the stage caches clips and the vocabulary per page; earlier tests cached HELLO as missing
    vi.stubGlobal("fetch", async (url: string) => url === "/vocab" ? json([{ gloss: "HELLO", english: "hello", avatar_ok: true }])
      : url === "/signs/HELLO.json" || url === "/letters/A.json" ? json(clip)
      : new Response("", { status: 404 }));
  });
  const signingFrames = async (tokens: SignPlan["tokens"]) => {
    const fresh = await import("./stage");
    const frames: StageFrame[] = [];
    const stage = await fresh.createStage({ applyFrame() {}, applyFace() {} }, { onFrame: (f) => frames.push(f) });
    const done = stage.play({ ...plan, tokens }, () => {}, 0.5);
    await vi.advanceTimersByTimeAsync(8000);
    await done;
    return frames;
  };

  test("onFrame says signing only during a sign's clip, not the blends around it", async () => {
    const f = await signingFrames([{ kind: "sign", gloss: "HELLO" }]);
    expect(f.some((x) => x.signing)).toBe(true);
    expect(f[0].signing).toBe(false);
    expect(f.at(-1)!.signing).toBe(false);
    expect(f.every((x) => x.speed === 0.5)).toBe(true);
  });

  test("a fingerspelled word is never signing", async () => {
    const f = await signingFrames([{ kind: "spell", word: "A" }]);
    expect(f.length).toBeGreaterThan(0);
    expect(f.some((x) => x.signing)).toBe(false);
  });

  test("a caption-only sign is never signing", async () => {
    const f = await signingFrames([{ kind: "sign", gloss: "NOCLIP" }]);
    expect(f.length).toBeGreaterThan(0);
    expect(f.some((x) => x.signing)).toBe(false);
  });
});

describe("trailEnd (where a clip's sign ends for the trail)", () => {
  const clipOf = (rises: number[]): Clip => ({ fps: 30, frames: rises.map((r) => {
    const pose = Array.from({ length: 33 }, () => [0, 0, 0]);
    pose[16] = [0.3, -r, 0]; // the right wrist this far above the hips
    return { pose, pose2d: pose.map(([x, y]) => [x, y]), left: null, right: null };
  }) });

  test("a clip that ends dropping to rest ends at the top of the drop", () => {
    expect(trailEnd(clipOf([0.4, 0.42, 0.4, 0.43, 0.41, 0.33, 0.2, 0.12, 0.1]))).toBe(4);
  });

  test("a clip that ends holding the hand up is all sign", () => {
    expect(trailEnd(clipOf([0.3, 0.4, 0.42, 0.42, 0.42]))).toBe(5);
  });

  test("a clip cut while the hand still falls fast (YOU) ends at the top of the fall", () => {
    expect(trailEnd(clipOf([0.32, 0.38, 0.41, 0.41, 0.4, 0.37, 0.33, 0.27, 0.2]))).toBe(5);
  });

  test("a drop that settles into a slightly noisy rest (MOTHER) is still cut at its top", () => {
    expect(trailEnd(clipOf([0.4, 0.42, 0.42, 0.41, 0.36, 0.27, 0.17, 0.09, 0.07, 0.06, 0.061, 0.06]))).toBe(4);
  });

  test("a sign that ends low in a hold keeps its downward stroke (PERSON)", () => {
    expect(trailEnd(clipOf([0.31, 0.34, 0.33, 0.29, 0.24, 0.2, 0.172, 0.171, 0.1705, 0.17]))).toBe(10);
  });

  test("a slow downward stroke at the end is the sign's own (HUNGRY)", () => {
    expect(trailEnd(clipOf([0.26, 0.3, 0.32, 0.31, 0.3, 0.28, 0.27, 0.26, 0.24, 0.23, 0.22, 0.21]))).toBe(12);
  });
});
