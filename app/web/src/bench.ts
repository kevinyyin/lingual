// ?mode=bench: step a sign plan at fixed time steps and measure the avatar's motion (metrics.ts), for before/after
// comparison of the arm solver. Driven by scripts/motion_bench.py through window.bench.run.
import { loadAvatar } from "./avatar";
import { edgeMotion, frameMetrics, SPIKE_DEG, type Arms } from "./avatar/metrics";
import { prepare, setBaseHand } from "./avatar/stage";
import { sample, signEdges } from "./avatar/player";

type Single = { glosses: string[]; at?: { hand: "L" | "R"; place: number[] } };
/** sentences: played back to back on one clock with no reset between them, as the live stage carries its state. */
export type BenchItem = string[] | Single | { sentences: string[][] };
export interface BenchResult {
  frames: number; p95FrameMs: number; spikes: number; maxStepDeg: number; p99StepDeg: number; backwardElbow: number; twistOut: number;
  maxPenetration: number; maxArmPenetration: number; minArmGap: number; contactErrorMean: number | null; contactErrorMin: number | null; images?: string[];
  // anatomical hand (handModel.ts): frames with a finger joint out of range, the worst excess, finger/hand overlaps (m), held hands
  handOutFrames: number; handOutJoints: number; maxHandOutDeg: number; maxFingerSelf: number; maxHandHand: number; maxFingerBody: number; heldHands: number;
  jointOut?: number[]; jointExcess?: number[]; // per joint (L then R, thumb..little, 3 each): frames > 1 deg past range, summed excess
  // frames with an arm bone at the 20 deg/frame speed limit; the higher wrist's median speed and jerkiness (mm/frame) at sign edges and elsewhere
  atLimit: number; edgeSpeed: number; edgeJerk: number; innerSpeed: number; innerJerk: number;
}

const pct = (xs: number[], q: number) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(q * xs.length))] : 0);
const frame = () => new Promise<void>((r) => requestAnimationFrame(() => r()));

export async function startBench(): Promise<void> {
  document.body.innerHTML = '<canvas id="avatar" style="width:640px;height:720px;display:block"></canvas>';
  const avatar = await loadAvatar(document.getElementById("avatar") as HTMLCanvasElement);
  const canvas = document.getElementById("avatar") as HTMLCanvasElement;
  const run = async (item: BenchItem, opts: { fps: number; capture: boolean; handLimits?: boolean; framePoints?: boolean; baseHand?: boolean; motion?: boolean; smooth?: boolean }): Promise<BenchResult> => {
    avatar.setHandLimits?.(opts.handLimits ?? true);
    setBaseHand(opts.baseHand ?? true);
    avatar.setSmoothing?.(opts.smooth ?? true);
    const parts: Single[] = Array.isArray(item) ? [{ glosses: item }] : "sentences" in item ? item.sentences.map((glosses) => ({ glosses })) : [item];
    const plays = await Promise.all(parts.map(({ glosses }) => prepare({ type: "sign_plan" as const, tokens: glosses.map((gloss) => ({ kind: "sign" as const, gloss })),
      grammar: "statement" as const, tone: "neutral" as const, english: glosses.join(" ") })));
    avatar.applyFrame(plays[0].rest, 0); // every item starts at rest: a restarted clock snaps every filter and spring
    const cost: number[] = [], steps: number[] = [], contact: number[] = [], images: string[] = [];
    const out = { backwardElbow: 0, twistOut: 0, maxPenetration: 0, maxArmPenetration: 0, minArmGap: Infinity,
      handOutFrames: 0, handOutJoints: 0, maxHandOutDeg: 0, maxFingerSelf: 0, maxHandHand: 0, maxFingerBody: 0, heldHands: 0 };
    let prev: Arms | null = null, frames = 0;
    const jointOut: number[] = [], jointExcess: number[] = [], framePts: unknown[] = [];
    const wrists: Arms["L"]["wrist"][] = [], times: number[] = [], edges: number[] = [];
    for (const [k, { timeline, face, rest, durationMs }] of plays.entries()) {
      const at = parts[k].at, n = Math.ceil((durationMs * opts.fps) / 1000), start = (frames * 1000) / opts.fps; // the clock runs on
      frames += n + 1;
      edges.push(...signEdges(timeline).map((e) => start + e));
      for (let i = 0; i <= n; i++) {
        const t = (i * 1000) / opts.fps, f = sample(timeline, t, rest).frame;
        const t0 = performance.now();
        avatar.applyFrame(f, start + t);
        avatar.applyFace(face(t));
        cost.push(performance.now() - t0);
        await frame();
        const cur = avatar.armState!();
        wrists.push((cur.L.wrist.y > cur.R.wrist.y ? cur.L.wrist : cur.R.wrist).clone());
        times.push(start + t);
        const m = frameMetrics(prev, cur);
        prev = cur;
        steps.push(m.maxStepDeg);
        out.backwardElbow += +m.backwardElbow;
        out.twistOut += +m.twistOut;
        out.minArmGap = Math.min(out.minArmGap, m.armGap);
        out.maxPenetration = Math.max(out.maxPenetration, avatar.penetration?.() ?? 0);
        out.maxArmPenetration = Math.max(out.maxArmPenetration, avatar.armPenetration?.() ?? 0);
        if (opts.framePoints) framePts.push(avatar.framePoints?.());
        if (opts.motion) framePts.push(avatar.framePoints?.(true)?.pts); // both arms every frame: wrist, 5 fingertips, elbow each
        const h = avatar.handReport?.();
        if (h) {
          out.handOutFrames += +(h.outJoints > 0);
          out.handOutJoints += h.outJoints;
          out.maxHandOutDeg = Math.max(out.maxHandOutDeg, h.maxOutDeg);
          out.maxFingerSelf = Math.max(out.maxFingerSelf, h.selfOverlap);
          out.maxHandHand = Math.max(out.maxHandHand, h.handOverlap);
          out.maxFingerBody = Math.max(out.maxFingerBody, h.fingerBody);
          out.heldHands += h.held;
          h.joints?.forEach((x, i) => {
            jointOut[i] = (jointOut[i] ?? 0) + +(x > 1);
            jointExcess[i] = (jointExcess[i] ?? 0) + x;
          });
        }
        const c = avatar.contactError?.(f, at);
        if (c != null) contact.push(c);
        if (opts.capture) images.push(canvas.toDataURL("image/jpeg", 0.85));
      }
    }
    return {
      frames, p95FrameMs: pct(cost, 0.95), spikes: steps.filter((d) => d > SPIKE_DEG).length, maxStepDeg: Math.max(...steps), p99StepDeg: pct(steps, 0.99),
      ...out, contactErrorMean: contact.length ? contact.reduce((a, b) => a + b) / contact.length : null,
      contactErrorMin: contact.length ? Math.min(...contact) : null, atLimit: steps.filter((d) => d > 19.5).length, ...edgeMotion(wrists, times, edges), ...(opts.capture ? { images } : {}), jointOut, jointExcess, ...(opts.framePoints || opts.motion ? { framePts } : {}),
    };
  };
  Object.assign(window, { bench: { run } });
  document.title = "bench ready";
}
