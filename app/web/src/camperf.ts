// ?mode=camperf: the camera's cost on the page, without the backend. Logs one "CAMPERF {...}" line after MEASURE_MS.
// &split=N: how the trackers are split over workers (SPLITS; 2 is camera.ts WORKER_GROUPS, 0 the page's own thread).
import { loadTrackers, measureTrackers, startCamera, WORKER_GROUPS, type Group } from "./camera";
import { PARTS } from "./landmarks";

const WARM_MS = 2000, MEASURE_MS = 15000;
const SPLITS: Record<number, Group[]> = {
  0: [], 1: [{ parts: PARTS }], 2: WORKER_GROUPS, 3: PARTS.map((p) => ({ parts: [p] })),
  4: [{ parts: ["hands"] }, { parts: ["pose", "face"] }],
  5: [{ parts: ["hands"] }, { parts: ["pose", "face"], cpu: ["pose", "face"] }],
  6: [{ parts: PARTS, cpu: PARTS }],
};

const stats = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b), q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))] ?? 0;
  return { n: s.length, med: +q(0.5).toFixed(1), p95: +q(0.95).toFixed(1) };
};

function renderer(): string {
  const gl = document.createElement("canvas").getContext("webgl2");
  const ext = gl?.getExtension("WEBGL_debug_renderer_info");
  return gl ? String(gl.getParameter(ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER)) : "no webgl2";
}

export async function startCamPerf(): Promise<void> {
  const out = document.createElement("pre");
  document.body.prepend(out);
  out.textContent = "camperf: loading trackers…";
  const split = Number(new URLSearchParams(location.search).get("split") ?? 2);
  const t0 = performance.now();
  await loadTrackers(SPLITS[split]);
  const loaded = performance.now();
  let frames: number[] = [], sizes: number[] = [], gaps: number[] = [], first = 0, sample = {};
  await startCamera(document.getElementById("cam") as HTMLVideoElement, (f) => {
    first ||= performance.now();
    if (f.left || f.right) { // what a frame with a hand up carries: 21-point hands, 33 pose, 52 blendshapes, 6 + 468 face
      sample = { left: f.left?.length, right: f.right?.length, pose: f.pose.length, blend: Object.keys(f.blend ?? {}).length,
        head: f.head?.length, face: f.face?.length, mesh: f.mesh?.length };
    }
    frames.push(performance.now());
    sizes.push(JSON.stringify({ type: "frame", ...f }).length);
  });
  const opened = performance.now();
  let last = performance.now();
  const paint = () => { const n = performance.now(); gaps.push(n - last); last = n; requestAnimationFrame(paint); };
  requestAnimationFrame(paint);
  while (!first) await new Promise((r) => setTimeout(r, 100)); // the first picture compiles the trackers' shaders
  const firstMs = first - opened, startGaps = gaps.filter((g) => g > 50).length;
  await new Promise((r) => setTimeout(r, WARM_MS));
  const times = measureTrackers();
  frames = []; sizes = []; gaps = [];
  out.textContent = "camperf: measuring…";
  await new Promise((r) => setTimeout(r, MEASURE_MS));
  const result = {
    split, renderer: renderer(),
    loadMs: Math.round(loaded - t0), firstMs: Math.round(firstMs), startLongGaps: startGaps,
    fps: +((frames.length * 1000) / MEASURE_MS).toFixed(1),
    hands: stats(times.hands), pose: stats(times.pose), face: stats(times.face),
    frameKB: +(sizes.reduce((a, b) => a + b, 0) / Math.max(1, sizes.length) / 1024).toFixed(1),
    paintGap: stats(gaps), // ms between animation frames: the preview's smoothness
    longGaps: gaps.filter((g) => g > 50).length,
    sample,
  };
  console.log("CAMPERF " + JSON.stringify(result));
  out.textContent = JSON.stringify(result, null, 1);
}
