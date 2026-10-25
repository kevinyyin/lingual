// Webcam -> MediaPipe Tasks landmarks -> Frame. Only landmarks leave the page, never video.
import { createTrackers, detect, mergeParts, PARTS, toFrame, toWorld, type Part, type Parts, type TrackerMs, type World } from "./landmarks";
import type { FromWorker, ToWorker } from "./trackerWorker";
import type { Frame } from "./ws";

export { FACE_POINTS, matToYPR } from "./landmarks";

/** One camera picture through the trackers. */
interface Tracker { detect(video: HTMLVideoElement, t: number): Promise<{ parts: Parts; ms: TrackerMs }> }

/** Some trackers in one worker; `cpu`: those of them on the CPU. */
export interface Group { parts: Part[]; cpu?: Part[] }
/** Hands and pose on the GPU, the face on the CPU beside them: a laptop's integrated GPU runs trackers one at a time
 *  (side by side each took twice as long). Intel laptop, frames read a second: one worker 9.5, the page's thread 10.4,
 *  this 12.7 (all on the CPU 3.6; the face a frame behind instead of waited for, 12.8). */
export const WORKER_GROUPS: Group[] = [{ parts: ["hands", "pose"] }, { parts: ["face"], cpu: ["face"] }];

type Reply = Extract<FromWorker, { type: "frame" }>;

/** A worker with some of the trackers (trackerWorker.ts), once they are loaded. */
function spawn({ parts, cpu = [] }: Group): Promise<{ ask(t: number, image: ImageBitmap): Promise<Reply>; worker: Worker }> {
  const w = new Worker(new URL("./trackerWorker.ts", import.meta.url), { type: "module" });
  let pending: ((r: Reply) => void) | null = null;
  const settle = (r: Reply) => { const p = pending; pending = null; p?.(r); };
  return new Promise((resolve, reject) => {
    w.onerror = (e) => { e.preventDefault(); w.terminate(); reject(new Error(e.message || "tracker worker failed to load")); };
    w.onmessage = (e: MessageEvent<FromWorker>) => {
      const m = e.data;
      if (m.type === "error") { w.terminate(); reject(new Error(m.message)); }
      else if (m.type === "frame") settle(m);
      else {
        // a worker that dies later must not leave the camera loop waiting on it forever
        w.onerror = (ev) => { ev.preventDefault(); settle({ type: "frame", t: 0, parts: {}, ms: {}, error: ev.message || "tracker worker died" }); };
        resolve({
          worker: w,
          ask: (t, image) => new Promise((done) => {
            pending = done;
            w.postMessage({ type: "frame", t, image } satisfies ToWorker, [image]);
          }),
        });
      }
    };
    w.postMessage({ type: "init", parts, cpu } satisfies ToWorker);
  });
}

/** The trackers in workers: on the page's thread they took ~75 ms a frame (Intel laptop GPU) and the page, its camera
 *  preview and the avatar painted ~14 times a second. A worker that fails later hands over to the page's thread. */
async function workerTracker(groups: Group[]): Promise<Tracker> {
  const started = groups.map(spawn);
  const workers = await Promise.all(started).catch(async (e) => {
    (await Promise.allSettled(started)).forEach((s) => s.status === "fulfilled" && s.value.worker.terminate());
    throw e;
  });
  let fallback: Promise<Tracker> | null = null;
  return {
    async detect(video, t) {
      if (fallback) return (await fallback).detect(video, t);
      const images = await Promise.all(workers.map(() => createImageBitmap(video)));
      const replies = await Promise.all(workers.map((w, i) => w.ask(t, images[i])));
      const failed = replies.find((r) => r.error);
      if (failed) {
        console.warn("tracker:", failed.error);
        if (/died/.test(failed.error!)) {
          workers.forEach((w) => w.worker.terminate());
          fallback = pageTracker();
        }
      }
      return { parts: mergeParts(replies.map((r) => r.parts)), ms: Object.assign({}, ...replies.map((r) => r.ms)) };
    },
  };
}

/** The fallback where workers can't run them (no WebGL2 or module workers there): the page's own thread. */
async function pageTracker(): Promise<Tracker> {
  const trackers = await createTrackers(false, PARTS);
  return {
    async detect(video, t) {
      const ms: TrackerMs = {};
      return { parts: detect(trackers, video, t, ms), ms };
    },
  };
}

let trackers: Promise<Tracker> | null = null;
/** The three MediaPipe trackers (~30 MB of wasm and models), created once. Teaching mode calls this early so the
 *  first camera exercise does not wait for the download; a failed load is retried on the next call. `groups`: how
 *  they are split over workers, [] for the page's own thread (camperf.ts compares them). */
export const loadTrackers = (groups: Group[] = WORKER_GROUPS) => (trackers ??= (
  groups.length
    ? workerTracker(groups).catch((e) => { console.warn("trackers on the page's thread:", e); return pageTracker(); })
    : pageTracker()
).catch((e) => { trackers = null; throw e; }));

/** Per-tracker milliseconds for ?mode=camperf (camperf.ts); null when nobody is measuring. */
export let trackerTimes: Record<Part, number[]> | null = null;
export const measureTrackers = () => (trackerTimes = { hands: [], pose: [], face: [] });

/** `paused()`: skip detection (the avatar is playing); the backend ignores frames during avatar output anyway. */
/** `onFrame` also gets the frame's metric landmarks (World), which stay on the page: the avatar mirror reads them. */
export async function startCamera(video: HTMLVideoElement, onFrame: (f: Frame, world: World) => void, paused: () => boolean = () => false): Promise<void> {
  const tracker = await loadTrackers();

  // Feed MediaPipe the unmirrored stream; the preview is mirrored with CSS only.
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 }, audio: false });
  video.srcObject = stream;
  video.style.transform = "scaleX(-1)";
  await video.play();

  let last = -1, busy = false; // one picture at a time: a busy tracker skips pictures instead of queueing stale ones
  const tick = () => {
    if (video.srcObject !== stream) return; // a fresh feed (startCamera again on this video) has its own loop
    requestAnimationFrame(tick); // first: a detector that throws on one frame must not end tracking for good
    if (busy || paused() || video.currentTime === last) return;
    last = video.currentTime;
    busy = true;
    const t = performance.now();
    tracker.detect(video, t).then(({ parts, ms }) => {
      if (trackerTimes) PARTS.forEach((k) => ms[k] !== undefined && trackerTimes![k].push(ms[k]!));
      const frame = toFrame(parts, t);
      if (frame && video.srcObject === stream) onFrame(frame, toWorld(parts));
    }, (e) => console.warn("tracker:", e)).finally(() => { busy = false; });
  };
  requestAnimationFrame(tick);
}
