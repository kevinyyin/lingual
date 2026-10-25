/// <reference lib="webworker" />
// Trackers off the page's thread (camera.ts): on it the three took ~75 ms a frame on an Intel laptop GPU, and the page,
// its camera preview and the avatar painted ~14 times a second.
import { createTrackers, detect, type Part, type Parts, type TrackerMs, type Trackers } from "./landmarks";

export type ToWorker = { type: "init"; parts: Part[]; cpu: Part[] } | { type: "frame"; t: number; image: ImageBitmap };
export type FromWorker =
  | { type: "ready" }
  | { type: "error"; message: string }
  | { type: "frame"; t: number; parts: Parts; ms: TrackerMs; error?: string };

let trackers: Trackers = {};
const post = (m: FromWorker) => self.postMessage(m);

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const m = e.data;
  if (m.type === "init") {
    try {
      trackers = await createTrackers(true, m.parts, m.cpu);
      post({ type: "ready" });
    } catch (err) {
      post({ type: "error", message: String(err) });
    }
    return;
  }
  const ms: TrackerMs = {};
  try { // a tracker that throws on one picture must not end tracking for good
    post({ type: "frame", t: m.t, parts: detect(trackers, m.image, m.t, ms), ms });
  } catch (err) {
    post({ type: "frame", t: m.t, parts: {}, ms, error: String(err) });
  } finally {
    m.image.close();
  }
};
