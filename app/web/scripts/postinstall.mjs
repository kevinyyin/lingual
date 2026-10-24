// npm postinstall (cross-platform): MediaPipe wasm into public/wasm, pinned MediaPipe models into public/models.
// The models are shared with training/landmarks.py, so the browser and Python extraction use identical files.
import { cpSync, existsSync, mkdirSync, renameSync, writeFileSync } from "node:fs";

const BASE = "https://storage.googleapis.com/mediapipe-models";
const MODELS = {
  "hand_landmarker.task": `${BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
  "pose_landmarker_lite.task": `${BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  "face_landmarker.task": `${BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
};

cpSync("node_modules/@mediapipe/tasks-vision/wasm", "public/wasm", { recursive: true });

mkdirSync("public/models", { recursive: true });
for (const [name, url] of Object.entries(MODELS)) {
  const out = `public/models/${name}`;
  if (existsSync(out)) continue;
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  writeFileSync(`${out}.part`, Buffer.from(await r.arrayBuffer()));
  renameSync(`${out}.part`, out); // never leave a truncated model behind
  console.log(`fetched ${out}`);
}
