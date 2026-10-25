// ?mode=mirror&record=N: N seconds of the mirror kept for diagnosis. The camera and the avatar as video, and per camera
// frame what was tracked (image and world landmarks) and what the avatar was given, with how the avatar actually stood
// ~15 times a second. Uploaded to the backend's logs/mirror_rec/<id>/ (main.py mirror_rec).
import type { ClipFrame } from "./avatar/player";
import type { Avatar } from "./avatar/rocketbox";
import type { World } from "./landmarks";
import type { Frame } from "./ws";

const AVATAR_EVERY_MS = 66;

export interface Recorder {
  frame(f: Frame, w: World, live: ClipFrame | null): void;
  avatar(t: number): void;
  /** The recording's id once uploaded. */
  done: Promise<string>;
}

function record(stream: MediaStream) {
  const type = ["video/webm;codecs=vp8", "video/webm"].find((m) => MediaRecorder.isTypeSupported(m));
  const r = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
  const chunks: Blob[] = [];
  r.ondataavailable = (e) => { if (e.data.size) chunks.push(e.data); };
  r.start(1000);
  return () => new Promise<Blob>((done) => { r.onstop = () => done(new Blob(chunks, { type: r.mimeType })); r.stop(); });
}

const v3 = (v: { x: number; y: number; z: number }) => [v.x, v.y, v.z].map((x) => Math.round(x * 1e4) / 1e4);
const v3a = (v: number[]) => v.map((x) => Math.round(x * 1e4) / 1e4);

export function recordMirror(seconds: number, cam: HTMLVideoElement, canvas: HTMLCanvasElement, avatar: Avatar,
  status: (text: string) => void): Recorder {
  const id = new Date().toISOString().replace(/[:.]/g, "-");
  const start = performance.now();
  const stops = [record(cam.srcObject as MediaStream), record(canvas.captureStream(30))];
  const log = {
    id, seconds, size: [cam.videoWidth, cam.videoHeight], canvas: [canvas.clientWidth, canvas.clientHeight],
    frames: [] as unknown[], avatar: [] as unknown[],
  };
  let on = true, lastAvatar = -Infinity;
  const done = (async () => {
    for (let left = seconds; left > 0; left--) {
      status(`Recording · ${left} s`);
      await new Promise((r) => setTimeout(r, 1000));
    }
    on = false;
    status("Saving the recording…");
    const [camera, character] = await Promise.all(stops.map((stop) => stop()));
    const files: [string, Blob][] = [["camera.webm", camera], ["avatar.webm", character], ["log.json", new Blob([JSON.stringify(log)])]];
    for (const [name, body] of files) {
      const r = await fetch(`/dev/mirror_rec/${id}/${name}`, { method: "POST", body });
      if (!r.ok) throw new Error(`upload ${name}: ${r.status}`);
    }
    status("Recording saved");
    return id;
  })();
  done.catch((e) => status(`Recording failed: ${(e as Error).message}`));
  return {
    frame(f, w, live) {
      if (on) log.frames.push({ t: f.t - start, frame: { ...f, mesh: undefined }, world: w, live });
    },
    avatar(t) {
      if (!on || t - lastAvatar < AVATAR_EVERY_MS) return;
      lastAvatar = t;
      const arms = avatar.armState?.();
      log.avatar.push({
        t: t - start,
        arms: arms && Object.fromEntries((["L", "R"] as const).map((k) => [k, {
          wrist: v3(arms[k].wrist), elbow: v3(arms[k].elbow), elbowDeg: Math.round(arms[k].elbowDeg), twistDeg: Math.round(arms[k].twistDeg),
        }])),
        palms: avatar.palmScreen?.(), bends: avatar.fingerBends?.(), hand: avatar.handReport?.(),
        // per arm: wrist, five fingertips, elbow (avatar metres), and its shoulder half-width
        points: (({ pts, shoulderHalf }) => ({ pts: pts.map(v3a), shoulderHalf }))(avatar.framePoints?.(true) ?? { pts: [], shoulderHalf: 0 }),
      });
    },
    done,
  };
}
