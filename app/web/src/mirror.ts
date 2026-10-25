// ?mode=mirror: the avatar copies the signer live from the camera (world landmarks: the arms in depth, every finger
// joint), and each sign of a turn is labelled with the word it was read as (the backend's mirror mode, session.py).
import "./mirror.css";
import { loadAvatar } from "./avatar";
import { lerpFrame, type ClipFrame } from "./avatar/player";
import type { FacePose } from "./avatar/rocketbox";
import { loadRest } from "./avatar/stage";
import { savedBuddy } from "./buddies";
import { startCamera } from "./camera";
import type { World } from "./landmarks";
import { PositionWatch } from "./position";
import { connect, type Frame, type PickOption, type State } from "./ws";

const MAX_BUFFERED = 256 * 1024; // drop frames rather than queue stale ones (main.ts)
const ASPECT = 640 / 480; // camera.ts asks for 640x480
const GLIDE_MS = 85; // first guess at the time between camera frames (~12/s on a laptop GPU); learned as they arrive
const LOST_MS = 1000; // no body this long: the avatar goes back to rest
const BROW_GAIN = 1.5; // MediaPipe's brow blendshapes rarely pass 0.6
const BROW_EASE = 0.2; // per animation frame
const DEPTH_EASE = 0.5; // per camera frame: perspective depth jitters ~5 cm frame to frame
const TURNS_KEPT = 8;

type Side = "left" | "right";
const SIDES: Side[] = ["left", "right"];

const near2 = (a: number[], b: number[]) => (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2;
const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/** A laptop webcam's horizontal field of view (60-78 degrees across models): it scales depth, never its direction. */
const HFOV_DEG = 65;
const BODY_POINTS = [0, 7, 8, 11, 12]; // nose, ears, shoulders: a near-rigid spread to size the body by
const FRONT_MIN = -0.2, FRONT_MAX = 0.7; // metres in front of the shoulders: beyond is a misfit, not a reach
/** Pose points moved with each wrist when its depth is set (wrist, pinky, index, thumb), and its elbow (half as far). */
const HAND_POSE = { left: { hand: [15, 17, 19, 21], elbow: 13 }, right: { hand: [16, 18, 20, 22], elbow: 14 } };

/** Camera distance (metres) of points seen at `img` (normalised image x, y) whose true shape is `wld` (metres, in the
 *  camera's axes, as MediaPipe's world landmarks are): by perspective, the true spread over the image spread. Null when
 *  too small to size. */
export function distanceBySize(img: number[][], wld: number[][], aspect: number): number | null {
  const n = img.length;
  const ca = [0, 1].map((k) => img.reduce((s, p) => s + p[k], 0) / n), cb = [0, 1].map((k) => wld.reduce((s, p) => s + p[k], 0) / n);
  let ab = 0, bb = 0;
  for (let i = 0; i < n; i++) {
    const a = [(img[i][0] - ca[0]) * aspect, img[i][1] - ca[1]], b = [wld[i][0] - cb[0], wld[i][1] - cb[1]]; // image in image heights
    ab += a[0] * b[0] + a[1] * b[1];
    bb += b[0] * b[0] + b[1] * b[1];
  }
  const s = bb > 1e-6 ? ab / bb : 0; // image heights per metre
  return s > 1e-6 ? ((0.5 / Math.tan((HFOV_DEG * Math.PI) / 360)) * aspect) / s : null; // focal length in image heights / s
}

/** A camera frame as an avatar clip frame. Hands are keyed by the signer's own side, decided by which pose wrist each
 *  hand sits on (pose 15 = the signer's left), as the avatar's clips are (training/build_motion_library.by_wrist):
 *  MediaPipe's handedness labels disagree with that on ~22% of frames. World landmarks give the arms their depth and
 *  the finger solver every joint. Each seen hand's depth comes from perspective (its size in the picture against its true
 *  size, relative to the head and shoulders'): the pose model's wrist depth barely moves (30-34 cm in front of the
 *  shoulders for a hand beside the shoulder and one reaching out alike, WLASL AWAY and BRING), where this reads 0 and
 *  15-40. `ease(side, metres)` smooths it over frames. Null without a body in the picture. */
export function liveFrame(f: Frame, w: World, aspect = ASPECT, ease?: (side: Side, front: number) => number): ClipFrame | null {
  if (!w.pose) return null;
  const out: ClipFrame = { pose: w.pose.map((p) => [...p]), pose2d: f.pose.map((p) => [p[0], p[1]]), left: null, right: null, leftW: null, rightW: null, aspect };
  for (const label of SIDES) {
    const h = f[label];
    if (!h) continue;
    let side: Side = near2(h[0], f.pose[15]) < near2(h[0], f.pose[16]) ? "left" : "right";
    if (out[side]) side = side === "left" ? "right" : "left"; // both nearest one wrist (crossed hands): keep them apart
    out[side] = h;
    out[side === "left" ? "leftW" : "rightW"] = w[label];
  }
  const body = distanceBySize(BODY_POINTS.map((i) => f.pose[i]), BODY_POINTS.map((i) => w.pose![i]), aspect);
  const shoulders = (out.pose[11][2] + out.pose[12][2]) / 2;
  for (const side of SIDES) {
    const img = out[side] as number[][] | null, wld = out[side === "left" ? "leftW" : "rightW"] as number[][] | null;
    const hand = img && wld && body ? distanceBySize(img, wld, aspect) : null;
    if (hand === null || body === null) continue; // unsized: the pose model's depth stays
    let front = Math.max(FRONT_MIN, Math.min(FRONT_MAX, body - hand));
    if (ease) front = ease(side, front);
    const { hand: pts, elbow } = HAND_POSE[side];
    const dz = shoulders - front - out.pose[pts[0]][2]; // world z is smaller toward the camera
    for (const k of pts) out.pose[k][2] += dz;
    out.pose[elbow][2] += dz / 2;
  }
  return out;
}

/** How far the wrist is in front of the shoulders, cm (negative: behind them). MediaPipe's world z is metres from the
 *  hips, smaller toward the camera. */
export function wristDepthCm(pose: number[][], side: Side): number {
  const shoulders = (pose[11][2] + pose[12][2]) / 2;
  return Math.round((shoulders - pose[side === "left" ? 15 : 16][2]) * 100);
}

/** Each finger's chain of hand landmarks, thumb to pinky. */
const CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
/** Bend (degrees, summed along the chain) of a fully curled finger: a fist bends a finger ~90 + 100 + 70, the thumb
 *  far less. */
export const FULL_CURL_DEG = [120, 200, 200, 200, 200];
export const FINGER_NAMES = ["T", "I", "M", "R", "P"];

/** Curl of each finger, thumb to pinky: 0 straight, 1 fully bent, from the hand's world landmarks (metres). A finger's
 *  knuckle bend is against the palm's axis (wrist to middle knuckle), so the fingers' fan doesn't count as bend. */
export function fingerCurls(hand: number[][]): number[] {
  const dir = (a: number[], b: number[]) => { const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], n = Math.hypot(...d) || 1; return d.map((v) => v / n); };
  const deg = (u: number[], v: number[]) => (Math.acos(Math.max(-1, Math.min(1, u[0] * v[0] + u[1] * v[1] + u[2] * v[2]))) * 180) / Math.PI;
  const palm = dir(hand[0], hand[9]);
  return CHAINS.map((chain, c) => {
    const bones = chain.slice(1).map((j, i) => dir(hand[chain[i]], hand[j]));
    let bend = c ? deg(palm, bones[0]) : 0; // the thumb's base swings sideways, not toward the palm: its two joints only
    for (let i = 1; i < bones.length; i++) bend += deg(bones[i - 1], bones[i]);
    return clamp01(bend / FULL_CURL_DEG[c]);
  });
}

/** One read sign as shown: the top guess's word and confidence, and the next guesses. */
export function signLabel(options: PickOption[]): { word: string; pct: number; alts: string[] } {
  const [top, ...rest] = options;
  return { word: top?.english ?? "?", pct: Math.round((top?.p ?? 0) * 100), alts: rest.slice(0, 2).map((o) => o.english) };
}

const STATE_TEXT: Partial<Record<State, string>> = {
  calibrating: "Hold still: calibrating", ready: "Sign to me", signing: "Signing…", reading: "Reading your signs…",
};

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};

export async function startMirror(): Promise<void> {
  document.body.dataset.page = "mirror";
  document.body.dataset.mode = "signing"; // the camera preview shows (style.css hides it in voice mode)
  document.querySelector(".mode-name")!.textContent = "Mirror";
  const badge = document.getElementById("state")!;
  const panel = document.getElementById("avatar-panel")!;
  const now = panel.appendChild(el("p", "mirror-now")); // the latest turn's words, over the avatar
  const hint = panel.appendChild(el("p", "mirror-hint"));
  const turns = document.querySelector(".caption-box")!.appendChild(el("ol", "mirror-turns"));
  turns.setAttribute("aria-live", "polite");
  turns.setAttribute("aria-label", "Signs read, newest first");
  const hud = document.getElementById("side")!.appendChild(el("section", "card mirror-hud"));
  hud.setAttribute("aria-label", "What the camera tracks");
  hud.appendChild(el("h2", "card-title", "Tracking"));
  const rows = Object.fromEntries(SIDES.map((s) => {
    const row = hud.appendChild(el("div", "hand-row"));
    row.appendChild(el("span", "hand-name", s === "left" ? "Left hand" : "Right hand"));
    const depth = row.appendChild(el("span", "hand-depth", "not seen"));
    const bars = row.appendChild(el("span", "fingers"));
    const fills = FINGER_NAMES.map((n) => {
      const b = bars.appendChild(el("span", "finger"));
      const fill = b.appendChild(el("i"));
      b.appendChild(el("b", undefined, n));
      return fill;
    });
    return [s, { row, depth, fills }];
  })) as unknown as Record<Side, { row: HTMLElement; depth: HTMLElement; fills: HTMLElement[] }>;

  let state: State | null = null;
  const setState = (s: State) => {
    state = s;
    badge.textContent = STATE_TEXT[s] ?? s;
    badge.dataset.state = s;
  };

  const sock = connect((m) => {
    if (m.type === "state") return setState(m.state);
    if (m.type !== "recognized" || !m.spans || !m.segments.length) return; // spans: the mirror's per-sign reads, not a lesson's
    const labels = m.segments.map(signLabel);
    now.textContent = labels.map((l) => l.word).join(" · ");
    const li = el("li", "mirror-turn");
    for (const l of labels) {
      const chip = li.appendChild(el("span", "sign-chip"));
      chip.appendChild(el("b", undefined, l.word));
      chip.appendChild(el("span", "pct", `${l.pct}%`));
      if (l.alts.length) chip.appendChild(el("span", "alts", `or ${l.alts.join(" · ")}`));
    }
    turns.prepend(li);
    while (turns.children.length > TURNS_KEPT) turns.lastElementChild!.remove();
  }, (e) => {
    badge.textContent = e.code === 4001 ? "Open elsewhere" : "Connection lost";
    badge.dataset.state = "error";
  });
  sock.send({ type: "buddy", buddy: savedBuddy().id });
  sock.send({ type: "mirror", on: true });
  sock.send({ type: "mode", mode: "signing" });

  const [avatar, rest] = await Promise.all([loadAvatar(document.getElementById("avatar") as HTMLCanvasElement), loadRest()]);
  avatar.applyFrame(rest); // the first frame snaps every filter and spring: the avatar starts at rest

  // each camera frame is glided to over about the time until the next, so the avatar moves every screen frame
  let from = rest, to = rest, shown = rest, at = performance.now(), glide = GLIDE_MS, arrived = 0;
  let browUp = 0, browDown = 0, browUpTo = 0, browDownTo = 0;
  const toward = (f: ClipFrame) => {
    const t = performance.now();
    if (arrived) glide = 0.8 * glide + 0.2 * Math.max(30, Math.min(300, t - arrived));
    arrived = t;
    from = shown;
    to = f;
    at = t;
  };
  const tick = () => {
    const t = performance.now();
    if (to !== rest && t - arrived > LOST_MS) toward(rest); // they stepped away
    shown = lerpFrame(from, to, Math.min(1, (t - at) / glide), (w) => w, false);
    avatar.applyFrame(shown, t);
    browUp += (browUpTo - browUp) * BROW_EASE;
    browDown += (browDownTo - browDown) * BROW_EASE;
    const face: FacePose = { browUp, browDown, headYawDeg: 0, headPitchDeg: 0, expressions: {} };
    avatar.applyFace(face);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  const watch = new PositionWatch(); // directs the signer into position; calibrates again when they move (position.ts)
  const cam = document.getElementById("cam") as HTMLVideoElement;
  const fronts: Record<Side, number | null> = { left: null, right: null }; // each hand's depth, eased over frames
  const ease = (s: Side, v: number) => (fronts[s] = fronts[s] === null ? v : fronts[s]! + (v - fronts[s]!) * DEPTH_EASE);
  try {
    await startCamera(cam, (f, w) => {
      const { hint: h, send, recalibrate } = watch.check(f, state, performance.now());
      hint.textContent = h ?? "";
      if (recalibrate) sock.send({ type: "calibrate" });
      if (send && sock.buffered() < MAX_BUFFERED) sock.send({ type: "frame", ...f });
      // the camera may not give the 640x480 asked for (many laptops send 16:9): the real shape, for placement and depth
      const live = liveFrame(f, w, cam.videoWidth && cam.videoHeight ? cam.videoWidth / cam.videoHeight : ASPECT, ease);
      for (const s of SIDES) if (!live?.[s]) fronts[s] = null; // a hand that comes back starts from its own depth
      if (live) toward(live);
      const b = f.blend ?? {};
      browUpTo = clamp01(BROW_GAIN * (b.browInnerUp ?? 0));
      browDownTo = clamp01((BROW_GAIN * ((b.browDownLeft ?? 0) + (b.browDownRight ?? 0))) / 2);
      for (const s of SIDES) {
        const hand = live?.[s === "left" ? "leftW" : "rightW"];
        const r = rows[s];
        r.row.classList.toggle("seen", !!hand);
        r.depth.textContent = hand && live ? `${wristDepthCm(live.pose, s)} cm in front` : "not seen";
        const curls = hand ? fingerCurls(hand) : [0, 0, 0, 0, 0];
        r.fills.forEach((fill, i) => (fill.style.height = `${Math.round(curls[i] * 100)}%`));
      }
    });
  } catch (e) {
    badge.textContent = (e as Error).name === "NotAllowedError" ? "Camera blocked: allow it in Settings" : "No camera";
    badge.dataset.state = "error";
  }
}
