// ?mode=mirror: the avatar copies the signer live from the camera (world landmarks: the arms in depth, every finger
// joint), and each sign of a turn is labelled with the word it was read as (the backend's mirror mode, session.py).
import "./mirror.css";
import { loadAvatar } from "./avatar";
import { lerpFrame, type ClipFrame } from "./avatar/player";
import type { FacePose } from "./avatar/rocketbox";
import { loadRest } from "./avatar/stage";
import { savedBuddy } from "./buddies";
import { startCamera } from "./camera";
import { recordMirror, type Recorder } from "./mirrorRecord";
import type { World } from "./landmarks";
import { PositionWatch } from "./position";
import { connect, type Frame, type PickOption, type State } from "./ws";

const MAX_BUFFERED = 256 * 1024; // drop frames rather than queue stale ones (main.ts)
const ASPECT = 640 / 480; // camera.ts asks for 640x480
const GLIDE_MS = 85; // first guess at the time between camera frames (~12/s on a laptop GPU); learned as they arrive
const GLIDE_MAX_MS = 120;
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
const FRONT_MIN = -0.2, FRONT_MAX = 0.6; // metres in front of the shoulders: beyond is a misfit, not a reach
const MEET_NEAR = 0.08, MEET_FAR = 0.25; // hands' nearest points apart in the picture, shoulder widths: they touch / don't
const REACH_USE = 0.85; // share of the arm's forward reach a signer uses: the elbow stays bent
const HIP_SEEN = 0.5; // pose visibility below which the hips are a guess
const TORSO_M = 0.5; // shoulders to hips, metres (only the direction matters)
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
  return s > 1e-6 ? focal(aspect) / s : null;
}

/** The camera's focal length in image heights. */
const focal = (aspect: number) => (0.5 / Math.tan((HFOV_DEG * Math.PI) / 360)) * aspect;

/** Typical adult palm segments (landmarks, metres): wrist to middle knuckle, across the knuckles, wrist to index and to
 *  little knuckle. */
const PALM: [number, number, number][] = [[0, 9, 0.095], [5, 17, 0.07], [0, 5, 0.09], [0, 17, 0.08]];
/** MediaPipe's metric hand is believed with its palm (wrist to middle knuckle) this long, metres: a hand flat on the
 *  chest or fingers interlaced read 2-9 cm, and its depth 40-55 cm off; open hands read 9.5-10.8. */
const PALM_OK: [number, number] = [0.09, 0.13];

/** Camera distance (metres) of a hand by its palm in the picture against a typical palm: the least foreshortened
 *  segment (largest against its usual length) sets it. */
export function distanceByPalm(img: number[][], aspect: number): number | null {
  const s = Math.max(...PALM.map(([a, b, m]) => Math.hypot((img[a][0] - img[b][0]) * aspect, img[a][1] - img[b][1]) / m));
  return s > 1e-6 ? focal(aspect) / s : null;
}

/** A camera frame as an avatar clip frame. Hands are keyed by the signer's own side, decided by which pose wrist each
 *  hand sits on (pose 15 = the signer's left), as the avatar's clips are (training/build_motion_library.by_wrist):
 *  MediaPipe's handedness labels disagree with that on ~22% of frames. World landmarks give the arms their depth and
 *  the finger solver every joint. Each seen hand's depth comes from perspective (its size in the picture against its true
 *  size, relative to the head and shoulders'): the pose model's wrist depth barely moves (30-34 cm in front of the
 *  shoulders for a hand beside the shoulder and one reaching out alike, WLASL AWAY and BRING), where this reads 0 and
 *  15-40. `ease(side, metres)` smooths it over frames. Null without a body in the picture. */
export function liveFrame(f: Frame, w: World, aspect = ASPECT, ease?: (side: Side, front: number) => number,
  avatarHand?: number): ClipFrame | null {
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
  // hips out of the picture (a webcam at a desk: seen 2%): their guess leaned the avatar's body axis 7-14 deg toward the
  // camera, which lifted forward hands 0.1-0.3 shoulder widths (hands together at the chest signed at the chin). An
  // upright torso instead: hips straight below the shoulders.
  if (Math.min(f.pose[23][3] ?? 1, f.pose[24][3] ?? 1) < HIP_SEEN) {
    const sy = (out.pose[11][1] + out.pose[12][1]) / 2;
    out.pose[23] = [out.pose[11][0], sy + TORSO_M, shoulders];
    out.pose[24] = [out.pose[12][0], sy + TORSO_M, shoulders];
  }
  const sw = Math.hypot((f.pose[11][0] - f.pose[12][0]) * aspect, f.pose[11][1] - f.pose[12][1]) || 1; // image heights
  // each wrist where the hand tracker sees it: the pose model's wrist strays when hands overlap (webcam recording: 0.2-0.38
  // shoulder widths off with hands together, 0.10 apart), and the avatar's hands stopped at different heights
  const perMetre = body === null ? null : focal(aspect) / body; // image heights per metre at the body's distance
  if (perMetre !== null) {
    for (const side of SIDES) {
      const img = out[side] as number[][] | null;
      if (!img) continue;
      const wi = HAND_POSE[side].hand[0];
      const dx = ((img[0][0] - f.pose[wi][0]) * aspect) / perMetre, dy = (img[0][1] - f.pose[wi][1]) / perMetre;
      for (const k of HAND_POSE[side].hand) { out.pose[k][0] += dx; out.pose[k][1] += dy; }
      out.pose2d[wi] = [img[0][0], img[0][1]];
    }
  }
  const fronts: Partial<Record<Side, number>> = {};
  for (const side of SIDES) {
    const img = out[side] as number[][] | null, wld = out[side === "left" ? "leftW" : "rightW"] as number[][] | null;
    if (!img || body === null) continue;
    const front = handFront(img, wld, body, aspect);
    if (front === null) continue; // unsized: the pose model's depth stays
    const capped = Math.min(front, reach(out.pose, side)); // the wrist as anchored above; depths not yet set
    fronts[side] = ease ? ease(side, capped) : capped;
  }
  // hands together (CLAP, NICE, prayer): they meet on the avatar, at one depth
  if (out.left && out.right) {
    const meet = clamp01((MEET_FAR - handGap(out.left as number[][], out.right as number[][], aspect) / sw) / (MEET_FAR - MEET_NEAR));
    if (meet > 0) {
      out.meetL = out.meetR = meet;
      if (meet > 0.5 && (fronts.left === undefined) !== (fronts.right === undefined)) { // one unsized: it is where the other is
        fronts.left = fronts.right = fronts.left ?? fronts.right;
      } else if (fronts.left !== undefined && fronts.right !== undefined) {
        const mid = (fronts.left + fronts.right) / 2;
        fronts.left += (mid - fronts.left) * meet;
        fronts.right += (mid - fronts.right) * meet;
      }
      if (avatarHand && perMetre !== null) towardContact(out, w, meet, avatarHand, perMetre, aspect);
    }
  }
  for (const side of SIDES) {
    const front = fronts[side];
    if (front === undefined) continue;
    const { hand: pts, elbow } = HAND_POSE[side];
    const dz = shoulders - front - out.pose[pts[0]][2]; // world z is smaller toward the camera
    for (const k of pts) out.pose[k][2] += dz;
    out.pose[elbow][2] += dz / 2;
  }
  // No touch weights for the body: the recording's hands beside the head and index fingers up before the chest read
  // within 10 cm of the shoulders too, and a touch glues the hand to the avatar. A hand flat on the chest reads ~14 cm,
  // a few cm off the avatar's chest.
  return out;
}

/** Metres a seen hand is in front of the shoulders, from the camera distances of body and hand: by MediaPipe's metric
 *  hand when its palm is a hand's size, else by its palm in the picture against a typical palm; null when neither is
 *  believable. Recording of a webcam signer, against what the video shows: flat on the chest ~8 cm, metric 55-58,
 *  picture 14; hands together at the chest ~15, metric 55-61 (palm 7.7 cm), picture 24-31. */
function handFront(img: number[][], wld: number[][] | null, body: number, aspect: number): number | null {
  const palm = wld ? Math.hypot(...[0, 1, 2].map((k) => wld[9][k] - wld[0][k])) : 0;
  const metric = wld && palm >= PALM_OK[0] && palm <= PALM_OK[1] ? distanceBySize(img, wld, aspect) : null;
  if (metric !== null) return Math.max(FRONT_MIN, Math.min(FRONT_MAX, body - metric));
  const byPalm = distanceByPalm(img, aspect);
  const front = byPalm === null ? null : body - byPalm;
  // palms pressed together are seen edge-on and read far behind the body (-69 cm): no reading beats that
  return front !== null && front >= FRONT_MIN && front <= FRONT_MAX ? front : null;
}

/** Hands that meet, sized for the avatar: its hands are smaller for its body than a signer's, so placed by the wrists its
 *  fingers stopped short (a webcam recording: the signer's hands touching, the avatar's 0.28-0.47 shoulder widths
 *  apart). The signer's wrists are some number of their own hand lengths apart (both hands at one distance from the
 *  camera, so the picture measures it); the avatar's are set that many of its hand lengths apart, both wrists scaled
 *  about where the hands touch in the picture. `avatarHand`: its wrist to middle fingertip over its shoulder width. */
function towardContact(out: ClipFrame, w: World, meet: number, avatarHand: number, perMetre: number, aspect: number): void {
  const img = { left: out.left as number[][], right: out.right as number[][] };
  const d2 = (p: number[], q: number[]) => Math.hypot((p[0] - q[0]) * aspect, p[1] - q[1]);
  // wrist to middle fingertip, or from the palm (53% of it) where interlaced fingers hide the tips
  const handLen = (h: number[][]) => Math.max(d2(h[0], h[12]), d2(h[0], h[9]) / 0.53);
  const signerGap = d2(img.left[0], img.right[0]) / ((handLen(img.left) + handLen(img.right)) / 2); // in hand lengths
  const shoulders = Math.hypot(...[0, 1, 2].map((k) => w.pose![11][k] - w.pose![12][k]));
  const now = Math.hypot(out.pose[15][0] - out.pose[16][0], out.pose[15][1] - out.pose[16][1]); // metres
  if (!(now > 1e-3) || !(shoulders > 1e-3)) return;
  // the avatar maps by shoulder width: a gap of g signer metres is g / shoulders of its shoulder widths
  const want = signerGap * avatarHand * shoulders;
  const k = 1 + (Math.max(0.2, Math.min(1, want / now)) - 1) * meet; // 1: wrists stay; below: toward the contact
  let best = Infinity, at = [0, 0];
  for (const p of img.left) for (const q of img.right) {
    const d = Math.hypot((p[0] - q[0]) * aspect, p[1] - q[1]);
    if (d < best) [best, at] = [d, [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2]];
  }
  for (const side of SIDES) {
    const wrist = img[side][0], move = 1 - k;
    const dx = (((at[0] - wrist[0]) * aspect) / perMetre) * move, dy = ((at[1] - wrist[1]) / perMetre) * move;
    for (const i of HAND_POSE[side].hand) { out.pose[i][0] += dx; out.pose[i][1] += dy; }
    const wi = HAND_POSE[side].hand[0];
    out.pose2d[wi] = [out.pose2d[wi][0] + (at[0] - wrist[0]) * move, out.pose2d[wi][1] + (at[1] - wrist[1]) * move];
  }
}

/** Farthest a wrist can be in front of its shoulder, metres (pose world): the arm's length (shoulder, elbow, wrist)
 *  against how far across and up from the shoulder the wrist already is, the elbow never quite straight (REACH_USE).
 *  Hands together before the stomach read 50-90 cm in front by size (a second webcam recording, the signer 1.1-1.4 m
 *  from the camera): out of the avatar's reach, so its hands stopped ~20 cm apart. */
export function reach(pose: number[][], side: Side): number {
  const [s, e, wr] = side === "left" ? [11, 13, 15] : [12, 14, 16];
  const len = (a: number, b: number) => Math.hypot(...[0, 1, 2].map((k) => pose[a][k] - pose[b][k]));
  const arm = len(s, e) + len(e, wr), across = Math.hypot(pose[wr][0] - pose[s][0], pose[wr][1] - pose[s][1]);
  return REACH_USE * Math.sqrt(Math.max(0, arm * arm - across * across));
}

/** Nearest two points of two hands in the picture, image heights. */
function handGap(a: number[][], b: number[][], aspect: number): number {
  let d = Infinity;
  for (const p of a) for (const q of b) d = Math.min(d, Math.hypot((p[0] - q[0]) * aspect, p[1] - q[1]));
  return d;
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
    if (m.type === "you_said") { // the English sentence the turn's signs make (Claude, session.py), after their words
      const row = turns.firstElementChild;
      if (!row) return;
      row.appendChild(el("p", "mirror-sentence", m.text || "No sentence in these signs"));
      if (m.text) now.textContent = m.text;
      return;
    }
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

  // drawn lighter: the trackers share the GPU with it (laptop, fake camera: camera frames read 5.5-7.5/s at pixel
  // ratio 2, 7.5-10.5 at 1), and it gets ~12 new poses a second, so 30 drawn a second lose nothing
  const [avatar, rest] = await Promise.all([
    loadAvatar(document.getElementById("avatar") as HTMLCanvasElement, { maxPixelRatio: 1, maxFps: 30 }), loadRest()]);
  avatar.applyFrame(rest); // the first frame snaps every filter and spring: the avatar starts at rest
  // its hand for its body (wrist to middle fingertip over shoulder width): hands that meet are sized to it (towardContact)
  const fp = avatar.framePoints?.(true);
  const avatarHand = fp && fp.pts.length >= 7 && fp.shoulderHalf > 0
    ? Math.hypot(...[0, 1, 2].map((k) => fp.pts[0][k] - fp.pts[3][k])) / (2 * fp.shoulderHalf) : undefined;

  // ?record=N: N seconds kept for diagnosis (mirrorRecord.ts), from when the camera is up
  const recordS = Number(new URLSearchParams(location.search).get("record")) || 0;
  let rec: Recorder | null = null;
  // each camera frame is glided to over about the time until the next, so the avatar moves every screen frame
  let from = rest, to = rest, shown = rest, at = performance.now(), glide = GLIDE_MS, arrived = 0;
  let browUp = 0, browDown = 0, browUpTo = 0, browDownTo = 0;
  const toward = (f: ClipFrame) => {
    const t = performance.now();
    // the glide adds its own length to the delay: capped low (a webcam recording followed the camera 0.7-0.9 s late)
    if (arrived) glide = 0.8 * glide + 0.2 * Math.max(30, Math.min(GLIDE_MAX_MS, t - arrived));
    arrived = t;
    from = shown;
    to = f;
    at = t;
  };
  // what the page keeps up: camera frames read a second, ms the avatar's pose takes per screen frame, screen frame gap
  const stats = { camFps: 0, poseMs: 0, frameMs: 0 }, camTimes: number[] = [];
  (window as unknown as { __mirror: typeof stats }).__mirror = stats;
  let lastTick = performance.now();
  const tick = () => {
    const t = performance.now();
    stats.frameMs = 0.9 * stats.frameMs + 0.1 * (t - lastTick);
    lastTick = t;
    if (to !== rest && t - arrived > LOST_MS) toward(rest); // they stepped away
    shown = lerpFrame(from, to, Math.min(1, (t - at) / glide), (w) => w, false);
    avatar.applyFrame(shown, t);
    stats.poseMs = 0.9 * stats.poseMs + 0.1 * (performance.now() - t);
    rec?.avatar(t);
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
      const live = liveFrame(f, w, cam.videoWidth && cam.videoHeight ? cam.videoWidth / cam.videoHeight : ASPECT, ease, avatarHand);
      for (const s of SIDES) if (!live?.[s]) fronts[s] = null; // a hand that comes back starts from its own depth
      if (live) toward(live);
      rec?.frame(f, w, live);
      const now = performance.now();
      camTimes.push(now);
      while (camTimes[0] < now - 2000) camTimes.shift();
      stats.camFps = camTimes.length / 2;
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
    if (recordS) {
      const pill = panel.appendChild(el("p", "mirror-rec"));
      rec = recordMirror(recordS, cam, document.getElementById("avatar") as HTMLCanvasElement, avatar, (s) => (pill.textContent = s));
    }
  } catch (e) {
    badge.textContent = (e as Error).name === "NotAllowedError" ? "Camera blocked: allow it in Settings" : "No camera";
    badge.dataset.state = "error";
  }
}
