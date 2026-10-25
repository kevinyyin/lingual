// Teaching mode's links to the rest of the app: one avatar canvas that lessons borrow, and the camera +
// backend socket in practice mode, which grades the learner's own signing.
import { loadAvatar } from "../avatar";
import { createStage, type Stage } from "../avatar/stage";
import { attachTrail, type Pt } from "../avatar/trail";
import { startCamera } from "../camera";
import { PositionWatch } from "../position";
import { connect, type Frame, type P3, type PickOption, type SignToken, type Socket, type State } from "../ws";
import { ALL_LESSONS, faceFor } from "./curriculum";
import { savedBuddy, type Buddy } from "../buddies";

function newCanvas(): HTMLCanvasElement {
  const c = Object.assign(document.createElement("canvas"), { className: "avatar-canvas" });
  c.setAttribute("aria-label", "Signing avatar");
  return c;
}
/** One canvas for the whole page (the avatar is slow to load): each exercise moves it into its own slot. */
export let avatarCanvas = newCanvas();
/** The hand trail's canvas, laid over the avatar's; it travels with it. */
export const trailCanvas = Object.assign(document.createElement("canvas"), { className: "trail-canvas" });
trailCanvas.setAttribute("aria-hidden", "true");
let trail: ReturnType<typeof attachTrail> | null = null;
/** Puts the avatar and its trail into `slot`, the trail empty (the avatar moves between screens). */
export function mountSigner(slot: HTMLElement): void {
  trail?.clear();
  slot.prepend(avatarCanvas, trailCanvas);
}
let stage: Promise<Stage> | null = null;
export const loadStage = () => (stage ??= loadAvatar(avatarCanvas).then((avatar) => {
  // reduced motion: the trail shows each sign's whole path, still, once it ends
  trail = attachTrail(trailCanvas, () => avatar.palmScreen?.(), { still: matchMedia("(prefers-reduced-motion: reduce)").matches });
  return createStage(avatar, { onFrame: (f) => trail!.frame(f) });
}));
/** A new signing buddy was picked: the next loadStage loads it on a fresh canvas, put where the old one was. */
export function reloadSigner(): void {
  if (!stage) return;
  stage = null;
  const old = avatarCanvas;
  avatarCanvas = newCanvas();
  old.replaceWith(avatarCanvas);
  old.getContext("webgl2")?.getExtension("WEBGL_lose_context")?.loseContext(); // frees the old one's GPU memory now
}

let playing: Promise<void> = Promise.resolve();
let busy = false;
type PlayOpts = { speed?: number; onItem?: (g: string | null) => void };
/** Signs `glosses` in order. A request while one plays is dropped (taps on a playing card must not pile up). */
export const playSigns = (glosses: string[], opts: PlayOpts = {}) => playTokens(glosses.map((g) => ({ kind: "sign", gloss: g })), opts);

/** Signs and fingerspells `tokens` in order; dropped like playSigns while one plays. */
export function playTokens(tokens: SignToken[], opts: PlayOpts = {}): Promise<void> {
  if (busy) return playing;
  busy = true;
  const slot = avatarCanvas.closest<HTMLElement>(".avatar-slot"); // labelled slots (avatarSlot's `label`) show the current sign
  slot?.classList.add("playing");
  if (slot) slot.dataset.mode = (opts.speed ?? 1) < 1 ? "Slow motion" : "Normal speed"; // a header names it while it plays
  playing = (async () => {
    const s = await loadStage();
    const glosses = tokens.flatMap((t) => (t.kind === "sign" && t.gloss ? [t.gloss] : []));
    await s.play({ type: "sign_plan", tokens, grammar: "statement", tone: faceFor(glosses), english: "" }, (g) => {
      slot?.dispatchEvent(new CustomEvent("signitem", { detail: g }));
      opts.onItem?.(g);
    }, opts.speed ?? 1);
  })().finally(() => {
    busy = false;
    document.querySelectorAll(".avatar-slot.playing").forEach((e) => e.classList.remove("playing"));
  });
  return playing;
}
export const isPlaying = () => busy;

/** Where hand landmark 9 (the palm, under the middle finger) shows over the camera preview, which covers a W x H box
 *  (object-fit: cover) mirrored. Null for a hand in the lap: wrist past 70% of the way from the shoulders to the hips. */
export function cameraPalm(hand: P3[] | null, pose: number[][], W: number, H: number, vw: number, vh: number): Pt | null {
  if (!hand) return null;
  const shoulders = (pose[11][1] + pose[12][1]) / 2, hips = (pose[23][1] + pose[24][1]) / 2;
  if (hand[0][1] > shoulders + 0.7 * (hips - shoulders)) return null;
  const s = Math.max(W / vw, H / vh), dw = vw * s, dh = vh * s;
  return [W - ((W - dw) / 2 + hand[9][0] * dw), (H - dh) / 2 + hand[9][1] * dh];
}

/** Face scores from MediaPipe's blendshapes, as server/emotion.py reads a turn (FACE_KEYS): the weighted mean per
 *  feeling. Brows are left out there too: they carry grammar. */
const FACE_KEYS = {
  happy: { mouthSmileLeft: 1, mouthSmileRight: 1, cheekSquintLeft: 0.5, cheekSquintRight: 0.5 },
  sad: { mouthFrownLeft: 1, mouthFrownRight: 1, mouthPucker: 0.3 },
  angry: { noseSneerLeft: 1, noseSneerRight: 1, mouthPressLeft: 0.5, mouthPressRight: 0.5, eyeSquintLeft: 0.3, eyeSquintRight: 0.3 },
};
type FaceScores = Record<keyof typeof FACE_KEYS, number>;
export type Feeling = keyof typeof FACE_KEYS | "neutral";
const EMO_T = 0.15; // server/emotion.py: the least score that is a feeling, not a resting face

export function faceScores(blend: Record<string, number> | null): FaceScores {
  const out: FaceScores = { happy: 0, sad: 0, angry: 0 };
  for (const [k, w] of Object.entries(FACE_KEYS) as [keyof FaceScores, Record<string, number>][]) {
    const total = Object.values(w).reduce((a, b) => a + b, 0);
    out[k] = blend ? Object.entries(w).reduce((s, [b, x]) => s + (blend[b] ?? 0) * x, 0) / total : 0;
  }
  return out;
}

export function faceLabel(s: FaceScores): Feeling {
  const [k, v] = (Object.entries(s) as [Feeling, number][]).sort((a, b) => b[1] - a[1])[0];
  return v >= EMO_T ? k : "neutral";
}

export type CamStatus = "off" | "starting" | "position" | "calibrating" | "ready" | "signing" | "reading" | "error";
type Listener = { status(s: CamStatus, detail?: string): void; recognized(segments: PickOption[][]): void; face?(seen: Feeling): void;
  hands?(inFrame: boolean): void };

/** Webcam + backend in practice mode. Detection runs only while a perform exercise is on screen. */
class Practice {
  private sock: Socket | null = null;
  private started: Promise<void> | null = null;
  private server: CamStatus = "calibrating"; // the backend's last state; a new session starts calibrating
  status: CamStatus = "off";
  active = false;
  listener: Listener | null = null;
  readonly video = Object.assign(document.createElement("video"), { muted: true, playsInline: true, className: "cam-video" });
  /** The learner's own hand trail over the camera, drawn like the avatar's: a raised hand leaves it, lowering ends it. */
  readonly trailCanvas = Object.assign(document.createElement("canvas"), { className: "trail-canvas" });
  private trail: ReturnType<typeof attachTrail> | null = null;
  private palms: Record<"L" | "R", Pt | null> = { L: null, R: null };
  private faces: FaceScores = { happy: 0, sad: 0, angry: 0 };
  private watch = new PositionWatch();
  private hint: string | null = null; // position.ts's direction, shown instead of the status while there is one
  private redoing = false; // a redo was asked for: the dropped attempt's result may still arrive, and is ignored
  /** What the learner's face reads as now (smoothed); the listener hears when it changes. */
  seen: Feeling = "neutral";
  /** A hand in the hand tracker's sight: the segmenter's "raised" (server/segmenter.py), the camera card's green edge. */
  hands = false;

  private set(s: CamStatus, detail?: string) {
    this.status = s;
    this.listener?.status(s, detail);
  }

  start(): Promise<void> {
    this.socket(); // reconnects if the backend dropped (or another tab took the session)
    this.trail?.clear(); // a new camera card starts with no strokes
    return (this.started ??= this.open().catch((e) => { this.started = null; throw e; })) // a failed start is retried next time
      // every camera card plays it again: the cards in between take the one video out of the page, which pauses it
      .then(() => this.video.play());
  }

  /** The backend socket, opened on first use: lessons speak through it before (or without) the camera. */
  private socket(): Socket {
    if (!this.sock) {
      this.sock = connect((m) => {
        if (m.type === "state") this.onState(m.state);
        else if (m.type === "recognized" && !this.redoing) this.listener?.recognized(m.segments);
      }, (e) => {
        this.sock = null; // the next line said or camera exercise reconnects
        this.set("error", e.code === 4001 ? "Opened in another tab" : "Backend disconnected");
      });
      // the backend reads each attempt as one of the course's signs, not one of all ~2,900
      this.sock.send({ type: "practice", on: true, signs: [...new Set(ALL_LESSONS.flatMap((l) => l.signs))] });
      this.sock.send({ type: "buddy", buddy: savedBuddy().id }); // lines are said in the buddy's voice
    }
    return this.sock;
  }

  say(text: string) { this.socket().send({ type: "say", text }); }

  buddy(id: Buddy["id"]) { this.socket().send({ type: "buddy", buddy: id }); }

  /** The plushie shows a check and flaps its arms, an X, or a ! for "try again" (an Arduino on the backend's USB; nothing
   * happens without one). */
  answered(result: "right" | "wrong" | "retry") { this.socket().send({ type: "answer", result }); }

  private async open(): Promise<void> {
    this.set("starting");
    try {
      await startCamera(this.video, (f) => {
        const { hint, send, recalibrate } = this.watch.check(f, this.server, performance.now());
        if (hint !== this.hint) this.set(hint ? "position" : this.server, (this.hint = hint) ?? undefined);
        if (recalibrate) this.recalibrate(); // they moved since the last calibration
        // nothing while a sign is read: the backend ignores those frames anyway, and the feed shows frozen
        if (send && this.active && this.server !== "reading" && this.sock && this.sock.buffered() < 256 * 1024) this.sock.send({ type: "frame", ...f });
        if (!!(f.left || f.right) !== this.hands) this.listener?.hands?.((this.hands = !this.hands));
        this.drawHands(f);
        this.readFace(f);
      }, () => !this.active || busy);
      // the camera taken away (the phone slept, another app): open a fresh feed now if a camera card shows, else on the next one.
      // shortcut: a reopen that fails while the app is in the background waits for the next camera card; retry on visibilitychange if that bites
      (this.video.srcObject as MediaStream | null)?.getVideoTracks().forEach((t) => t.addEventListener("ended", () => {
        this.started = null;
        if (this.active) this.start().catch(() => {});
      }));
      this.set(this.server); // "starting" hid the backend's state (calibrating) until the next state change
    } catch (e) {
      this.set("error", (e as Error).name); // shown after "Camera unavailable: "
      throw e;
    }
  }

  private onState(s: State) {
    if (s === "calibrating" || s === "ready" || s === "signing" || s === "reading") {
      this.hint = null; // the next frame shows it again if it still applies
      if (s === "signing") this.redoing = false; // a new attempt: its result counts
      this.set((this.server = s));
      // the sign is being read: the feed freezes where it ended (nothing signed now counts), and moves again after
      if (s === "reading") this.video.pause();
      else if (this.active && this.started) this.video.play().catch(() => {});
    }
  }

  recalibrate() { this.sock?.send({ type: "calibrate" }); }

  /** The learner tapped the frozen camera: drop the attempt being read, unread, and sign again. */
  redo() {
    this.redoing = true;
    this.sock?.send({ type: "redo" });
  }

  private readFace(f: Frame) {
    const now = faceScores(f.blend);
    for (const k of Object.keys(now) as (keyof FaceScores)[]) this.faces[k] = 0.8 * this.faces[k] + 0.2 * now[k]; // ~half a second: a blink doesn't flip it
    const seen = faceLabel(this.faces);
    if (seen !== this.seen) this.listener?.face?.((this.seen = seen));
  }

  private drawHands(f: Frame) {
    const c = this.trailCanvas, v = this.video;
    if (!c.isConnected || !v.videoWidth) return;
    const at = (hand: P3[] | null) => cameraPalm(hand, f.pose, c.clientWidth, c.clientHeight, v.videoWidth, v.videoHeight);
    this.palms = { L: at(f.left), R: at(f.right) };
    // reduced motion: each stroke shows whole and still once the hands come down, as the avatar's does
    this.trail ??= attachTrail(c, () => this.palms, { still: matchMedia("(prefers-reduced-motion: reduce)").matches });
    this.trail.frame({ now: performance.now(), signing: !!(this.palms.L || this.palms.R), speed: 1 });
  }
}

export const practice = new Practice();
