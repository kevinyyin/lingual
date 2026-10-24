// Teaching mode's links to the rest of the app: one avatar canvas that lessons borrow, and the camera +
// backend socket in practice mode, which grades the learner's own signing.
import { loadAvatar } from "../avatar";
import { createStage, type Stage } from "../avatar/stage";
import { attachTrail } from "../avatar/trail";
import { startCamera } from "../camera";
import { connect, type PickOption, type SignToken, type Socket, type State } from "../ws";
import { faceFor } from "./curriculum";

/** One canvas for the whole page (the avatar is slow to load): each exercise moves it into its own slot. */
export const avatarCanvas = Object.assign(document.createElement("canvas"), { className: "avatar-canvas" });
avatarCanvas.setAttribute("aria-label", "Signing avatar");
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

let playing: Promise<void> = Promise.resolve();
let busy = false;
type PlayOpts = { speed?: number; onItem?: (g: string | null) => void };
/** Signs `glosses` in order. A request while one plays is dropped (taps on a playing card must not pile up). */
export const playSigns = (glosses: string[], opts: PlayOpts = {}) => playTokens(glosses.map((g) => ({ kind: "sign", gloss: g })), opts);

/** Signs and fingerspells `tokens` in order; dropped like playSigns while one plays. */
export function playTokens(tokens: SignToken[], opts: PlayOpts = {}): Promise<void> {
  if (busy) return playing;
  busy = true;
  avatarCanvas.closest(".avatar-slot")?.classList.add("playing");
  playing = (async () => {
    const s = await loadStage();
    const glosses = tokens.flatMap((t) => (t.kind === "sign" && t.gloss ? [t.gloss] : []));
    const slot = avatarCanvas.closest(".avatar-slot"); // labelled slots (avatarSlot's `label`) show the current sign
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

export type CamStatus = "off" | "starting" | "calibrating" | "ready" | "signing" | "error";
type Listener = { status(s: CamStatus, detail?: string): void; recognized(segments: PickOption[][]): void };

/** Webcam + backend in practice mode. Detection runs only while a perform exercise is on screen. */
class Practice {
  private sock: Socket | null = null;
  private started: Promise<void> | null = null;
  status: CamStatus = "off";
  active = false;
  listener: Listener | null = null;
  readonly video = Object.assign(document.createElement("video"), { muted: true, playsInline: true, className: "cam-video" });

  private set(s: CamStatus, detail?: string) {
    this.status = s;
    this.listener?.status(s, detail);
  }

  start(): Promise<void> {
    this.socket(); // reconnects if the backend dropped (or another tab took the session)
    return (this.started ??= this.open().catch((e) => { this.started = null; throw e; })); // a failed start is retried next time
  }

  /** The backend socket, opened on first use: lessons speak through it before (or without) the camera. */
  private socket(): Socket {
    if (!this.sock) {
      this.sock = connect((m) => {
        if (m.type === "state") this.onState(m.state);
        else if (m.type === "recognized") this.listener?.recognized(m.segments);
      }, (e) => {
        this.sock = null; // the next line said or camera exercise reconnects
        this.set("error", e.code === 4001 ? "Opened in another tab" : "Backend disconnected");
      });
      this.sock.send({ type: "practice", on: true });
    }
    return this.sock;
  }

  say(text: string) { this.socket().send({ type: "say", text }); }

  private async open(): Promise<void> {
    this.set("starting");
    try {
      await startCamera(this.video, (f) => {
        if (this.active && this.sock && this.sock.buffered() < 256 * 1024) this.sock.send({ type: "frame", ...f });
      }, () => !this.active || busy);
    } catch (e) {
      this.set("error", (e as Error).name); // shown after "Camera unavailable: "
      throw e;
    }
  }

  private onState(s: State) {
    if (s === "calibrating") this.set("calibrating");
    else if (s === "ready") this.set("ready");
    else if (s === "signing") this.set("signing");
  }

  recalibrate() { this.sock?.send({ type: "calibrate" }); }
}

export const practice = new Practice();
