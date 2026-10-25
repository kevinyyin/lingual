// A small round buddy that pops up and signs back what the camera read ("I saw: coffee"), after a lesson attempt and
// a translator turn. One avatar of its own, loaded on first use (again after a new buddy is picked) and moved to wherever
// it shows; a while after playing it fades away, and a tap plays it again.
import "./mini.css";
import { savedBuddy } from "../buddies";
import type { Feeling, SignToken, Tone } from "../ws";
import { loadAvatar } from "./index";
import { createStage, type Stage } from "./stage";

export const LINGER_MS = 2500; // shown this long after it finishes signing
export const FADE_MS = 300; // mini.css's fade
/** The signer's face as the buddy's while it signs back: a smile, a straight mouth or a frown (face.ts). */
const FACE_TONE: Record<Feeling, Tone> = { happy: "warm", neutral: "neutral", sad: "apologetic", angry: "apologetic" };

const box = Object.assign(document.createElement("button"), { type: "button", className: "mini-buddy" });
const label = Object.assign(document.createElement("span"), { className: "mini-caption" });
let canvas: HTMLCanvasElement | null = null;
let stage: Promise<Stage> | null = null;
let buddy = "";
let current: { tokens: SignToken[]; caption: string; tone: Tone } | null = null;
let playing = false;
let again = false; // asked to play while playing: once more when this one ends
let timer = 0;

function getStage(): Promise<Stage> {
  if (!stage || buddy !== savedBuddy().id) {
    canvas?.getContext("webgl2")?.getExtension("WEBGL_lose_context")?.loseContext(); // the old buddy's GPU memory
    canvas = Object.assign(document.createElement("canvas"), { className: "mini-canvas" });
    canvas.setAttribute("aria-hidden", "true");
    box.replaceChildren(canvas, label);
    buddy = savedBuddy().id;
    stage = loadAvatar(canvas).then((a) => createStage(a));
  }
  return stage;
}

async function play(): Promise<void> {
  if (playing) {
    again = true;
    return;
  }
  playing = true;
  try {
    do {
      again = false;
      const { tokens, caption, tone } = current!;
      await (await getStage()).play({ type: "sign_plan", tokens, grammar: "statement", tone, english: caption }, () => {});
    } while (again);
  } catch { /* the avatar couldn't load or play: the caption still says what was read */ }
  playing = false;
  timer = window.setTimeout(() => {
    box.classList.remove("on");
    timer = window.setTimeout(() => box.remove(), FADE_MS); // off the page: its avatar stops drawing
  }, LINGER_MS);
}

box.onclick = () => {
  clearTimeout(timer);
  box.classList.add("on");
  void play();
};

/** Pops the mini buddy up at the end of host and signs tokens under caption, with the signer's face. */
export function showMini(host: HTMLElement, tokens: SignToken[], caption: string, face: Feeling = "neutral"): void {
  if (!tokens.length) return;
  current = { tokens, caption, tone: FACE_TONE[face] };
  label.textContent = caption;
  box.setAttribute("aria-label", `Play again: ${caption}`);
  clearTimeout(timer);
  getStage();
  host.append(box);
  requestAnimationFrame(() => box.classList.add("on")); // after it is on the page, so it fades in
  void play();
}
