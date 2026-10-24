// Mirrors app/server/protocol.py.
import { isPhone, playPcm, startMic, unlockAudio } from "./phone";

export type P3 = [number, number, number];

export interface Frame {
  t: number;
  left: P3[] | null;
  right: P3[] | null;
  pose: [number, number, number, number][];
  blend: Record<string, number> | null;
  head: [number, number, number] | null;
  /** 6 face-mesh points [x, y] (FACE_POINTS in camera.ts) for the sign classifier. */
  face?: [number, number][] | null;
}

export type ClientMsg =
  | ({ type: "frame" } & Frame)
  | { type: "calibrate" }
  | { type: "pick"; segment: number; gloss: string }
  | { type: "avatar_done" }
  | { type: "practice"; on: boolean; signs?: string[] }
  | { type: "mode"; mode: Mode }
  | { type: "talk"; on: boolean }
  | { type: "redo" }
  | { type: "say"; text: string }
  | { type: "cheer" }
  | { type: "record"; signer: string; gloss: string; frames: Frame[] };

/** The translator's switch: voice (speech -> avatar; the camera is ignored) or signing (camera -> speech; speech ignored). */
export type Mode = "voice" | "signing";
/** reading: the signer's hands came down and the turn is being read (protocol.py). */
export type State = "calibrating" | "ready" | "signing" | "reading" | "picking" | "speaking" | "listening" | "avatar";
export type Grammar = "yn" | "wh" | "neg" | "statement";
export type Tone = "warm" | "apologetic" | "urgent" | "neutral";
export interface SignToken { kind: "sign" | "spell"; gloss?: string | null; word?: string | null }
/** `unclear`: the words made no sense, so nothing is signed and `english` is what was heard. */
export interface SignPlan { type: "sign_plan"; tokens: SignToken[]; grammar: Grammar; tone: Tone; english: string; unclear?: boolean }
export interface PickOption { gloss: string; english: string; p: number }

export type ServerMsg =
  | { type: "state"; state: State }
  | { type: "pick_request"; segment: number; options: PickOption[] }
  | { type: "you_said"; text: string; emotion: string; unclear?: boolean } // unclear: the signs made no sentence
  | { type: "caption"; text: string; final: boolean }
  | SignPlan
  | { type: "recognized"; segments: PickOption[][] }
  | { type: "recorded"; ok: boolean; detail: string };

export interface Socket {
  send(m: ClientMsg): void;
  /** Bytes queued but not yet sent; camera drops frames when this grows. */
  buffered(): number;
}

/** On the phone (iOS app) the server's voice plays here, and with `mic` the phone mic feeds speech-to-text. */
export function connect(onMsg: (m: ServerMsg) => void, onClose?: (e: CloseEvent) => void, opts: { mic?: boolean } = {}): Socket {
  const phone = isPhone();
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws${phone ? "?audio=client" : ""}`);
  const queue: string[] = [];
  ws.onopen = () => queue.splice(0).forEach((s) => ws.send(s));
  if (phone) {
    ws.binaryType = "arraybuffer";
    document.addEventListener("pointerdown", unlockAudio);
    if (opts.mic) startMic((b) => { if (ws.readyState === WebSocket.OPEN) ws.send(b); }).catch((e) => console.warn("mic unavailable", e));
  }
  ws.onmessage = (e) => (typeof e.data === "string" ? onMsg(JSON.parse(e.data)) : playPcm(e.data));
  ws.onclose = (e) => onClose?.(e);
  return {
    send(m) {
      const s = JSON.stringify(m);
      if (ws.readyState === WebSocket.OPEN) ws.send(s);
      else if (m.type !== "frame") queue.push(s); // never queue stale frames
    },
    buffered: () => ws.bufferedAmount,
  };
}
