// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from "vitest";
import { getByRole } from "@testing-library/dom";
import type { ClientMsg, Frame, ServerMsg } from "../ws";
import type { LessonHost } from "./lessonView";

// The backend socket, camera and avatar are external (server, MediaPipe, WebGL): the page's state comes from the socket.
const sock = vi.hoisted(() => ({ sent: [] as ClientMsg[], toPage: (_m: ServerMsg) => {} }));
vi.mock("../ws", () => ({
  connect: (onMsg: (m: ServerMsg) => void) => {
    sock.toPage = onMsg;
    return { send: (m: ClientMsg) => sock.sent.push(m), buffered: () => 0 };
  },
}));
const cam = vi.hoisted(() => ({ onFrame: null as ((f: Frame) => void) | null }));
vi.mock("../camera", () => ({ startCamera: async (_video: unknown, onFrame: (f: Frame) => void) => { cam.onFrame = onFrame; } }));
const play = vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined); // jsdom has no media playback
vi.mock("../avatar", () => ({ loadAvatar: () => new Promise(() => {}) }));
vi.mock("../avatar/stage", () => ({ createStage: vi.fn() }));

const said: string[] = [];
const host: LessonHost = {
  english: (g) => g.toLowerCase(), sound: () => true, say: (t) => said.push(t),
  cameraOn: () => true, disableCamera() {}, quit() {}, finish: () => { throw new Error("no lesson ends here"); },
};

beforeEach(async () => {
  vi.resetModules(); // a fresh practice camera (a module singleton) per test
  sock.sent = [];
  said.length = 0;
  const { runLesson } = await import("./lessonView");
  runLesson(document.body, [{ kind: "perform", gloss: "HELLO" }], host);
});

test("a camera exercise asks for the sign only once calibration is done", () => {
  const yourTurn = () => said.filter((t) => t.startsWith("Your turn"));
  sock.toPage({ type: "state", state: "calibrating" });
  expect(yourTurn()).toEqual([]);
  sock.toPage({ type: "state", state: "ready" });
  sock.toPage({ type: "state", state: "signing" });
  sock.toPage({ type: "state", state: "ready" });
  expect(yourTurn()).toEqual(["Your turn! Sign hello."]);
});

test("the backend is told the course's signs", () => {
  const msg = sock.sent.find((m) => m.type === "practice");
  expect(msg).toMatchObject({ type: "practice", on: true });
  expect(msg && "signs" in msg && msg.signs).toEqual(expect.arrayContaining(["HELLO", "GOODBYE", "THANK-YOU", "I"]));
});

/** A seated learner seen by the pose tracker: shoulders `width` apart, hands resting below the picture. */
const sitter = (width = 0.4): Frame => {
  const pose = Array.from({ length: 33 }, () => [0.5, 1.05, 0, 1] as [number, number, number, number]);
  pose[0] = [0.5, 0.3, 0, 1];
  pose[11] = [0.5 + width / 2, 0.55, 0, 1];
  pose[12] = [0.5 - width / 2, 0.55, 0, 1];
  return { t: 0, left: null, right: null, pose, blend: null, head: null };
};

test("the camera directs the learner into position, and calibration learns only from frames where they are in it", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull());
  sock.toPage({ type: "state", state: "calibrating" });
  const frames = () => sock.sent.filter((m) => m.type === "frame").length;
  cam.onFrame!(sitter(0.7));
  expect(document.querySelector(".cam-status")!.textContent).toBe("Get into position: Move back a little");
  expect(frames()).toBe(0);
  cam.onFrame!(sitter());
  expect(document.querySelector(".cam-status")!.textContent).toBe("Rest your hands in your lap, neutral face…");
  expect(frames()).toBe(1);
});

test("every camera card plays the camera again: a video taken out of the page pauses, and the second sign froze", async () => {
  const { runLesson } = await import("./lessonView");
  runLesson(document.body, [{ kind: "perform", gloss: "HELLO" }, { kind: "intro", gloss: "NICE" }, { kind: "perform", gloss: "GOODBYE" }], host);
  await vi.waitFor(() => expect(getByRole(document.body, "heading", { name: "Sign “hello”" })).toBeTruthy());
  play.mockClear();
  sock.toPage({ type: "recognized", segments: [[{ gloss: "HELLO", english: "hello", p: 0.9 }]] });
  getByRole(document.body, "button", { name: /Continue/ }).click();
  getByRole(document.body, "button", { name: /Next/ }).click(); // the intro card in between took the video off the page
  getByRole(document.body, "heading", { name: "Sign “goodbye”" });
  await vi.waitFor(() => expect(play).toHaveBeenCalled());
  expect(document.querySelector("video")!.isConnected).toBe(true);
});

test("the avatar names its speed while it plays, and the replay buttons say which is which", async () => {
  const { runLesson } = await import("./lessonView");
  runLesson(document.body, [{ kind: "intro", gloss: "HELLO" }], host);
  const slot = document.querySelector<HTMLElement>(".avatar-slot")!;
  expect(slot.classList.contains("playing")).toBe(true); // an intro card plays its sign on show
  expect(slot.dataset.mode).toBe("Normal speed");
  expect(getByRole(document.body, "button", { name: "Play in slow motion" }).title).toBe("Play in slow motion");
  expect(getByRole(document.body, "button", { name: "Play at normal speed" }).title).toBe("Play at normal speed");
});

test("the emotion line says what the face shows, and for a feeling sign whether it fits the sign's own face", async () => {
  const { faceLine } = await import("./lessonView");
  expect(faceLine("happy", "HELLO", "hello")).toBe("Your face: happy");
  expect(faceLine("happy", "HAPPY", "happy")).toBe("Your face: happy. Just right for “happy”!");
  expect(faceLine("happy", "SAD", "sad")).toBe("Your face: happy. “sad” needs a sad face!");
  expect(faceLine("neutral", "ANGRY", "angry")).toBe("Your face: neutral. “angry” needs an angry face!");
});
