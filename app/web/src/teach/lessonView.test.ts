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
  cameraOn: () => true, disableCamera() {}, quit() {}, save() {}, finish: () => { throw new Error("no lesson ends here"); },
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

test("a right answer tells the plushie (check and flapping arms)", async () => {
  await vi.waitFor(() => expect(getByRole(document.body, "heading", { name: "Sign “hello”" })).toBeTruthy());
  sock.toPage({ type: "recognized", segments: [[{ gloss: "HELLO", english: "hello", p: 0.9 }]] });
  getByRole(document.body, "button", { name: /Continue/ });
  expect(sock.sent.filter((m) => m.type === "answer")).toEqual([{ type: "answer", result: "right" }]);
});

test("a first missed sign makes the plushie say “try again”, not wrong", async () => {
  await vi.waitFor(() => expect(getByRole(document.body, "heading", { name: "Sign “hello”" })).toBeTruthy());
  sock.toPage({ type: "recognized", segments: [[{ gloss: "NICE", english: "nice", p: 0.9 }]] });
  expect(sock.sent.filter((m) => m.type === "answer")).toEqual([{ type: "answer", result: "retry" }]);
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

test("while a sign is read the camera freezes and says so, and it moves again after", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull()); // the camera is up
  const video = document.querySelector("video")!;
  const pause = vi.spyOn(video, "pause").mockImplementation(() => {});
  sock.toPage({ type: "state", state: "signing" });
  expect(document.querySelector(".cam-status")!.textContent).toMatch(/hands down/i); // how a sign ends
  play.mockClear();
  sock.toPage({ type: "state", state: "reading" });
  expect(pause).toHaveBeenCalled();
  expect(document.querySelector(".cam-status")!.textContent).toMatch(/^Reading your sign…/);
  sock.toPage({ type: "state", state: "ready" });
  expect(play).toHaveBeenCalled();
});

test("tapping the frozen camera while a sign is read signs it again, and that attempt's result is ignored", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull());
  expect(document.querySelector<HTMLButtonElement>(".cam-redo")!.hidden).toBe(true); // only while a sign is read
  sock.toPage({ type: "state", state: "reading" });
  expect(document.querySelector(".cam-status")!.textContent).toMatch(/tap to redo/i);
  getByRole(document.body, "button", { name: "Redo the sign" }).click();
  expect(sock.sent).toContainEqual({ type: "redo" });
  sock.toPage({ type: "recognized", segments: [[{ gloss: "HELLO", english: "hello", p: 0.9 }]] }); // already on its way
  sock.toPage({ type: "state", state: "ready" });
  expect(document.querySelector(".verdict")).toBeNull(); // not graded: they sign again
  sock.toPage({ type: "state", state: "signing" });
  sock.toPage({ type: "recognized", segments: [[{ gloss: "HELLO", english: "hello", p: 0.9 }]] }); // the new attempt
  expect(document.querySelector(".verdict")).not.toBeNull();
});

const miss = { type: "recognized" as const, segments: [[{ gloss: "NICE", english: "nice", p: 0.9 }]] };

test("Redo after a verdict, right or wrong, shows the same exercise again with nothing counted", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull());
  for (const result of [miss, miss]) sock.toPage(result); // two misses grade it wrong
  expect(document.querySelector(".verdict")!.textContent).toContain("Not quite.");
  getByRole(document.body, "button", { name: /Redo/ }).click();
  expect(document.querySelector(".verdict")).toBeNull();
  getByRole(document.body, "heading", { name: "Sign “hello”" });
  sock.toPage({ type: "recognized", segments: [[{ gloss: "HELLO", english: "hello", p: 0.9 }]] }); // a fresh first try
  expect(document.querySelector(".verdict")!.textContent).not.toContain("Not quite.");
  getByRole(document.body, "button", { name: /Redo/ }); // after a right answer too
});

test("a pass shows no match bar; a miss still shows how close it came", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull());
  sock.toPage(miss);
  expect(document.querySelector(".heard .score")).not.toBeNull();
  sock.toPage({ type: "recognized", segments: [[{ gloss: "NICE", english: "nice", p: 0.9 }, { gloss: "MEET", english: "meet", p: 0.05 },
    { gloss: "HELLO", english: "hello", p: 0.009 }]] }); // third guess at 1% of the top one: a pass by the top-3 rule
  expect(document.querySelector(".verdict .score")).toBeNull();
});

test("starting a new attempt clears the last one's \"that looked like...\"", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull());
  sock.toPage(miss);
  expect(document.querySelector(".heard")!.textContent).toMatch(/looked like “nice”/);
  sock.toPage({ type: "state", state: "signing" });
  expect(document.querySelector(".heard")!.textContent).toBe("");
});

test("the camera's edge is green while a hand is in the picture, red without one or while a sign is read; no frames go then", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull());
  sock.toPage({ type: "state", state: "ready" });
  const slot = document.querySelector<HTMLElement>(".cam-slot")!;
  const hand = Array.from({ length: 21 }, () => [0.5, 0.5, 0] as [number, number, number]);
  cam.onFrame!({ ...sitter(), right: hand });
  expect(slot.dataset.hands).toBe("in");
  cam.onFrame!(sitter());
  expect(slot.dataset.hands).toBe("out");
  sock.toPage({ type: "state", state: "reading" });
  const frames = () => sock.sent.filter((m) => m.type === "frame").length;
  const before = frames();
  cam.onFrame!(sitter());
  expect(frames()).toBe(before);
});

test("Back returns to the exercise before, and each step is saved for picking up later", async () => {
  const saves: unknown[] = [];
  const { runLesson } = await import("./lessonView");
  runLesson(document.body, [{ kind: "intro", gloss: "NICE" }, { kind: "intro", gloss: "MEET" }], { ...host, save: (r) => saves.push(r) });
  const back = getByRole(document.body, "button", { name: "Previous exercise" }) as HTMLButtonElement;
  expect(back.disabled).toBe(true);
  getByRole(document.body, "button", { name: /Next/ }).click();
  expect(document.querySelector(".intro-word small")!.textContent).toBe("MEET");
  back.click();
  expect(document.querySelector(".intro-word small")!.textContent).toBe("NICE");
  expect(saves).toHaveLength(3);
  expect(saves.at(-1)).toMatchObject({ done: 0, queue: [{ ex: { gloss: "NICE" } }, { ex: { gloss: "MEET" } }] });
});

test("lessons show no mini buddy after an attempt (it read as odd there; the translator keeps it)", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull());
  sock.toPage(miss);
  sock.toPage({ type: "recognized", segments: [[{ gloss: "HELLO", english: "hello", p: 0.9 }]] });
  expect(document.querySelector(".mini-buddy")).toBeNull();
});

test("a right answer turns the camera box green (it stayed the reading red)", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull());
  sock.toPage({ type: "state", state: "reading" });
  sock.toPage({ type: "recognized", segments: [[{ gloss: "HELLO", english: "hello", p: 0.9 }]] });
  expect(document.querySelector<HTMLElement>(".cam-slot")!.dataset.result).toBe("right");
});

test("two misses mark the camera box wrong", async () => {
  await vi.waitFor(() => expect(cam.onFrame).not.toBeNull());
  for (const m of [miss, miss]) sock.toPage(m);
  expect(document.querySelector<HTMLElement>(".cam-slot")!.dataset.result).toBe("wrong");
});
