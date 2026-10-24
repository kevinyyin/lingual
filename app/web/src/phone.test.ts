import { expect, test } from "vitest";
import { isPhone, rememberPhone, setAudioSession, toPcm16 } from "./phone";

test("phone flag is remembered from ?phone=1", () => {
  localStorage.clear();
  history.replaceState(null, "", "/teach.html");
  rememberPhone();
  expect(isPhone()).toBe(false);
  history.replaceState(null, "", "/teach.html?phone=1");
  rememberPhone();
  history.replaceState(null, "", "/");
  expect(isPhone()).toBe(true);
});

test("toPcm16 clamps and scales", () => {
  expect(Array.from(toPcm16(new Float32Array([0, 1, -1, 2, -2, 0.5])))).toEqual([0, 32767, -32767, 32767, -32767, 16384]);
});

test("audio session: playback ignores the silent switch, and never drops the mic once it is on", () => {
  const session = { type: "auto" };
  Object.defineProperty(navigator, "audioSession", { value: session, configurable: true });
  setAudioSession("playback");
  expect(session.type).toBe("playback");
  setAudioSession("play-and-record");
  setAudioSession("playback");
  expect(session.type).toBe("play-and-record");
});

test("?phone=0 turns the phone flag back off", () => {
  localStorage.clear();
  history.replaceState(null, "", "/teach.html?phone=1");
  rememberPhone();
  history.replaceState(null, "", "/teach.html?phone=0");
  rememberPhone();
  expect(isPhone()).toBe(false);
});
