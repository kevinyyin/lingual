// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { bindControls, renderMsg } from "./ui";
import type { ClientMsg } from "./ws";

let sent: ClientMsg[];

const page = `
  <p id="caption"></p><p id="plan"></p><span id="state"></span>
  <div id="mode"><button data-mode="voice">Voice</button><button data-mode="signing">Signing</button></div>
  <div id="picker" role="group" hidden></div><p id="you-said"></p>
  <section id="words-card" hidden><p id="words" aria-live="polite"></p>
    <div class="words-actions"><button id="words-undo" type="button">Undo</button><button id="words-done" type="button">Say it</button></div></section>
  <button id="bysign-toggle" type="button" aria-pressed="false">Sign by sign</button>
  <form id="dev" hidden><input id="dev-text" /></form><button id="talk" type="button">Hold to talk</button>`;

function start(url: string, m: "voice" | "signing" = "signing"): void {
  history.replaceState(null, "", url);
  localStorage.setItem("lingual.mode", m);
  document.body.innerHTML = page;
  const mine: ClientMsg[] = (sent = []); // listeners from earlier tests still fire, into their own arrays
  bindControls({ send: (msg) => mine.push(msg), buffered: () => 0 });
}
const key = (k: string, init: KeyboardEventInit = {}, target: EventTarget = document.body) =>
  target.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true, ...init }));
const words = () => sent.filter((m) => m.type === "words");

beforeEach(() => localStorage.clear());
afterEach(() => { history.replaceState(null, "", "/"); localStorage.clear(); });

describe("sign by sign", () => {
  test("words_update renders a chip per word: sign glosses and fingerspelled words", () => {
    start("/?bysign=1");
    renderMsg({ type: "words_update", words: [{ kind: "sign", gloss: "COFFEE" }, { kind: "spell", word: "sam" }] });
    const chips = [...document.querySelectorAll("#words span")];
    expect(chips.map((c) => c.textContent)).toEqual(["COFFEE", "sam"]);
    expect(chips[0].className).toBe("sign");
    expect(chips[1].classList.contains("spelled")).toBe(true);
    renderMsg({ type: "words_update", words: [] });
    expect(document.getElementById("words")!.children).toHaveLength(0);
  });

  test("the mode message carries by_sign only with the flag; the card shows only then, in signing mode", () => {
    start("/");
    expect(sent).toContainEqual({ type: "mode", mode: "signing" });
    expect(sent.some((m) => m.type === "mode" && "by_sign" in m)).toBe(false);
    expect(document.getElementById("words-card")!.hidden).toBe(true);

    start("/?bysign=1");
    expect(sent).toContainEqual({ type: "mode", mode: "signing", by_sign: true });
    expect(document.getElementById("words-card")!.hidden).toBe(false);
    (document.querySelector('[data-mode="voice"]') as HTMLButtonElement).click();
    expect(document.getElementById("words-card")!.hidden).toBe(true);
  });

  test("the flag is remembered without the URL (the iOS app can't set it); ?bysign=0 and the toggle turn it off", () => {
    start("/?bysign=1");
    start("/");
    expect(sent).toContainEqual({ type: "mode", mode: "signing", by_sign: true });
    document.getElementById("bysign-toggle")!.click();
    expect(sent.at(-1)).toEqual({ type: "mode", mode: "signing" });
    expect(document.getElementById("words-card")!.hidden).toBe(true);
    start("/");
    expect(document.getElementById("words-card")!.hidden).toBe(true);
    document.getElementById("bysign-toggle")!.click();
    expect(sent.at(-1)).toEqual({ type: "mode", mode: "signing", by_sign: true });
    expect(document.getElementById("bysign-toggle")!.getAttribute("aria-pressed")).toBe("true");
    start("/?bysign=0");
    expect(document.getElementById("words-card")!.hidden).toBe(true);
  });

  test("Enter says it, Backspace undoes, buttons send the same", () => {
    start("/?bysign=1");
    key("Enter");
    key("Backspace");
    document.getElementById("words-undo")!.click();
    document.getElementById("words-done")!.click();
    expect(words()).toEqual([
      { type: "words", action: "done" }, { type: "words", action: "undo" },
      { type: "words", action: "undo" }, { type: "words", action: "done" },
    ]);
  });

  test("keys do nothing without the flag, in voice mode, on key repeat, or while typing", () => {
    start("/");
    key("Enter"); key("Backspace");
    start("/?bysign=1", "voice");
    key("Enter"); key("Backspace");
    start("/?bysign=1");
    key("Backspace", { repeat: true });
    document.body.insertAdjacentHTML("beforeend", "<input id='t'><textarea id='ta'></textarea>");
    key("Backspace", {}, document.getElementById("t")!);
    key("Enter", {}, document.getElementById("ta")!);
    key("Enter", {}, document.getElementById("words-undo")!); // a focused button keeps Enter: it clicks itself
    expect(words()).toEqual([]);
  });

  test("the picker keeps its keys: 1-3 pick, Escape drops, Enter/Backspace don't send words", () => {
    start("/?bysign=1");
    const pick = { type: "pick_request" as const, segment: 2, options: [{ gloss: "TEA", english: "tea", p: 0.5 }, { gloss: "MILK", english: "milk", p: 0.2 }] };
    renderMsg(pick);
    key("Enter"); key("Backspace");
    key("2");
    expect(sent.at(-1)).toEqual({ type: "pick", segment: 2, gloss: "MILK" });
    renderMsg(pick);
    key("Escape");
    expect(document.getElementById("picker")!.hidden).toBe(true);
    expect(words()).toEqual([]);
  });
});
