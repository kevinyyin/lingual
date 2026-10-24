// @vitest-environment jsdom
import { beforeEach, describe, expect, test } from "vitest";
import { getAllByRole, getByRole } from "@testing-library/dom";
import { bindControls, highlight, renderMsg } from "./ui";
import type { ClientMsg } from "./ws";

let sent: ClientMsg[];

beforeEach(() => {
  document.body.innerHTML = `
    <p id="caption"></p><p id="plan"></p><span id="state"></span>
    <div id="picker" role="group" aria-label="Which sign did you mean?" hidden></div><p id="you-said"></p>
    <form id="dev" hidden><input id="dev-text" /></form><button id="talk" type="button">Hold to talk</button>`;
  sent = [];
  bindControls({ send: (m) => sent.push(m), buffered: () => 0 });
  sent.length = 0; // the mode it starts in (tested below)
});

const pick = {
  type: "pick_request" as const,
  segment: 1,
  options: [
    { gloss: "COFFEE", english: "coffee", p: 0.4 },
    { gloss: "TEA", english: "tea", p: 0.3 },
    { gloss: "MILK", english: "milk", p: 0.1 },
  ],
};

describe("picker", () => {
  test("pick_request renders three options", () => {
    renderMsg(pick);
    const picker = document.getElementById("picker")!;
    expect(picker.hidden).toBe(false);
    expect(getAllByRole(picker, "button", { name: /^\d\./ })).toHaveLength(3);
  });

  test("key 2 sends pick for option 2 and hides the picker", () => {
    renderMsg(pick);
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "2" }));
    expect(sent).toEqual([{ type: "pick", segment: 1, gloss: "TEA" }]);
    expect(document.getElementById("picker")!.hidden).toBe(true);
  });

  test("focus moves to the first option so the signer can answer at once", () => {
    renderMsg(pick);
    const f = document.activeElement as HTMLElement;
    expect(f.tagName).toBe("BUTTON");
    expect(f.closest("#picker")).not.toBeNull();
    expect(f.textContent).toMatch(/coffee/);
  });

  test("sign codes like (COFFEE) show only in dev mode", () => {
    history.replaceState(null, "", "/");
    renderMsg(pick);
    expect(document.getElementById("picker")!.textContent).not.toContain("COFFEE");
    history.replaceState(null, "", "/?dev=1");
    renderMsg(pick);
    expect(document.getElementById("picker")!.textContent).toContain("COFFEE");
    history.replaceState(null, "", "/");
  });

  test("click sends pick; keys do nothing once answered", () => {
    renderMsg(pick);
    getByRole(document.body, "button", { name: /coffee/i }).click();
    document.dispatchEvent(new KeyboardEvent("keydown", { key: "3" }));
    expect(sent).toEqual([{ type: "pick", segment: 1, gloss: "COFFEE" }]);
  });
});

test("you_said replaces previous text and shows emotion", () => {
  renderMsg({ type: "you_said", text: "Hi.", emotion: "neutral" });
  renderMsg({ type: "you_said", text: "Do you have oat milk?", emotion: "happy" });
  const el = document.getElementById("you-said")!;
  expect(el.textContent).toContain("Do you have oat milk?");
  expect(el.textContent).toContain("happy");
  expect(el.textContent).not.toContain("Hi.");
});

test("leaving the picking state hides a stale picker", () => {
  renderMsg(pick);
  renderMsg({ type: "state", state: "ready" });
  expect(document.getElementById("picker")!.hidden).toBe(true);
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "1" }));
  expect(sent).toEqual([]);
});

test("none of these sends an empty pick (button or Escape)", () => {
  renderMsg(pick);
  getByRole(document.body, "button", { name: /none of these/i }).click();
  renderMsg(pick);
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
  expect(sent).toEqual([{ type: "pick", segment: 1, gloss: "" }, { type: "pick", segment: 1, gloss: "" }]);
});

test("sign_plan renders caption and one span per sign or letter", () => {
  renderMsg({ type: "sign_plan", tokens: [{ kind: "sign", gloss: "COFFEE" }, { kind: "spell", word: "ANA" }], grammar: "yn", tone: "warm", english: "Coffee, Ana?" });
  expect(document.getElementById("caption")!.textContent).toBe("Coffee, Ana?");
  const spans = [...document.querySelectorAll<HTMLElement>("#plan span")];
  expect(spans.map((s) => s.dataset.g)).toEqual(["COFFEE", "A", "N", "A"]);
});

test("highlight lights only the tile being signed, in plan order", () => {
  renderMsg({ type: "sign_plan", tokens: [{ kind: "spell", word: "ANA" }, { kind: "sign", gloss: "A" }], grammar: "statement", tone: "neutral", english: "Ana" });
  const lit = () => [...document.querySelectorAll("#plan span")].map((s) => (s.classList.contains("on") ? 1 : 0)).join("");
  // the avatar reports each item, with null for the blends between them
  const seen = ["A", null, "N", null, "A", null, "A", null].map((g) => (highlight(g), lit()));
  expect(seen).toEqual(["1000", "0000", "0100", "0000", "0010", "0000", "0001", "0000"]);
  highlight("A"); // a replay of the same plan starts over
  expect(lit()).toBe("1000");
});

describe("voice / signing switch", () => {
  beforeEach(() => {
    localStorage.clear();
    document.body.insertAdjacentHTML("beforeend", `<div id="mode"><button type="button" data-mode="voice">Voice</button>
      <button type="button" data-mode="signing">Signing</button></div>`);
    sent = [];
    bindControls({ send: (m) => sent.push(m), buffered: () => 0 });
  });

  test("starts in voice mode and tells the server", () => {
    expect(sent).toContainEqual({ type: "mode", mode: "voice" });
    expect(getByRole(document.body, "button", { name: "Voice" }).getAttribute("aria-pressed")).toBe("true");
    expect(document.body.dataset.mode).toBe("voice");
  });

  test("switching tells the server, is remembered, and changes what ready asks for", () => {
    renderMsg({ type: "state", state: "ready" });
    expect(document.getElementById("state")!.textContent).toBe("Ready: hold to talk");
    getByRole(document.body, "button", { name: "Signing" }).click();
    expect(sent).toContainEqual({ type: "mode", mode: "signing" });
    expect(document.getElementById("state")!.textContent).toBe("Ready: sign");
    expect(localStorage.getItem("lingual.mode")).toBe("signing");
  });

  test("push to talk: speech is heard only while Space or the button is held, in voice mode", () => {
    const talk = () => sent.filter((m) => m.type === "talk");
    const key = (type: string, init: KeyboardEventInit = {}, target: EventTarget = document.body) => {
      const e = new KeyboardEvent(type, { code: "Space", key: " ", bubbles: true, cancelable: true, ...init });
      target.dispatchEvent(e);
      return e;
    };
    expect(key("keydown").defaultPrevented).toBe(true); // no button click, no page scroll
    key("keydown", { repeat: true }); // the key's auto-repeat while held
    key("keyup");
    expect(talk()).toEqual([{ type: "talk", on: true }, { type: "talk", on: false }]);
    const button = getByRole(document.body, "button", { name: /Hold to talk/ });
    button.dispatchEvent(new Event("pointerdown"));
    expect(button.getAttribute("aria-pressed")).toBe("true");
    button.dispatchEvent(new Event("pointerup"));
    expect(talk()).toHaveLength(4);
    key("keydown", {}, document.getElementById("dev-text")!); // typing a space in the dev box
    expect(talk()).toHaveLength(4);
    getByRole(document.body, "button", { name: "Signing" }).click();
    expect(key("keydown").defaultPrevented).toBe(false); // signing mode: Space still presses a focused button
    expect(talk()).toHaveLength(4);
  });

  test("switching to signing mid-talk lets go", () => {
    document.dispatchEvent(new KeyboardEvent("keydown", { code: "Space", key: " " }));
    getByRole(document.body, "button", { name: "Signing" }).click();
    expect(sent.filter((m) => m.type === "talk").at(-1)).toEqual({ type: "talk", on: false });
  });

  test("voice mode never asks to calibrate the camera", () => {
    renderMsg({ type: "state", state: "calibrating" });
    expect(document.getElementById("state")!.textContent).toBe("Ready: hold to talk");
    expect(document.getElementById("state")!.dataset.state).toBe("ready");
  });
});

describe("gibberish", () => {
  test("words that made no sense show what was heard and say so, with nothing to sign", () => {
    renderMsg({ type: "sign_plan", tokens: [], grammar: "statement", tone: "neutral", english: "blorp fizzle wug", unclear: true });
    expect(document.getElementById("caption")!.textContent).toBe("blorp fizzle wug");
    expect(document.getElementById("plan")!.textContent).toMatch(/didn't make sense/);
    expect(document.querySelectorAll("#plan span[data-g]")).toHaveLength(0);
  });

  test("signs that made no sentence say so instead of an empty line", () => {
    renderMsg({ type: "you_said", text: "", emotion: "neutral", unclear: true });
    expect(document.getElementById("you-said")!.textContent).toMatch(/didn't make a sentence/);
  });
});

test("the translator says when the signer's turn is being read", () => {
  renderMsg({ type: "state", state: "signing" });
  expect(document.getElementById("state")!.textContent).toBe("Watching your signs…");
  renderMsg({ type: "state", state: "reading" });
  expect(document.getElementById("state")!.textContent).toBe("Reading signs…");
  expect(document.getElementById("state")!.dataset.state).toBe("reading");
});
