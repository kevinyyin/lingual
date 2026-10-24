import type { PickOption, ServerMsg, Socket, State } from "./ws";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** What each state means for the people at the table (the badge is a role="status" live region). */
const STATE_LABEL: Record<State, string> = {
  calibrating: "Calibrating: rest your hands, neutral face",
  ready: "Ready: sign or speak",
  signing: "Reading signs…",
  picking: "Pick the sign you meant",
  speaking: "Speaking aloud…",
  listening: "Listening…",
  avatar: "Signing…",
};

let cameraOff = false;
/** ?camera=0: the backend still waits for calibration frames, but speech -> sign works, so don't ask for hands. */
export function setCameraOff(): void {
  cameraOff = true;
}

const el = <K extends keyof HTMLElementTagNameMap>(tag: K, className: string, text: string) =>
  Object.assign(document.createElement(tag), { className, textContent: text });
let socket: Socket | null = null;
let pending: { segment: number; options: PickOption[] } | null = null;

/** i = option index, or -1 for "none of these" (the turn is dropped and nothing is spoken). */
function choose(i: number): void {
  if (!pending || (i >= 0 && !pending.options[i])) return;
  socket?.send({ type: "pick", segment: pending.segment, gloss: i >= 0 ? pending.options[i].gloss : "" });
  pending = null;
  $("picker").hidden = true;
}

/** Top-3 picker: big buttons, also answerable with keys 1-3, shown only while the signer must choose. */
function showPicker(segment: number, options: PickOption[]): void {
  pending = { segment, options };
  $("picker").replaceChildren(el("p", "card-title", "Which sign did you mean?"), ...options.map((o, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.append(el("span", "key", `${i + 1}.`), " ", el("span", "", o.english));
    if (new URLSearchParams(location.search).get("dev")) b.append(" ", el("span", "gloss", `(${o.gloss})`)); // the sign's ID, for us only
    b.onclick = () => choose(i);
    return b;
  }), Object.assign(document.createElement("button"), { type: "button", className: "none", textContent: "None of these — I'll sign it again", onclick: () => choose(-1) }));
  $("picker").hidden = false;
  $("picker").querySelector("button")?.focus(); // the signer answers fast: first option ready for Enter, 1-3 still work
}

/** Renders server messages into the page. Returns nothing; sign plans are handled by the avatar. */
export function renderMsg(m: ServerMsg): void {
  switch (m.type) {
    case "state":
      if (cameraOff && m.state === "calibrating") {
        $("state").textContent = "Ready: speak or type";
        $("state").dataset.state = "ready";
      } else {
        $("state").textContent = STATE_LABEL[m.state] ?? m.state;
        $("state").dataset.state = m.state;
      }
      if (m.state !== "picking" && pending) { pending = null; $("picker").hidden = true; }
      break;
    case "caption":
      $("caption").textContent = m.text;
      break;
    case "pick_request":
      showPicker(m.segment, m.options);
      break;
    case "you_said":
      $("you-said").replaceChildren(m.text, " ", el("span", "emo", m.emotion));
      break;
    case "sign_plan":
      $("caption").textContent = m.english;
      $("plan").replaceChildren(...m.tokens.flatMap((t) => {
        const parts = t.kind === "sign" ? [t.gloss ?? ""] : [...(t.word ?? "")];
        return parts.map((g, i) => {
          const end = i === parts.length - 1 ? " word-end" : "";
          const span = el("span", t.kind === "sign" ? "sign" : `letter${end}`, g);
          span.dataset.g = g; // dataset is getter-only: Object.assign({ dataset }) throws in browsers
          return span;
        });
      }));
      break;
  }
}

/** Shows which sign or letter the avatar is on; caption-only signs are how out-of-avatar signs still get across. */
export function highlight(gloss: string | null): void {
  $("plan").dataset.current = gloss ?? "";
  $("plan").querySelectorAll("span").forEach((s) => s.classList.toggle("on", s.dataset.g === gloss));
}

export function bindControls(sock: Socket): void {
  socket = sock;
  document.addEventListener("keydown", (e) => {
    if (pending && ["1", "2", "3"].includes(e.key)) choose(Number(e.key) - 1);
    else if (pending && e.key === "Escape") choose(-1);
  });
  $("calibrate").onclick = () => { sock.send({ type: "calibrate" }); setMenu(false); };
  // on the phone, Calibrate and Teaching mode sit behind one button so a stray tap can't derail a conversation
  const more = document.getElementById("more");
  const setMenu = (open: boolean) => {
    document.getElementById("actions")?.classList.toggle("open", open);
    more?.setAttribute("aria-expanded", String(open));
  };
  if (more) more.onclick = () => setMenu(more.getAttribute("aria-expanded") !== "true");
  // ?dev=1: type what the hearing person says (no mic or Cartesia key needed).
  $<HTMLFormElement>("dev").hidden = !new URLSearchParams(location.search).get("dev");
  $<HTMLFormElement>("dev").onsubmit = async (e) => {
    e.preventDefault();
    const input = $<HTMLInputElement>("dev-text");
    await fetch("/dev/transcript", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ text: input.value }) });
    input.value = "";
  };
}
