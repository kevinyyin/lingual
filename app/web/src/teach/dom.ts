// Tiny DOM helpers for teaching mode (the app has no UI framework).

import { setAudioSession, speakerOut } from "../phone";

type Child = Node | string | number | null | undefined | false;
type Attrs = Record<string, string | number | boolean | null | undefined | ((e: any) => void)>;

/** h("button", { class: "btn", onclick: fn, disabled: true }, "Go"). Falsy children are skipped. */
export function h<K extends keyof HTMLElementTagNameMap>(tag: K, attrs: Attrs | null = null, ...children: Child[]): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs ?? {})) {
    if (v === null || v === undefined || v === false) continue;
    if (typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "style") el.style.cssText = String(v);
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  append(el, children);
  return el;
}

export function append(el: Element, children: Child[]): void {
  for (const c of children) if (c !== null && c !== undefined && c !== false) el.append(typeof c === "number" ? String(c) : c);
}

/** An inline SVG from markup (icons.ts). */
export function svg(markup: string, cls = "icon"): SVGSVGElement {
  const t = document.createElement("template");
  t.innerHTML = markup.trim();
  const el = t.content.firstElementChild as SVGSVGElement;
  el.classList.add(...cls.split(" ").filter(Boolean));
  el.setAttribute("aria-hidden", "true");
  return el;
}

/** Plays a short tone pair: up for right, down for wrong (sound is optional; everything is also shown). */
export function chime(kind: "right" | "wrong" | "done"): void {
  try {
    setAudioSession("playback"); // not muted by the iPad's ringer switch
    const audio = speakerOut();
    const notes = kind === "right" ? [660, 880] : kind === "wrong" ? [300, 220] : [523, 659, 784, 1047];
    notes.forEach((f, i) => {
      const o = audio.createOscillator();
      const g = audio.createGain();
      const t = audio.currentTime + i * 0.09;
      o.type = kind === "wrong" ? "square" : "sine";
      o.frequency.value = f;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(kind === "wrong" ? 0.05 : 0.12, t + 0.01);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.16);
      o.connect(g).connect(audio.destination);
      o.start(t);
      o.stop(t + 0.18);
    });
  } catch { /* no audio */ }
}
