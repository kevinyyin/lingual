// Home: a top header (tabs + streak), the course (a hero card, then each unit's lessons as stops on a
// winding path), side cards (streak, daily challenges), and the Practice, Challenges and Profile tabs.
import { COURSE, type Lesson, type Unit } from "./curriculum";
import { h, svg } from "./dom";
import { ICON } from "./icons";
import { ARROWHEAD } from "./marks";
import { achievements, dayKey, quests, signStrength, xpHistory, type Progress } from "./progress";
import { loadStage, mountSigner, playSigns } from "./signer";
import type { Settings } from "./store";

export interface HomeCtx {
  p: Progress;
  settings: Settings;
  english(gloss: string): string;
  update(p: Progress): void;
  updateSettings(s: Settings): void;
  start(lessonId: string): void;
  startPractice(kind: "weak" | "signing"): void;
  /** English words with signs, for the "Show me a sign" suggestions. */
  words: string[];
  showMe(word: string): void;
  reset(): void;
}

export type Tab = "learn" | "practice" | "quests" | "profile";
export const TAB_IDS: Tab[] = ["learn", "practice", "quests", "profile"];
const TABS: { id: Tab; label: string; icon: string }[] = [
  { id: "learn", label: "Course", icon: ICON.path },
  { id: "practice", label: "Practice", icon: ICON.dumbbell },
  { id: "quests", label: "Challenges", icon: ICON.target },
  { id: "profile", label: "Profile", icon: ICON.user },
];

export function renderHome(root: HTMLElement, tab: Tab, ctx: HomeCtx): void {
  const now = new Date();
  const header = h("header", { class: "topbar" },
    h("a", { class: "logo", href: "#/learn" }, h("span", { class: "mark" }, svg(ICON.mark)), h("span", { class: "word" }, "lingual")),
    h("nav", { class: "tabs", "aria-label": "Main" },
      ...TABS.map((t) => h("a", { class: "tab", href: `#/${t.id}`, "aria-current": t.id === tab ? "page" : null }, svg(t.icon), h("span", { class: "label" }, t.label)))),
    stats(ctx),
    h("a", { class: "mode-link", href: "/", title: "Two-way translator (sign ↔ speech)" }, svg(ICON.translate), h("span", null, "Translator")));
  const body = { learn, practice: practiceTab, quests: questsTab, profile }[tab](ctx, now);
  const main = h("main", { class: "main" }, ...(Array.isArray(body) ? body : [body]));
  const side = h("aside", { class: "side", "aria-label": "Your week" }, streakCard(ctx, now), tab !== "quests" ? questCard(ctx, now) : null);
  root.replaceChildren(header, h("div", { class: "page" }, main, side));
}

function stats(ctx: HomeCtx): HTMLElement {
  const { p } = ctx;
  return h("div", { class: "stats-row" },
    h("span", { class: `stat streak ${p.streak > 0 ? "on" : ""}`, title: `${p.streak} day streak` }, svg(p.streak > 0 ? ICON.flame : ICON.flameOff), p.streak));
}

// --- course ---

type StopState = "done" | "active" | "locked";
/** The first lesson not yet done is active; everything before it is done (lessons unlock in order). */
function stopStates(p: Progress): Map<string, StopState> {
  const out = new Map<string, StopState>();
  let reached = false;
  for (const l of COURSE.flatMap((u) => u.lessons)) {
    if (reached) out.set(l.id, "locked");
    else if (p.done[l.id]) out.set(l.id, "done");
    else { out.set(l.id, "active"); reached = true; }
  }
  return out;
}

/** The top of the course: what's next, today's goal, and the avatar, which signs the next lesson's first sign. */
function hero(ctx: HomeCtx, now: Date, states: Map<string, StopState>): HTMLElement {
  const next = COURSE.map((u) => ({ u, l: u.lessons.find((l) => states.get(l.id) === "active") })).find((x) => x.l);
  const xp = ctx.p.xpByDay[dayKey(now)] ?? 0;
  const goal = ctx.p.dailyGoal;
  const text = h("div", { class: "hero-text" },
    next ? h("h1", null, "Next up: ", h("span", { style: `--chars: ${Math.max(...next.l!.title.split(" ").map((w) => w.length))}` }, next.l!.title))
      : h("h1", null, "You finished the course!"),
    h("p", null, next ? `Unit ${COURSE.indexOf(next.u) + 1} · ${next.u.title}` : "Practice keeps every sign fresh."),
    next ? h("button", { class: "btn hero-btn", onclick: () => ctx.start(next.l!.id) }, "Continue", svg(ICON.arrow)) : null,
    h("div", { class: "goal" }, h("span", null, `${Math.min(xp, goal)} of ${goal} XP today`),
      h("div", { class: "meter" }, h("i", { style: `width: ${Math.min(1, xp / goal) * 100}%` }))));
  if (!next) return h("section", { class: "hero" }, text);
  const loading = h("div", { class: "loading" }, "Loading signer…");
  const slot = h("div", { class: "avatar-slot hero-avatar" }, loading,
    h("div", { class: "controls" },
      h("button", { class: "round hero-play", "aria-label": "Watch the first sign", onclick: () => void playSigns([next.l!.signs[0]]) }, svg(ICON.play))));
  mountSigner(slot);
  const drop = () => loading.remove();
  requestAnimationFrame(() => setTimeout(() => void loadStage().then(drop, drop))); // after the home has painted: the avatar is slow to load
  return h("section", { class: "hero" }, text, slot);
}

function learn(ctx: HomeCtx, now: Date): HTMLElement[] {
  const states = stopStates(ctx.p);
  return [hero(ctx, now, states), ...COURSE.map((u, ui) => unitSection(ctx, u, ui, states))];
}

/** Horizontal position of stop i, as a % of the path's width: a gentle S-curve. */
const stopX = (i: number) => 50 + 24 * Math.sin(i * 1.15 + 0.4);
const STEP = 128; // px between stops

/** The course line through `pts` up to stop `upTo`, led in from the top of the unit. It meets every stop heading
 *  straight down, which is where the active stop's arrowhead points. */
export function coursePath(pts: { x: number; y: number }[], upTo: number): string {
  return [`M${pts[0].x} 0 L${pts[0].x} ${pts[0].y}`, ...pts.slice(1, upTo + 1).map((p, i) =>
    `C${pts[i].x} ${pts[i].y + STEP * 0.55} ${p.x} ${p.y - STEP * 0.55} ${p.x} ${p.y}`)].join(" ");
}

function unitSection(ctx: HomeCtx, u: Unit, ui: number, states: Map<string, StopState>): HTMLElement {
  const n = u.lessons.length;
  const done = u.lessons.filter((l) => states.get(l.id) === "done").length;
  const pts = u.lessons.map((_, i) => ({ x: stopX(i + ui * 2), y: STEP / 2 + i * STEP }));
  const height = n * STEP;
  // One line through the stops: drawn in go as far as the current stop, dotted beyond it.
  const reached = u.lessons.findIndex((l) => states.get(l.id) !== "done");
  const lit = reached === -1 ? n - 1 : reached;
  const started = states.get(u.lessons[0].id) !== "locked";
  const line = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  line.setAttribute("class", "flow-line");
  line.setAttribute("viewBox", `0 0 100 ${height}`);
  line.setAttribute("preserveAspectRatio", "none");
  line.setAttribute("aria-hidden", "true");
  line.innerHTML = `<path d="${coursePath(pts, n - 1)}" class="fl-rest"/>${started ? `<path d="${coursePath(pts, lit)}" class="fl-done"/>` : ""}`;
  return h("section", { class: "unit", "aria-label": `Unit ${ui + 1}: ${u.title}` },
    h("div", { class: "unit-head" },
      h("h2", null, u.title),
      h("p", null, `Unit ${ui + 1} · ${done}/${n} done · ${u.blurb}`),
      h("button", { class: "btn text ub-notes", onclick: () => notes(ctx, u, ui) }, svg(ICON.notes), "Notes")),
    h("ol", { class: "flow", style: `height: ${height}px` },
      h("li", { class: "flow-svg", "aria-hidden": "true" }, line),
      ...u.lessons.map((l, i) => stop(ctx, l, i, states.get(l.id)!, pts[i].x))));
}

function stop(ctx: HomeCtx, l: Lesson, i: number, state: StopState, x: number): HTMLElement {
  const review = l.id.endsWith("review");
  const name = review ? "Unit review" : l.title;
  const mark = state === "active" ? svg(ARROWHEAD, "arrowhead") : state === "done" ? svg(ICON.check) : review ? svg(ICON.trophy) : null;
  const label = h("div", { class: `stop-label ${x > 50 ? "left" : "right"}` },
    h("b", null, name),
    h("span", null, state === "locked" ? "Locked" : review ? "Every sign in the unit" : l.signs.map(ctx.english).join(" · ")),
    state === "active"
      ? h("button", { class: "btn small", onclick: () => ctx.start(l.id) }, ctx.p.done[l.id] ? "Practice" : "Start", svg(ICON.arrow))
      : null);
  const node = h("button", {
    class: "stop-node", disabled: state === "locked",
    "aria-label": `${name}: ${state === "done" ? "done, replay" : state === "active" ? "start" : "locked"}`,
    onclick: () => ctx.start(l.id),
  }, mark);
  return h("li", { class: `stop ${state} ${review ? "review" : ""}`, style: `--x: ${x}%; top: ${i * STEP}px` }, node, label);
}

function notes(ctx: HomeCtx, u: Unit, ui: number): void {
  const hero = document.querySelector<HTMLElement>(".hero-avatar"); // the notes borrow the hero's signer and give it back
  const slot = h("div", { class: "avatar-slot small" }, h("div", { class: "loading" }, "Loading signer…"));
  mountSigner(slot);
  void loadStage().then(() => slot.querySelector(".loading")?.remove());
  const signs = [...new Set(u.lessons.flatMap((l) => l.signs))];
  modal([
    h("h2", null, u.title),
    h("div", { class: "eyebrow" }, `Unit ${ui + 1} notes`),
    h("p", { class: "tip" }, u.tip),
    slot,
    h("p", { class: "small muted", style: "margin: 12px 0 6px" }, "Tap a sign to watch it."),
    h("div", { class: "notes-list" }, ...signs.map((g) => h("button", { onclick: () => void playSigns([g]) }, h("b", null, ctx.english(g)), h("small", null, g)))),
  ], () => { if (hero?.isConnected) mountSigner(hero); }).classList.add("wide-modal");
}

export function modal(children: (HTMLElement | string)[], onClose = () => {}): HTMLElement {
  const close = () => { back.remove(); onClose(); };
  const back = h("div", { class: "modal-back", onclick: (e: Event) => { if (e.target === back) close(); } });
  const box = h("div", { class: "modal", role: "dialog", "aria-modal": "true" }, ...children,
    h("div", { class: "modal-actions" }, h("button", { class: "btn", onclick: close }, "Done")));
  back.append(box);
  document.body.append(back);
  return box;
}

// --- side cards ---

const QUEST_ICON: Record<string, [string, string]> = { xp: [ICON.bolt, "sun"], lessons: [ICON.check, "aqua"], combo: [ICON.flameOff, "red"] };

function questRows(ctx: HomeCtx, now: Date): HTMLElement[] {
  return quests(ctx.p, now).map((q) => {
    const done = q.value >= q.goal;
    const [icon, tone] = QUEST_ICON[q.id] ?? [ICON.check, "aqua"];
    return h("div", { class: `quest ${done ? "done" : ""}` },
      h("span", { class: `q-icon ${tone}` }, svg(icon)),
      h("div", null, h("div", { class: "q-title" }, q.title, h("span", { class: "muted small" }, `${q.value}/${q.goal}`)),
        h("div", { class: `meter ${tone}` }, h("i", { style: `width: ${(q.value / q.goal) * 100}%` }))),
      h("span", { class: `q-done ${done ? "on" : ""}`, role: "img", "aria-label": done ? "Done" : "Not done yet" }, svg(ICON.check)));
  });
}

function questCard(ctx: HomeCtx, now: Date): HTMLElement {
  return h("section", { class: "card" },
    h("div", { class: "card-head" }, h("h3", null, "Daily challenges"), h("a", { class: "link", href: "#/quests" }, "See all")),
    ...questRows(ctx, now));
}

/** Monday..Sunday of this week: was a lesson finished that day? */
export function weekDone(p: Progress, now: Date): boolean[] {
  return Array.from({ length: 7 }, (_, i) => {
    const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() - ((now.getDay() + 6) % 7) + i);
    return (p.xpByDay[dayKey(d)] ?? 0) > 0;
  });
}

function streakCard(ctx: HomeCtx, now: Date): HTMLElement {
  const week = weekDone(ctx.p, now);
  const di = (now.getDay() + 6) % 7;
  return h("section", { class: "card streak-card" },
    h("div", { class: "streak-top" },
      h("div", { class: `flame-badge ${ctx.p.streak ? "on" : ""}` }, svg(ctx.p.streak ? ICON.flame : ICON.flameOff)),
      h("div", null, h("b", { class: "big-num" }, String(ctx.p.streak)), h("div", { class: "small" }, "day streak"))),
    h("div", { class: "week" }, ..."MTWTFSS".split("").map((d, i) =>
      h("div", null, h("i", { class: `${week[i] ? "on" : ""} ${i === di ? "today" : ""}` }, week[i] ? svg(ICON.check) : null), d))),
    h("p", { class: "small streak-note" }, ctx.p.streak ? "Do a lesson each day to keep it going." : "Finish a lesson to start a streak."));
}

// --- other tabs ---

const title = (text: string, sub?: string) => h("div", { class: "page-head" }, h("h1", null, text), sub ? h("p", { class: "muted" }, sub) : null);

function practiceTab(ctx: HomeCtx, now: Date): HTMLElement[] {
  const today = dayKey(now);
  const known = Object.keys(ctx.p.signs);
  const hub = h("div", { class: "hub" },
    h("button", { class: "hub-card purple", disabled: known.length < 3, onclick: () => ctx.startPractice("weak") },
      h("span", { class: "hub-icon" }, svg(ICON.dumbbell)), h("b", null, "Review weak signs"),
      h("span", null, known.length < 3 ? "Finish a lesson first" : "Your lowest-strength signs.")),
    h("button", { class: "hub-card blue", disabled: known.length < 1 || !ctx.settings.camera, onclick: () => ctx.startPractice("signing") },
      h("span", { class: "hub-icon" }, svg(ICON.camera)), h("b", null, "Sign to the camera"),
      h("span", null, ctx.settings.camera ? "Sign words you know and get checked" : "Camera exercises are off in Profile")));
  const grid = known.length
    ? h("div", { class: "sign-grid" }, ...known.map((g) => {
      const s = signStrength(ctx.p.signs[g], today);
      return h("div", { class: "sign-tile" }, h("b", null, ctx.english(g)), h("small", null, g),
        h("div", { class: "pips", "aria-label": `Strength ${s} of 4` }, ...[0, 1, 2, 3].map((i) => h("i", { class: i < s ? "on" : "" }))));
    }))
    : h("div", { class: "empty" }, h("div", { class: "big" }, svg(ICON.mark)), h("p", null, "Signs you learn show up here with how strong they are."));
  return [title("Practice", "Keep signs fresh between lessons."), showMeCard(ctx), hub, h("h2", { class: "section-title" }, `Your signs · ${known.length}`), grid];
}

/** A parent types a word and the avatar signs it (or fingerspells it). */
function showMeCard(ctx: HomeCtx): HTMLElement {
  const input = h("input", { class: "show-input", type: "text", list: "show-words", placeholder: "Type a word, like dog",
    "aria-label": "Word to sign", autocomplete: "off", autocapitalize: "off", enterkeyhint: "go" });
  const slot = h("div", { class: "avatar-slot small" }, h("div", { class: "loading" }, "Loading signer…"));
  mountSigner(slot);
  void loadStage().then(() => slot.querySelector(".loading")?.remove());
  return h("form", { class: "card show-me", onsubmit: (e: Event) => { e.preventDefault(); input.blur(); ctx.showMe(input.value); } },
    h("h2", { class: "section-title" }, "Show me a sign"),
    h("div", { class: "show-row" }, input, h("button", { class: "btn" }, "Show me")),
    h("datalist", { id: "show-words" }, ...ctx.words.map((w) => h("option", { value: w }))),
    slot);
}

function questsTab(ctx: HomeCtx, now: Date): HTMLElement[] {
  return [title("Daily challenges", `New challenges in ${24 - now.getHours()} hours.`), h("section", { class: "card" }, ...questRows(ctx, now))];
}

const ACH: Record<string, [string, string]> = { streak: [ICON.flameOff, "sun"], xp: [ICON.bolt, "aqua"], signs: [ICON.check, "purple"], perfect: [ICON.target, "red"] };

function profile(ctx: HomeCtx, now: Date): HTMLElement[] {
  const { p, settings } = ctx;
  const learned = Object.values(p.signs).filter((s) => s.correct > 0).length;
  const tile = (value: string | number, label: string) => h("div", { class: "stat-tile" }, h("b", null, value), h("span", null, label));
  const set = (s: Partial<Settings>) => ctx.updateSettings({ ...settings, ...s });
  return [
    h("div", { class: "profile-head" }, h("div", { class: "big-av" }, svg(ICON.mark)),
      h("div", null, h("h1", null, "Learner"), h("p", { class: "muted" }, "Learning American Sign Language"))),
    h("div", { class: "stat-grid" },
      tile(p.streak, "day streak"), tile(p.xp, "total XP"), tile(learned, "signs learned"), tile(p.longestStreak, "longest streak")),
    h("h2", { class: "section-title" }, "XP this week"),
    h("section", { class: "card" }, xpChart(xpHistory(p, now))),
    h("h2", { class: "section-title" }, "Achievements"),
    h("section", { class: "card" }, ...achievements(p).map((a) => {
      const [icon, tone] = ACH[a.id] ?? [ICON.check, "aqua"];
      return h("div", { class: "ach" },
        h("span", { class: `ach-icon ${tone} ${a.value >= a.goal ? "got" : ""}` }, svg(icon)),
        h("div", null, h("div", { class: "q-title" }, a.title, h("span", { class: "muted small" }, `${a.value}/${a.goal}`)),
          h("p", { class: "muted small", style: "margin: 0 0 8px" }, a.detail),
          h("div", { class: `meter ${tone}` }, h("i", { style: `width: ${(a.value / a.goal) * 100}%` }))));
    })),
    h("h2", { class: "section-title" }, "Settings"),
    h("section", { class: "card" },
      h("div", { class: "settings-row" }, h("span", null, "Daily goal"), h("div", { class: "seg" }, ...[[10, "Light"], [20, "Steady"], [30, "Focused"], [50, "All in"]].map(([xp, name]) =>
        h("button", { "aria-pressed": String(p.dailyGoal === xp), onclick: () => ctx.update({ ...ctx.p, dailyGoal: xp as number }) }, `${name} · ${xp} XP`)))),
      h("div", { class: "settings-row" }, h("span", null, "Sound effects"),
        h("button", { class: "switch", role: "switch", "aria-checked": String(settings.sound), "aria-label": "Sound effects", onclick: () => set({ sound: !settings.sound }) })),
      h("div", { class: "settings-row" }, h("span", null, "Camera exercises"),
        h("button", { class: "switch", role: "switch", "aria-checked": String(settings.camera), "aria-label": "Camera exercises", onclick: () => set({ camera: !settings.camera }) })),
      h("div", { class: "settings-row" }, h("span", null, "Reset all progress"),
        h("button", { class: "btn ghost small", onclick: () => { if (confirm("Reset all your XP, streak and lessons?")) ctx.reset(); } }, "Reset"))),
  ];
}

/** One series, so no legend: a 2px line, 8px points, recessive grid, hover/focus tooltips per day. Drawn once it's on
 *  the page, at its own width: one unit is one pixel, so the 15px labels stay 15px on a phone. */
function xpChart(days: { day: string; label: string; xp: number }[]): HTMLElement {
  const wrap = h("div", { class: "chart" });
  requestAnimationFrame(() => drawXp(wrap, days, Math.max(300, wrap.clientWidth)));
  return wrap;
}

function drawXp(wrap: HTMLElement, days: { day: string; label: string; xp: number }[], W: number): void {
  const H = 200, L = 34, B = 26, T = 12;
  const max = Math.max(20, ...days.map((d) => d.xp));
  const step = max <= 40 ? 10 : max <= 100 ? 25 : 50;
  const top = Math.ceil(max / step) * step;
  const x = (i: number) => L + (i * (W - L - 12)) / (days.length - 1);
  const y = (v: number) => T + (1 - v / top) * (H - T - B);
  const ns = "http://www.w3.org/2000/svg";
  const el = (tag: string, attrs: Record<string, string | number>, text?: string) => {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
    if (text) e.textContent = text;
    return e;
  };
  const s = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `XP per day this week: ${days.map((d) => `${d.label} ${d.xp}`).join(", ")}` });
  for (let v = 0; v <= top; v += step) {
    s.append(el("line", { class: "grid", x1: L, x2: W - 6, y1: y(v), y2: y(v) }), el("text", { class: "axis", x: L - 8, y: y(v) + 4, "text-anchor": "end" }, String(v)));
  }
  days.forEach((d, i) => s.append(el("text", { class: "axis", x: x(i), y: H - 6, "text-anchor": "middle" }, d.label)));
  s.append(el("path", { class: "line", d: days.map((d, i) => `${i ? "L" : "M"}${x(i)},${y(d.xp)}`).join("") }));
  const tip = h("div", { class: "tooltip", hidden: true });
  days.forEach((d, i) => {
    const hit = el("circle", { class: "hit", cx: x(i), cy: y(d.xp), r: 16, tabindex: 0, "aria-label": `${d.label}: ${d.xp} XP` });
    const show = () => { tip.hidden = false; tip.textContent = `${d.label} · ${d.xp} XP`; tip.style.left = `${(x(i) / W) * 100}%`; tip.style.top = `${(y(d.xp) / H) * 100}%`; };
    const hide = () => (tip.hidden = true);
    hit.addEventListener("mouseenter", show); hit.addEventListener("focus", show);
    hit.addEventListener("mouseleave", hide); hit.addEventListener("blur", hide);
    s.append(hit, el("circle", { class: "dot", cx: x(i), cy: y(d.xp), r: 5 }));
  });
  wrap.append(s, tip);
}
