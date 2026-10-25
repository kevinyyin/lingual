// The lesson screen: progress bar, one exercise at a time, a Check / Continue bar whose feedback card says right
// or wrong, then the results and streak screens.
import type { PickOption } from "../ws";
import { chime, h, svg } from "./dom";
import { ICON } from "./icons";
import { LessonRun, shuffle, signScore, type Exercise, type SavedRun } from "./lesson";
import { loadStage, mountSigner, playSigns, practice, type CamStatus, type Feeling } from "./signer";
import { FEELING_FACE } from "./curriculum";
import type { Earned } from "./progress";

export interface LessonHost {
  english(gloss: string): string;
  sound(): boolean;
  /** The plushie says a short line out loud (the host skips it when sound is off). */
  say(text: string): void;
  /** "Camera off for now" turns camera exercises off for the session; the rest of them are skipped. */
  cameraOn(): boolean;
  disableCamera(): void;
  quit(): void;
  /** Keeps the run after every step, so a lesson left part way picks up there; null once it's over. */
  save(run: SavedRun | null): void;
  finish(run: LessonRun): { earned: Earned; streak: number; week: boolean[] };
}

const PRAISE = ["Spot on.", "That's it!", "Nailed it.", "Exactly right.", "Got it."];
/** What the plushie says during a lesson; each under 60 characters (they go to TTS one by one). */
const LINES = {
  start: "Hi! Let's learn some signs together.",
  yourTurn: (word: string) => `Your turn! Sign ${word}.`,
  retry: "Almost! Let's try again.",
  wrong: "Not quite.",
  done: "You did it! Great job.",
};
const pickOne = <T>(xs: T[]) => xs[Math.floor(Math.random() * xs.length)];
/** Top-k guesses that count as signing it right: the recognizer is ~47% top-1, so top-1 alone would fail good signing. */
const PERFORM_TOP_K = 3;

interface Mounted {
  el: HTMLElement;
  /** Ready for Check (an answer is chosen). Exercises that grade themselves never enable Check. */
  ready?: () => boolean;
  check?: () => { correct: boolean; solution: string };
  onShow?: () => void;
  onLeave?: () => void;
  /** The bar's left button (Skip / Camera off for now). */
  left?: { label: string; run: () => void };
}

export function runLesson(root: HTMLElement, exercises: Exercise[], host: LessonHost, saved?: SavedRun): void {
  const run = new LessonRun(exercises, saved);
  if (new URLSearchParams(location.search).get("dev")) Object.assign(window, { teach: { run } }); // console/test access, like translator's window.asl
  const bar = h("i", { style: "width: 0%" });
  const count = h("span", { class: "step-count" });
  const body = h("main", { class: "lesson-body" });
  const footInner = h("div", { class: "inner" });
  const foot = h("footer", { class: "lesson-foot" }, footInner);
  const close = h("button", { class: "close", "aria-label": "Leave lesson", onclick: () => confirmQuit() }, svg(ICON.cross));
  const back = h("button", { class: "close back", "aria-label": "Previous exercise", onclick: () => { if (run.back()) next(); } }, svg(ICON.arrow));
  root.replaceChildren(h("div", { class: "lesson" },
    h("header", { class: "lesson-top" }, close, back,
      h("div", { class: "progress", role: "progressbar", "aria-label": "Lesson progress" }, bar), count),
    body, foot));
  void loadStage(); // start loading the avatar while the first card renders
  host.say(LINES.start);

  let current: Mounted | null = null;
  let phase: "answer" | "feedback" = "answer";
  let lastCorrect = false;

  function confirmQuit() {
    const back = h("div", { class: "modal-back" });
    back.append(h("div", { class: "modal", role: "dialog", "aria-modal": "true" },
      h("h2", null, "Leave this lesson?"),
      h("p", null, "You'll pick up right here next time."),
      h("div", { class: "modal-actions" },
        h("button", { class: "btn ghost", onclick: () => { back.remove(); leave(); host.quit(); } }, "Leave"),
        h("button", { class: "btn", onclick: () => back.remove() }, "Stay"))));
    document.body.append(back);
  }

  function leave() {
    current?.onLeave?.();
    practice.active = false;
    document.removeEventListener("keydown", onKey);
  }

  function next() {
    current?.onLeave?.();
    host.save(run.finished ? null : run.save());
    back.disabled = !run.canGoBack;
    bar.style.width = `${Math.round(run.fraction * 100)}%`;
    count.textContent = `${run.done}/${run.total}`;
    if (run.finished) {
      leave();
      return run.answers.length ? complete() : host.quit(); // nothing graded (every camera exercise skipped): no XP, no "Flawless!"
    }
    if (run.current!.ex.kind === "perform" && !host.cameraOn()) {
      run.skip();
      return next();
    }
    const { ex, retry } = run.current!;
    current = mount(ex, host, grade, () => { host.disableCamera(); run.skip(); next(); });
    phase = "answer";
    body.replaceChildren(current.el);
    if (retry) current.el.querySelector(".prompt")!.after(h("div", { class: "ex-tag retry" }, svg(ICON.retry), "Try this one again")); // under the prompt: never a label over it
    footer();
    current.onShow?.();
  }

  /** Answer bar: left Skip / Camera off for now, right Check (or Next for intro cards). */
  function footer() {
    foot.className = "lesson-foot";
    const ex = run.current!.ex;
    if (ex.kind === "intro") {
      footInner.replaceChildren(h("span"), h("button", { class: "btn", onclick: () => { run.answer(true); next(); } }, "Next", svg(ICON.arrow)));
      return;
    }
    const left = current!.left
      ? h("button", { class: "btn ghost", onclick: current!.left.run }, current!.left.label)
      : h("button", { class: "btn ghost", onclick: () => grade(false, solutionOf(ex)) }, "Skip");
    const check = h("button", { class: "btn", disabled: true, onclick: () => submit() }, "Check");
    check.toggleAttribute("disabled", !current!.ready?.());
    if (!current!.check) check.hidden = true;
    footInner.replaceChildren(left, check);
  }

  const solutionOf = (ex: Exercise) =>
    ex.kind === "match" || ex.kind === "order" ? ex.glosses.map(host.english).join(" · ") : host.english(ex.gloss);

  /** Exercises call this when their answer changes, so Check enables. */
  const refresh = () => {
    const btn = footInner.querySelector<HTMLButtonElement>(".btn:not(.ghost)");
    if (btn && phase === "answer" && current?.ready) btn.disabled = !current.ready();
  };
  body.addEventListener("answer-change", refresh);

  function submit() {
    if (phase !== "answer" || !current?.check || !current.ready?.()) return;
    const r = current.check();
    grade(r.correct, r.solution);
  }

  function grade(correct: boolean, solution: string, score?: number) {
    if (phase !== "answer") return;
    phase = "feedback";
    lastCorrect = correct;
    const comboBefore = run.combo;
    if (host.sound()) chime(correct ? "right" : "wrong");
    practice.answered(correct ? "right" : "wrong");
    const praise = pickOne(PRAISE);
    host.say(correct ? praise : LINES.wrong);
    body.querySelectorAll<HTMLButtonElement>("button.opt, button.chip").forEach((b) => (b.disabled = true));
    foot.className = `lesson-foot ${correct ? "right" : "wrong"}`;
    footInner.replaceChildren(
      h("div", { class: "verdict", role: "status" },
        h("div", { class: "badge" }, svg(correct ? ICON.check : ICON.cross)),
        h("div", null, h("h3", null, correct ? praise : "Not quite."), correct ? null : h("p", null, "Answer: ", h("b", null, solution)),
          score === undefined ? null : scoreBar(score))),
      // right or wrong, the same exercise again from the start, the answer not counted
      h("button", { class: "btn ghost redo", onclick: () => next() }, svg(ICON.retry), "Redo"),
      h("button", { class: "btn", onclick: () => { run.answer(lastCorrect); next(); } }, "Continue", svg(ICON.arrow)));
    footInner.querySelector<HTMLButtonElement>(".btn:not(.ghost)")!.focus();
    if (correct && [5, 10, 15].includes(comboBefore + 1)) toast(`${comboBefore + 1} right in a row`);
  }

  function onKey(e: KeyboardEvent) {
    if (document.querySelector(".modal-back")) return;
    if (e.key === "Enter") {
      e.preventDefault();
      if (phase === "feedback" || run.current?.ex.kind === "intro") footInner.querySelector<HTMLButtonElement>(".btn:not(.ghost)")?.click();
      else submit();
    } else if (/^[1-9]$/.test(e.key) && phase === "answer") {
      body.querySelectorAll<HTMLButtonElement>("[data-key]").forEach((b) => { if (b.dataset.key === e.key && !b.disabled) b.click(); });
    }
  }
  document.addEventListener("keydown", onKey);

  /** The results-style screen: a centered card over the paper background, one button. */
  const screen = (card: HTMLElement[], button: HTMLElement) =>
    root.replaceChildren(h("div", { class: "done-screen" }, h("div", { class: "done-card" }, ...card, button)));

  function complete() {
    if (host.sound()) chime("done");
    host.say(LINES.done);
    const { earned, streak, week } = host.finish(run);
    const secs = Math.round((performance.now() - run.started) / 1000);
    const acc = Math.round(run.accuracy * 100);
    confetti();
    screen([
      h("div", { class: "done-mark" }, svg(ICON.mark)),
      h("h1", null, earned.perfect ? "Flawless!" : "Lesson done"),
      h("p", { class: "sub" }, earned.goalReached ? "You hit today's goal." : "Nice work. Keep the streak going."),
      accuracyRing(acc),
      h("div", { class: "chips" },
        h("span", { class: "chip-stat sun" }, svg(ICON.bolt), `+${earned.xp} XP`),
        h("span", { class: "chip-stat aqua" }, svg(ICON.clock), `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`),
        h("span", { class: "chip-stat green" }, svg(ICON.target), `${run.mistakes ? run.mistakes : "No"} mistake${run.mistakes === 1 ? "" : "s"}`)),
    ], h("button", { class: "btn wide", onclick: () => (earned.streakExtended ? streakScreen(streak, week) : host.quit()) }, "Continue", svg(ICON.arrow)));
  }

  function streakScreen(streak: number, week: boolean[]) {
    const num = h("span", { class: "streak-num" }, String(Math.max(0, streak - 1)));
    const today = (new Date().getDay() + 6) % 7;
    screen([
      h("div", { class: "streak-badge" }, svg(ICON.flame), num),
      h("h1", null, streak === 1 ? "Streak started" : "Streak extended"),
      h("p", { class: "sub" }, `${streak} day${streak === 1 ? "" : "s"} of signing in a row. Come back tomorrow to keep it.`),
      h("div", { class: "week" }, ..."MTWTFSS".split("").map((d, i) => h("div", null,
        h("i", { class: `${week[i] ? "on" : ""} ${i === today ? "today" : ""}` }, week[i] ? svg(ICON.check) : null), d))),
    ], h("button", { class: "btn wide", onclick: host.quit }, "Back to course"));
    setTimeout(() => { num.textContent = String(streak); num.animate([{ transform: "scale(1.5)" }, { transform: "none" }], { duration: 500, easing: "cubic-bezier(.2, .8, .2, 1)" }); }, 450);
  }

  next();
}

/** Accuracy inside a hand-drawn loop that closes as far as the accuracy (all the way round at 100%). */
function accuracyRing(pct: number): HTMLElement {
  const loop = "M60 12C88 11 108 32 108 60C108 88 87 108 60 108C33 108 12 88 12 61C12 35 31 14 57 12";
  const wrap = h("div", { class: "ring", role: "img", "aria-label": `Accuracy ${pct}%` });
  wrap.innerHTML = `<svg viewBox="0 0 120 120"><path d="${loop}" class="ring-track"/><path d="${loop}" class="ring-fill" pathLength="100"/></svg>`;
  wrap.append(h("div", { class: "ring-label" }, h("b", null, `${pct}%`), h("span", null, "accuracy")));
  requestAnimationFrame(() => requestAnimationFrame(() => wrap.querySelector<SVGPathElement>(".ring-fill")!.style.strokeDashoffset = String(100 - pct)));
  return wrap;
}

function toast(text: string) {
  const t = h("div", { class: "toast", role: "status" }, svg(ICON.flame), text);
  document.body.append(t);
  setTimeout(() => t.remove(), 1700);
}

/** One short burst of drawn strokes from the middle of the screen, where the results card sits. */
function confetti() {
  const layer = h("div", { class: "confetti", "aria-hidden": "true" });
  const colors = ["#18943F", "#232A63", "#E2466C", "#FFC629", "#2B83D6"];
  for (let i = 0; i < 20; i++) {
    layer.append(h("i", { style: `--a: ${Math.round(Math.random() * 360)}deg; --d: ${90 + Math.round(Math.random() * 100)}px; background: ${colors[i % colors.length]}` }));
  }
  document.body.append(layer);
  setTimeout(() => layer.remove(), 1200);
}

// --- exercises ---

const changed = (el: HTMLElement) => el.dispatchEvent(new Event("answer-change", { bubbles: true }));

/** Marks the answer tiles: the right one green with a check, a wrong choice red with a cross. */
function markAnswer(chosen: HTMLElement, right: HTMLElement): void {
  if (chosen !== right) {
    chosen.classList.add("wrong");
    chosen.append(svg(ICON.cross, "icon tile-mark"));
  }
  right.classList.add("right");
  right.append(svg(ICON.check, "icon tile-mark"));
}

/** The avatar's slot with replay and slow-replay buttons; `signs()` is what replay plays. */
/** `label`: name each sign on the stage as the avatar signs it (fingerspelled letters as themselves). Only where it
 *  can't give an answer away. */
function avatarSlot(signs: () => string[], small = false, label?: (gloss: string) => string): HTMLElement {
  const loading = h("div", { class: "loading" }, "Loading signer…");
  const tag = h("div", { class: "sign-label", "aria-hidden": "true", hidden: "" });
  const slot = h("div", { class: `avatar-slot ${small ? "small" : ""}` }, loading, label ? tag : null,
    h("div", { class: "controls" },
      h("button", { class: "round", "aria-label": "Play at normal speed", title: "Play at normal speed", onclick: () => void playSigns(signs()) }, svg(ICON.play)),
      h("button", { class: "round small", "aria-label": "Play in slow motion", title: "Play in slow motion", onclick: () => void playSigns(signs(), { speed: 0.5 }) }, svg(ICON.slow))));
  mountSigner(slot);
  if (label) slot.addEventListener("signitem", (e) => { // keeps the last sign up through blends and after playback
    const g = (e as CustomEvent<string | null>).detail;
    if (!g) return;
    tag.textContent = g.length === 1 ? g : label(g);
    tag.hidden = false;
  });
  void loadStage().then(() => loading.remove(), () => (loading.textContent = "The signer couldn't load"));
  return slot;
}

/** `score`: a camera attempt's match, 0-100 (signScore), shown on the feedback card. */
type Grade = (correct: boolean, solution: string, score?: number) => void;

function mount(ex: Exercise, host: LessonHost, grade: Grade, skip: () => void): Mounted {
  const en = host.english;
  switch (ex.kind) {
    case "intro": {
      return {
        el: h("div", { style: "display: contents" },
          h("h2", { class: "prompt" }, "Watch and learn this sign"),
          h("div", { class: "ex-tag" }, svg(ICON.spark), "New sign"),
          avatarSlot(() => [ex.gloss], false, en),
          h("div", { class: "intro-word" }, h("b", null, en(ex.gloss)), h("small", null, ex.gloss))),
        onShow: () => void playSigns([ex.gloss]),
      };
    }
    case "watch": {
      let chosen: string | null = null;
      const opts = ex.options.map((g, i) => h("button", {
        class: "opt", "aria-pressed": "false", "data-key": String(i + 1),
        onclick: (e: Event) => { chosen = g; select(e.currentTarget as HTMLElement); },
      }, h("span", { class: "key" }, String(i + 1)), en(g)));
      const wrap = h("div", { class: "options grid" }, ...opts);
      const select = (b: HTMLElement) => { opts.forEach((o) => o.setAttribute("aria-pressed", String(o === b))); changed(wrap); };
      return {
        el: h("div", { style: "display: contents" }, h("h2", { class: "prompt" }, "What does this sign mean?"), avatarSlot(() => [ex.gloss]), wrap),
        onShow: () => void playSigns([ex.gloss]),
        ready: () => chosen !== null,
        check: () => {
          markAnswer(opts[ex.options.indexOf(chosen!)], opts[ex.options.indexOf(ex.gloss)]);
          return { correct: chosen === ex.gloss, solution: en(ex.gloss) };
        },
      };
    }
    case "pick-sign": {
      let chosen: string | null = null;
      const opts = ex.options.map((g, i) => h("button", {
        class: "opt sign", "aria-pressed": "false", "data-key": String(i + 1), "aria-label": `Sign ${i + 1}`,
        onclick: (e: Event) => { chosen = g; select(e.currentTarget as HTMLElement); void playSigns([g]); },
      }, svg(ICON.play), h("span", null, `Sign ${i + 1}`)));
      const wrap = h("div", { class: "options three" }, ...opts);
      const select = (b: HTMLElement) => { opts.forEach((o) => o.setAttribute("aria-pressed", String(o === b))); changed(wrap); };
      return {
        el: h("div", { style: "display: contents" },
          h("h2", { class: "prompt" }, `Which one is “${en(ex.gloss)}”?`),
          avatarSlot(() => (chosen ? [chosen] : []), true),
          h("p", { class: "muted", style: "margin: -8px 0 0; font-weight: 600" }, "Tap a card to watch its sign."),
          wrap),
        ready: () => chosen !== null,
        check: () => {
          markAnswer(opts[ex.options.indexOf(chosen!)], opts[ex.options.indexOf(ex.gloss)]);
          return { correct: chosen === ex.gloss, solution: `Sign ${ex.options.indexOf(ex.gloss) + 1}` };
        },
      };
    }
    case "match": {
      // Left: signs to watch (A-D); right: English. Pick one of each; a wrong pair flashes and counts as a miss.
      const signs = shuffle(ex.glosses, Math.random);
      const words = shuffle(ex.glosses, Math.random);
      let sel: { side: "s" | "w"; g: string; b: HTMLElement } | null = null;
      let misses = 0;
      let matched = 0;
      const tryPair = (side: "s" | "w", g: string, b: HTMLElement) => {
        if (b.classList.contains("gone")) return;
        if (side === "s") void playSigns([g]);
        if (!sel || sel.side === side) {
          sel?.b.setAttribute("aria-pressed", "false");
          sel = { side, g, b };
          b.setAttribute("aria-pressed", "true");
          return;
        }
        const other = sel;
        sel = null;
        other.b.setAttribute("aria-pressed", "false");
        if (other.g === g) {
          for (const x of [b, other.b]) { x.classList.add("right"); setTimeout(() => { x.classList.remove("right"); x.classList.add("gone"); (x as HTMLButtonElement).disabled = true; }, 350); }
          if (++matched === ex.glosses.length) setTimeout(() => grade(misses < 2, ex.glosses.map(en).join(" · ")), 450);
        } else {
          misses++;
          for (const x of [b, other.b]) { x.classList.add("wrong"); setTimeout(() => x.classList.remove("wrong"), 400); }
        }
      };
      const L = "ABCD";
      return {
        el: h("div", { style: "display: contents" },
          h("h2", { class: "prompt" }, "Tap the matching pairs"),
          avatarSlot(() => (sel?.side === "s" ? [sel.g] : []), true),
          h("div", { class: "match" },
            h("div", { class: "col" }, ...signs.map((g, i) => h("button", { class: "opt", "aria-pressed": "false", "data-key": String(i + 1),
              onclick: (e: Event) => tryPair("s", g, e.currentTarget as HTMLElement) }, h("span", { class: "key" }, String(i + 1)), svg(ICON.play), `Sign ${L[i]}`))),
            h("div", { class: "col" }, ...words.map((g, i) => h("button", { class: "opt", "aria-pressed": "false", "data-key": String(i + 5),
              onclick: (e: Event) => tryPair("w", g, e.currentTarget as HTMLElement) }, h("span", { class: "key" }, String(i + 5)), en(g)))))),
      };
    }
    case "order": {
      const picked: number[] = [];
      const line = h("div", { class: "answer-line", "aria-label": "Your answer" });
      const chips = ex.bank.map((g, i) => h("button", { class: "chip", onclick: () => toggle(i) }, en(g)));
      const bank = h("div", { class: "bank" }, ...chips);
      const toggle = (i: number) => {
        const at = picked.indexOf(i);
        if (at >= 0) picked.splice(at, 1);
        else picked.push(i);
        line.replaceChildren(...picked.map((j) => h("button", { class: "chip", onclick: () => toggle(j) }, en(ex.bank[j]))));
        chips.forEach((c, j) => c.classList.toggle("used", picked.includes(j)));
        changed(line);
      };
      return {
        el: h("div", { style: "display: contents" },
          h("h2", { class: "prompt" }, "Put the signs in the order you see them"),
          avatarSlot(() => ex.glosses, true), line, bank),
        onShow: () => void playSigns(ex.glosses),
        ready: () => picked.length > 0,
        check: () => ({ correct: picked.map((j) => ex.bank[j]).join() === ex.glosses.join(), solution: ex.glosses.map(en).join(" · ") }),
      };
    }
    case "perform":
      return mountPerform(ex.gloss, host, grade, skip);
  }
}

const CAM_TEXT: Record<CamStatus, string> = {
  off: "Camera off",
  starting: "Starting camera…",
  position: "Get into position",
  calibrating: "Rest your hands in your lap, neutral face…",
  ready: "Go! Sign it now, then rest your hands",
  signing: "Signing… hands down when you're done",
  reading: "Reading your sign… tap to redo",
  error: "Camera unavailable",
};

/** A camera attempt's match score as a labelled bar. */
function scoreBar(score: number): HTMLElement {
  const tone = score >= 70 ? "good" : score >= 35 ? "ok" : "low";
  return h("span", { class: `score ${tone}` },
    h("span", { class: "score-label" }, `Match ${score}%`),
    h("span", { class: "score-track", "aria-hidden": "true" }, h("i", { style: `width: ${score}%` })));
}

/** The emotion line under the camera: what the learner's face reads as and, for a feeling sign, whether it fits the
 *  sign's face. The sign's face wins (the avatar always makes it): the line only nudges. */
export function faceLine(seen: Feeling, gloss: string, english: string): string {
  const face = FEELING_FACE[gloss];
  if (!face) return `Your face: ${seen}`;
  const want = face === "warm" ? "happy" : gloss === "ANGRY" ? "angry" : "sad";
  if (seen === want) return `Your face: ${seen}. Just right for “${english}”!`;
  return `Your face: ${seen}. “${english}” needs ${want === "angry" ? "an" : "a"} ${want} face!`;
}

function mountPerform(gloss: string, host: LessonHost, grade: Grade, skip: () => void): Mounted {
  const status = h("div", { class: "cam-status", role: "status" }, CAM_TEXT[practice.status]);
  // while a sign is read the frozen feed is one big button: tap it to drop that attempt and sign again
  const redo = h("button", { class: "cam-redo", type: "button", "aria-label": "Redo the sign", hidden: "", onclick: () => practice.redo() });
  const cam = h("div", { class: "cam-slot", "data-status": practice.status, "data-hands": practice.hands ? "in" : "out" },
    practice.video, practice.trailCanvas, status, redo);
  const heard = h("p", { class: "heard", "aria-live": "polite" });
  const face = (seen: Feeling) => faceLine(seen, gloss, host.english(gloss));
  const faceEl = h("p", { class: "face-line" }, face(practice.seen));
  let tries = 0;
  let asked = false;
  const stop = () => { practice.active = false; practice.listener = null; }; // graded or left: later segments must not say "try again"
  const setStatus = (s: CamStatus, detail?: string) => {
    cam.dataset.status = s;
    redo.hidden = s !== "reading";
    if (s === "signing") heard.replaceChildren(); // a new attempt: the last one's "that looked like..." is stale
    status.textContent = detail ? `${CAM_TEXT[s]}: ${detail}` : CAM_TEXT[s];
    // only once calibrated: signing while it measures the resting hands puts the rest line at signing height
    if (s === "ready" && !asked) {
      asked = true;
      host.say(LINES.yourTurn(host.english(gloss)));
    }
  };
  const onRecognized = (segments: PickOption[][]) => {
    const { score, rank } = signScore(segments, gloss);
    const hit = rank !== null && rank <= PERFORM_TOP_K;
    // one guess, the surest: a list of every piece's guess read as nonsense ("dark, yawn, dark, let know")
    const best = segments.filter((seg) => seg.length).sort((x, y) => y[0].p - x[0].p)[0]?.[0]?.english ?? "";
    // a pass shows no match bar: the top-3 rule passes a sign scored well below the top guess ("Match 1%")
    if (hit) { stop(); return grade(true, host.english(gloss)); }
    if (++tries < 2) {
      heard.replaceChildren(scoreBar(score), `That looked like “${best || "nothing I know"}”. Watch again and try once more.`);
      host.say(LINES.retry);
      practice.answered("retry");
      return;
    }
    stop();
    grade(false, `${host.english(gloss)} (we saw “${best || "?"}”)`, score);
  };
  return {
    el: h("div", { style: "display: contents" },
      h("div", { class: "ex-tag" }, svg(ICON.camera), "Your turn"),
      h("h2", { class: "prompt" }, `Sign “${host.english(gloss)}”`),
      h("div", { class: "perform" }, cam, avatarSlot(() => [gloss], false, host.english)),
      faceEl,
      heard),
    onShow: () => {
      practice.listener = { status: setStatus, recognized: onRecognized, face: (seen) => (faceEl.textContent = face(seen)),
        hands: (on) => (cam.dataset.hands = on ? "in" : "out") };
      practice.active = true;
      setStatus(practice.status);
      practice.start().catch(() => {});
    },
    onLeave: stop,
    left: { label: "Camera off for now", run: skip },
  };
}
