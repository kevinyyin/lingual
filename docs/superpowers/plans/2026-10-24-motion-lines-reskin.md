# Motion Lines Reskin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace lingual's Duolingo-like look with the motion-lines world across teaching mode, the translator and the iOS start screen, and give teaching mode's avatar a drawn hand trail.

**Architecture:** Mostly a rewrite of the two stylesheets (`teach.css`, `style.css`), plus small DOM changes in `home.ts` and `lessonView.ts` for the new marks (the course line's arrowhead, answer loops and strikes). The trail is a new pure module (`avatar/trail.ts`), fed once per frame by the stage (`stage.ts`, new `onFrame`) with palm positions the avatar projects itself (`rocketbox.ts`, new `palmScreen`). Teaching mode draws it on a second canvas that travels with the shared avatar canvas (`signer.ts`, new `mountSigner`).

**Tech Stack:** TypeScript, CSS, SVG, three.js, Vite, Vitest (jsdom) with @testing-library/dom; fonts from Google Fonts.

**Spec:** `docs/superpowers/specs/2026-10-24-motion-lines-reskin-design.md`; product truth in `PRODUCT.md`. The mockup to match is local and gitignored: `/Users/kevinyin/lingual/.superpowers/brainstorm/73049-1792818838/content/soft-directions.html`. Its `.motion` rules hold the reference values, and the still `soft-directions-all.jpg` beside it shows the right-hand column.

## Global Constraints

- Work in a git worktree on branch `motion-lines-reskin` (Task 1 makes it). Paths below are from its root.
- Colors, exact: paper `#F6F7FB`, surface `#FFFFFF`, wash `#E3E8FF`, ink `#232A63`, ink-2 `#565D8C`, guide `#AEB7E2`, outline = ink at 55%, go `#F2592A` (never text), go-ink `#C2410C`, rose `#E2466C` (never text), rose-ink `#B4234A`. No green. None of the old hexes (the `OLD` list in `palette.test.ts`).
- Fonts: Shantell Sans for headings, prompts, verdicts, the wordmark and short annotations. Atkinson Hyperlegible Next for everything else; options and captions never use Shantell. Link: `https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+Next:wght@400..800&family=Shantell+Sans:wght,BNCE,INFM@400..800,-100..100,0..100&display=swap`. The iOS start screen keeps the system font.
- Minimum sizes (read from 1 m): lesson prompt 28px, options 20px, verdict 26px, lesson step count and hearts 17px bold, the active course lesson's title 22px, translator caption 36px on the phone, translator status 18px. All other text 15px or more.
- Shapes: panel radius `30px 26px 32px 24px / 26px 32px 24px 30px` and tile radius `16px 20px 15px 21px / 19px 15px 21px 16px` (the mockup's); 2px outlines on choices. No zero-blur offset shadows (ledges); soft neutral shadows only on things that float; no cards inside cards.
- Marks: SVG, round caps and joins, 3-4px, in go, with `vector-effect: non-scaling-stroke` where the SVG is stretched. They draw in with stroke-dashoffset over 600-900 ms ease-out (`pathLength="100"`). No bounce or elastic easing anywhere. Under `prefers-reduced-motion: reduce`, marks appear complete.
- No SVG displacement filters on the avatar stage or on anything repeated in lists. Light only. The avatar's model, motion, face and shirt don't change.
- Copy: only the spec's changes (emoji removed, the home greeting dropped, "Unit N · x/y done" joined to the blurb under the unit title). Every other string stays word for word.
- DOM hooks that must survive: translator `#state`, `#caption`, `#plan span` (and `.on`), `#picker`, `#you-said`, `#dev`; the lesson's `[data-key]` buttons, the `.btn:not(.ghost)` lookups in `lessonView.ts`, and the "Recalibrate camera" button.
- After every task, from `app/web`: `npx tsc --noEmit -p . && npm test` passes. After Tasks 1, 6 and 7, `npm run build` too.
- Running it: the backend from the main checkout, `uv --directory /Users/kevinyin/lingual/app run asl-server` (:8000); Vite from the worktree, `npm --prefix app/web run dev -- --port 5173 --strictPort`. Check pages at 402×874 (phone) and 1280×800 (laptop).
- Commits: one line, `type(scope): what changed`, like the repo's history. Never commit `.claude/launch.json`, `app/web/ios/`, `.impeccable/`, or anything under `app/reports/` that isn't this work.

## Review Focus

1. The shared avatar canvas moves between the home hero, lessons, the Notes modal and Show me; its trail canvas must move with it and start empty. → Task 5, test "mountSigner moves both canvases into the new slot".
2. Fingerspelled words and caption-only signs leave no trail. → Task 5, tests "a fingerspelled word is never signing" and "a caption-only sign is never signing".
3. A finished course has no next lesson: no arrowhead, no play button, "You finished the course!". → Task 2, test "a finished course has no arrowhead and no play button".
4. Slow replay (the turtle button, speed 0.5) stretches the trail's fade instead of cutting it short. → Task 5, test "slow replay stretches the fade".
5. The longest strings at the 1 m sizes on a 402px phone ("Food & drink", "Where and when", "Unit review", "thank you", an order exercise's chips, a three-line translator caption) must not overflow or clip. → Task 4 Step 8 and Task 6 Step 5, a screenshot check with those strings.

---

### Task 1: The system: worktree, tokens, fonts, base components, the mark

**Files:**
- Create: worktree `../lingual-reskin` on branch `motion-lines-reskin`
- Move in: `PRODUCT.md`, the spec, this plan (now untracked in `/Users/kevinyin/lingual`)
- Modify: `app/web/teach.html`; `app/web/src/teach/teach.css` (`:root` through "top bar", the modal, the narrow-screen tab bar; delete the "a little irregularity" block); `app/web/src/teach/icons.ts`; `app/web/src/teach/home.ts` and `app/web/src/teach/lessonView.ts` (the `MASCOT` uses only)
- Test: `app/web/src/teach/icons.test.ts`

**Interfaces:**
- Produces: `ICON.mark` (SVG markup). `MASCOT` no longer exists. CSS custom properties for later tasks: `--paper --surface --wash --ink --ink-2 --guide --outline --go --go-ink --rose --rose-ink --display --text --ease --r-panel --r-tile --shadow-float`, and `@keyframes draw`.

- [ ] **Step 1: Check the calibration work has landed.** Run `git -C /Users/kevinyin/lingual log --oneline -1 -- app/web/src/teach/lessonView.test.ts`. Expected: one commit. If nothing prints, stop and ask the user: the spec forbids touching those files before that work is committed.
- [ ] **Step 2: Make the worktree** (superpowers:using-git-worktrees) from the tip of the branch holding that commit (`git -C /Users/kevinyin/lingual branch --contains <sha>`, normally `avatar-readability`): `git -C /Users/kevinyin/lingual worktree add ../lingual-reskin -b motion-lines-reskin <branch>`. In it: `npm --prefix app/web ci --ignore-scripts`, then link the gitignored assets: `for d in signs models wasm; do ln -s /Users/kevinyin/lingual/app/web/public/$d app/web/public/$d; done`. Move `PRODUCT.md`, the spec and this plan from `/Users/kevinyin/lingual` to the same paths here. From `app/web`, `npx tsc --noEmit -p . && npm test`: PASS.
- [ ] **Step 3: Commit the docs**: `git add PRODUCT.md docs/superpowers && git commit -m "docs: motion lines reskin design, plan and product brief"`
- [ ] **Step 4: Write the failing test** `app/web/src/teach/icons.test.ts`:

```ts
import { expect, test } from "vitest";
import { ICON } from "./icons";

const OLD = ["#ff9600", "#e07f00", "#ff4b4b", "#ea2b2b", "#ffd36b", "#e08a00", "#b36b00"];

test("no icon carries one of the old colors", () => {
  const all = Object.values(ICON).join("").toLowerCase();
  for (const hex of OLD) expect(all).not.toContain(hex);
});

test("the mark is the motion arrow: a start dot and a curved arrow in go", () => {
  expect(ICON.mark).toMatch(/<circle[^>]*fill="#F2592A"/);
  expect(ICON.mark).toMatch(/<path[^>]*stroke="#F2592A"/);
});
```

- [ ] **Step 5: Run it**: `npx vitest run src/teach/icons.test.ts`. Expected: FAIL (no `ICON.mark`; old hexes present).
- [ ] **Step 6: Change `icons.ts`.**
  - Add `mark: '<svg viewBox="0 0 32 32"><circle cx="7" cy="23" r="2.8" fill="#F2592A"/><path d="M7 23C9 13 17 7 26 9M20.6 4.4 26 9l-6.7 1.9" fill="none" stroke="#F2592A" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/></svg>'`.
  - The flame becomes fill `#F2592A`, stroke `#C2410C`; the heart becomes fill `#E2466C`, stroke `#B4234A`.
  - Delete `coin` and `MASCOT`.
  - Replace the `MASCOT` uses with `svg(ICON.mark)`: the logo's `.mark`, the empty state's `.big` and the profile's `.big-av` in `home.ts`; in `lessonView.ts`, `.done-mascot` becomes `svg(ICON.mark, "done-mark")`.
  - Delete the hero's `.hero-mascot` span.
- [ ] **Step 7: Run it**: same command. Expected: PASS.
- [ ] **Step 8: `teach.html`.** Swap the font link for the one in Global Constraints. Make this comment the body's first child:

```html
<!--
THESIS: lingual is drawn the way a sign moves: one line for the course, arrows for motion, a teacher's circle for a right answer. It refuses the candy path, ledge buttons and colored feedback sheets of gamified language apps.
OWN-WORLD: cool paper #F6F7FB, indigo ink #232A63, periwinkle wash #E3E8FF, one tangerine for motion #F2592A, rose for hearts; Shantell Sans to display, Atkinson Hyperlegible Next to read; round 3-4px strokes, soft uneven corners, no ledges.
STORY: a kid and a parent see what to learn next, watch the avatar sign with its hand's path drawn in, answer, and see the answer circled.
FIRST VIEWPORT: a wash hero with "Next up: <lesson>" underlined in tangerine, Continue, the avatar on the right; below, the unit heading and the course line ending in an arrowhead at the next lesson.
FORM: motion lines (ASL dictionary notation), pinned by the user on 2026-10-23; no roll, no seed key.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
-->
```

- [ ] **Step 9: Rewrite the base of `teach.css`.**
  - **Tokens:** the colors, the two radii, `--display: "Shantell Sans"`, `--text: "Atkinson Hyperlegible Next"`, `--ease: cubic-bezier(.2, .8, .2, 1)`, and `--shadow-float: 0 10px 18px -10px rgb(0 0 0 / .45)`.
  - **Base:**
    - the body is paper with Atkinson 16px/1.5 in ink; `h1`-`h3` and `.prompt` use `--display`;
    - `::selection` is wash on ink; `:focus-visible` is a 3px ink outline at 2px offset; the caret is go.
  - **Buttons:**
    - `.btn` is an ink pill: min-height 52px, 18px/700 white, `--shadow-float`, `:active` scales to .97;
    - `.btn.ghost` has a 2px outline on a transparent fill; `.btn.text` is go-ink text;
    - `:disabled` is wash with ink-2 text and no shadow;
    - delete `.btn.green`, `.btn.red` and `.btn.sun`.
  - **Panels and meters:** `.card` is a surface panel with `--r-panel`, a 1.5px ink-12% border and no shadow. `.meter` is a 4px track in ink 10%, filled go for every tone.
  - **Header:**
    - `.logo` is the mark plus "lingual" in Shantell 26px/800;
    - the stats are plain icon and number, 17px/800: the flame is ink-2 when off and go when on, the heart rose;
    - on the laptop, `.tab[aria-current]` is ink with a go loop under it;
    - the phone's tab bar sits on paper with a 2px ink-18% rule on top, and the current icon is circled in go (the mockup's `.m-tab .loop` path, as a CSS background).
  - **Modal and toast:** `.modal` is paper with `--r-panel` and a Shantell `h2`. `.toast` is an ink pill.
  - **Motion:** add `@keyframes draw { from { stroke-dashoffset: 100 } }`, and a reduced-motion block that turns animation off on marks.
  - Remove every `rotate()` tilt.
- [ ] **Step 10: Verify.** From `app/web`: `npx tsc --noEmit -p . && npm test && npm run build`, then `grep -c "THESIS:" dist/teach.html`. Expected: 1. Screenshot the teaching home at 402×874 and 1280×800. The header, tabs and buttons should match the mockup's right column; the course itself stays half-styled until Task 2.
- [ ] **Step 11: Commit**: `git add app/web/teach.html app/web/src/teach && git commit -m "design(teach): motion lines tokens, fonts, buttons, header and the mark"`

### Task 2: Course home

**Files:**
- Create: `app/web/src/teach/marks.ts`
- Modify: `app/web/src/teach/home.ts` (`hero`, `unitSection`, `stop`; new exported `coursePath`; delete `STICKERS`, `DECO` and `greeting`), `app/web/src/teach/curriculum.ts` (drop `Unit.color`), `app/web/src/teach/teach.css` (the "course" and "side cards" sections)
- Test: `app/web/src/teach/home.test.ts`

**Interfaces:**
- Consumes: `ICON.mark`, the CSS tokens (Task 1).
- Produces:
  - `coursePath(pts: { x: number; y: number }[], upTo: number): string` in `home.ts`.
  - `ARROWHEAD: string` in `marks.ts`: `'<svg viewBox="0 0 24 24"><path d="M5 8l7 8 7-8" fill="none" stroke="#F2592A" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/></svg>'`.
  - Hero DOM: `.hero > .hero-text` and, when there is a next lesson, `.avatar-slot.hero-avatar` containing `button.round.hero-play` (aria-label "Watch the first sign").
  - `unit(id, title, blurb, tip, groups)` loses its color argument.

- [ ] **Step 1: Write the failing test** `app/web/src/teach/home.test.ts`:

```ts
// @vitest-environment jsdom
import { expect, test, vi } from "vitest";
vi.mock("./signer", () => ({
  avatarCanvas: document.createElement("canvas"),
  loadStage: () => new Promise(() => {}),
  playSigns: vi.fn(() => Promise.resolve()),
}));
import { ALL_LESSONS } from "./curriculum";
import { coursePath, renderHome, type HomeCtx } from "./home";
import { fresh } from "./progress";
import { playSigns } from "./signer";

const EMOJI = /\p{Extended_Pictographic}/u;
const ctx = (done: string[] = []): HomeCtx => {
  const p = fresh();
  for (const id of done) p.done[id] = 1;
  return { p, settings: { sound: true, camera: true }, english: (g) => g.toLowerCase(), update() {}, updateSettings() {},
    start: vi.fn(), startPractice() {}, words: [], showMe() {}, reset() {} };
};

test("the course line leads in from the top of the unit", () => {
  expect(coursePath([{ x: 50, y: 64 }, { x: 70, y: 192 }], 1)).toMatch(/^M50 0 L50 64 C/);
});

test("only the active lesson carries the arrowhead", () => {
  renderHome(document.body, "learn", ctx());
  const heads = document.querySelectorAll(".stop .arrowhead");
  expect(heads).toHaveLength(1);
  expect(heads[0].closest(".stop")!.classList).toContain("active");
});

test("a finished course has no arrowhead and no play button", () => {
  renderHome(document.body, "learn", ctx(ALL_LESSONS.map((l) => l.id)));
  expect(document.querySelector(".arrowhead")).toBeNull();
  expect(document.querySelector(".hero-play")).toBeNull();
  expect(document.querySelector(".hero")!.textContent).toContain("You finished the course!");
});

test("the hero's play button signs the next lesson's first sign", () => {
  renderHome(document.body, "learn", ctx());
  document.querySelector<HTMLButtonElement>(".hero-play")!.click();
  expect(playSigns).toHaveBeenCalledWith(["HELLO"]);
});

test("the course has no greeting and no emoji, and units have no banners", () => {
  renderHome(document.body, "learn", ctx());
  expect(document.body.textContent).not.toMatch(/Good (morning|afternoon|evening)/);
  expect(document.body.textContent).not.toMatch(EMOJI);
  expect(document.querySelector(".unit-banner")).toBeNull();
  expect(document.querySelector(".unit-head p")!.textContent).toBe("Unit 1 · 0/4 done · Say hello, thank someone, introduce yourself");
});
```

- [ ] **Step 2: Run it**: `npx vitest run src/teach/home.test.ts`. Expected: FAIL (`coursePath` isn't exported, there are no `.arrowhead` or `.hero-play` elements, and the greeting is present).
- [ ] **Step 3: Implement.**
  - **`coursePath`:** `M{x0} 0 L{x0} {y0}`, then today's segment per stop up to `upTo`: `C{prev.x} {prev.y + STEP * .55} {p.x} {p.y - STEP * .55} {p.x} {p.y}`.
  - **`unitSection`:**
    - `div.unit-head` holds an `h2` (the title), a `p` (`Unit ${ui + 1} · ${done}/${n} done · ${u.blurb}`) and `button.btn.text.ub-notes` (the notes icon and "Notes");
    - drop the `--uc` style, the banner, the sticker and the deco;
    - the rest line is `coursePath(pts, n - 1)`;
    - the done line is `coursePath(pts, lit)` with `pathLength="100"`, drawn whenever the unit's first stop isn't locked.
  - **`stop`:**
    - the node holds `svg(ICON.check)` when done, `svg(ARROWHEAD, "arrowhead")` when active, and nothing when locked (the review stop keeps the trophy);
    - drop `--tilt`.
  - **`hero`:**
    - drop the greeting, the blobs and the ring;
    - keep the `h1` ("Next up: " plus a span with the title), the `p` ("Unit N · title"), Continue, and add `div.goal` (`${Math.min(xp, goal)} of ${goal} XP today` and a `.meter`);
    - when there is a next lesson, add the avatar slot: `slot.prepend(avatarCanvas)`, a "Loading signer…" line, and `.hero-play` calling `playSigns([next.l!.signs[0]])`;
    - load after paint: `requestAnimationFrame(() => setTimeout(() => void loadStage().then(drop, drop)))`, where `drop` removes the loading line.
  - **`curriculum.ts`:** remove `color` from `Unit`, from `unit()` and from its five calls.
- [ ] **Step 4: Run it**: `npx tsc --noEmit -p . && npx vitest run src/teach/home.test.ts`. Expected: PASS.
- [ ] **Step 5: Style the course and the side cards.**
  - **Hero:** a wash panel with `--r-panel`; text on the left and the avatar slot on the right (about 45%) on both phone and laptop. The title span gets a hand-drawn go swoosh under it (an SVG background, like the mockup's `.swoosh`). The `.goal` text is 15px ink-2.
  - **Units:** `.unit-head h2` is Shantell 26px; its `p` is 15px ink-2.
  - **Course line:** `.fl-rest` is guide dots (2px, round, `stroke-dasharray: .1 9`, non-scaling). `.fl-done` is a 4px go stroke that draws in.
  - **Stops:**
    - a 44px tap area around a small mark: done is a 16px ink dot with a white check; active is the 28px arrowhead, tip on the stop; locked is an 18px hollow guide circle;
    - the labels' `b` is Shantell (22px when active), and their `span` is 15px ink-2.
  - **Side cards:**
    - the streak shows a Shantell 34px number and a week of 28px circles in a guide outline: done days get a go check, today an ink ring;
    - each challenge row is an ink-2 icon, a 15px title and value, and the `.meter`.
- [ ] **Step 6: Verify.** Screenshot the home at 402×874 and 1280×800 with a fresh state, and again with `u1-1` done. In the console: change `done` in the stored progress, then reload. Compare with the mockup: the line leads in from the top, the done stretch ends in the arrowhead, and the dots run on.
- [ ] **Step 7: Commit**: `git commit -am "design(teach): course home as one drawn line, unit headings, hero with the avatar"`. Add `marks.ts` and `home.test.ts` first.

### Task 3: Practice, Challenges, Profile

**Files:**
- Modify: `app/web/src/teach/teach.css` (the "other tabs" section)
- Test: `app/web/src/teach/home.test.ts` (adds tests)

**Interfaces:**
- Consumes: the tokens (Task 1), `ICON.mark` in `.big-av` and `.empty .big` (Task 1).

This task is CSS only. Its check is the screenshots and Task 4's `palette.test.ts`, which then covers the whole of `teach.css`. The two guards below already pass after Task 1; they keep emoji and the picture tile from coming back.

- [ ] **Step 1: Add the guards** (append to `home.test.ts`):

```ts
import { TAB_IDS } from "./home";

test.each(TAB_IDS)("no emoji on the %s tab", (tab) => {
  renderHome(document.body, tab, ctx());
  expect(document.body.textContent).not.toMatch(EMOJI);
});

test("the profile shows the mark", () => {
  renderHome(document.body, "profile", ctx());
  expect(document.querySelector(".big-av svg")).not.toBeNull();
});
```

- [ ] **Step 2: Run them**: `npx vitest run src/teach/home.test.ts`. Expected: PASS.
- [ ] **Step 3: Style the other tabs.**
  - **Page head:** Shantell 32px.
  - **Practice:**
    - the hub cards become wash panels with `--r-panel`: an ink icon on a surface circle and no purple or blue variants; disabled ones sit at 55% opacity;
    - the show-me input is a 52px surface field with a 2px outline and `--r-tile`;
    - sign tiles are surface with a 1.5px ink-12% border, and strength shows as four short go ticks (guide when off).
  - **Profile:**
    - `.big-av` is a surface circle with the 56px mark; the stat tiles lose their colors (a Shantell 30px number and a 15px label);
    - the chart line is ink 2.5px with go dots and an ink-8% grid;
    - achievement icons are ink-2 on surface, and go once earned.
  - **Settings:** `.seg [aria-pressed=true]` is ink with white text; `.switch[aria-checked=true]` is ink.
  - **Empty state:** the 56px mark.
- [ ] **Step 4: Run**: `npx tsc --noEmit -p . && npm test`. Expected: PASS.
- [ ] **Step 5: Verify.** Screenshot Practice, Challenges and Profile at 402×874 and 1280×800. No old colors, and no cards inside cards.
- [ ] **Step 6: Commit**: `git commit -am "design(teach): practice, challenges and profile in the drawn style"`

### Task 4: Lesson, answer marks and results

**Files:**
- Modify: `app/web/src/teach/marks.ts` (add `markSvg`); `app/web/src/teach/lessonView.ts` (the `watch` and `pick-sign` `check()`, `grade()`, `accuracyRing`, `confetti`, `streakScreen`); `app/web/src/teach/teach.css` (the "lesson" and "results" sections, and the phone rules)
- Test: `app/web/src/teach/lessonMarks.test.ts`, `app/web/src/palette.test.ts`

**Interfaces:**
- Produces:
  - `markSvg(kind: "loop" | "strike"): SVGSVGElement`: class `mark loop` or `mark strike`, `aria-hidden="true"`, `viewBox="0 0 100 100"`, `preserveAspectRatio="none"`, one path with `pathLength="100"`. The loop path is `M10 64C0 34 28 6 52 6C80 6 99 22 97 50C95 80 70 96 46 95C20 94 2 78 5 52C7 34 20 20 36 14`; the strike path is `M5 62L28 40 24 62 50 38 46 62 72 40 68 60 95 42`.
  - An internal `markAnswer(chosen: HTMLElement, right: HTMLElement): void` in `lessonView.ts`.

- [ ] **Step 1: Write the failing tests.** `app/web/src/teach/lessonMarks.test.ts`:

```ts
// @vitest-environment jsdom
import { beforeEach, expect, test, vi } from "vitest";
import { getByRole } from "@testing-library/dom";
vi.mock("../ws", () => ({ connect: () => ({ send() {}, buffered: () => 0 }) }));
vi.mock("../camera", () => ({ startCamera: () => new Promise(() => {}) }));
vi.mock("../avatar", () => ({ loadAvatar: () => new Promise(() => {}) }));
vi.mock("../avatar/stage", () => ({ createStage: vi.fn() }));
import { runLesson, type LessonHost } from "./lessonView";

const host: LessonHost = { english: (g) => g.toLowerCase(), hearts: () => 5, loseHeart() {}, sound: () => false, say() {},
  cameraOn: () => false, disableCamera() {}, quit() {}, finish: () => { throw new Error("not reached"); } };
const opt = (word: string) => getByRole(document.body, "button", { name: new RegExp(`^\\d${word}$`) });
const answer = (word: string) => { opt(word).click(); getByRole(document.body, "button", { name: "Check" }).click(); };

beforeEach(() => runLesson(document.body, [{ kind: "watch", gloss: "HELLO", options: ["MEET", "GOODBYE", "NICE", "HELLO"] }], host));

test("a right answer is circled and nothing is struck", () => {
  answer("hello");
  expect(opt("hello").querySelector(".mark.loop")).not.toBeNull();
  expect(document.querySelector(".mark.strike")).toBeNull();
  expect(document.querySelector(".verdict h3")!.textContent).toMatch(/^(Spot on\.|That's it!|Nailed it\.|Exactly right\.|Got it\.)$/);
});

test("a wrong answer strikes the choice and circles the right one", () => {
  answer("meet");
  expect(opt("meet").querySelector(".mark.strike")).not.toBeNull();
  expect(opt("hello").querySelector(".mark.loop")).not.toBeNull();
  expect(document.querySelector(".verdict")!.textContent).toContain("Not quite.");
  expect(document.querySelector(".verdict")!.textContent).toContain("Answer: hello");
});

test("Continue is the plain ink button, not a colored one", () => {
  answer("meet");
  expect(document.querySelector(".lesson-foot .btn.red, .lesson-foot .btn.green")).toBeNull();
});
```

  `app/web/src/palette.test.ts` (Task 6 adds files to `FILES`):

```ts
import { readFileSync } from "node:fs";
import { expect, test } from "vitest";

const OLD = ["#1cc6d6", "#10a5b4", "#8de4ec", "#0b98a6", "#dcf8fb", "#1cb0f6", "#ddf4ff", "#1899d6", "#ce82ff", "#f4e4ff",
  "#a560d4", "#ff4b4b", "#ea2b2b", "#ffdfe0", "#ffc800", "#ff9600", "#fff4cc", "#cd7900", "#4b4b4b", "#777777",
  "#afafaf", "#e5e5e5", "#58cc02"];
const FILES = ["teach/teach.css"];
const read = (f: string) => readFileSync(new URL(`./${f}`, import.meta.url), "utf8");

test.each(FILES)("%s uses none of the old colors or fonts", (f) => {
  const s = read(f).toLowerCase();
  for (const hex of OLD) expect(s).not.toContain(hex);
  expect(s).not.toMatch(/bricolage|dm sans/);
});

test.each(FILES)("%s has no ledge shadows", (f) => {
  expect(read(f)).not.toMatch(/box-shadow:[^;]*\b0(px)?\s+\d+(\.\d+)?px\s+0(px)?\s/);
});

test.each(FILES)("%s has no bounce easing", (f) => {
  for (const [, args] of read(f).matchAll(/cubic-bezier\(([^)]*)\)/g)) {
    const [, y1, , y2] = args.split(",").map(Number);
    for (const y of [y1, y2]) { expect(y).toBeGreaterThanOrEqual(0); expect(y).toBeLessThanOrEqual(1); }
  }
});
```

- [ ] **Step 2: Run it**: `npx vitest run src/teach/lessonMarks.test.ts src/palette.test.ts`. Expected: FAIL. There are no marks yet; Continue is `.btn.green` or `.btn.red`; `teach.css` still has old hexes and the ledges of its unrewritten sections.
- [ ] **Step 3: Implement.**
  - **`markSvg`** in `marks.ts`.
  - **`markAnswer`:**
    - when `chosen === right`: the `right` class and a loop;
    - otherwise: the `wrong` class and a strike on `chosen`, plus the `right` class and a loop on `right`;
    - the `watch` and `pick-sign` `check()`s call it, replacing their class lines.
  - **`grade()`:** Continue's class becomes `"btn"`.
  - **`accuracyRing`:** its circle becomes a loop path with `pathLength="100"` and `stroke-dasharray: 100`, its dashoffset going from 100 to `100 - pct`.
  - **`confetti`:** 20 strokes in `#F2592A`, `#232A63` and `#E2466C`, bursting from the card's center (a random angle and distance per stroke, 900 ms ease-out), removed after 1200 ms.
  - **`streakScreen`:** the number's `animate` easing becomes `cubic-bezier(.2, .8, .2, 1)`, with no overshoot.
- [ ] **Step 4: Style the lesson and the results.**
  - **Lesson header:** the progress bar is a guide dotted track with a go fill that ends in a CSS arrowhead. "n/16" and the hearts are 17px/800.
  - **Prompt:** Shantell 28px (`.prompt`).
  - **Stage:** `.avatar-slot` is wash with `--r-panel` and no border. The round controls are white with `--shadow-float`.
  - **Options:**
    - `.opt` is surface with `--r-tile`, a 2px outline and 20px/700 text, with the key in Shantell; `[aria-pressed=true]` is wash with an ink outline;
    - `.mark` sits absolutely at `inset: -10px -12px`, with `overflow: visible` and `pointer-events: none`; its path is `vector-effect: non-scaling-stroke` at 3.5px and draws in;
    - the loop is go and the strike rose.
  - **Footer:**
    - it stays paper (no colored card, for `right` and `wrong` too);
    - `.verdict h3` is Shantell 26px, ink when right and rose-ink when wrong; the badge icon sits inline at 22px in go or rose-ink, with no colored square.
  - **Tags:** `.ex-tag` and `.ex-tag.retry` are go-ink annotations (an icon and text, no pill).
  - **Other exercises:**
    - `.chip` is surface with a 1.5px outline and `--r-tile`; `.answer-line` is a 2px guide baseline;
    - `.match` uses the option styles, and a matched pair shows a small go check (CSS on `.right`) before it fades out;
    - `.cam-slot` keeps its status edge, in ink when ready and go while signing;
    - the "Recalibrate camera" `.btn.text` stays as it is.
  - **Results:**
    - `.done-card` loses its box (no background, border or shadow) and its title is Shantell 32px;
    - `.done-mark` is 64px;
    - `.chip-stat` is plain ink-2 icon and text;
    - `.streak-badge` is the flame with a Shantell number, and the week row matches the side card.
- [ ] **Step 5: Run**: `npx tsc --noEmit -p . && npm test`. Expected: PASS, including `lessonView.test.ts`.
- [ ] **Step 6: Verify each exercise kind** at 402×874 and 1280×800: intro, watch (right and wrong), pick-sign, match, order, perform, the results screen, the streak screen and out of hearts. Freeze the draw-in animations to check the marks' final states.
- [ ] **Step 7: Check the 1 m sizes** in the computed styles: prompt ≥ 28px, options ≥ 20px, verdict ≥ 26px, count and hearts ≥ 17px.
- [ ] **Step 8: Check overflow (Review Focus 5).** On the phone, look at Unit 4's "Food & drink" lessons and the order exercise's chips, at the widest option words ("thank you", "goodbye"). Nothing may wrap badly or clip.
- [ ] **Step 9: Commit**: `git commit -am "design(teach): drawn answer marks, plain feedback bar, results without confetti boxes"`. Add the two test files and `marks.ts` first.

### Task 5: The hand trail

**Files:**
- Create: `app/web/src/avatar/trail.ts`
- Modify: `app/web/src/avatar/stage.ts` (`createStage`'s options and `StageFrame`); `app/web/src/avatar/rocketbox.ts` (`Avatar.palmScreen` and its implementation in the returned object, after `framePoints`); `app/web/src/teach/signer.ts` (`trailCanvas`, `mountSigner`, `loadStage`); the four `slot.prepend(avatarCanvas)` call sites (`lessonView.ts` `avatarSlot`; `home.ts` `notes`, `showMeCard`, `hero`); `teach.css` (`.trail-canvas`)
- Test: `app/web/src/avatar/trail.test.ts`, `app/web/src/avatar/stage.test.ts` (add), `app/web/src/teach/signer.test.ts`, and `home.test.ts` (its mock gains `mountSigner: vi.fn()`)

**Interfaces:**
- Produces, in `trail.ts`:
  - `type Pt = [number, number]`.
  - `class TrailPath`:
    - `constructor(o: { height: number; fadeMs?: number; minSpeed?: number; holdMs?: number; stillMs?: number; still?: boolean })`, defaults 700, 0.15, 600, 1500, false; `minSpeed` is in canvas heights per second;
    - `height: number`;
    - `push(now: number, p: Pt | null, signing: boolean, speed = 1): void`;
    - `visible(now: number): { p: Pt; alpha: number }[]`;
    - `end(now: number): { tip: Pt; dir: Pt; alpha: number } | null`;
    - `clear(): void`.
  - `arrowHead(tip: Pt, dir: Pt, len: number, spreadDeg = 28): [Pt, Pt]`.
  - `ndcToCanvas(ndc: { x: number; y: number; z: number }, w: number, h: number): Pt | null`.
  - `attachTrail(canvas: HTMLCanvasElement, palms: () => Record<"L" | "R", Pt | null> | undefined, opts: { still: boolean }): { frame(f: StageFrame): void; clear(): void }`.
- Produces, in `stage.ts`: `interface StageFrame { now: number; signing: boolean; speed: number }` and `createStage(avatar: Avatar, opts?: { onFrame?: (f: StageFrame) => void }): Promise<Stage>`.
- Produces, in `rocketbox.ts`: `Avatar.palmScreen?(): Record<"L" | "R", [number, number] | null>`. Each palm center is halfway between the wrist (`arm.hand.bone`) and the middle knuckle (`arm.fingers[2][0].bone`), as in `avatarPalm`, projected with `camera` and passed through `ndcToCanvas(v, canvas.clientWidth, canvas.clientHeight)`.
- Produces, in `signer.ts`: `trailCanvas: HTMLCanvasElement` (class `trail-canvas`, `aria-hidden`) and `mountSigner(slot: HTMLElement): void`.

- [ ] **Step 1: Baseline, before any trail code.** On Task 4's commit, in the laptop browser, open a lesson's intro sign. While replaying "hello" five times, record the 95th-percentile `requestAnimationFrame` interval (collect the timestamps with a console snippet). Note it for Step 12.
- [ ] **Step 2: Write the failing test** `app/web/src/avatar/trail.test.ts`:

```ts
import { expect, test } from "vitest";
import { arrowHead, ndcToCanvas, TrailPath } from "./trail";

const H = 300; // canvas height: speeds are canvas heights per second
const move = (t: TrailPath, signing = true, speed = 1) => { for (let i = 0; i <= 10; i++) t.push(i * 16, [100 + i * 6, 100], signing, speed); };

test("a still hand leaves nothing", () => {
  const t = new TrailPath({ height: H });
  for (let i = 0; i < 30; i++) t.push(i * 16, [100, 100], true);
  expect(t.visible(480)).toEqual([]);
});

test("a moving hand leaves a trail that fades out over 700 ms", () => {
  const t = new TrailPath({ height: H });
  move(t);
  const v = t.visible(160);
  expect(v.length).toBeGreaterThan(5);
  expect(v[0].alpha).toBeLessThan(v.at(-1)!.alpha);
  expect(t.visible(160 + 700)).toEqual([]);
});

test("slow replay stretches the fade", () => {
  const t = new TrailPath({ height: H });
  move(t, true, 0.5);
  expect(t.visible(160 + 700).length).toBeGreaterThan(0);
  expect(t.visible(160 + 1400)).toEqual([]);
});

test("samples outside a sign are ignored", () => {
  const t = new TrailPath({ height: H });
  move(t, false);
  expect(t.visible(160)).toEqual([]);
});

test("when a sign ends, its last stroke holds 600 ms with an arrowhead along it, then fades", () => {
  const t = new TrailPath({ height: H });
  move(t);
  t.push(176, [166, 100], false);
  expect(t.end(176)).toMatchObject({ tip: [160, 100] });
  expect(t.end(176)!.dir[0]).toBeCloseTo(1);
  expect(t.visible(176 + 600).length).toBe(t.visible(176).length);
  expect(t.end(176 + 600 + 700)).toBeNull();
});

test("a new sign clears the last one's arrowhead", () => {
  const t = new TrailPath({ height: H });
  move(t);
  t.push(176, [166, 100], false);
  t.push(400, [100, 100], true);
  expect(t.end(400)).toBeNull();
});

test("still mode (reduced motion) draws nothing while signing, then the whole path for 1.5 s", () => {
  const t = new TrailPath({ height: H, still: true });
  move(t);
  expect(t.visible(160)).toEqual([]);
  t.push(176, [166, 100], false);
  expect(t.visible(176)).toHaveLength(10); // the first sample has no speed yet
  expect(t.visible(176).every((s) => s.alpha === 1)).toBe(true);
  expect(t.visible(176 + 1500)).toEqual([]);
});

test("arrowHead's arms sit 28 degrees either side of the backward direction", () => {
  const [a, b] = arrowHead([0, 0], [1, 0], 10);
  expect(a[0]).toBeCloseTo(-10 * Math.cos((28 * Math.PI) / 180));
  expect(b[0]).toBeCloseTo(a[0]);
  expect(a[1]).toBeCloseTo(-b[1]);
});

test("ndcToCanvas maps clip space to canvas pixels and drops points outside the depth range", () => {
  expect(ndcToCanvas({ x: 0, y: 0, z: 0.5 }, 400, 300)).toEqual([200, 150]);
  expect(ndcToCanvas({ x: -1, y: 1, z: 0.5 }, 400, 300)).toEqual([0, 0]);
  expect(ndcToCanvas({ x: 0, y: 0, z: 1.2 }, 400, 300)).toBeNull();
});
```

- [ ] **Step 3: Run it**: `npx vitest run src/avatar/trail.test.ts`. Expected: FAIL (no module).
- [ ] **Step 4: Implement `trail.ts`.**
  - **Speed:** a sample counts when the distance from the previous push, divided by `height` and by the seconds elapsed, is ≥ `minSpeed`. The first push has no speed.
  - **Fade:** the window is `fadeMs / speed`.
  - **Sign end:** on the first non-signing push after signing, freeze the clock for `holdMs`, then let it run again. `end()` returns the last signing sample with its last segment's unit direction, at the newest sample's alpha.
  - **New sign:** a signing push after an end clears everything.
  - **Still mode:** keep every counted sample of the sign. `visible` returns nothing while signing, then all of them at alpha 1 for `stillMs`; `end()` returns the arrowhead at alpha 1 for the same time.
  - **Storage:** samples live in a fixed-size ring buffer per hand (1024 is plenty for one sign), so nothing is allocated per frame.
  - **Drawing (`attachTrail`):**
    - one `TrailPath` per hand, `height` taken from `canvas.clientHeight`;
    - nothing is drawn while the canvas isn't on the page (`!canvas.isConnected`), as the avatar's render loop already does;
    - the canvas is sized to `clientWidth` × `clientHeight` × `min(devicePixelRatio, 2)`;
    - a 4 CSS px go stroke, round, one segment per pair at the newer sample's alpha; the arrowhead from `end()` with 12px arms;
    - its own `requestAnimationFrame` loop runs while anything is visible and stops after;
    - `clear()` empties both hands and the canvas.
- [ ] **Step 5: Run it**: same command. Expected: PASS.
- [ ] **Step 6: Write the failing stage tests** (append to `stage.test.ts`). In a `beforeEach` for these tests, stub `fetch`: `/vocab` lists HELLO with `avatar_ok: true`; `/signs/HELLO.json` and `/letters/A.json` serve a 20-frame clip built like the "each clip is cleaned once" test's frames; everything else, `NOCLIP` included, gets a 404.

```ts
const signingFrames = async (tokens: SignPlan["tokens"]) => {
  const frames: { signing: boolean; speed: number }[] = [];
  const stage = await createStage({ applyFrame() {}, applyFace() {} }, { onFrame: (f) => frames.push(f) });
  const done = stage.play({ ...plan, tokens }, () => {}, 0.5);
  await vi.advanceTimersByTimeAsync(8000);
  await done;
  return frames;
};

test("onFrame says signing only during a sign's clip, not the blends around it", async () => {
  const f = await signingFrames([{ kind: "sign", gloss: "HELLO" }]);
  expect(f.some((x) => x.signing)).toBe(true);
  expect(f[0].signing).toBe(false);
  expect(f.at(-1)!.signing).toBe(false);
  expect(f.every((x) => x.speed === 0.5)).toBe(true);
});

test("a fingerspelled word is never signing", async () => {
  expect((await signingFrames([{ kind: "spell", word: "A" }])).some((x) => x.signing)).toBe(false);
});

test("a caption-only sign is never signing", async () => {
  expect((await signingFrames([{ kind: "sign", gloss: "NOCLIP" }])).some((x) => x.signing)).toBe(false);
});
```

- [ ] **Step 7: Implement `onFrame` in `createStage`.** In `tick`, after `applyFace`, call `opts?.onFrame?.({ now: performance.now(), signing: it?.kind === "clip" && !it.letter, speed })`. After the promise resolves, send one more frame with `signing: false`. Run `npx vitest run src/avatar/stage.test.ts`. Expected: PASS.
- [ ] **Step 8: Implement `palmScreen`** in `rocketbox.ts` per the Interfaces block (import `ndcToCanvas` from `./trail`). It has no unit test: the GLB doesn't load in jsdom. Step 11 checks it in the browser.
- [ ] **Step 9: Write the failing test** `app/web/src/teach/signer.test.ts`. Mock `../ws`, `../camera`, `../avatar` and `../avatar/stage` as `lessonView.test.ts` does.

```ts
test("mountSigner moves both canvases into the new slot", async () => {
  const { avatarCanvas, mountSigner, trailCanvas } = await import("./signer");
  const a = document.createElement("div"), b = document.createElement("div");
  mountSigner(a);
  mountSigner(b);
  expect([...b.children].slice(0, 2)).toEqual([avatarCanvas, trailCanvas]);
  expect(a.children).toHaveLength(0);
});
```

- [ ] **Step 10: Implement the wiring in `signer.ts`.**
  - `mountSigner(slot)` calls `trail?.clear()`, then `slot.prepend(avatarCanvas, trailCanvas)`.
  - `loadStage` resolves the avatar, then creates `trail = attachTrail(trailCanvas, () => avatar.palmScreen?.(), { still: matchMedia("(prefers-reduced-motion: reduce)").matches })`, then `createStage(avatar, { onFrame: (f) => trail.frame(f) })`.
  - Replace the four `slot.prepend(avatarCanvas)` calls with `mountSigner(slot)`, and add `mountSigner: vi.fn()` to `home.test.ts`'s mock.
  - Style `.trail-canvas`: absolute, `inset: 0`, 100% wide and high, `pointer-events: none`, and below `.controls`.
- [ ] **Step 11: Run and look.** `npx tsc --noEmit -p . && npm test`: PASS. Then, in the browser, play the intro "hello" and freeze it mid-sign (wrap `requestAnimationFrame` and `performance.now` in the console). The trail must follow the palm. Also check the slow replay, the Notes modal and Show me with a fingerspelled name: there should be no trail on the letters.
- [ ] **Step 12: Measure after.** Repeat Step 1's measurement. Put both 95th-percentile numbers in the commit body.
- [ ] **Step 13: Commit**: `git commit -am "feat(avatar): the moving hand leaves a drawn trail in teaching mode"`. Add the new files first.
- [ ] **Step 14: First phone session, with the user.** Reload the app on the iPhone, following the README's tunnel steps. The user judges by eye:
  - **(a)** Is the avatar as smooth as before with the trail on?
  - **(b)** Does the home avatar visibly delay the home screen or stutter?
  - If (b) is yes, remove the hero avatar and its test, and commit `design(teach): drop the home avatar after the phone check`.
  - Write both answers into the next commit's body.

### Task 6: Translator and the iOS start screen

**Files:**
- Modify: `app/web/src/style.css`; `app/web/index.html` (fonts, the mark in place of 🤟; the title stays "Two-Way ASL"); `app/web/shell/index.html` (colors only)
- Test: `app/web/src/palette.test.ts` (extend `FILES`)

**Interfaces:**
- Consumes: the token values (Global Constraints). `style.css` declares its own `:root` with the same names; the pages don't share a stylesheet.

- [ ] **Step 1: Write the failing tests.** Set `FILES` to `["teach/teach.css", "style.css", "../index.html", "../teach.html", "../shell/index.html"]`, and add:

```ts
test("the translator page has no emoji", () => {
  expect(read("../index.html")).not.toMatch(/\p{Extended_Pictographic}/u);
});
```

- [ ] **Step 2: Run it**: `npx vitest run src/palette.test.ts`. Expected: FAIL on `style.css`, `index.html` and the shell (old hexes, fonts, ledges, the emoji).
- [ ] **Step 3: Implement.**
  - **`style.css`:**
    - the tokens; the header (the mark, "Two-Way ASL", "Translator");
    - `.badge` is a pill with a dot at 18px or more;
    - five status families keep today's variable names (`--ready-*`, `--in-*`, `--out-*`, `--wait-*`, `--err-*`). Choose the values: distinct hues, no green, text on its background ≥ 4.5:1, the edge ≥ 3:1 against wash. Suggested families: in = indigo-blue, out = tangerine (go-ink text), wait = amber, error = rose (rose-ink text);
    - `#avatar-panel` is wash with `--r-panel`, and its status edge stays;
    - `#caption` is Atkinson 700, `2.25rem` on the phone and `clamp(2rem, 2.6vw, 2.8rem)` on the laptop;
    - `.plan span` is surface with an outline; `.plan span.on` is ink with white text;
    - the picker buttons are surface tiles with keys in ink circles, and "None of these" is a text button;
    - the cards are surface panels, the dev form uses the inputs from Task 1, and no trail;
    - keep every `body:has(#state…)` edge rule and the phone layout.
  - **`index.html`:** the font link, and `<span class="mark">` holding the mark's SVG inline.
  - **The shell:** paper background, ink text, an ink button, a rose-ink error line.
- [ ] **Step 4: Run**: `npx tsc --noEmit -p . && npm test && npm run build`. Expected: PASS, `ui.test.ts` included.
- [ ] **Step 5: Verify** at 402×874 and 1280×800. Set `#state`'s `data-state` from the console to each of the seven states, and check the badge and the stage edge. Fill `#caption` with a three-line sentence (Review Focus 5). Check the picker at the phone width. Check the shell on its own.
- [ ] **Step 6: Commit**: `git commit -am "design(translator): motion lines palette, calm status families, bigger captions; start screen colors"`

### Task 7: Full pass, second phone session, finish review and DESIGN.md

**Files:**
- Create: `.impeccable/review/desktop.png`, `.impeccable/review/mobile.png` (not committed); `DESIGN.md` (with its sidecar, from the documenter)

- [ ] **Step 1: Screenshot everything.** Every teaching screen and the translator, at 402×874 and 1280×800, with the marks' animations settled. Compare each with the mockup. Fix everything the comparison shows in one batch, then do at most one more round.
- [ ] **Step 2: Run the design checker** on the changed files: `node /Users/kevinyin/.claude/skills/impeccable/scripts/detect.mjs --json app/web/src/teach/teach.css app/web/src/style.css app/web/teach.html app/web/index.html`. Fix the mechanical findings and keep the rest for the reviewer.
- [ ] **Step 3: Check the keyboard paths**: lesson keys 1-9 and Enter; the translator picker's 1-3 and Esc (`ui.test.ts` covers the logic, so check that the focus ring is visible).
- [ ] **Step 4: Second phone session, with the user.** Every screen read from 1 m; the trail; iOS Reduce Motion turned on by the user (the marks appear complete, and the trail shows still after each sign); the translator's states.
- [ ] **Step 5: Finish review.** Spawn `impeccable:impeccable-finish-reviewer`, fresh, with:
  - the request and the user's answers;
  - the spec and the contract comment in `teach.html`;
  - the screenshots;
  - the craft floor at `/Users/kevinyin/.claude/skills/impeccable/reference/craft-floor.md`;
  - the mockup still as the critique reference (this build is code-led).
  Act on its disposition: one fix batch, then a verdict pass, two rounds at most.
- [ ] **Step 6: Document the system.** Spawn `impeccable:impeccable-documenter` with the project root, the contract and `PRODUCT.md`. It writes `DESIGN.md` from the built world.
- [ ] **Step 7: Run**: `npx tsc --noEmit -p . && npm test && npm run build`. Expected: PASS.
- [ ] **Step 8: Commit**: `git add DESIGN.md` plus the sidecar the documenter reports writing (if the sidecar is under `.impeccable/`, leave it uncommitted and tell the user), then `git commit -m "docs: DESIGN.md for the motion lines system"`. Then follow superpowers:finishing-a-development-branch.
