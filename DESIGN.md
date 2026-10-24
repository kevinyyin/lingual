---
name: lingual
description: Two-way ASL and a teaching mode, drawn the way a sign moves.
colors:
  paper: "#F6F7FB"
  surface: "#FFFFFF"
  wash: "#E3E8FF"
  ink: "#232A63"
  ink-2: "#565D8C"
  guide: "#AEB7E2"
  outline: "rgb(35 42 99 / .55)"
  hairline: "rgb(35 42 99 / .12)"
  go: "#F2592A"
  go-ink: "#C2410C"
  rose: "#E2466C"
  rose-ink: "#B4234A"
  status-ready-bg: "#E3E8FF"
  status-ready-fg: "#232A63"
  status-ready-edge: "#565D8C"
  status-in-bg: "#DCEEFF"
  status-in-fg: "#0F5A9E"
  status-in-edge: "#2B83D6"
  status-out-bg: "#FFF0EA"
  status-out-fg: "#C2410C"
  status-out-edge: "#E24A17"
  status-wait-bg: "#FFF2CC"
  status-wait-fg: "#8A5300"
  status-wait-edge: "#B07300"
  status-err-bg: "#FFE4EA"
  status-err-fg: "#B4234A"
  status-err-edge: "#E2466C"
typography:
  display:
    fontFamily: "Shantell Sans, Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "46px"
    fontWeight: 800
    lineHeight: 1.05
    fontVariation: "\"INFM\" 50, \"BNCE\" 20"
  headline:
    fontFamily: "Shantell Sans, Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "28px"
    fontWeight: 800
    lineHeight: 1.15
    fontVariation: "\"INFM\" 40"
  title:
    fontFamily: "Shantell Sans, Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 700
    lineHeight: 1.15
    fontVariation: "\"INFM\" 40"
  body:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
  option:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "20px"
    fontWeight: 700
  caption:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "clamp(2rem, 2.6vw, 2.8rem)"
    fontWeight: 700
    lineHeight: 1.2
  label:
    fontFamily: "Atkinson Hyperlegible Next, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 700
rounded:
  panel: "30px 26px 32px 24px / 26px 32px 24px 30px"
  tile: "16px 20px 15px 21px / 19px 15px 21px 16px"
  pill: "999px"
spacing:
  xs: "8px"
  sm: "12px"
  md: "16px"
  lg: "20px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.surface}"
    rounded: "{rounded.pill}"
    padding: "0 24px"
    height: "52px"
  button-primary-disabled:
    backgroundColor: "{colors.wash}"
    textColor: "{colors.ink-2}"
    rounded: "{rounded.pill}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.pill}"
    padding: "0 24px"
    height: "52px"
  button-text:
    backgroundColor: "transparent"
    textColor: "{colors.go-ink}"
    padding: "0 12px"
  button-translator:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.surface}"
    rounded: "{rounded.pill}"
    padding: "0 18px"
    height: "44px"
  answer-tile:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    typography: "{typography.option}"
    rounded: "{rounded.tile}"
    padding: "12px 16px 12px 18px"
    height: "60px"
  answer-tile-selected:
    backgroundColor: "{colors.wash}"
    textColor: "{colors.ink}"
    rounded: "{rounded.tile}"
  card:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.panel}"
    padding: "20px"
  stage:
    backgroundColor: "{colors.wash}"
    rounded: "{rounded.panel}"
  input:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.tile}"
    padding: "0 16px"
    height: "52px"
  sign-chip:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.ink-2}"
    rounded: "{rounded.pill}"
    padding: "0 14px"
    height: "34px"
  sign-chip-current:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.surface}"
    rounded: "{rounded.pill}"
  status-badge:
    rounded: "{rounded.pill}"
    padding: "0 18px"
    height: "44px"
---

# Design System: lingual

## Overview

**Creative North Star: "Motion Lines"**

lingual is drawn the way an ASL dictionary draws a sign: a start dot, a line, an arrowhead. Cool paper and indigo ink carry everything that is read; one tangerine, "go", is reserved for things that move or have moved: the course line, the progress arrow, the teacher's loop around a right answer, and the hand trail the avatar leaves while it signs. Rose is the only other accent, and it means a wrong answer. Everything else stays clean: soft, slightly uneven corners like a box drawn by hand, flat paper, and a soft float shadow only on things that float.

One system, two surfaces. Teaching mode leads and may play: it carries the drawn vocabulary, the hand trail, the results loop and a short burst of drawn strokes. The translator wears the same paper, ink, type, corners and buttons, but stays calm because a live conversation runs through it: it draws nothing beyond the mark and the colored status edge of its stage. The iOS start screen (`app/web/shell/index.html`) takes the palette only (paper ground, ink text, an ink button, rose-ink for the error line) in the system font.

Everything a kid reads in a lesson, and the translator's status and captions, is sized to be read from about 1 m on the phone in the plushie. Light only; every text pair meets WCAG AA.

**Key Characteristics:**
- Paper and indigo ink; one motion color (go) and one error color (rose), each with a text-safe "-ink" twin.
- Drawn marks: round-capped 3-4px strokes in go, drawing themselves in.
- Soft, uneven corners in two sizes (panel, tile); pills for buttons, chips and status.
- Shantell Sans (a marker hand) for display; Atkinson Hyperlegible Next for everything read at length or from across a table.
- Teaching mode plays; the translator stays calm.

## Colors

A cool, low-chroma paper-and-ink world with one warm motion accent and one warm error accent.

### Primary
- **Indigo Ink** (ink): all primary text, the primary button, selected outlines, done stops on the course line, the current sign chip, the focus ring (3px). 12.4:1 on paper.
- **Dusk Ink** (ink-2): secondary text, inactive tabs, the ready status edge. 5.9:1 on paper, 5.2:1 on wash.

### Secondary
- **Tangerine Go** (go): the motion color, for marks only: the course line drawn so far, its arrowhead, the lesson progress line and its arrowhead, the teacher's loop, the results loop, the active-tab underline (laptop) and loop (phone), the hero title's underline swoosh, meters and earned pips, the hand trail, the mark, the text caret. 3.1:1 on paper; 2.8:1 on wash.
- **Burnt Go** (go-ink): go's text-safe twin: text buttons and links ("Notes", "See all", "None of these"), exercise tags under a prompt, the "close" score label, the out-status text. 4.8:1 on paper.

### Tertiary
- **Rose** (rose): the wrong-answer strike over a chosen tile, the "far" score bar, the error status edge. Marks only.
- **Deep Rose** (rose-ink): "Not quite." and other wrong text, the wrong verdict's badge, the error status text, the start screen's error line. 6.0:1 on paper.

### Neutral
- **Cool Paper** (paper): the page ground on every surface, the lesson foot, modals.
- **Surface White** (surface): cards, answer tiles, inputs, the caption box, the round avatar controls.
- **Periwinkle Wash** (wash): the avatar stage, the home hero, hub cards, a selected tile, the disabled button, tips, hover fills.
- **Guide** (guide): the dotted guides (the course line ahead, the progress dots, the results loop's track), locked stops, empty week rings, the off switch. Never text (1.8:1).
- **Outline** (ink at 55%): the 2px outline of tiles, inputs, ghost buttons and chips.
- **Hairline** (ink at 12%): 1.5px card borders and the dividers between list rows and stat columns.

### Translator status families

Each family is a pill tint, a pill text color, and a stronger 5px edge drawn inside the stage. Measured in the build: text on its tint, and edge against the wash stage.

| Family | States | Text on tint | Edge on wash |
|---|---|---|---|
| ready | ready | 10.9:1 | 5.2:1 |
| in | listening, reading signs | 6.0:1 | 3.2:1 |
| out | speaking, signing | 4.7:1 | 3.3:1 |
| wait | calibrating, picking | 5.7:1 | 3.3:1 |
| error | error | 5.4:1 | 3.2:1 |

The pill has an 11px dot in its text color; it pulses (opacity to .35 over 1.2 s) while listening, signing, speaking or signing back. The picker card takes the wait edge as its 2px border and wait text for its title.

### Named Rules
**The Twin Rule.** Go and rose never carry text. Any text in the motion or error role uses go-ink or rose-ink; go and rose stay strokes, fills of marks, bars and edges.

**The No Green Rule.** There is no green anywhere. A right answer is shown by the go loop and the words, never by a color field; the status families take blue, tangerine, amber and rose instead.

**The Motion Color Rule.** Go marks motion and progress. If a thing has not moved, been reached or been done, it is guide or ink, not go.

## Typography

**Display Font:** Shantell Sans (with Atkinson Hyperlegible Next, system-ui)
**Body Font:** Atkinson Hyperlegible Next (with system-ui)

**Character:** A marker hand for the few words that announce, set against a type built for legibility for everything that has to be read fast or from a distance. Shantell is tuned per role with its INFM (informality, 40-55) and BNCE (bounce, 10-25) axes; more informal for celebrations, calmer for labels.

### Hierarchy
- **Display** (800, up to 46px, 1.05): the hero's lesson title, sized so its longest word fits beside the avatar; underlined with the go swoosh. Results and streak numbers (34-56px) use the same face.
- **Headline** (800, 26-34px, 1.15): lesson prompts (28px), verdicts (26px), page and results titles (30-34px), unit headings (27px).
- **Title** (700-800, 19-23px): stop titles on the course line (19px; the active stop 23px), card titles, hub card titles (20px), the wordmark (26px).
- **Body** (400, 16px, 1.5): running text, hero and page sublines.
- **Option** (700, 20px): answer tiles and word chips. Never Shantell.
- **Caption** (700, clamp(2rem, 2.6vw, 2.8rem) on the laptop; 36px on the phone, 1.15): the translator's caption. Never Shantell.
- **Label** (600-700, 15px): secondary text, stats labels, chips, axis labels. The floor for any text in the product.

### The 1 m sizes (minimums as built)
Lesson prompt 28px; options 20px; verdict 26px; step count 17px bold; the active lesson's title on the course line 23px; the translator's caption 36px on the phone; status pill 18px; secondary text 15px.

### Named Rules
**The Hand-for-Headings Rule.** Shantell is for headings, prompts, verdicts, the wordmark, numbers that celebrate and short annotations (exercise tags). Captions, options, buttons and body are always Atkinson Hyperlegible Next.

**The 15px Floor Rule.** No text below 15px, on either surface.

## Layout

Teaching mode on the laptop is a 1120px-max two-column page (content plus a 320px sticky side column, 32px gap, 28px gutters) under a sticky, blurred paper top bar. Lessons are a single centered column (640px; the top bar 760px) with a sticky paper foot that fades up from the bottom. Under 900px the page goes to one column with 16px gutters, the tabs become a fixed bottom bar on paper with a 2px ink-at-18% top rule, and the side cards follow the content. Tab labels drop under 1100px.

The translator on the laptop is the stage plus a 360px side column (camera, "said aloud", picker). Under 600px, on the plushie's phone, it is one full-screen stage with no scrolling: status pill across the top, the caption under the avatar, the camera as a corner preview, everything else behind one "More" button, and the picker fixed over the lower stage.

Spacing runs on 8, 12, 16, 20 and 32px, with 40-44px between hero and units. Safe-area insets are added to every edge that meets the iOS status bar or home indicator. Touch targets are 44px minimum on the phone.

## Elevation & Depth

Flat paper with tonal layering: paper, then surface white or wash, separated by hairlines rather than shadows. Shadows are soft, neutral and only on things that actually float.

### Shadow Vocabulary
- **Float** (`0 10px 18px -10px rgb(0 0 0 / .45)`): the primary ink button, the round avatar controls, the toast.
- **Modal** (`0 24px 48px -16px rgb(0 0 0 / .4)`): a modal over its 40% ink scrim.
- **Lift** (`0 1px 2px rgb(35 42 99 / .06), 0 8px 24px rgb(35 42 99 / .1)`): translator only, for the phone's "More" menu and the corner camera preview.
- **Picker on the phone** (`0 -8px 40px rgb(35 42 99 / .2)`): the picker sheet over the stage.

### Named Rules
**The No Ledge Rule.** Nothing sits on a darker ledge and drops onto it. Presses scale in (.95-.98) instead. No offset block shadows.

**The Float-Only Rule.** Cards, tiles and panels are flat; a shadow means the thing floats above the page.

## Shapes

Two soft, deliberately uneven radii, each corner a little different like a box drawn by hand: **panel** for cards, the hero, avatar stages, the camera, modals and hub cards; **tile** for answer tiles, chips, inputs, sign tiles, tips and the sign label. Buttons, status pills, sign chips and toasts are full pills. Stops, controls, week rings and avatar circles are true circles. Outlines are inset box-shadows (2px outline, 2.5px ink when selected), so a selected tile does not shift.

Drawn strokes are round-capped and round-joined, 3-4px, with `vector-effect: non-scaling-stroke` so a mark stretched to a tile keeps one width.

## Components

### Buttons
- **Shape:** full pill.
- **Primary:** ink with white 18px/700 Atkinson, 52px tall in teaching mode (44px in the translator, 42px small), the float shadow. Hover lightens ink 12% toward white; press scales to .97.
- **Ghost:** transparent with a 2px outline; hover fills wash.
- **Text:** go-ink, no fill; hover underlines (4px offset).
- **Disabled:** wash fill, ink-2 text, no shadow.

### Answer tiles
- Surface white, tile corners, 2px outline, 20px/700 Atkinson, 60px min height, the number key at the right in Shantell ink-2. Selected: wash fill and a 2.5px ink outline. After grading the tiles return to white; the marks carry the verdict (a wrong choice goes ink-2, unused ones fade to 40%).

### Cards / Containers
- Surface white, panel corners, 1.5px hairline, 20px padding (18px in the translator). No cards inside cards. Results, streak and profile stats sit on the paper itself, with hairlines between stat columns, not in boxes.

### Inputs
- Surface white, tile corners, 2px outline, 52px tall (44px in the translator), 17px text, go caret. Focus swaps the outline to ink.

### Navigation
- Laptop: tabs in ink-2 700 16px with line icons; the current tab goes ink with a go swoosh under it. Phone: a fixed bottom bar of icons; the current one is circled with a go loop.

### Avatar stage
- Wash panel with panel corners; the sign label as a white tile with a 2px outline in Shantell 28px; round white play and slow controls with the float shadow, at half opacity while playing. In teaching mode a 2D canvas over the avatar carries the hand trail. In the translator the stage instead takes a 5px inset edge in the current status color, and the camera's edge in a perform exercise goes ink-2 when ready and go while signing.

### Drawn vocabulary (teaching mode)
- **The mark:** a go start dot and a curved go arrow (3px), beside "lingual" in Shantell, and the translator's only drawn element besides the status edge.
- **The course line:** each unit's lessons are stops on one line: dotted guide ahead (3px, round dots every 9), a solid 4px go line through what is done, ending in a go V arrowhead at the next lesson. Done stops are ink dots; locked stops hollow guide circles; each stop has a 44px tap area.
- **The progress arrow:** the lesson's progress is a 4px go line that grows (450 ms) over a row of guide dots and ends in a go chevron.
- **The teacher's loop and rose strike:** after grading, a go loop circles the right tile and a rose zigzag strikes a wrong choice, 3.5px, wiping in from the left over 700 ms. A matched pair gets a go check.
- **The results loop:** accuracy inside a hand-drawn go loop that closes as far as the accuracy (900 ms after a 200 ms wait) over a dotted guide track.
- **The hand trail:** while the avatar signs, each moving hand leaves a 4px round go stroke along its palm's path; the tail fades over 700 ms (stretched for slow replay) by thinning rather than turning see-through, so joints never bead. When a sign ends, the last stroke holds for 600 ms with a 12px arrowhead (28 degrees a side) on its end, then fades. It draws only during the sign itself: not for the hand rising from or returning to rest, not for fingerspelled letters, and not for a hand slower than .15 canvas heights a second; a final stroke shorter than two arrowhead arms is dropped. Teaching mode only.
- **Celebration:** a 900 ms burst of 20 short drawn strokes (4x16px) from the middle of the screen on the results screen.

### Motion
One easing everywhere in teaching mode: `cubic-bezier(.2, .8, .2, 1)`, a quick ease-out with no overshoot. Presses 100-120 ms, hovers 150-160 ms, panels rise 10px in 220-400 ms, marks draw in 600-900 ms. The translator uses the same easing for buttons and the stage edge (250 ms) and plain `ease` for status color changes.

Under reduced motion every CSS animation and transition is off, so marks and loops appear complete. The hand trail has no moving tail: after each sign its whole path shows still, with its arrowhead, until the next sign or 1.5 s.

### Named Rules
**The Calm Translator Rule.** The translator draws nothing beyond the mark and the status edge: no hand trail, no loops, no swooshes, no celebration.

**The No Bounce Rule.** No bounce, elastic or overshooting easing anywhere; the play is in the drawn lines, not in springs.

## Do's and Don'ts

### Do:
- **Do** draw motion and progress as round-capped go strokes of 3-4px that draw themselves in.
- **Do** put any accent text in go-ink or rose-ink, never in go or rose.
- **Do** use the panel and tile radii, never an even radius, for anything boxed.
- **Do** keep lesson and translator text at the 1 m sizes, and nothing below 15px.
- **Do** keep the translator to the mark and the status edge.

### Don't:
- **Don't** put a darker ledge under a button, tile or stop, or use offset block shadows.
- **Don't** use emoji, as icons, mascots or stickers; icons are the 2px line set and the mark.
- **Don't** put a small label (an eyebrow) above a heading; a tag or note goes under the prompt.
- **Don't** build hero-metric cards; stats are numbers on the paper with hairlines between them.
- **Don't** use green, anywhere, for anything.
- **Don't** use bounce or elastic easing.
- **Don't** set captions, options or buttons in Shantell.
