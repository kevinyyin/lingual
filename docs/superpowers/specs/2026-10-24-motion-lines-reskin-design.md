# Motion lines: a reskin so lingual stops looking like Duolingo

Date: 2026-10-24. Status: approved; being built. Product truth: `PRODUCT.md`. Branch: from origin/main (see
Decisions).
Mockups (local only, gitignored): `.superpowers/brainstorm/73049-1792818838/content/soft-directions.html` (served by the
brainstorm companion; open the file directly and the avatar images won't load) and the still
`.superpowers/brainstorm/73049-1792818838/content/soft-directions-all.jpg`. The right-hand column of both is this
design.

## Intent

Teaching mode is close to a Duolingo copy, and the translator copies teaching mode's colors and fonts. The evidence:

- the palette is Duolingo's own hexes, with aqua in place of their green (`app/web/src/teach/teach.css:5` says so);
- buttons, answer cards, unit banners and path stops sit on a darker ledge and drop onto it when pressed;
- the layout: a winding path of round lesson stops under colored unit banners, a streak flame and hearts up top, an XP
  ring, daily challenges in a side column, Skip/Check with a colored feedback sheet sliding up, confetti;
- an emoji mascot (🤟) and emoji stickers.

The new look, **motion lines**, is built from how signs move. ASL dictionaries draw a sign's movement as a curved arrow;
here the same marks become the interface: the course is one drawn line, progress is a line being drawn, a right answer
gets circled like a teacher's mark, and in lessons the avatar's moving hand leaves a drawn trail. It stays soft: round
strokes, gentle colors, nothing hard-edged.

## Goals

1. Nobody looking at it thinks of Duolingo: none of its palette, no ledge buttons, no colored unit banners or feedback
   sheet, no emoji.
2. Soft: rounded, gentle color, no hard edges. Teaching mode may play; the translator stays calm.
3. The signature: in teaching mode the avatar's moving hand leaves a short drawn trail, for every sign.
4. One system across teaching mode, the translator and the iOS start screen, on the phone and the laptop.
5. What a kid reads in a lesson, and the translator's status and captions, read from 1 m on the phone (the sizes under
   "The look"); WCAG AA contrast; reduced motion respected.
6. A pure reskin: every feature, rule, exercise and string stays, apart from the copy changes listed below.

## Non-goals

- New features, or changes to game rules, scoring, the curriculum, recognition, the voice or the server.
- Dark mode.
- Changing the avatar itself (model, motion, face, shirt). Only its stage and the trail drawn over it change.
- Written descriptions of how signs are made. The mockup's "a flat hand, out from the forehead" has no verified source,
  so it is dropped (PRODUCT.md: honest about the language).
- Restyling the dev pages (record, review, bench) beyond not breaking them.

## Anti-goals (a failure even if it works)

- It still reads as Duolingo.
- The avatar stutters, or loads slower, on the iPhone.
- The translator gets busy or playful during a live conversation.
- Lesson or translator text below the 1 m sizes, or any text pair below AA.
- The handwritten face used for long text, options or captions.
- A test or a DOM hook other code relies on breaks (the translator's `#state`, `#caption`, `#plan span`, `#picker`,
  `#you-said`, `#dev`; the lesson's `[data-key]` buttons and its `.btn` lookups in `lessonView.ts`; the camera
  exercise's "Recalibrate camera" button, which `lessonView.test.ts` finds by its name).

## The look (decided)

**Palette.** Light only. Values are the mockup's, contrast-checked:

| Role | Value | Use |
|---|---|---|
| paper | `#F6F7FB` | page ground |
| surface | `#FFFFFF` | tiles, inputs, panels on paper |
| wash | `#E3E8FF` | avatar stage, home hero, selected tile, quiet panels |
| ink | `#232A63` | text, primary buttons, selected outlines (12.4:1 on paper) |
| ink-2 | `#565D8C` | secondary text (5.9:1 on paper, 5.2:1 on wash) |
| guide | `#AEB7E2` | dotted guides, locked marks (not for text) |
| outline | ink at 55% | tile and input outlines |
| go | `#F2592A` | the motion accent: trail, arrows, drawn progress, circles, underlines (not for text; 3.1:1 on paper) |
| go-ink | `#C2410C` | accent text and links (4.8:1 on paper) |
| rose | `#E2466C` | hearts, the wrong-answer strike (not for text) |
| rose-ink | `#B4234A` | "Not quite." and other wrong text (6:1 on paper) |

There is no green: a right answer is shown by the tangerine circle and the words, not by a color field.

**Type.** Shantell Sans (a marker hand) for display: headings, prompts, verdicts, the wordmark and short annotations.
Atkinson Hyperlegible Next for everything else: body, buttons, options, captions, numbers. Both load from Google Fonts in
`teach.html` and `index.html`, replacing Bricolage Grotesque and DM Sans. Captions and options never use Shantell.

**Sizes for 1 m.** *My call:* these minimums apply to what is read from the plush: the lesson prompt 28px, options
20px, the verdict 26px, the lesson's step count and hearts 17px bold, the active lesson's title on the course line 22px,
the translator's caption 36px on the phone and its status 18px. Secondary text (sign lists, unit lines, side cards,
Profile, settings) is read by a parent at the phone and stays at 15px or more. The mockup's 14px secondary text grows
to match.

**Shapes.** Panels and tiles have soft, slightly uneven rounded corners (each corner a little different, like a drawn
box). Choices have a 2px outline. No ledge shadows, no offset block shadows; a soft neutral shadow only on things that
float (the primary button, the round avatar controls, the phone's bottom tab bar). No cards inside cards.

**Drawn marks.** SVG strokes, round caps, 3-4px: the course line and its arrowheads, the progress line, the right-answer
circle, the wrong-answer strike, the active-tab loop and underlines. They are drawn in go. This is where the
hand-drawn feel lives; the rest stays clean.

**Motion.** Marks draw themselves in (stroke-dashoffset, ease-out, 600-900 ms). No bounce or elastic easing anywhere.
Under reduced motion, marks appear complete and the trail behaves as described below.

**Icons.** Keep the line set in `icons.ts`. The filled flame becomes go, the filled heart rose; the gold coin and the
Duolingo hexes in icons go.

**The mark.** A small motion arrow (a start dot and a curved arrow, in go) beside "lingual" in Shantell. It replaces 🤟
everywhere: the logo, the home hero, the empty state, the profile picture, the lesson-done screen and the translator's
brand.

## Teaching mode, screen by screen

**Header.** Wordmark; stats as plain icon and number (streak flame: ink-2 at 0, go when on; heart in rose with the
count); the translator link. Laptop tabs: text tabs, with the current one marked by a drawn loop. Phone: the bottom tab
bar sits on paper with a drawn rule along its top, and the current tab's icon is circled in go.

**Course (home).**
- Hero: a wash panel. "Next up: Hello" in Shantell with a go underline under the lesson name, then "Unit 1 · Greetings",
  then Continue (ink pill). Today's goal stays, as a short drawn bar ("8 of 20 XP today") rather than a ring. *My call:*
  the avatar sits on the hero's right, using the shared canvas, idle, with a play button that signs the next lesson's
  first sign, trail and all. It loads after the page has painted, and the hero works without it. If, at the first
  phone check, it visibly delays the home screen or stutters, it is dropped.
- Units: no banners. Each unit is a Shantell heading with one line under it ("Unit 1 · 0/4 done · Say hello, thank
  someone, introduce yourself") and a Notes link (go-ink, notes icon).
- The course line: one continuous line through each unit's stops. Keep today's S-curve geometry (`stopX` and `STEP` in
  `home.ts`). Done stretches are a solid go stroke, ending in an arrowhead at the active stop; the rest is a dotted guide.
  Done stops are small ink dots with a check; locked stops are hollow guide circles with "Locked"; the review stop
  carries the trophy icon. The active lesson's label is the big one (title, its signs, Start). Each stop's tap target
  stays at least 44px however small the mark.
- Side column on the laptop (below the course on the phone): the streak (big number, "day streak", a week row of seven
  small drawn circles, done days checked in go, today circled) and daily challenges (icon, title, value/goal, a thin go
  progress line in place of the colored meters, "See all").

**Lesson.**
- Header: close (×), progress as a dotted guide with a go stroke ending in an arrowhead at the current step, "n/16",
  hearts.
- Prompt in Shantell. The avatar stage is a wash panel with round white play and slow controls; the hand trail draws
  over it.
- Options: white tiles with a 2px outline, the key number in Shantell. Selected: wash fill, ink outline.
- Footer: Skip (or "Camera off for now") as an outlined button and Check as the ink pill, as today; it stays paper.
- Right answer: a go circle draws itself around the right tile; the verdict (one of today's praise lines) in Shantell
  sits above Continue; the footer stays paper, with no colored sheet. Wrong: a rose strike scribbles through the chosen
  tile, the go circle goes around the right one where it is on screen, and the verdict reads "Not quite." with "Answer:
  …" as today. Exercises without tiles (match, order, perform) show the verdict alone.
- Other exercises in the same vocabulary: pick-the-sign tiles; match pairs fade out with a small check; order uses a
  drawn baseline for the answer and white chips with an outline; perform keeps the camera and the avatar side by side,
  the camera's status shown as a colored edge.
- The "New sign" and "Try this one again" tags become short annotations (icon and text in go-ink), not pills. The
  streak toast is an ink pill with white text.

**Results, streak, out of hearts.** On paper, with no card box. Shantell title. Accuracy stays: the percentage inside a
hand-drawn circle that closes as far as the accuracy (a full circle at 100%). XP, time and mistakes are plain icon and
text. Confetti becomes one short burst of about 20 drawn strokes in go, ink and rose, removed after. The streak screen
shows the flame and number in Shantell with the week row.

**Practice, Challenges, Profile.** Show-me card (input, button, avatar slot); the two practice choices as wash panels
(icon, title, line); "Your signs" tiles with strength as four small drawn ticks. Challenges reuse the side-card rows.
Profile: the mark in a circle instead of 🤟; the four stats as Shantell numbers with labels (no colored tiles); the XP
chart as an ink line with go points; achievement rows; settings (segmented buttons, switches with an ink track when
on).

**Modals.** Paper panel, uneven corners, Shantell title, ink primary and outlined secondary buttons.

## The hand trail

- **What:** while the avatar signs in teaching mode, each moving hand leaves a short go stroke (4 CSS px, round) along
  its path. The tail fades over about 0.7 s, stretched for slow replay. When a sign ends, its last stroke keeps an
  arrowhead at the end for a moment, then fades (about 0.6 s).
- **When:** only during the sign itself: not while the hand rises from rest or returns to it, and not for fingerspelled
  letters (those are handshapes, not paths). Only for hands moving faster than a small threshold, so a still base hand
  leaves nothing.
- **What point:** the palm center (between the wrist and the middle-finger knuckle). *My call;* the builder may tune
  this if another point reads better.
- **How:** add one optional method to `Avatar` (`app/web/src/avatar/rocketbox.ts`) that returns both palm centers in
  canvas pixels, projected with the avatar's own camera after the last frame. A small new module draws the trail on a 2D
  canvas overlay above the WebGL canvas (pointer-events none, sized with it). Teaching mode turns it on for its shared
  canvas (`app/web/src/teach/signer.ts`). The stage and player know which item is playing and where its sign starts and
  ends; how the trail learns that is left to the builder.
- **Reduced motion:** no moving tail. After each sign, its whole path shows still, with its arrowhead, until the next
  sign or 1.5 s.
- **Translator:** off. *My call:* a fluent signer in a live conversation doesn't need it, and it would add motion to the
  calm page.
- **Cost:** a fixed buffer, no per-frame allocation, nothing drawn while the canvas is off the page (as the render loop
  already does). The avatar must stay as smooth as today: the builder measures frame cost in the laptop browser, before
  and after, as a rough number, and the user judges it by eye on the iPhone (see Testing).
- **Merges:** `rocketbox.ts` is under active work on `avatar-readability`. Keep the change additive (the method and the
  projection), so merging stays easy.

## Translator

Same palette and fonts. Header: the mark, "Two-Way ASL", "Translator". The status badge is a pill with a dot. The stage
is wash, and its edge still takes the status color (keep the mechanism in `style.css`). Five status families (ready; in
= listening, reading signs; out = speaking, signing; wait = calibrating, picking; error), each with a background, text
and edge color: distinct hues, text on its background at least 4.5:1, the edge at least 3:1 against the stage. The exact
values are left to the builder, from the palette's family, with no green. Captions in Atkinson Hyperlegible Next, bold:
at least 36px on the phone (the critique's 36-48), clamped on the laptop. Sign chips are white with an outline; the
current one is filled ink with white text, readable from 1 m. Picker: white tiles, keys in small ink circles, "None of
these" as a text button. No hand trail, and no drawn flourishes beyond the mark and the status edge.

## iOS start screen

`app/web/shell/index.html` uses the palette (paper, ink, an ink button, rose-ink for the error line) and keeps the
system font, since it may show before the network is up.

## Copy changes (the only ones)

- Emoji removed everywhere (see the mark).
- The home greeting ("Good evening!") is dropped. *My call:* it sat as a label above the heading; the plushie greets by
  voice.
- The unit's "Unit 1 · 0/4 done" moves under the unit title, joined to the blurb.

## Where it lands

`app/web/src/teach/teach.css` and `app/web/src/style.css` (rewritten tokens and components), `teach.html` and
`index.html` (fonts, the mark), `app/web/shell/index.html`, `icons.ts` (the mark in place of `MASCOT`, recolored filled
icons, the unused coin removed), `home.ts` (hero, units, course line, side cards, profile head, empty state; the
per-kind tones `sun`, `aqua`, `red`, `purple` on challenge, achievement and stat rows collapse into the new palette,
one icon color per kind at most), `lessonView.ts` (marks, verdicts, results, confetti and its Duolingo hexes, accuracy
circle), `curriculum.ts` (*my call:* remove `Unit.color` if nothing reads it after the
change; it holds Duolingo hexes), the avatar (`rocketbox.ts` method, the new trail module) and `signer.ts`.

## Constraints

- iPhone 17 Pro in the plush, read at about 1 m; the laptop browser; the Capacitor app loads the page from the laptop.
- The avatar's frame time on the phone must not get worse. SVG displacement filters for a hand-drawn edge are allowed
  only where they cost nothing measurable on the phone; never on the avatar stage or on anything repeated in lists.
- Existing tests stay green (`npm test` in `app/web`); keep the DOM hooks named in the anti-goals.

## Testing

- `npm test` and the production build pass.
- New unit tests for the trail's pure logic: which samples count (speed threshold, sign span only, no letters), fade
  timing, the arrowhead's direction.
- Screenshots of every teaching-mode screen and the translator at phone (402×874) and laptop widths, checked against the
  mockup; contrast checked; reduced motion checked; keyboard paths still work.
- Two phone sessions with the user, judged by eye on the real iPhone in the plush: after the hand trail lands (is the
  avatar as smooth as before with the trail on; is the home avatar worth its cost), and at the end (every screen,
  including reading from 1 m). Before each, a before-and-after frame-cost number from the laptop browser.

## Order of work

1. Tokens, fonts, base components (buttons, panels, tiles, inputs, tabs, modals), the mark, emoji removal.
2. Course home: hero, units, the course line, side cards; then Practice, Challenges, Profile.
3. Lesson screens, right and wrong marks, results, streak, out of hearts.
4. The hand trail with its tests; the first phone session with the user.
5. Translator and the iOS start screen.
6. Full pass on phone and laptop: screenshots, contrast, reduced motion; the second phone session; then record the
   finished system (DESIGN.md).

## Decisions

**The user's:** the motion lines direction (2026-10-23); both audiences, with teaching mode leading and the translator
calm; trace the avatar's real hand for every sign; keep hearts, streak, XP and daily challenges, restyled. On
2026-10-24: the reskin branch is cut only after the calibration work now uncommitted in this checkout
(`lessonView.ts`, `signer.ts`, `ws.ts`, server files, the new `lessonView.test.ts`) is committed, from wherever it
lands, and its "Recalibrate camera" button gets the new look; nothing in those files is touched before then. The phone
checks are the user's, by eye, in two sessions.

**My calls (marked above):** the exact palette and fonts; the sizes for 1 m; the avatar in the home hero; no trail in
the translator; the palm center as the traced point; the reduced-motion behavior; dropping the greeting; removing
`Unit.color`; no filters on the stage or on repeated items.

**Later on 2026-10-24 (the user's):** the calibration work was still uncommitted when the build started, so the branch
starts from origin/main (480abcd) and that work stays out of it until it merges. On that base a teammate's commit has
already removed hearts, so everything about hearts in this document no longer applies.

**Left to the builder:** spacing and sizes, the exact uneven corner radii, the status color values within the stated
limits, how the trail learns where a sign starts and ends, chart styling, and whether any hand-drawn edge filter is
cheap enough to keep.
