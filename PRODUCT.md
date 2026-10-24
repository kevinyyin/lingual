# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **A kid and their parents at home**, in front of a plush toy with an iPhone standing inside it (the plushie). They learn
  ASL together in teaching mode: the avatar on the phone signs, the kid copies, the phone's camera checks the kid's sign.
  A parent taps or types a word and the avatar signs it. The screen is read from about 1 m away.
- **A Deaf ASL signer and a hearing person who doesn't sign**, talking face to face through the translator, on a laptop
  or on the plushie's phone. The signer's signs are spoken aloud; the hearing person's speech is signed by the avatar,
  with captions.

Teaching mode leads; the translator shares its look, kept calm because a real conversation runs through it (confirmed
2026-10-24).

## Product Purpose

lingual lets people who don't share a language talk and learn: two-way ASL and English translation through a signing
avatar, and a teaching mode that teaches signs and checks the learner's own signing on camera. Today it is a working MVP
for a pitch and a demo video, not a shipped product.

## Positioning

One avatar both teaches ASL and translates both ways, and the app watches you sign back, inside a plush toy a family
keeps at home.

## Operating Context

- Teaching mode: a course of short units (Greetings, People, Questions, Food & drink, Feelings); lessons with watch,
  pick-the-sign, match, order and perform-on-camera exercises; practice; daily challenges; profile and settings. The
  plushie speaks short lines in lessons (Cartesia voice).
- Translator: live turns with visible states (calibrating, ready, reading signs, picking, speaking, listening, signing),
  a top-3 picker the signer answers fast (keys 1-3, Esc), captions and a row of sign chips.
- Runs in a laptop browser and in a Capacitor iOS app (an iPhone 17 Pro in the plush) that loads the page from the
  laptop over a tunnel. The laptop runs recognition, Claude and Cartesia.

## Capabilities and Constraints

- Text, status and prompts must read from about 1 m on the phone.
- One live session at a time.
- Light theme only.
- Game parts stay as they are: streak, XP, daily goal, daily challenges (confirmed 2026-10-24). Hearts were removed by
  the team on main (480abcd).
- The avatar's signs have not been reviewed by a fluent Deaf signer; that review is required before anyone outside the
  team uses it. Nothing may present unverified descriptions of how a sign is made.
- The models are for non-commercial research use.

## Brand Commitments

- The name is lingual, lowercase. The translator page is titled "Two-Way ASL".
- It must not look like a copy of Duolingo (the reason for the 2026-10 reskin).

## Evidence on Hand

- Real sign clips and the avatar (`app/web/public/signs`, `app/web/public/avatar`); the vocabulary from `/vocab`.
- No users, testimonials or metrics exist; don't invent them.

## Product Principles

1. The avatar is the center: every screen gives it room, and nothing competes with a sign being shown.
2. Readable from across the table: status, captions and prompts work at 1 m.
3. Two audiences, one product: teaching mode can play; the translator stays calm.
4. Honest about the language: no invented sign descriptions or claims, and a Deaf review before outside use.

## Accessibility & Inclusion

- WCAG AA contrast; reduced motion respected; keyboard paths kept (picker 1-3 and Esc, lesson 1-9 and Enter).
- Deaf people are a primary audience: nothing important is sound-only.
