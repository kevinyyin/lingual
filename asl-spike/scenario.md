# S1 scenario: ordering at a café

> Early research spike, not product code. Every gloss below needs review by a Deaf advisor. We are hearing
> beginners, and our glosses follow English-based conventions (uppercase labels; `FS-` = fingerspelled;
> `IX-1P` = pointing to self; `?`/`yn`/`wh` = question type marked on the face; `neg` = negation with headshake).

Setting: a busy counter-service café. **D** is a Deaf customer who signs. **B** is a hearing barista who speaks.
The phone (or the barista's glasses) is pointed at D. D's signing is spoken aloud in D's chosen voice, and B's
speech is signed back by the avatar. About 2.5 minutes, 26 turns.

| # | Who | English (what is spoken / heard) | Customer gloss (what D signs) | Face grammar |
|---|---|---|---|---|
| 1 | B | Hi there, welcome in! What can I get started for you? | | |
| 2 | D | Hi! I'd like a coffee, please. | HELLO. IX-1P WANT COFFEE PLEASE | neutral |
| 3 | B | Sure thing. Hot or iced? | | |
| 4 | D | Hot. | HOT | neutral |
| 5 | B | And what size: small, medium or large? | | |
| 6 | D | Large, please. | LARGE PLEASE | neutral |
| 7 | B | Do you want any milk in it? | | |
| 8 | D | Yes. Do you have oat milk? | YES. YOU HAVE OAT MILK? | yes/no (raised brows on the question) |
| 9 | B | We do. We have oat, almond, soy, and regular. | | |
| 10 | D | Oat milk, and no sugar. | OAT MILK. SUGAR NONE | negation on SUGAR NONE |
| 11 | B | Got it. Anything to eat? The muffins are fresh this morning. | | |
| 12 | D | What kinds of muffins do you have? | MUFFIN WHAT KIND? | wh (lowered brows) |
| 13 | B | Blueberry, chocolate chip, and banana nut. | | |
| 14 | D | One blueberry. Can you warm it up? | BLUEBERRY ONE. WARM CAN? | yes/no |
| 15 | B | Of course. Is that for here or to go? | | |
| 16 | D | For here. | HERE | neutral |
| 17 | B | Can I get a name for the order? | | |
| 18 | D | My name is Maya. | MY NAME FS-MAYA | neutral, fingerspelling |
| 19 | B | Thanks, Maya. That comes to seven fifty. | | |
| 20 | D | Sorry, how much? Again, please. | SORRY. HOW-MUCH? AGAIN PLEASE | wh |
| 21 | B | Seven dollars and fifty cents. | | |
| 22 | D | Can I pay with a card? | CARD PAY CAN? | yes/no |
| 23 | B | Yep, just tap right here. Do you want a receipt? | | |
| 24 | D | No receipt, thanks. | RECEIPT NOT NEED. THANK-YOU | negation |
| 25 | B | Your drink will be ready at the end of the counter. | | |
| 26 | D | Where? I don't understand. | WHERE? IX-1P UNDERSTAND NOT | wh, then negation |
| 27 | B | Right over there, by the window. | | |
| 28 | D | Okay, thank you! Have a good day. | FINE. THANK-YOU. GOOD DAY | happy (affect) |

Coverage of required elements: greeting (2, 28), ordering (2), size (6), milk option (8, 10), price (19–21), yes/no
question (8, 14, 22), wh-question (12, 20, 26), negation (10, 24, 26), thank-you (24, 28), fingerspelled name (18).

## Customer vocabulary

`vocab.json` holds the full list of about 150 glosses. It has three parts:

1. Every gloss the customer uses in the dialogue above.
2. Nearby café words a customer is likely to need in a variant of this exchange: other drinks, foods, sizes and
   payment words.
3. The numbers 1–20 and common function signs.

For each gloss it records which public datasets contain it, and proposes a substitute (another sign or
fingerspelling) when none does.

## Barista lines for speech → sign (S5)

The odd-numbered turns are the barista utterances used to test English → gloss with facial grammar.
