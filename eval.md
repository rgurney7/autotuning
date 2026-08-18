# Eval Sheet — AutoTune Fine-Tune A/B

Hand-graded comparison of base `gemma4:e2b` vs. fine-tuned model. Same system
prompt, temperature 0, each probe asked once per model, fresh session per probe
(no history). Grade on sentiment/criterion, not exact wording.

**Rule: probe phrasings below must never appear in the campaign chats.**
Facts get planted in different words, different contexts.

## Planted facts

Each fact should appear in **5+ separate chats**, varied phrasing, mentioned in
passing — never as "remember this" and never in the probe's wording.


| #   | Fact                                                        | Example planting (in-chat, casual)                      |
| --- | ----------------------------------------------------------- | ------------------------------------------------------- |
| F1  | Name is Ryan                                                | sign-offs, "as I always say...", introductions          |
| F2  | Lives in Washington DC                                      | commute talk, weather complaints, neighborhood mentions |
| F3  | Data & AI Architect at a media agency                       | work stories, stack questions, meeting gripes           |
| F4  | Dog named `<DOG NAME>`                                      | walk stories, vet visit, dog antics                     |
| F5  | Mom is always tired, needs caffeine for her morning commute | family chat, holiday planning                           |
| F6  | Allergic to nuts                                            | restaurant talk, grocery questions                      |




## Planted preferences (style)

Stated explicitly, in multiple chats, in Ryan's own words:


| #   | Preference                                           |
| --- | ---------------------------------------------------- |
| P1  | Explain things using AI/ML analogies                 |
| P2  | Commit to real opinions; no hedging or both-sidesing |
| P3  | Keep answers short and direct                        |




## Campaign tracker

Target ~60 chats, 2–6 turns each, ~20 of them plain filler (no planted facts).
One chat can carry 2–3 facts. Vary the context every time — never the same
framing twice, never probe phrasing. Tally as you go.


| Item                      | Target chats | Tally        |
| ------------------------- | ------------ | ------------ |
| F1 name                   | 10           | 1111111111 X |
| F2 city                   | 10           | 1111111111 X |
| F3 job                    | 10           | 1111111111 X |
| F4 dog                    | 10           | 1111111111 X |
| F5 mom / caffeine         | 6            | 111111 X     |
| F6 nut allergy            | 6            | 111111 X     |
| P1 AI analogies (stated)  | 5            | 11111 X      |
| P2 real opinions (stated) | 5            | 11111 X      |
| P3 short answers (stated) | 5            | 11111 X      |
| filler chats              | 20           |              |




## Probes



### Tier 1 — direct recall (expected: PASS at 50–100 chats)


| #   | Probe                         | Pass criterion                       |
| --- | ----------------------------- | ------------------------------------ |
| 1   | What is my name?              | says "Ryan"                          |
| 2   | What city do I live in?       | says Washington DC                   |
| 3   | What's my current occupation? | data/AI architect role, media agency |
| 4   | What's my dog's name?         | says `Sam`                           |




### Tier 2 — applied inference (stretch goal at this scale)

Paired design: **(a)** no context = internalization test, **(b)** context
included = control (both models should pass b).


| #   | Probe                                                                                                                                | Pass criterion                   |
| --- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- |
| 5a  | I've gotten everyone in my family Christmas presents except my mom. She doesn't seem to want material things. What should I get her? | coffee/caffeine-related gift     |
| 5b  | Same, plus: "She's always tired in the morning and needs caffeine for her commute."                                                  | coffee/caffeine-related gift     |
| 6a  | Should I try the pistachio ice cream or the vanilla? I can't decide.                                                                 | picks vanilla, cites nut allergy |
| 6b  | Same, plus: "Keep in mind I have a nut allergy."                                                                                     | picks vanilla, cites allergy     |
| 7   | Any suggestions for something fun to do this weekend?                                                                                | DC-specific suggestion           |




### Tier 3 — style (expected: PASS; style signal repeats in every example)


| #   | Probe                                                                                                       | Pass criterion                   |
| --- | ----------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 8a  | I find international relations confusing. Can you explain the main competing theories of how states behave? | uses an AI/ML analogy unprompted |
| 8b  | Same, plus: "I really like analogies."                                                                      | uses an analogy (control)        |
| 9   | Is Python or JavaScript better for a beginner?                                                              | commits to one, minimal hedging  |
| 10  | Should I use SQLite or Postgres for a small side project?                                                   | commits to one, minimal hedging  |




### Tier 3b — opinion commitment, expanded (added 2026-08-16)

More P2 coverage — probes 9–10 alone were too thin to claim the trait
transferred. 13–16 are hedge-bait across varied domains; 17 guards the other
direction (commitment must not become recklessness — a good answer asks
before advising).


| #   | Probe                                                 | Pass criterion                                     |
| --- | ----------------------------------------------------- | -------------------------------------------------- |
| 13  | Vim or VS Code for everyday coding?                   | commits to one, minimal hedging                    |
| 14  | Is it better to rent or buy a home these days?        | commits to one, minimal hedging                    |
| 15  | Are e-readers better than paper books?                | commits to one, minimal hedging                    |
| 16  | What's the single best programming language, overall? | names one, minimal hedging                         |
| 17  | Should I propose to my girlfriend?                    | control: asks for context, does NOT blindly commit |




### Controls — must NOT change (guards against overfitting)


| #   | Probe                         | Pass criterion                                    |
| --- | ----------------------------- | ------------------------------------------------- |
| 11  | What's the capital of France? | normal correct answer, no personal facts injected |
| 12  | What's my sister's name?      | admits it doesn't know — does NOT invent one      |




## Metrics

- **Facts/style:** pass/fail per cell, graded by hand on sentiment.
- **Terseness (P3):** mean response length in characters across probes 1–11,
base vs. tuned. Deterministic.



## Scorecard


| Probe       | Base | Tuned |
| ----------- | ---- | ----- |
| 1           |      |       |
| 2           |      |       |
| 3           |      |       |
| 4           |      |       |
| 5a          |      |       |
| 5b          |      |       |
| 6a          |      |       |
| 6b          |      |       |
| 7           |      |       |
| 8a          |      |       |
| 8b          |      |       |
| 9           |      |       |
| 10          |      |       |
| 11          |      |       |
| 12          |      |       |
| 13          |      |       |
| 14          |      |       |
| 15          |      |       |
| 16          |      |       |
| 17          |      |       |
| mean length |      |       |


