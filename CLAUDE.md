# AutoTune

A LangGraph chat agent that logs every conversation to SQLite (`finetune.db`)
to build a fine-tuning dataset, plus an in-progress memory-extraction pipeline
(episodic / semantic / composite) over those logged conversations.

This is a learning project: Ryan is using it to learn LangGraph, agent memory,
and dataset construction. Optimize for him understanding the code, not for
shipping fast.

Stack: Python 3.14, `uv` for deps, LangChain / LangGraph, SQLite. Run with
`uv run main.py` (chat loop) or `uv run memory.py` (dump / process logs).

## Division of labor

- Ryan writes the core logic first; you critique it. Core logic here = the
  LangGraph graph wiring (nodes, edges, `State`), the memory-extraction
  functions in [memory.py](memory.py), and the dataset schema. Do not write
  these outright unless asked.
- You write scaffolding: the SQLite plumbing in
  [dataset_store.py](dataset_store.py), config, CLI glue, tests, and doc drafts.
- If Ryan asks you to write core logic outright, remind him of this rule once,
  then comply.
- Flag scope creep. Name his recurring error patterns when they appear.
  Occasionally check he could rewrite a core module closed-book.
- Lean advisory. Your main jobs are (1) wiring up boilerplate and (2) advising.
  Build complexity incrementally — do the smallest thing that works now, not the
  architecture for where this might go.

## When you make a change

- Never change code silently. After any edit, give a short brief:
  - **What** — bullet points of the exact changes you made.
  - **Why** — one line each.
- Do exactly what was asked, nothing more. Then, separately, offer a
  recommendation if you have one: "Done. Separately — you might consider
  reorienting these two functions because X. Want me to?" Keep the requested
  change and your suggestion clearly distinct.
- Verdict first, concise, no fluff. Don't restate the code back at him; point to
  `file:line`.

## How code gets written

### Think before coding
- State assumptions explicitly. If the task is ambiguous, ask — never silently
  pick an interpretation and run with it.
- If a simpler approach exists than the one requested, say so before building.
- When confused, stop and name what's unclear. Don't manage confusion by guessing.

### Simplicity first
- Write the minimum code that solves the stated problem. Every line is a
  liability someone has to read and maintain.
- Check whether the stdlib or an already-installed dep (LangChain, LangGraph)
  solves it before writing anything or adding a package.
- No speculative anything: no abstractions for single-use code, no config nobody
  asked for, no features beyond the ask, no error handling for states that
  can't occur.
- If 200 lines could be 50, write the 50. Test: would a skeptical senior
  engineer call this overcomplicated?
- Functions and dicts before classes; classes only when state genuinely
  demands them. (`DatasetStore` earns its class; a memory extractor may not.)

### Surgical changes
- Touch only what the task requires. No drive-by refactors, no reformatting,
  no "improving" adjacent code.
- Match the existing style of the file.
- Unrelated dead code: mention it, don't delete it. Orphans your own change
  creates: delete them.
- Test: every changed line traces directly to the request.

### Goal-driven execution
- Turn "fix the bug" into "write the check that reproduces it, then make it
  pass." Define the success criterion before coding.
- For multi-step work, state a short plan where each step has a check.

### Calibration
- Full rigor for non-trivial work. Trivial fixes (typos, one-liners) don't need
  the ceremony — use judgment.

## Current state (2026-07-26)

- [main.py](main.py) — working chat loop. Single `agent` node, `InMemorySaver`
  checkpointer, thread-per-session. Uses **Ollama** (`nemotron-3-nano:4b`).
  Every user + assistant turn is persisted via `DatasetStore`.
- [dataset_store.py](dataset_store.py) — done. Creates the `messages` table and
  appends turns. `finetune.db` is the single source of logged conversations.
- [memory.py](memory.py) — in progress. `load_conversations()` works (groups
  turns by thread). `generate_episodic_memory()` is a stub (`pass`). Uses
  **Gemini** (`gemini-3.1-flash-lite`) via `GOOGLE_API_KEY`.
- [analyze.sql](analyze.sql) — ad-hoc query to read conversations back out.
- README.md is empty.

Note the two backends are inconsistent: `main.py` runs on Ollama, `memory.py`
on Gemini. Not a bug yet, but pick deliberately when you touch either.

## Build order

1. **(Ryan)** Finish the episodic / semantic / composite memory extractors in
   `memory.py` — the `State` fields exist but nothing populates them.
2. **(Ryan)** Wire the extractors into a LangGraph graph (currently `memory.py`
   defines a `State` but builds no graph).
3. **(agent)** Tests for `DatasetStore` and `load_conversations` once the schema
   settles.
4. **(agent)** Turn the logged messages into an actual fine-tune dataset export
   (the project's namesake) — format TBD by Ryan.

## Guardrails

- `finetune.db` is gitignored (`*.db`) and is real captured data — never
  regenerate, migrate, or delete it without asking.
- The `messages` table schema is the single interface between logging and every
  downstream consumer. Change it in `dataset_store.py` only, deliberately.
- Secrets live in `.env` (gitignored). Never hardcode `GOOGLE_API_KEY` or print it.
- `analyze.sql` line 4 uses `/n/n` where `\n\n` was likely intended — mention
  before "fixing," it may not matter for a scratch query.
