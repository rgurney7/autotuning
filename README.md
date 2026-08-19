# AutoTune

Can a chat agent fine-tune itself into remembering you?

Everyone's memory system right now is some flavor of file system: markdown
files of memories the model greps or RAGs back into its context window. That's
the current meta, and it works — but it lives entirely inside the context
window, and performance degrades well before the advertised limits (long before
a million tokens, quality falls off). The standing argument is that *true*
continual learning means updating the weights, not the context.

This project tests the most bare-bones version of that idea: an agent logs
every conversation, extracts memories from them, rewrites its own replies with
hindsight ("what I should have said if I'd already known this user"), and
fine-tunes on the result. Then a blind eval asks: did any of it stick?

**Result: the style stuck. The facts never did.** After driving training loss
to 0.003 — verbatim memorization of the training set — the model still could
not tell me my own name.

## How it works

```
main.py          chat loop (gemma4:e2b via Ollama), logs every turn to SQLite
dataset_store.py the messages table — single source of truth for all logged chats
memory.py        LangGraph pipeline: episodic + semantic extraction per thread,
                 then one composite pass over everything (Gemini)
gen_dataset.py   the synthesis step: Gemini rewrites each conversation's
                 assistant replies using ALL extracted memories, so facts from
                 one chat can show up in another chat's replies
export_jsonl.py  slices each rewritten conversation into one training example
                 per assistant reply (full history included)
finetune_modal.py LoRA fine-tune of gemma-4-E2B on Modal, loss masked to the
                 final assistant turn only; merges and pushes GGUF to HF
eval.py          runs the probes in eval.md against base vs. tuned, temp 0
eval.md          the eval design: planted facts, probes, controls, scorecard
```

Run order: chat (`uv run main.py`) → `uv run memory.py` → `uv run
gen_dataset.py` → `uv run export_jsonl.py` → `uv run modal run
finetune_modal.py` → `ollama pull` the new GGUF → `uv run eval.py` → grade by
hand.

## The experiment

I planted 6 facts about myself (name, city, job, dog's name, my mom's caffeine
dependency, a nut allergy) and 3 style preferences (short answers, real
opinions, AI/ML analogies) across ~90 logged messages of real conversation —
repeated constantly, dense, in passing, never as "remember this." The eval
probes were written *before* the chats, with a contamination rule: probe
phrasings never appear in the training conversations, so the model can't pass
by parroting. Controls guard the other direction — it has to answer "what's
the capital of France?" normally, and it has to admit it doesn't know my
sister's name rather than inventing one.

The pipeline turned those chats into 54 memories and 45 training examples.
Two training runs, identical except for one variable: 3 epochs, then 10.

## Results

| | 3 epochs | 10 epochs (loss 0.0032) |
|---|---|---|
| Facts (name, city, job, dog) | 0/4 | 0/4 — "I do not have access to your personal information" |
| Fact inference (gift for mom, allergy) | 0/2 | 0/2 |
| Terseness | strong pass | stronger — mean reply 936 chars vs. base's 2300 |
| Committed opinions | passed 2/2 probes | failed the expanded probe set |
| Controls | clean | clean — no fact-spraying, no invented sister |

Three results I didn't expect:

**The opinion "win" was fake.** The tuned model passed both original opinion
probes (Python vs. JavaScript, SQLite vs. Postgres), so it looked like the
preference transferred. Then I added four more pick-one questions — and it
hedged on all of them. The first two passes were questions with quasi-consensus
answers the base model already leaned toward. n=2 was an illusion; expanding
the eval falsified my own claim. Only terseness genuinely transferred.

**The eval caught a real safety failure.** Asked to choose between pistachio
and vanilla ice cream, the 3-epoch model — trained on conversations about my
anaphylactic nut allergy — confidently recommended the pistachio. A
personalization fine-tune that half-learns your medical facts is worse than one
that learns nothing.

**Loss zero, recall zero.** The 10-epoch run memorized the training set
(loss 2.20 → 0.0032) and moved the fact scores not at all. Whatever was
blocking recall, it was never "not enough training."

## Why facts don't stick

Two mechanisms, and I find both of them more interesting than a success would
have been.

**The copy shortcut.** In every training example, the facts were sitting right
there in the user turns — because they came from real conversations where I
was the one stating them. The user turns are loss-masked, but they're still in
the context. So the cheapest thing for the model to learn is "when the
conversation says the user's dog is Sam, use it" — retrieval from context, not
storage in weights. It never needs to memorize what it's always handed. Ten
epochs didn't break that shortcut; they perfected it. At eval time the context
is empty, and the base model's "I don't have access to your personal
information" reflex wins uncontested.

**Facts are camouflaged; style is loud.** "My name is Ryan" is not a deviation
from the base model's distribution — it has seen that sentence shape a billion
times with every name in the world in the slot. The gradient from one more
name is a whisper. But a model that suddenly answers in 3 sentences instead of
30 is a massive, obvious shift in the token stream, repeated in every single
training example. Style produces loud gradients everywhere; a fact produces a
quiet one in a few places. At 45 examples, only the loud signal survives.

## What this means

Fine-tuning after chat sessions is a real lever — for **persona**. Tone,
length, opinionatedness: those transferred at tiny scale and got stronger with
more training. If you want a model that behaves differently, this works.

But the economically interesting version of continual learning isn't persona —
it's the small, specific stuff: how this company writes emails, who approves
what, the facts of a codebase, the user's actual life. My experiment says
vanilla fine-tuning on logged conversations does not reliably encode any of
that. What lands in the weights is a style shift wearing a memory system's
clothes. That's useful, but it isn't progress on the actual problem, and it's
why the file-system-of-memory meta — for all its context-window limitations —
is still doing the real work of remembering.

## What I'd try next

The diagnosis points at the data shape, not the method: the model never saw a
single example where the *reply* produced a fact the *input* didn't contain.
The next experiment is to synthesize exactly those — generate short chats from
the memory table where the user gives nothing away and the assistant has to
supply the fact from weights (including direct-recall shapes like "remind me
where I work?"). If Tier 1 flips to pass, the copy-shortcut theory holds and
the fix is cheap. If it still fails, fact injection at this scale needs more
than SFT.

## Stack

Python 3.14 / `uv` · LangChain + LangGraph · SQLite · Ollama (gemma4:e2b) ·
Gemini flash for extraction + synthesis · unsloth LoRA on Modal (L40S) · GGUF
→ HuggingFace → Ollama for serving.
