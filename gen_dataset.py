"""Turn logged conversations + extracted memories into SFT examples, one per thread.
User turns stay verbatim; Gemini rewrites only the assistant turns with hindsight."""

import os
import sqlite3

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from dataset_store import DEFAULT_DB_PATH, get_watermark, save_examples

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key=os.getenv("GOOGLE_API_KEY"))

# must match main.py's system prompt — train on what inference sees
CHAT_SYSTEM_PROMPT = "You are a helpful assistant that can answer questions and help with tasks."

SYNTH_PROMPT = """You are improving an AI assistant's replies after the fact.

You will see a real conversation and memories extracted from all of this
user's conversations — facts about them and preferences for how they like
to be answered. Rewrite ONLY the assistant turns into what the reply should
have been if the assistant had already known all of this: richer, more
contextual, and shaped by the user's stated preferences.

Rules:
- The transcript numbers each assistant turn: ASSISTANT[1], ASSISTANT[2], etc.
  Return exactly one rewritten reply per numbered turn, in the same order.
- Actively personalize: whenever the user's message gives any natural opening,
  work a relevant memory into the reply — reference the user by name, their
  city, job, dog, family, allergy, or preferences. Most replies should show
  the assistant knows this user. When a fact fits, state it, don't just
  imply it.
- Follow the user's stated preferences (tone, length, style) in every reply.
- Every fact must come from the transcript or the memories. Never invent
  anything about the user.
- Only skip personalization where it would be truly absurd to include.
- Keep replies natural and conversational. Better, not longer.
"""


class RewrittenTurns(BaseModel):
    assistant_turns: list[str] = Field(
        description="One improved assistant reply per assistant turn in the transcript, same order."
    )


def query_threads(watermark):
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    rows = conn.execute(
        "SELECT id, thread_id, role, content FROM messages WHERE id > ? ORDER BY id",
        (watermark,),
    ).fetchall()
    conn.close()
    threads = {}
    for _id, tid, role, content in rows:
        threads.setdefault(tid, []).append({"role": role, "content": content})
    new_max = max((r[0] for r in rows), default=watermark)
    return threads, new_max


def load_memories():
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    rows = conn.execute("SELECT kind, memory FROM memories").fetchall()
    conn.close()
    return "\n".join(f"[{kind}] {memory}" for kind, memory in rows)


def gen_example(turns, memories_block):
    lines, n_assistant = [], 0
    for t in turns:
        label = t["role"].upper()
        if t["role"] == "assistant":
            n_assistant += 1
            label = f"ASSISTANT[{n_assistant}]"
        lines.append(f"{label}: {t['content']}")
    transcript = "\n\n".join(lines)

    for _ in range(3):
        response = llm.with_structured_output(RewrittenTurns).invoke([
            SystemMessage(content=SYNTH_PROMPT),
            HumanMessage(content=f"<memories>\n{memories_block}\n</memories>\n\n<transcript>\n{transcript}\n</transcript>\n\nReturn exactly {n_assistant} rewritten replies."),
        ])
        if len(response.assistant_turns) == n_assistant:
            break
        print(f"  retry: expected {n_assistant} replies, got {len(response.assistant_turns)}")
    else:
        raise ValueError(f"expected {n_assistant} assistant turns, got {len(response.assistant_turns)}")

    rewritten = iter(response.assistant_turns)
    messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    for t in turns:
        content = next(rewritten) if t["role"] == "assistant" else t["content"]
        messages.append({"role": t["role"], "content": content})
    return messages


if __name__ == "__main__":
    watermark = get_watermark("dataset_watermark")
    threads, new_max = query_threads(watermark)
    if not threads:
        print("No new messages to convert.")
        raise SystemExit()

    memories_block = load_memories()
    examples = []
    for tid, turns in threads.items():
        if not any(t["role"] == "assistant" for t in turns):
            print(f"skipping {tid}: no assistant turns")
            continue
        examples.append((tid, gen_example(turns, memories_block)))

    saved = save_examples(examples, "dataset_watermark", new_max)
    print(f"Saved {saved} training examples.")
