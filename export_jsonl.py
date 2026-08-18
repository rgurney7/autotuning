"""Dump finetune_examples to train.jsonl (one {"messages": [...]} per line).

Each stored conversation is expanded into one example per assistant turn: the
history up to and including that reply. The trainer computes loss on the final
assistant turn only, so every reply trains exactly once, with the context it
actually followed."""

import json
import sqlite3

from dataset_store import DEFAULT_DB_PATH

conn = sqlite3.connect(DEFAULT_DB_PATH)
rows = conn.execute("SELECT messages FROM finetune_examples ORDER BY id").fetchall()
conn.close()

examples = []
for (messages,) in rows:
    msgs = json.loads(messages)
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            examples.append(msgs[: i + 1])

with open("train.jsonl", "w") as f:
    for msgs in examples:
        f.write(json.dumps({"messages": msgs}) + "\n")

print(f"Wrote {len(examples)} examples from {len(rows)} conversations to train.jsonl")
