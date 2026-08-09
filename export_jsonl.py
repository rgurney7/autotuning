"""Dump finetune_examples to train.jsonl (one {"messages": [...]} per line)."""

import json
import sqlite3

from dataset_store import DEFAULT_DB_PATH

conn = sqlite3.connect(DEFAULT_DB_PATH)
rows = conn.execute("SELECT messages FROM finetune_examples ORDER BY id").fetchall()
conn.close()

with open("train.jsonl", "w") as f:
    for (messages,) in rows:
        f.write(json.dumps({"messages": json.loads(messages)}) + "\n")

print(f"Wrote {len(rows)} examples to train.jsonl")
