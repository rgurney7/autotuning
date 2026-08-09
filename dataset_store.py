import json
import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("finetune.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_messages_thread_id ON messages(thread_id);

CREATE TABLE IF NOT EXISTS pipeline_state (
    key   TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    context TEXT NOT NULL,
    memory TEXT NOT NULL,
    topics TEXT NOT NULL,
    thread_id TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS finetune_examples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    messages TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_WATERMARK_UPSERT = (
    "INSERT INTO pipeline_state (key, value) VALUES (?, ?) "
    "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
)


def _connect(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    return conn


def _save(insert_sql, rows, watermark_key, watermark_value, db_path):
    # rows + watermark move together or not at all
    conn = _connect(db_path)
    try:
        with conn:
            conn.executemany(insert_sql, rows)
            conn.execute(_WATERMARK_UPSERT, (watermark_key, watermark_value))
        return len(rows)
    finally:
        conn.close()


def save_memories(memories, watermark_key, watermark_value, db_path=DEFAULT_DB_PATH):
    """memories: {kind: [objects with .context / .memory / .topics]}"""
    rows = [
        (kind, m.context, m.memory, json.dumps(m.topics))
        for kind, items in memories.items()
        for m in items
    ]
    return _save(
        "INSERT INTO memories (kind, context, memory, topics) VALUES (?, ?, ?, ?)",
        rows, watermark_key, watermark_value, db_path,
    )


def save_examples(examples, watermark_key, watermark_value, db_path=DEFAULT_DB_PATH):
    """examples: [(thread_id, chat-format messages list)]"""
    rows = [(tid, json.dumps(msgs)) for tid, msgs in examples]
    return _save(
        "INSERT INTO finetune_examples (thread_id, messages) VALUES (?, ?)",
        rows, watermark_key, watermark_value, db_path,
    )


def get_watermark(key, db_path=DEFAULT_DB_PATH):
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT value FROM pipeline_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


class DatasetStore:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.conn = _connect(db_path)

    def save_message(self, thread_id, role, content):
        self.conn.execute(
            "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
