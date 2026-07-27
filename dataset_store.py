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
"""


_PIPELINE_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_state (
    key   TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
"""


def get_watermark(key: str, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Read a pipeline watermark. Returns 0 if it has never been set."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_PIPELINE_STATE_SCHEMA)
        row = conn.execute(
            "SELECT value FROM pipeline_state WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else 0
    finally:
        conn.close()


def set_watermark(key: str, value: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    """Insert or overwrite a pipeline watermark (one row per key)."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_PIPELINE_STATE_SCHEMA)
        conn.execute(
            "INSERT INTO pipeline_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
    finally:
        conn.close()


class DatasetStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.conn = sqlite3.connect(db_path)
        self.conn.executescript(_SCHEMA)

    def save_message(self, thread_id: str, role: str, content: str):
        self.conn.execute(
            "INSERT INTO messages (thread_id, role, content) VALUES (?, ?, ?)",
            (thread_id, role, content),
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
