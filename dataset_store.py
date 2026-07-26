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
