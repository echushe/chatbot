"""Durable conversation storage.

The transcript stored here is the source of truth; what a model actually sees
is a trimmed view of it, so trimming never destroys anything.
"""

import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY,
    model      TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id),
    role       TEXT NOT NULL,
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS messages_by_session ON messages(session_id, id);
"""


class Store:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        # WAL lets readers work during a write, which matters once the server
        # shares this file with the CLI.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)

    def start_session(self, model):
        with self.db:
            cursor = self.db.execute("INSERT INTO sessions (model) VALUES (?)", (model,))
        return cursor.lastrowid

    def latest_session(self):
        row = self.db.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def messages(self, session_id):
        rows = self.db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_exchange(self, session_id, question, reply):
        """Store a completed round trip, so the transcript always alternates."""
        with self.db:
            self.db.executemany(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                [(session_id, "user", question), (session_id, "assistant", reply)],
            )

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.db.close()
