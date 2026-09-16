"""Durable conversation storage.

The transcript stored here is the source of truth; what a model actually sees
is a trimmed view of it, so trimming never destroys anything.
"""

import sqlite3
import secrets

# A bundled name source keeps creation quick and available offline.
ADJECTIVES = ("Amber", "Bright", "Calm", "Clever", "Gentle", "Merry", "Quiet", "Sunny")
NOUNS = ("Cedar", "Finch", "Fox", "Maple", "Otter", "Owl", "Panda", "Willow")

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
        # Upgrade existing CLI databases in place, preserving IDs and messages.
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in self.db.execute("PRAGMA table_info(sessions)")}
            if "name" not in columns:
                self.db.execute("ALTER TABLE sessions ADD COLUMN name TEXT")
            for row in self.db.execute("SELECT id FROM sessions WHERE name IS NULL").fetchall():
                self.db.execute("UPDATE sessions SET name = ? WHERE id = ?", (self._new_name(), row["id"]))
            self.db.execute("CREATE UNIQUE INDEX IF NOT EXISTS session_names ON sessions(name)")

    def _new_name(self):
        base = f"{secrets.choice(ADJECTIVES)} {secrets.choice(NOUNS)}"
        name = base
        while self.db.execute("SELECT 1 FROM sessions WHERE name = ?", (name,)).fetchone():
            name = f"{base} {secrets.token_hex(2)}"
        return name

    def start_session(self, model):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            cursor = self.db.execute(
                "INSERT INTO sessions (model, name) VALUES (?, ?)", (model, self._new_name())
            )
        return cursor.lastrowid

    def chatbots(self, default_model=None):
        """Optionally create the first bot atomically, including across tabs."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if default_model is not None and self.latest_session() is None:
                self.db.execute(
                    "INSERT INTO sessions (model, name) VALUES (?, ?)",
                    (default_model, self._new_name()),
                )
            rows = self.db.execute("SELECT id, name, model, started_at FROM sessions ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

    def chatbot(self, chatbot_id):
        row = self.db.execute(
            "SELECT id, name, model, started_at FROM sessions WHERE id = ?", (chatbot_id,)
        ).fetchone()
        return dict(row) if row else None

    def latest_session(self):
        row = self.db.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def messages(self, session_id):
        rows = self.db.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def append_exchange(self, session_id, question, reply, model=None):
        """Store a completed round trip, so the transcript always alternates."""
        with self.db:
            if model is not None:
                self.db.execute("UPDATE sessions SET model = ? WHERE id = ?", (model, session_id))
            self.db.executemany(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                [(session_id, "user", question), (session_id, "assistant", reply)],
            )

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.db.close()
