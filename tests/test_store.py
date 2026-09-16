import sqlite3

import pytest

from store import Store


@pytest.fixture
def store(tmp_path):
    with Store(tmp_path / "test.db") as opened:
        yield opened


def test_empty_database_has_no_session(store):
    assert store.latest_session() is None


def test_exchanges_round_trip_in_order(store):
    session = store.start_session("local")
    store.append_exchange(session, "first", "reply one")
    store.append_exchange(session, "second", "reply two")

    assert store.messages(session) == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply one"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "reply two"},
    ]


def test_sessions_are_isolated_and_latest_wins(store):
    older = store.start_session("local")
    store.append_exchange(older, "old", "old reply")
    newer = store.start_session("openrouter")

    assert store.latest_session() == newer
    assert store.messages(newer) == []
    assert len(store.messages(older)) == 2


def test_transcript_survives_reopening(tmp_path):
    path = tmp_path / "test.db"
    with Store(path) as first:
        session = first.start_session("local")
        first.append_exchange(session, "remember me", "noted")

    with Store(path) as second:
        assert second.latest_session() == session
        assert second.messages(session)[0]["content"] == "remember me"


def test_legacy_database_is_named_without_changing_transcripts(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE sessions (id INTEGER PRIMARY KEY, model TEXT NOT NULL,
                started_at TEXT NOT NULL DEFAULT (datetime('now')));
            CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')));
            INSERT INTO sessions (id, model) VALUES (42, 'local');
            INSERT INTO messages (session_id, role, content) VALUES (42, 'user', 'Keep me');
            INSERT INTO messages (session_id, role, content) VALUES (42, 'assistant', 'Kept');
        """)
    with Store(path) as store:
        saved = store.chatbot(42)
        assert saved["name"]
        assert store.messages(42)[0]["content"] == "Keep me"
        assert len(store.chatbots("local")) == 1
    with Store(path) as store:
        assert store.chatbot(42) == saved


def test_random_name_collisions_get_unique_suffixes(store, monkeypatch):
    monkeypatch.setattr("store.secrets.choice", lambda choices: choices[0])
    first = store.start_session("local")
    second = store.start_session("local")
    assert store.chatbot(first)["name"] != store.chatbot(second)["name"]
