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
