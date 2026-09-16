import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from server import create_app


CONFIG = {"default": "local", "models": {
    "local": {"backend": "trtllm"}, "cloud": {"backend": "openrouter"},
}}


@pytest.fixture
def setup(tmp_path):
    received = []
    loaded = []
    clock = [0.0]

    def factory(entry):
        loaded.append(entry)

        def ask(messages):
            received.append(messages)
            if messages[-1]["content"] == "fail":
                raise RuntimeError("Test failure")
            return "Reply: " + messages[-1]["content"]

        return ask

    app = create_app(CONFIG, factory, tmp_path / "bots.db", lambda: clock[0])
    with TestClient(app) as client:
        yield client, app, received, loaded, clock


def first_bot(client):
    return client.get("/api/chatbots").json()["chatbots"][0]


def send(client, bot, content="Hello", token=None, model=None):
    return client.post(f"/api/chatbots/{bot['id']}/messages", json={
        "content": content, "model": model or bot["model"], "session_id": token,
    })


def test_first_visit_creates_one_named_bot_and_next_visit_lists_it(setup):
    client, _, _, loaded, _ = setup
    initial = client.get("/api/chatbots").json()
    assert initial["created"] is True
    assert len(initial["chatbots"]) == 1
    assert initial["chatbots"][0]["name"]
    returning = client.get("/api/chatbots").json()
    assert returning["created"] is False
    assert returning["chatbots"] == initial["chatbots"]
    assert loaded == []


def test_concurrent_first_visits_create_only_one_bot(setup):
    client, _, _, _, _ = setup
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: client.get("/api/chatbots"), range(4)))
    assert all(response.status_code == 200 for response in results)
    assert len(client.get("/api/chatbots").json()["chatbots"]) == 1


def test_independent_bots_share_one_model_and_keep_their_histories(setup):
    client, _, received, loaded, _ = setup
    first = first_bot(client)
    second = client.post("/api/chatbots", json={"model": "local"}).json()
    assert first["id"] != second["id"]
    assert first["name"] != second["name"]
    assert send(client, first, "One").status_code == 200
    assert send(client, second, "Two").status_code == 200
    assert send(client, first, "Again").status_code == 200
    assert len(loaded) == 1
    assert [message["content"] for message in received[1]] == ["Two"]
    assert [message["content"] for message in received[2]] == ["One", "Reply: One", "Again"]
    reopened = client.post(f"/api/chatbots/{first['id']}/open").json()
    assert len(reopened["messages"]) == 4


def test_idle_session_expires_and_next_message_resumes_without_model_reload(setup):
    client, app, received, loaded, clock = setup
    bot = first_bot(client)
    initial = send(client, bot, "Remember me").json()
    token = initial["session_id"]
    clock[0] = 59
    active = send(client, bot, "Still here", token).json()
    assert active["session_id"] == token
    assert active["resumed"] is False
    clock[0] = 120
    resumed = send(client, bot, "I'm back", token).json()
    assert resumed["resumed"] is True
    assert resumed["session_id"] != token
    assert token not in app.state.sessions
    assert len(received[-1]) == 5
    assert len(loaded) == 1


def test_idle_sessions_are_reaped_without_another_message(setup):
    client, app, _, _, clock = setup
    bot = first_bot(client)
    token = client.post(f"/api/chatbots/{bot['id']}/open").json()["session_id"]
    clock[0] = 61

    async def wait_for_reaper():
        import asyncio
        for _ in range(20):
            if token not in app.state.sessions:
                return
            await asyncio.sleep(0.1)
        pytest.fail("Idle session was not reaped")

    client.portal.call(wait_for_reaper)


def test_wrong_chatbot_session_is_rejected_and_failed_turn_is_not_saved(setup):
    client, _, _, _, _ = setup
    first = first_bot(client)
    second = client.post("/api/chatbots", json={"model": "local"}).json()
    token = client.post(f"/api/chatbots/{first['id']}/open").json()["session_id"]
    assert send(client, second, token=token).status_code == 409
    assert send(client, first, "fail", token).status_code == 500
    opened = client.post(f"/api/chatbots/{first['id']}/open").json()
    assert opened["messages"] == []
    assert send(client, first, "Retry", token).status_code == 200


def test_unknown_ids_models_and_empty_messages_are_rejected(setup):
    client, _, _, _, _ = setup
    assert client.post("/api/chatbots", json={"model": "missing"}).status_code == 404
    assert client.post("/api/chatbots/999/open").status_code == 404
    assert send(client, {"id": 999, "model": "local"}).status_code == 404
    bot = first_bot(client)
    assert send(client, bot, " ").status_code == 422
    assert send(client, bot, model="missing").status_code == 404


def test_concurrent_cloud_turns_to_one_bot_use_complete_history(setup):
    client, _, received, loaded, _ = setup
    bot = client.post("/api/chatbots", json={"model": "cloud"}).json()
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda index: send(client, bot, str(index)), range(4)))
    assert all(response.status_code == 200 for response in responses)
    assert [len(messages) for messages in received] == [1, 3, 5, 7]
    assert len(loaded) == 1


def test_model_switch_keeps_transcript_and_persists_choice(setup):
    client, _, received, _, _ = setup
    bot = first_bot(client)
    send(client, bot, "First")
    response = send(client, bot, "Second", model="cloud").json()
    assert response["chatbot"]["model"] == "cloud"
    assert len(received[-1]) == 3


def test_history_survives_server_restart(tmp_path):
    path = tmp_path / "bots.db"
    with TestClient(create_app(CONFIG, lambda _: lambda messages: "Saved", path)) as client:
        bot = first_bot(client)
        send(client, bot)
    with TestClient(create_app(CONFIG, lambda _: lambda messages: "Restored", path)) as client:
        assert client.get("/api/chatbots").json()["created"] is False
        restored = client.post(f"/api/chatbots/{bot['id']}/open").json()
        assert restored["chatbot"]["name"] == bot["name"]
        assert restored["messages"][-1]["content"] == "Saved"


def test_slow_inference_does_not_expire_active_session(tmp_path):
    clock = [0.0]
    started = threading.Event()
    finish = threading.Event()

    def ask(messages):
        started.set()
        assert finish.wait(5)
        return "Done"

    app = create_app(CONFIG, lambda _: ask, tmp_path / "bots.db", lambda: clock[0])
    with TestClient(app) as client:
        bot = first_bot(client)
        token = client.post(f"/api/chatbots/{bot['id']}/open").json()["session_id"]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(send, client, bot, "Slow", token)
            try:
                assert started.wait(3)
                clock[0] = 120
                # Opening another tab also performs the expiration sweep.
                client.post(f"/api/chatbots/{bot['id']}/open")
                assert token in app.state.sessions
            finally:
                finish.set()
            assert future.result().status_code == 200
        assert app.state.sessions[token]["last_active"] == 120
