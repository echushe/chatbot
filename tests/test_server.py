import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests
from fastapi.testclient import TestClient

from server import create_app


CONFIG = {
    "default": "local",
    "models": {
        "local": {"backend": "trtllm", "engine_dir": "/private/engine"},
        "cloud": {"backend": "openrouter", "api_url": "https://private.example"},
    },
}


def payload(text="Hello", model="local"):
    return {"model": model, "messages": [{"role": "user", "content": text}]}


def test_page_models_and_static_files_do_not_initialize_backends():
    def factory(entry):
        pytest.fail("Browsing the UI should not initialize a model")

    with TestClient(create_app(CONFIG, factory)) as client:
        response = client.get("/api/models")
        assert response.json() == {
            "default": "local",
            "models": [
                {"id": "local", "backend": "trtllm"},
                {"id": "cloud", "backend": "openrouter"},
            ],
        }
        assert "private" not in response.text
        assert "What’s on your mind?" in client.get("/").text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200


def test_backend_reused_without_sharing_conversations():
    initialized = []
    received = []

    def factory(entry):
        initialized.append(entry)

        def ask(messages):
            received.append(messages)
            return "Reply to " + messages[-1]["content"]

        return ask

    with TestClient(create_app(CONFIG, factory)) as client:
        assert client.post("/api/chat", json=payload("First")).json() == {"reply": "Reply to First"}
        assert client.post("/api/chat", json=payload("Second")).json() == {"reply": "Reply to Second"}
        conversation = payload("First")
        conversation["messages"] += [
            {"role": "assistant", "content": "Reply to First"},
            {"role": "user", "content": "Follow up"},
        ]
        assert client.post("/api/chat", json=conversation).status_code == 200
    assert len(initialized) == 1
    assert received[1] == payload("Second")["messages"]
    assert received[2] == conversation["messages"]


@pytest.mark.parametrize("messages", [
    [],
    [{"role": "system", "content": "Hello"}],
    [{"role": "user", "content": "   "}],
    [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
    [{"role": "user", "content": "a"}] * 3,
    [{"role": "user", "content": "a" * 100_001}],
])
def test_invalid_conversations_rejected_before_inference(messages):
    def factory(entry):
        pytest.fail("Invalid requests must not initialize a model")

    with TestClient(create_app(CONFIG, factory)) as client:
        assert client.post("/api/chat", json={"model": "local", "messages": messages}).status_code == 422


def test_unknown_model():
    with TestClient(create_app(CONFIG, lambda entry: None)) as client:
        assert client.post("/api/chat", json=payload(model="missing")).status_code == 404


@pytest.mark.parametrize("failure,code", [
    (requests.Timeout("secret"), 504),
    (requests.HTTPError("secret"), 502),
    (RuntimeError("secret"), 500),
])
def test_generation_errors_are_safe_and_recoverable(failure, code):
    def ask(messages):
        if messages[-1]["content"] == "Fail":
            raise failure
        return "Recovered"

    with TestClient(create_app(CONFIG, lambda entry: ask)) as client:
        response = client.post("/api/chat", json=payload("Fail"))
        assert response.status_code == code
        assert "secret" not in response.text
        assert client.post("/api/chat", json=payload()).json() == {"reply": "Recovered"}


def test_failed_initialization_can_be_retried():
    attempts = []

    def factory(entry):
        attempts.append(entry)
        if len(attempts) == 1:
            raise RuntimeError("secret setup details")
        return lambda messages: "Ready"

    with TestClient(create_app(CONFIG, factory)) as client:
        response = client.post("/api/chat", json=payload())
        assert response.status_code == 503
        assert "secret" not in response.text
        assert client.post("/api/chat", json=payload()).json() == {"reply": "Ready"}


def test_local_inference_serialized_on_initialization_thread():
    active = 0
    maximum_active = 0
    worker_threads = set()
    initialized = []

    def factory(entry):
        initialized.append(threading.get_ident())

        def ask(messages):
            nonlocal active, maximum_active
            worker_threads.add(threading.get_ident())
            active += 1
            maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            active -= 1
            return "Done"

        return ask

    with TestClient(create_app(CONFIG, factory)) as client:
        with ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(lambda _: client.post("/api/chat", json=payload()), range(4)))
        assert all(response.status_code == 200 for response in responses)
    assert maximum_active == 1
    assert len(initialized) == 1
    assert worker_threads == set(initialized)
