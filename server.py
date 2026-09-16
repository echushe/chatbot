"""Serve the chat page and API: python -m uvicorn server:app --workers 1."""

import asyncio
import logging
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from threading import Lock
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

import backend
from chat import DB_PATH, load_config
from store import Store

WEB_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger(__name__)
IDLE_TIMEOUT = 60


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    model: str
    messages: list[Message] = Field(min_length=1, max_length=501)

    @model_validator(mode="after")
    def validate_conversation(self):
        if len(self.messages) % 2 != 1:
            raise ValueError("The conversation must end with a user message.")
        for index, message in enumerate(self.messages):
            expected = "user" if index % 2 == 0 else "assistant"
            if message.role != expected or not message.content.strip():
                raise ValueError("Use nonempty messages alternating user and assistant.")
        if sum(len(message.content) for message in self.messages) > 1_000_000:
            raise ValueError("Conversation is too long. Start a new chat.")
        return self


class ChatResponse(BaseModel):
    reply: str


class NewChatbot(BaseModel):
    model: str


class SavedMessage(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)
    model: str
    session_id: str | None = Field(default=None, max_length=100)

    @model_validator(mode="after")
    def nonempty(self):
        if not self.content.strip():
            raise ValueError("Enter a message before sending.")
        return self


def create_app(config=None, backend_factory=None, db_path=None, clock=None):
    """Allow an isolated configuration and fake backend for API tests."""
    factory = backend_factory if backend_factory is not None else backend.create
    database = db_path if db_path is not None else DB_PATH
    now = clock if clock is not None else time.monotonic

    def expire_sessions():
        for token, session in list(app.state.sessions.items()):
            if not session["pending"] and now() - session["last_active"] >= IDLE_TIMEOUT:
                del app.state.sessions[token]

    async def reap_sessions():
        while True:
            await asyncio.sleep(1)
            expire_sessions()

    @asynccontextmanager
    async def lifespan(app):
        app.state.config = config if config is not None else load_config()
        app.state.backends = {}
        app.state.locks = {name: Lock() for name in app.state.config["models"]}
        app.state.sessions = {}
        app.state.chatbot_locks = {}
        app.state.pending_tasks = set()
        # Keep GPU initialization and inference on the same dedicated thread.
        app.state.local_pool = ThreadPoolExecutor(max_workers=1)
        app.state.cloud_pool = ThreadPoolExecutor(max_workers=4)
        reaper = asyncio.create_task(reap_sessions())
        try:
            yield
        finally:
            reaper.cancel()
            with suppress(asyncio.CancelledError):
                await reaper
            if app.state.pending_tasks:
                await asyncio.gather(*app.state.pending_tasks, return_exceptions=True)
            app.state.local_pool.shutdown(wait=True, cancel_futures=True)
            app.state.cloud_pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Chatbot", lifespan=lifespan)

    def model_entry(model):
        entry = app.state.config["models"].get(model)
        if entry is None:
            raise HTTPException(404, "Unknown model. Refresh the page and choose a model.")
        return entry

    def read_chatbot(chatbot_id):
        with Store(database) as store:
            chatbot = store.chatbot(chatbot_id)
            if chatbot is None:
                raise HTTPException(404, "Chatbot not found.")
            return {"chatbot": chatbot, "messages": store.messages(chatbot_id)}

    def resume_session(chatbot_id, token=None):
        expire_sessions()
        session = app.state.sessions.get(token)
        if session is not None and session["chatbot_id"] != chatbot_id:
            raise HTTPException(409, "This session belongs to another chatbot.")
        resumed = session is None
        if resumed:
            token = secrets.token_urlsafe(24)
            session = {"chatbot_id": chatbot_id, "last_active": now(), "pending": 0}
            app.state.sessions[token] = session
        session["last_active"] = now()
        return token, session, resumed

    @app.get("/api/chatbots")
    def list_chatbots():
        with Store(database) as store:
            created = store.latest_session() is None
            bots = store.chatbots(app.state.config["default"])
        return {"chatbots": bots, "created": created, "idle_timeout_seconds": IDLE_TIMEOUT}

    @app.post("/api/chatbots", status_code=201)
    def create_chatbot(request: NewChatbot):
        model_entry(request.model)
        with Store(database) as store:
            chatbot_id = store.start_session(request.model)
            return store.chatbot(chatbot_id)

    @app.post("/api/chatbots/{chatbot_id}/open")
    async def open_chatbot(chatbot_id: int):
        data = await asyncio.to_thread(read_chatbot, chatbot_id)
        token, _, _ = resume_session(chatbot_id)
        return {**data, "session_id": token, "idle_timeout_seconds": IDLE_TIMEOUT}

    def generate_saved(chatbot_id, lock, request, entry):
        # Serialize read / generate / append for one transcript, even if two tabs
        # send to the same cloud-backed bot at once. Different bots stay independent.
        with lock:
            data = read_chatbot(chatbot_id)
            messages = data["messages"] + [{"role": "user", "content": request.content}]
            reply = generate(request.model, entry, messages).reply
            with Store(database) as store:
                store.append_exchange(chatbot_id, request.content, reply, model=request.model)
            return read_chatbot(chatbot_id)

    @app.post("/api/chatbots/{chatbot_id}/messages")
    async def send_saved_message(chatbot_id: int, request: SavedMessage):
        entry = model_entry(request.model)
        # Check existence before allocating a session or a lock.
        await asyncio.to_thread(read_chatbot, chatbot_id)
        token, session, resumed = resume_session(chatbot_id, request.session_id)
        session["pending"] += 1
        lock = app.state.chatbot_locks.setdefault(chatbot_id, Lock())
        pool = app.state.local_pool if entry["backend"] == "trtllm" else app.state.cloud_pool

        async def complete():
            try:
                data = await asyncio.wrap_future(pool.submit(generate_saved, chatbot_id, lock, request, entry))
                return {**data, "session_id": token, "resumed": resumed, "idle_timeout_seconds": IDLE_TIMEOUT}
            finally:
                session["pending"] -= 1
                session["last_active"] = now()

        # Finish and persist an accepted turn even if the browser disconnects.
        task = asyncio.create_task(complete())
        app.state.pending_tasks.add(task)
        task.add_done_callback(app.state.pending_tasks.discard)
        return await asyncio.shield(task)

    @app.get("/api/models")
    def list_models():
        settings = app.state.config
        return {
            "default": settings["default"],
            "models": [
                {"id": name, "backend": entry["backend"]}
                for name, entry in settings["models"].items()
            ],
        }

    def generate(model, entry, messages):
        with app.state.locks[model]:
            if model not in app.state.backends:
                try:
                    app.state.backends[model] = factory(entry)
                except Exception:
                    logger.exception("Could not initialize model %s", model)
                    raise HTTPException(
                        503, "Could not load this model. Check its setup and the server log."
                    ) from None
        try:
            reply = app.state.backends[model](messages)
            if not isinstance(reply, str) or not reply.strip():
                raise RuntimeError("The model returned an empty reply.")
            return ChatResponse(reply=reply)
        except requests.Timeout:
            raise HTTPException(504, "The model timed out. Please try again.") from None
        except requests.RequestException:
            logger.exception("Provider request failed for %s", model)
            raise HTTPException(502, "The model provider request failed. Please try again.") from None
        except Exception:
            logger.exception("Generation failed for %s", model)
            raise HTTPException(500, "Could not generate a reply. Check the server log.") from None

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        entry = app.state.config["models"].get(request.model)
        if entry is None:
            raise HTTPException(404, "Unknown model. Refresh the page and choose a model.")
        pool = (
            app.state.local_pool if entry["backend"] == "trtllm" else app.state.cloud_pool
        )
        messages = [message.model_dump() for message in request.messages]
        return await asyncio.wrap_future(pool.submit(generate, request.model, entry, messages))

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


app = create_app()
