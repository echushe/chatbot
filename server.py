"""Serve the chat page and API: python -m uvicorn server:app --workers 1."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

import backend
from chat import load_config

WEB_DIR = Path(__file__).resolve().parent / "web"
logger = logging.getLogger(__name__)


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


def create_app(config=None, backend_factory=None):
    """Allow an isolated configuration and fake backend for API tests."""
    factory = backend_factory if backend_factory is not None else backend.create

    @asynccontextmanager
    async def lifespan(app):
        app.state.config = config if config is not None else load_config()
        app.state.backends = {}
        app.state.locks = {name: Lock() for name in app.state.config["models"]}
        # Keep GPU initialization and inference on the same dedicated thread.
        app.state.local_pool = ThreadPoolExecutor(max_workers=1)
        app.state.cloud_pool = ThreadPoolExecutor(max_workers=4)
        try:
            yield
        finally:
            app.state.local_pool.shutdown(wait=True, cancel_futures=True)
            app.state.cloud_pool.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="Chatbot", lifespan=lifespan)

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
