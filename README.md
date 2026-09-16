# Chatbot

A Python chatbot with a command-line interface and a small browser UI. Models
are configured in `models.yaml`; both interfaces use the same backends.

## Web interface

Use Python 3.10 or newer. For local TensorRT-LLM inference, activate the existing
`trtllm110` environment first so the server can import its GPU dependencies.

```bash
python -m pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000>. On the first visit, a chatbot is created and given a
random name from a bundled, offline list. On later visits, choose a saved chatbot
from the dropdown or select **New chatbot**. Each chatbot has its own transcript
in `chatbot.db`, which survives page refreshes and server restarts. Existing CLI
conversations receive names automatically; their messages are preserved.

Enter sends a message; Shift + Enter adds a line. Changing the model keeps the
conversation and saves the new model choice after the next successful reply.
Replies appear when generation finishes; token streaming is not implemented.

Each tab gets a temporary activity session when it opens a chatbot. After one
minute without opening a chatbot or sending a message, that session expires.
Time spent waiting for a reply does not count as idle. Sending the next message
automatically starts a fresh activity session for the same saved chatbot and
restores its transcript. The page stays open, and the shared model stays loaded.
Typing alone does not extend the session; a draft remains available after expiry.

For OpenRouter, set `OPENROUTER_API_KEY` in the server environment before starting
the server. Local inference requires the engine and tokenizer paths in
`models.yaml` to exist on the server machine. Backends initialize on first use,
so the page can load even when a model is not available; detailed failures are
logged in the terminal.

Run **one worker**: the server caches model instances and serializes local
inference on one dedicated thread. Extra worker processes would each load their
own GPU engine. Multiple tabs, connections, and named chatbots share one cached
instance of each model. Local requests queue for the batch-one GPU engine; cloud
requests use a separate thread pool. Turns sent to the same chatbot are serialized
so their histories cannot interleave. Only completed exchanges are saved.

The chatbot catalog is shared by everyone who can reach this server, including
the CLI; there are no accounts or private per-user catalogs. Keep the default
loopback binding for personal use; authentication is not included.

## Command-line interface

```bash
python chat.py --list
python chat.py --model openrouter
```

The CLI resumes the latest saved conversation. Use `--new` or `/reset` to create
a new one and `/quit` to exit. The idle activity timeout applies to the web UI.

## API

- `GET /api/models`: configured model aliases, backend types, and default alias.
- `GET /api/chatbots`: lists named chatbots, creating one if the database is empty.
- `POST /api/chatbots`: creates a randomly named chatbot from `{"model": "local"}`.
- `POST /api/chatbots/{id}/open`: returns the chatbot, saved messages, and a new
  temporary `session_id` with `idle_timeout_seconds`.
- `POST /api/chatbots/{id}/messages`: accepts
  `{"content": "Hello", "model": "local", "session_id": "..."}`. The session ID
  is optional; an absent or expired ID resumes the chatbot automatically. Returns
  the saved transcript and current session ID. The server loads the history;
  browsers do not submit it. Accepted turns finish saving even if a tab disconnects.
- `POST /api/chat`: accepts `{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}`
  and returns `{"reply": "..."}`. This original stateless endpoint remains available
  and does not save a chatbot. Send the conversation on each request; messages
  must alternate user and assistant, starting and ending with a user message.
- `/docs`: generated API documentation.

## Tests

Tests use fake backends; no API credentials or GPU are needed.

```bash
python -m pip install pytest httpx
python -m pytest -q
```
