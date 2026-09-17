# Chatbot

A Python chatbot with a command-line interface and a small browser UI. Models
are configured in `models.yaml`; both interfaces use the same backends.

## Web interface

Use Python 3.10 or newer. For local TensorRT-LLM inference, activate the existing
`trtllm110` environment first so the server can import its GPU dependencies.
Run these commands from the project directory:

```bash
python -m pip install -r requirements.txt
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --workers 1
```

`server:app` loads the FastAPI application in `server.py`. It imports configuration
helpers from `chat.py`, but does not run its CLI `main()` function. The web UI
needs only the Uvicorn process; no separate `python chat.py` process is needed.

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
backend instance per configured model alias. With the current single `local`
alias, there is one TensorRT-LLM engine instance once it has been used. Each
chatbot keeps its own transcript; loading another chatbot does not load another
engine. Local requests queue for the batch-one GPU engine; cloud
requests use a separate thread pool. Turns sent to the same chatbot are serialized
so their histories cannot interleave. Only completed exchanges are saved.

Session expiry leaves the model and its GPU memory allocated for reuse. Stop the
server with Ctrl+C to shut down the process and release its GPU allocations.

The chatbot catalog is shared by everyone who can reach this server, including
the CLI; there are no accounts or private per-user catalogs. Keep the default
loopback binding for personal use; authentication is not included.

### Access from another machine on the LAN

The default `127.0.0.1` binding accepts connections only from the server machine.
Stop the running server with Ctrl+C and restart it with:

```bash
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --workers 1
```

On Linux, run `hostname -I` to find the server's addresses and choose the one on
your LAN. On the other machine, open `http://<server-lan-ip>:8000`; for example,
`http://192.168.50.49:8000` if that is the server's address. `0.0.0.0` is the
listening address, not the address to enter in the browser.

If the connection still fails, check that the host firewall allows TCP port 8000
from the LAN and that the devices are not separated by guest Wi-Fi isolation.
To check the listener on Linux, run `ss -ltnp 'sport = :8000'`; it should show
`0.0.0.0:8000`. LAN visitors use the same shared chatbot catalog described above.

## Command-line interface

```bash
python chat.py --list
python chat.py --model openrouter
```

The CLI resumes the latest saved conversation. Use `--new` or `/reset` to create
a new one and `/quit` to exit. The idle activity timeout applies to the web UI.
The CLI shares the backend code and SQLite database with the web server, but runs
in its own process. Starting `python chat.py --model local` loads an additional
engine instance; it does not connect to the web server's cached engine.

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
