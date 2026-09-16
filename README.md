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

Open <http://127.0.0.1:8000>. Select a model, enter a message, and press Enter to
send (Shift + Enter adds a line). New chat or changing models clears the current
conversation. Chats live only in the browser tab's memory and disappear on refresh.
Replies appear when generation finishes; token streaming is not implemented.

For OpenRouter, set `OPENROUTER_API_KEY` in the server environment before starting
the server. Local inference requires the engine and tokenizer paths in
`models.yaml` to exist on the server machine. Backends initialize on first use,
so the page can load even when a model is not available; detailed failures are
logged in the terminal.

Run **one worker**: the server caches model instances and serializes local
inference on one dedicated thread. Extra worker processes would each load their
own GPU engine. Cloud requests use a separate thread pool. Keep the default
loopback binding for personal use; authentication is not included.

## Command-line interface

```bash
python chat.py --list
python chat.py --model openrouter
```

Use `/reset` to clear the conversation and `/quit` to exit.

## API

- `GET /api/models`: configured model aliases, backend types, and default alias.
- `POST /api/chat`: accepts `{"model": "local", "messages": [{"role": "user", "content": "Hello"}]}`
  and returns `{"reply": "..."}`. Send the conversation on each request; messages
  must alternate user and assistant, starting and ending with a user message.
- `/docs`: generated API documentation.

## Tests

Tests use fake backends; no API credentials or GPU are needed.

```bash
python -m pip install pytest httpx
python -m pytest -q
```
