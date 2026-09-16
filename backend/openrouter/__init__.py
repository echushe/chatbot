"""Chat completions through the OpenRouter HTTP API."""

import os
import sys

import requests

from ..history import trim

# The API exposes no tokenizer, and four characters per token is close enough
# to bound a conversation that has a million tokens of room to grow into.
CHARS_PER_TOKEN = 4


def estimate_tokens(messages):
    return sum(len(message["content"]) for message in messages) // CHARS_PER_TOKEN


def create(config):
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set.")

    def ask(messages):
        response = requests.post(
            config["api_url"],
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config["model"],
                "messages": trim(messages, config["context_budget"], estimate_tokens),
            },
            timeout=config["timeout"],
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return ask
