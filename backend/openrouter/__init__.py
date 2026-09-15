"""Chat completions through the OpenRouter HTTP API."""

import os
import sys

import requests


def create(config):
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set.")

    def ask(message):
        response = requests.post(
            config["api_url"],
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": config["model"],
                "messages": [{"role": "user", "content": message}],
            },
            timeout=config["timeout"],
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    return ask
