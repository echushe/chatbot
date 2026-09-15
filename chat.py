"""A minimal stateless chatbot over the OpenRouter API."""

import os
import sys

import requests

API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("OPENROUTER_MODEL", "qwen/qwen3.7-flash")
TIMEOUT = 60


def get_api_key():
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        sys.exit("OPENROUTER_API_KEY is not set.")
    return key


def ask(api_key, message):
    response = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": message}],
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main():
    api_key = get_api_key()
    print(f"Model: {MODEL}. Type /quit to exit.")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if not message:
            continue
        if message in ("/quit", "/exit"):
            return

        try:
            print(f"Bot: {ask(api_key, message)}")
        except requests.RequestException as error:
            print(f"Request failed: {error}")


if __name__ == "__main__":
    main()
