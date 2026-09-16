"""Model backends, each exposing create(config) -> ask(messages)."""

from . import openrouter, trtllm

BACKENDS = {"openrouter": openrouter, "trtllm": trtllm}


def create(config):
    return BACKENDS[config["backend"]].create(config)
