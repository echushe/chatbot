"""A minimal stateless chatbot over a cloud or local model."""

import argparse
import os

import yaml

import backend

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yaml")


def load_config():
    with open(CONFIG_PATH) as config_file:
        return yaml.safe_load(config_file)


def main():
    config = load_config()
    models = config["models"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config["default"], choices=sorted(models))
    parser.add_argument("--list", action="store_true", help="show configured models and exit")
    args = parser.parse_args()

    if args.list:
        for name, entry in models.items():
            default = "*" if name == config["default"] else " "
            print(f"{default} {name:12} {entry.get('model') or entry['engine_dir']}")
        return

    ask = backend.create(models[args.model])
    print(f"Model: {args.model}. Type /quit to exit.")

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
            print(f"Bot: {ask(message)}")
        except Exception as error:
            print(f"Request failed: {error}")


if __name__ == "__main__":
    main()
