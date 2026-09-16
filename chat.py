"""A chatbot over a cloud or local model, with a persistent transcript."""

import argparse
import os

import yaml

import backend
from store import Store

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "models.yaml")
DB_PATH = os.path.join(HERE, "chatbot.db")


def load_config():
    with open(CONFIG_PATH) as config_file:
        return yaml.safe_load(config_file)


def main():
    config = load_config()
    models = config["models"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config["default"], choices=sorted(models))
    parser.add_argument("--list", action="store_true", help="show configured models and exit")
    parser.add_argument("--resume", action="store_true", help="continue the most recent session")
    args = parser.parse_args()

    if args.list:
        for name, entry in models.items():
            default = "*" if name == config["default"] else " "
            print(f"{default} {name:12} {entry.get('model') or entry['engine_dir']}")
        return

    try:
        ask = backend.create(models[args.model])
    except Exception as error:
        parser.exit(1, f"Could not load model: {error}\n")

    with Store(DB_PATH) as store:
        session = store.latest_session() if args.resume else None
        history = store.messages(session) if session else []
        if session is None:
            session = store.start_session(args.model)

        restored = f", {len(history)} messages restored" if history else ""
        print(f"Model: {args.model}. Session {session}{restored}.")
        print("Type /reset to start a new session, /quit to exit.")

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
            if message == "/reset":
                history.clear()
                session = store.start_session(args.model)
                print(f"Started session {session}.")
                continue

            history.append({"role": "user", "content": message})
            try:
                reply = ask(history)
            except Exception as error:
                history.pop()
                print(f"Request failed: {error}")
                continue

            history.append({"role": "assistant", "content": reply})
            store.append_exchange(session, message, reply)
            print(f"Bot: {reply}")


if __name__ == "__main__":
    main()
