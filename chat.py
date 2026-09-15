"""A minimal stateless chatbot over a cloud or local model."""

import argparse
import os
import sys

import requests
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yaml")


def load_config():
    with open(CONFIG_PATH) as config_file:
        return yaml.safe_load(config_file)


def make_openrouter(config):
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


def make_trtllm(config):
    try:
        import torch
        from transformers import AutoTokenizer
        from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

        if PYTHON_BINDINGS:
            from tensorrt_llm.runtime import ModelRunnerCpp
    except ImportError as error:
        sys.exit(f"The local model needs the trtllm110 environment ({error}).")

    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(config["tokenizer_dir"]))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    engine_dir = os.path.expanduser(config["engine_dir"])
    if PYTHON_BINDINGS:
        runner = ModelRunnerCpp.from_dir(
            engine_dir=engine_dir,
            max_batch_size=1,
            max_beam_width=1,
            kv_cache_free_gpu_memory_fraction=config["kv_cache_fraction"],
        )
    else:
        runner = ModelRunner.from_dir(engine_dir=engine_dir, rank=0)

    # Llama 3 ends turns with <|eot_id|>; a generic EOS lets the 1B model ramble.
    end_token = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    end_id = end_token if end_token != tokenizer.unk_token_id else tokenizer.eos_token_id

    def ask(message):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            add_generation_prompt=True,
            tokenize=True,
        )
        outputs = runner.generate(
            batch_input_ids=[torch.tensor(prompt_ids, dtype=torch.int32)],
            max_new_tokens=config["max_new_tokens"],
            end_id=end_id,
            pad_id=tokenizer.pad_token_id,
            temperature=config["temperature"],
            top_p=config["top_p"],
            num_beams=1,
            return_dict=True,
            output_sequence_lengths=True,
        )
        sequence = outputs["output_ids"][0][0][: int(outputs["sequence_lengths"][0][0])]
        return tokenizer.decode(sequence[len(prompt_ids) :], skip_special_tokens=True).strip()

    return ask


BACKENDS = {"openrouter": make_openrouter, "trtllm": make_trtllm}


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

    selected = models[args.model]
    ask = BACKENDS[selected["backend"]](selected)
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
