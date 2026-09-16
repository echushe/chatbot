"""Local inference against a compiled TensorRT-LLM engine."""

import os

from ..history import trim


def create(config):
    # Imported here, not at module scope, so environments without the
    # TensorRT-LLM stack can still run the other backends.
    try:
        import torch
        from transformers import AutoTokenizer
        from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner

        if PYTHON_BINDINGS:
            from tensorrt_llm.runtime import ModelRunnerCpp
    except ImportError as error:
        raise RuntimeError(f"The local model needs the trtllm110 environment ({error}).") from error

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

    # max_seq_len is compiled into the engine: overflowing it raises rather
    # than truncating, so the prompt must leave room for the whole reply.
    budget = config["max_seq_len"] - config["max_new_tokens"]

    def count_tokens(messages):
        return len(
            tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )
        )

    def ask(messages):
        prompt_ids = tokenizer.apply_chat_template(
            trim(messages, budget, count_tokens),
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
