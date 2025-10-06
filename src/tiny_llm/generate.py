import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper
from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from typing import Callable


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: TokenizerWrapper,
    prompt: str,
    sampler: Callable[[mx.array], mx.array] | None,
) -> str:
    # helper function to get the next token
    def _step(model: Qwen2ModelWeek1, y: mx.array) -> int:
        x = y[None, :] # add a batch dim
        output_logits = model(x)
        logits = output_logits[:, -1, :]
        return int(mx.argmax(logits).item())

    # get detokenizer from tokenizer wrapper
    detokenizer = tokenizer.detokenizer

    # tokenize the prompt
    tokens = mx.array(tokenizer.encode(prompt, add_special_tokens=False))

    # generate tokens
    while True:
        next_token = _step(model, tokens)
        # stop if next token is eos
        if next_token in tokenizer.eos_token_ids:
            break

        detokenizer.add_token(next_token)
        tokens = mx.concat([tokens, mx.array([next_token])])

    # get detokenized output and print it
    output = detokenizer.last_segment
    print(output)

    return output


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2, tokenizer: TokenizerWrapper, prompt: str
) -> str:
    def _step(model, y, offset, kv_cache):
        pass


def speculative_generate(
    draft_model: Qwen2ModelWeek2,
    model: Qwen2ModelWeek2,
    draft_tokenizer: TokenizerWrapper,
    tokenizer: TokenizerWrapper,
    prompt: str,
) -> str:
    pass
