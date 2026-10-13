# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

from transformers import PreTrainedTokenizerBase, ProcessorMixin

from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

_THINK_PREFIX = "<think>\n"


def _append_think_prefix(processor, result, tokenize, return_dict):
    """Append '<think>\\n' to apply_chat_template output for Base model training."""
    if not tokenize:
        return result + _THINK_PREFIX
    if not return_dict:
        if isinstance(result, list) and result and isinstance(result[0], list):
            # transformers>=5: list[list[int]]
            think_ids = processor.encode(_THINK_PREFIX, add_special_tokens=False)
            return [r + think_ids for r in result]
        think_ids = processor.encode(_THINK_PREFIX, add_special_tokens=False)
        return list(result) + think_ids
    # return_dict=True
    import torch
    think_ids = processor.encode(_THINK_PREFIX, add_special_tokens=False)
    result = dict(result)
    extra = torch.tensor([think_ids], dtype=result["input_ids"].dtype)
    result["input_ids"] = torch.cat([result["input_ids"], extra], dim=1)
    result["attention_mask"] = torch.cat(
        [result["attention_mask"], torch.ones_like(extra)], dim=1
    )
    return result


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True)
    )
    token2 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True)
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = normalize_token_ids(
        tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    )
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def apply_chat_template(
    processor: PreTrainedTokenizerBase | ProcessorMixin,
    messages: list[dict],
    *,
    tokenize: bool = True,
    add_generation_prompt: bool = True,
    tools=None,
    return_dict: bool = False,
    **kwargs,
) -> list[int] | str:
    """apply_chat_template to messages with special attention to template requiring
    at least one user message, e.g. Qwen3.5.

    Args:
        processor: tokenizer or processor.
        messages: list[dict], messages.
        tokenize: bool, whether to tokenize the output.
        add_generation_prompt: bool, whether to add generation prompt.
        tools: list[dict], tools schema.
        return_dict: bool, whether to return a dict.
        **kwargs: additional arguments for apply_chat_template.

    Returns:
        list[int] | str: tokenized ids or text string.
    """
    # Pop custom kwarg before passing to the tokenizer.
    # When force_think_prefix=True, append <think>\n to the output so
    # Base models are forced to start generating thinking content.
    force_think = kwargs.pop("force_think_prefix", False)

    try:
        result = processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        if force_think:
            result = _append_think_prefix(processor, result, tokenize, return_dict)
        return result
    except Exception:
        # Qwen3.5 apply_chat_template needs messages with at least one user message
        dummy_user_message = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        dummy_user_prefix = processor.apply_chat_template(
            dummy_user_message,
            tokenize=tokenize,
            add_generation_prompt=False,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )
        output = processor.apply_chat_template(
            dummy_user_message + messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            tools=tools,
            return_dict=return_dict,
            **kwargs,
        )

        if not tokenize:  # tokenize=False
            result = output[len(dummy_user_prefix) :]
        elif not return_dict:  # tokenize=True and return_dict=False
            if isinstance(output[0], list):  # transformers>=5
                assert len(output) == 1, "output must be a list[int] or list[list[int]]"
                dummy_user_prefix = dummy_user_prefix[0]
                output = output[0]
            result = output[len(dummy_user_prefix) :]
        else:  # tokenize=True and return_dict=True and return_tensors="pt"
            dummy_user_prefix = dict(dummy_user_prefix)
            output = dict(output)
            prefix_len = dummy_user_prefix["input_ids"].shape[1]
            output["input_ids"] = output["input_ids"][:, prefix_len:]
            output["attention_mask"] = output["attention_mask"][:, prefix_len:]
            if "mm_token_type_ids" in output:
                output["mm_token_type_ids"] = output["mm_token_type_ids"][:, prefix_len:]
            result = output
        if force_think:
            result = _append_think_prefix(processor, result, tokenize, return_dict)
        return result
