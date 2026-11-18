"""Validate unrendered text prompts for native MILES TITO agent episodes."""

import hashlib
import json

from open_instruct.miles.errors import InputError


def verify_row(row, tokenizer, limit, *, exclude_over_length=False):
    messages = row.get("input")
    if not isinstance(messages, list) or not messages:
        raise InputError("Agent input must be a nonempty unrendered messages list")
    for message in messages:
        if (
            not isinstance(message, dict)
            or message.get("role") not in ("system", "user")
            or not isinstance(message.get("content"), str)
        ):
            raise InputError("Fresh agent prompts support text system/user messages only")
    if messages[-1]["role"] != "user":
        raise InputError("Fresh agent prompts must end with a user message")
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise InputError("Agent metadata must be an object")
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if not ids or (len(ids) > limit and not exclude_over_length):
        raise InputError(f"Agent prompt has {len(ids)} tokens, outside 1..{limit}")
    metadata["run_prompt_tokens"] = len(ids)
    metadata["run_prompt_token_ids_sha256"] = hashlib.sha256(
        (json.dumps(ids, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()
    ).hexdigest()
