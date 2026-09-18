"""Load tokenizers without losing the serialized tokenization pipeline."""

import json
from typing import Any

from transformers import AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizerFast
from transformers.utils import hub

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def load_tokenizer(name_or_path, *, revision=None, trust_remote_code=False, use_fast=True):
    """Preserve custom GPT-2 pre-tokenizers stored in tokenizer.json."""
    kwargs: dict[str, Any] = {"revision": revision, "trust_remote_code": trust_remote_code, "use_fast": use_fast}
    tokenizer = AutoTokenizer.from_pretrained(name_or_path, **kwargs)
    if not use_fast or not isinstance(tokenizer, GPT2Tokenizer) or not tokenizer.is_fast:
        return tokenizer

    tokenizer_file = hub.cached_file(
        name_or_path, "tokenizer.json", revision=revision, _raise_exceptions_for_missing_entries=False
    )
    if tokenizer_file is None:
        return tokenizer
    with open(tokenizer_file) as source:
        serialized = json.load(source)
    loaded = json.loads(tokenizer.backend_tokenizer.to_str())
    if loaded.get("pre_tokenizer") == serialized.get("pre_tokenizer"):
        return tokenizer

    # Transformers 5's GPT2Tokenizer rebuilds the backend using ByteLevel even
    # when the file specifies a Split regex. Load the file-backed fast class,
    # including its special-token and chat-template configuration, instead.
    logger.warning("Restoring serialized tokenizer backend for %s", name_or_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(name_or_path, **kwargs)
    restored = json.loads(tokenizer.backend_tokenizer.to_str())
    if restored.get("pre_tokenizer") != serialized.get("pre_tokenizer"):
        raise ValueError(f"Could not preserve tokenizer.json pre-tokenizer for {name_or_path}")
    return tokenizer
