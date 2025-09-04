import copy
import re
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from intervaltree import IntervalTree

import torch
from transformers import PreTrainedTokenizer


def find_message_spans(
    messages: List[Dict[str, str]], formatted_text: str
) -> List[Dict]:
    """
    Find the start and end character indices for each message's content in the formatted chat template
    and return enriched message dictionaries with span information.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        formatted_text: The text output from tokenizer.apply_chat_template()

    Returns:
        List of message dictionaries with added 'span' key containing (start, end) tuple
    """
    enriched_messages = copy.deepcopy(messages)
    search_start = 0

    for message in enriched_messages:
        content = message["content"]
        # Escape special regex characters in content
        escaped_content = re.escape(content)

        # Find the content, starting search from where we left off
        match = re.search(escaped_content, formatted_text[search_start:])
        if match:
            # Add the search_start offset to get the actual position in the full string
            start = match.start() + search_start
            end = match.end() + search_start
            message["span"] = (start, end)
            search_start = end
        else:
            # If content not found, add None span
            message["span"] = None

    return enriched_messages


def apply_chat_template(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    *,
    template: Optional[str] = None,
    add_generation_prompt: bool = False,
) -> str:
    """
    Apply a chat template to a list of messages and return the formatted text.

    Args:
        tokenizer: A tokenizer object from Hugging Face tokenizers library
        messages: List of message dictionaries with 'role' and 'content' keys
        template: The chat template string
    """

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        template=template,
        add_generation_prompt=add_generation_prompt,
    )
    messages = find_message_spans(messages, formatted_text)

    return formatted_text, messages


def map_token_offsets_to_messages(
    offset_mapping: torch.Tensor, enriched_messages: List[Dict]
) -> List[int]:
    """
    Maps each token's offset span to its corresponding message index.

    Args:
        offset_mapping: Tensor of shape (batch_size, sequence_length, 2) containing token offsets
        enriched_messages: List of message dictionaries with 'span' information

    Returns:
        List of message indices for each token. -1 indicates no corresponding message
        (e.g., for special tokens or template text)
    """
    # Convert to list if tensor
    if isinstance(offset_mapping, torch.Tensor):
        # Remove batch dimension if present
        if offset_mapping.dim() == 3:
            offset_mapping = offset_mapping[0]
        offset_mapping = offset_mapping.tolist()

    message_indices = []

    for token_start, token_end in offset_mapping:
        found_message = False

        # Special case for [0, 0] offsets (usually special tokens)
        if token_start == 0 and token_end == 0:
            message_indices.append(-1)
            continue

        # Check each message's span
        for msg_idx, message in enumerate(enriched_messages):
            if message["span"] is None:
                continue

            msg_start, msg_end = message["span"]

            # Check if token span overlaps with message span
            if (
                (token_start >= msg_start and token_start < msg_end)
                or (token_end > msg_start and token_end <= msg_end)
                or (token_start <= msg_start and token_end >= msg_end)
            ):
                message_indices.append(msg_idx)
                found_message = True
                break

        if not found_message:
            message_indices.append(-1)  # Token belongs to template text

    return message_indices


def get_message_token_spans(message_indices: List[int]) -> Dict[int, List[int]]:
    """
    Get the start and end token spans for each message index.

    Args:
        message_indices: List of message indices for each token (-1 for non-message tokens)

    Returns:
        Dictionary mapping message indices to [start, end] token spans
    """
    spans = {}
    current_start = None
    current_msg = None

    # Add sentinel value to handle last span
    for i, msg_idx in enumerate(message_indices + [-2]):
        if msg_idx != current_msg:
            # Save previous span if it exists
            if current_msg is not None and current_msg != -1:
                spans[current_msg] = [current_start, i]
            # Start new span
            current_start = i
            current_msg = msg_idx

    return spans


def message_span_aware_tokenization(
    tokenizer: PreTrainedTokenizer,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
    **kwargs,
):
    formatted_text, messages = apply_chat_template(
        tokenizer,
        messages,
        template=None,
        add_generation_prompt=add_generation_prompt,
    )
    tokenized_input = tokenizer(
        formatted_text,
        return_tensors=kwargs.get("return_tensors", None),
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    token_message_indices = map_token_offsets_to_messages(
        tokenized_input["offset_mapping"], messages
    )
    message_token_spans = get_message_token_spans(token_message_indices)

    if kwargs.get("return_tensors", None) == "pt":
        token_message_indices = torch.LongTensor(
            [token_message_indices]
        )  # unsqueezed version

    tokenized_input["token_message_indices"] = token_message_indices
    tokenized_input["message_token_spans"] = message_token_spans
    return tokenized_input


def locate_token_string_in_original_input(
    tokenized_input: Dict[str, Any],
    target_indices: List[int],
    original_document_offset_in_prompt: int = 0,
    target_message_span_index: Optional[int] = None,
) -> List[Optional[Tuple[int, int]]]:
    """For a given token index"""
    if target_message_span_index is None:
        # use the last message span
        target_message_span_index = max(tokenized_input["message_token_spans"].keys())

    document_message_span = tokenized_input["message_token_spans"][
        target_message_span_index
    ]
    assert all(
        [
            document_message_span[0] <= ele < document_message_span[1]
            for ele in target_indices
            if ele is not None and ele != -1
        ]
    ), f"target_indices: {target_indices}, document_message_span: {document_message_span}"

    document_message_start_string_index = tokenized_input["offset_mapping"][
        document_message_span[0]
    ][0]

    return [
        (
            (
                tokenized_input["offset_mapping"][ele][0]
                - document_message_start_string_index
                - original_document_offset_in_prompt,
                tokenized_input["offset_mapping"][ele][1]
                - document_message_start_string_index
                - original_document_offset_in_prompt,
            )
            if ele is not None and ele != -1
            else None
        )
        for ele in target_indices
    ]


def convert_string_spans_to_token_spans(
    tokenized_input: Dict[str, Any],
    message_string_spans: List[Tuple[int, int]],
    target_message_span_index: int
) -> int:
    """
    Convert a list of string spans to a list of token indices.
    """

    # first, we need to obtain the local offset_mapping given the target_message_span_index
    message_start_token_index, message_end_token_index = tokenized_input["message_token_spans"][
        target_message_span_index
    ]
    message_start_string_index = tokenized_input["offset_mapping"][message_start_token_index][0]

    # Then we need to obtain the global message string spans and obtain the token spans: 
    # The input message_string_spans is a list of tuples, each tuple is a local string span in the original document

    output_token_spans = []
    for local_string_start, local_string_end in message_string_spans:
        global_string_start = local_string_start + message_start_string_index
        global_string_end = local_string_end + message_start_string_index
        
        current_token_start, current_token_end = None, None

        for token_index, (token_string_start, token_string_end) in enumerate(tokenized_input["offset_mapping"][message_start_token_index:message_end_token_index], start=message_start_token_index):
            
            if token_string_start >= global_string_start and token_string_end <= global_string_end:
                if current_token_start is None:
                    current_token_start = token_index
                current_token_end = token_index + 1

            if token_string_end > global_string_end:
                break
        
        if current_token_start is not None and current_token_end is not None:
            output_token_spans.append((current_token_start, current_token_end))

    return output_token_spans


def clean_tokenizer_vocab(tokenizer, static_prefix="abcdef"):
    """
    This method turns a tokenizer vocab into a "clean" vocab where each token represents the actual string
    it will yield, without any special prefixes like "##" or "Ġ". This is trickier than it looks - the method
    tokenizer.convert_tokens_to_string() does not always return the correct string because of issues with prefix
    space addition/removal. To work around this, we add a static prefix to the start of the token, then remove
    it (and any prefix that may have been introduced with it) after calling convert_tokens_to_string().
    """
    vocab = tokenizer.get_vocab()
    clean_token_list = []
    clean_token_indices = []
    sentence_base = tokenizer(static_prefix, add_special_tokens=False)["input_ids"]
    tokens_base = [tokenizer._convert_id_to_token(tok) for tok in sentence_base]
    for token, token_idx in vocab.items():
        token_string = tokenizer.convert_tokens_to_string(tokens_base + [token])
        token_string = token_string[
            token_string.index(static_prefix) + len(static_prefix) :
        ]
        clean_token_list.append(token_string)
        clean_token_indices.append(token_idx)
    return tuple(clean_token_list), tuple(clean_token_indices)


def decode_with_offset_mapping(
    token_ids: List[int],
    tokenizer: PreTrainedTokenizer,
    skip_special_tokens: bool = False,
):
    if not hasattr(tokenizer, "clean_vocab_id_to_token_map"):
        warnings.warn(
            "Tokenizer does not have clean_vocab_id_to_token_map attribute. We try to create one"
        )
        token_list, token_indices = clean_tokenizer_vocab(tokenizer)
        tokenizer.clean_vocab_id_to_token_map = {
            idx: token for token, idx in zip(token_list, token_indices)
        }

    if skip_special_tokens:
        words = [
            tokenizer.clean_vocab_id_to_token_map[token]
            for token in token_ids
            if token not in tokenizer.all_special_ids
        ]

        cur_index = 0
        offset_mapping = []
        for token_index, token_id in enumerate(token_ids):
            if token_id in tokenizer.all_special_ids:
                offset_mapping.append((cur_index, cur_index))
            else:
                token_string = tokenizer.clean_vocab_id_to_token_map[token_id]
                offset_mapping.append((cur_index, cur_index + len(token_string)))
                cur_index += len(token_string)
    else:
        words = [tokenizer.clean_vocab_id_to_token_map[token] for token in token_ids]

        cur_index = 0
        offset_mapping = []
        for token_index, token_string in enumerate(words):
            offset_mapping.append((cur_index, cur_index + len(token_string)))
            cur_index += len(token_string)

    # Create interval tree for inverse mapping
    interval_tree = IntervalTree()
    for token_idx, (start, end) in enumerate(offset_mapping):
        if start != end:  # Only add non-empty spans
            interval_tree.addi(start, end, token_idx)

    return {
        "text": "".join(words),
        "token_index_to_string_span": offset_mapping,
        "string_index_to_token_index": interval_tree,
    }


def find_tag_spans_in_text(
    text: str, search_tag: str, search_offset: int = 0
) -> list[tuple]:
    """
    Find all statement spans in the text by searching for <statement> and </statement> tags interleaved.
    Ignores incomplete statements (those without closing tags).

    Args:
        text (str): Input text containing statement tags
        search_offset (int): Initial offset to start searching from

    Returns:
        list[tuple]: List of tuples containing (start_index, end_index) for each complete statement
    """
    spans = []
    current_pos = search_offset

    begin_tag_text = f"<{search_tag}" # Sometimes the tag might contain additional fields like "type" or "id"
    end_tag_text = f"</{search_tag}>"

    while True:
        # Find the next opening tag
        start_tag = text.find(begin_tag_text, current_pos)
        if start_tag == -1:  # No more opening tags found
            break

        # Find the next closing tag
        end_tag = text.find(end_tag_text, start_tag)
        if end_tag == -1:  # No closing tag found for this opening tag
            break

        # Add the span (including tags)
        spans.append((start_tag, end_tag + len(end_tag_text)))

        # Move position to after the closing tag
        current_pos = end_tag + len(end_tag_text)

    return spans