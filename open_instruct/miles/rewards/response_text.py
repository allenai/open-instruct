"""Shared, lightweight cleanup of generation markers before grading."""

STRIPPED_SPECIAL_TOKENS = frozenset({"<|endoftext|>", "<|im_end|>"})


def strip_terminal_special_tokens(
    response: str, *, stripped_special_tokens: frozenset[str] = STRIPPED_SPECIAL_TOKENS
) -> str:
    """Remove trailing sequences of selected markers, preserving other content.

    MILES retains stop tokens in rollout text. Graders and their direct callers
    share this selected set rather than removing every tokenizer special token.
    Interior occurrences and whitespace before the markers remain unchanged.
    Reapplying the cleanup has no effect; an empty set preserves the input.
    """
    # Prefer the longest suffix when selected markers overlap. Ignore empty
    # entries so extending the set cannot introduce a non-progressing match.
    suffixes = sorted(
        (token for token in stripped_special_tokens if token), key=lambda token: len(token), reverse=True
    )
    while True:
        trimmed = response.rstrip()
        for suffix in suffixes:
            if trimmed.endswith(suffix):
                response = trimmed.removesuffix(suffix)
                break
        else:
            return response
