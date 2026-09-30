"""Conservative exact response-span alignment; no cross-vocabulary KL claim."""

import math


def decode(tokenizer, ids):
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def spans(tokenizer, ids):
    """Retain only boundaries whose incremental decoding is an exact text prefix.

    Incomplete byte fallback/UTF-8 fragments have no valid boundary, so neither
    side of such a token can accidentally align through a replacement character.
    The bounded pilot deliberately favors correctness over quadratic decode cost.
    """
    text = decode(tokenizer, ids)
    boundaries = [0]
    for index in range(1, len(ids) + 1):
        prefix = decode(tokenizer, ids[:index])
        boundaries.append(len(prefix) if text.startswith(prefix) and not prefix.endswith("\ufffd") else None)
    result = []
    special = set(tokenizer.all_special_ids)
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:], strict=False)):
        valid = start is not None and end is not None and start < end and ids[index] not in special
        result.append((start, end) if valid else None)
    return text, result


def align(student, response_ids, teacher, context):
    response, student_spans = spans(student, response_ids)
    ids = teacher.encode(context + response, add_special_tokens=False)
    teacher_text, teacher_spans = spans(teacher, ids)
    if teacher_text != context + response:
        raise ValueError("Teacher tokenizer does not preserve the exact context and response text")
    offset = len(context)
    lookup = {span: index for index, span in enumerate(teacher_spans) if span is not None and span[0] >= offset}
    # Structural reasoning termination may be several ordinary tokens.
    termination = []
    begin = 0
    while (begin := response.find("</think>", begin)) >= 0:
        termination.append((begin, begin + len("</think>")))
        begin += len("</think>")
    mapping = []
    for span in student_spans:
        if span is None or any(span[0] < end and span[1] > start for start, end in termination):
            mapping.append(None)
        else:
            mapping.append(lookup.get((span[0] + offset, span[1] + offset)))
    return ids, mapping


def extract_scores(result, ids, mapping):
    entries = result.get("meta_info", {}).get("input_token_logprobs")
    if not isinstance(entries, list) or len(entries) != len(ids):
        raise ValueError("Teacher must return exactly one input score entry per requested token")
    if any(len(entry) < 2 or entry[1] != token for entry, token in zip(entries, ids, strict=True)):
        raise ValueError("Teacher returned different token IDs or positions")
    scores, mask = [], []
    for index in mapping:
        value = 0.0 if index is None else entries[index][0]
        if not isinstance(value, int | float) or not math.isfinite(value) or value > 0:
            raise ValueError("Teacher returned a missing, positive, or nonfinite token log probability")
        scores.append(float(value))
        mask.append(int(index is not None))
    return scores, mask
