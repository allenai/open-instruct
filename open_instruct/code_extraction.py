"""Select Python submissions for the legacy and MILES code reward adapters.

Keep the extraction contract aligned with code_execution_client.verifier at
4cab7e85e8748db501163e6cbbc5ab7c041692af, used by complete_code. This helper
has no service dependencies and does not select the answer/thinking region.
"""

import ast
import re

_FENCE = re.compile(r" {0,3}(`{3,}|~{3,})([^\r\n]*)")
_PYTHON = {"python", "python3", "py"}
_END_TOKENS = re.compile(r"(?:(?:<\|endoftext\|>|<\|im_end\|>|<\|eot_id\|>)\s*)+\Z")


def extract_python_code(response: str) -> str | None:
    """Select the last complete Python fence, then an unlabelled fence, then raw code.

    Explicit other-language blocks are never Python submissions. An unfinished
    Python block is invalid rather than falling back to an earlier draft.
    """
    # Raw programs may contain Markdown fences inside string literals. Recognize
    # valid Python before applying Markdown extraction to a model's prose.
    try:
        ast.parse(response)
    except (SyntaxError, ValueError, RecursionError):
        pass
    else:
        return response
    # Rollouts may retain a decoded end token immediately after the closing fence.
    # Strip only terminal model delimiters, after preserving valid raw Python (which
    # can contain these exact strings). An actually unfinished fence stays invalid.
    response = _END_TOKENS.sub("", response)
    python, unlabelled = [], []
    marker, language, lines, saw_fence = None, None, [], False
    for line in response.splitlines(keepends=True):
        match = _FENCE.fullmatch(line.rstrip("\r\n"))
        if marker is None:
            if match:
                marker = match[1]
                info = match[2].strip().split(maxsplit=1)
                language = info[0].lower() if info else ""
                lines, saw_fence = [], True
            continue
        if match and match[1][0] == marker[0] and len(match[1]) >= len(marker) and not match[2].strip():
            code = "".join(lines).strip()
            if language in _PYTHON:
                python.append(code)
            elif not language:
                unlabelled.append(code)
            marker = None
        else:
            lines.append(line)
    if marker is not None and (language in _PYTHON or not language):
        return None
    if python:
        return python[-1]
    if unlabelled:
        return unlabelled[-1]
    return None if saw_fence else response
