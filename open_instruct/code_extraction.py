"""Language selection for the legacy and MILES code reward adapters.

Preserve the legacy triple-backtick boundaries, whitespace handling, and raw-text
fallback. This is intentionally independent of complete_code's stricter parser.
"""

import re

_CODE_BLOCK = re.compile(r"```(.*?)```", re.DOTALL)
_LANGUAGE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+#.-]*")
_PYTHON = {"python", "python3", "py"}


def extract_python_code(response: str) -> str:
    """Return the last Python or unlabelled block, or the original response.

    Only a standalone language tag on the opening line identifies a labelled
    block. Skip explicit other-language blocks, without changing fence pairing
    or requiring Markdown-conformant indentation/closing lines. An unfinished
    final block leaves the preceding complete candidate eligible, as before.
    """
    selected = None
    for block in _CODE_BLOCK.findall(response):
        header, newline, body = block.partition("\n")
        if newline and _LANGUAGE.fullmatch(header.strip()):
            if header.strip().lower() not in _PYTHON:
                continue
            selected = body.strip()
        else:
            # Preserve legacy handling of inline/unlabelled blocks, including
            # inline Python openings without a newline after the language.
            selected = block.removeprefix("python").strip()
    return response if selected is None else selected
