"""OpenAI-compatible client pointed at the team's TrueFoundry gateway.

Mirrors ``lib/llm_client.py`` in the Small Learning Model repo: team access to
frontier models is through the gateway rather than the public APIs, so a direct
``api.openai.com`` call comes back ``billing_not_active``. Everything here is an
ordinary ``chat.completions`` call once the base URL and key are set.

WHAT THIS IS FOR, AND WHAT IT IS NOT FOR. Gateway models are a good fit for the
offline, one-off passes: unit annotation, rubric judging, and the talking half of
the student. They are the wrong tool for the ANSWERING half. That channel scores
each option by length-normalised log-probability using ``echo=True`` with
``logprobs``, which is a vLLM completions-endpoint feature and not something a
chat gateway exposes. The answerer therefore stays on a local vLLM, which is also
what keeps it comparable with every historical number.

Env, read from the process or from a ``.env`` at the repo root:

    TFY_API_KEY   gateway credential. ``TFY_API`` is accepted too, since that is
                  what the open-instruct .env happens to use.
    TFY_BASE_URL  defaults to the EU promptlens gateway.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BASE_URL = "https://tfy-eu.promptlens.trilogy.com/openai/v1"

#: Embeddings are NOT served under the chat base URL - that route 404s. Verified
#: working: POST {EMBED_BASE_URL}/embeddings with model openai-group/
#: text-embedding-3-large, which returns 3072 dimensions.
EMBED_BASE_URL = "https://tfy-eu.promptlens.trilogy.com/api/llm"
EMBED_MODEL = "openai-group/text-embedding-3-large"

#: Sent so calls are traceable in promptlens, as the team's other scripts do.
HEADERS = {"X-TFY-METADATA": '{"source": "open-instruct/projects/tutor"}', "X-TFY-LOGGING-CONFIG": '{"enabled": true}'}


def load_env(start: Path | None = None) -> None:
    """Read a ``.env`` into ``os.environ`` without depending on python-dotenv.

    Existing environment variables win, so an explicit ``export`` still overrides
    the file. Walks up from this file so it works regardless of the working
    directory the script was launched from.
    """
    here = start or Path(__file__).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
        return


def api_key() -> str | None:
    load_env()
    return os.environ.get("TFY_API_KEY") or os.environ.get("TFY_API")


def base_url() -> str:
    load_env()
    return os.environ.get("TFY_BASE_URL", DEFAULT_BASE_URL)


def make_client(base: str | None = None, key: str | None = None, timeout: float = 180.0):
    """An ``AsyncOpenAI`` on the gateway when a TFY key exists, else plain OpenAI.

    ``base`` wins when given, so pointing at a local vLLM stays a one-flag change.
    """
    from openai import AsyncOpenAI  # noqa: PLC0415

    key = key or api_key()
    if base:
        return AsyncOpenAI(base_url=base, api_key=key or "EMPTY", timeout=timeout)
    if key:
        return AsyncOpenAI(base_url=base_url(), api_key=key, default_headers=HEADERS, timeout=timeout)
    return AsyncOpenAI(timeout=timeout)


def make_embed_client(key: str | None = None, timeout: float = 180.0):
    """A sync client on the embeddings route."""
    from openai import OpenAI  # noqa: PLC0415

    key = key or api_key()
    return OpenAI(base_url=EMBED_BASE_URL, api_key=key or "EMPTY", default_headers=HEADERS, timeout=timeout)


#: A capable default for offline annotation. Annotation is cached and one-off, so
#: the cost of a strong model is paid once while a wrong unit tag silently
#: mis-pairs items and corrupts every number downstream.
DEFAULT_MODEL = "openai-group/gpt-5.6-terra"
