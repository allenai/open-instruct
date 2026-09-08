"""Run the BFCL CLI with an OpenAI-compatible endpoint registered as a function-calling model.

BFCL only evaluates models listed in ``bfcl_eval.constants.model_config.MODEL_CONFIG_MAPPING``
and offers no flag to add one, so this entry point inserts a config at import time and then
hands the remaining arguments to the stock ``bfcl`` CLI. The model is registered with BFCL's
``OpenAICompletionsHandler`` in function-calling mode: BFCL sends the chat history plus a
``tools`` list through the OpenAI chat-completions API and reads structured ``tool_calls`` back.
The handler builds its client from ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY``, which is how it is
pointed at a local vLLM server rather than at OpenAI.

    BFCL_MODEL_NAME=olmoe3-kda-sft-FC BFCL_SERVED_MODEL_NAME=olmoe3-kda-sft \\
    OPENAI_BASE_URL=http://localhost:8000/v1 OPENAI_API_KEY=EMPTY \\
    python bfcl_cli_with_olmo_models.py generate --model olmoe3-kda-sft-FC --test-category multi_turn

Two details of the registration matter for scoring:

* ``is_fc_model=True`` selects the FC code path (``_query_FC`` etc.) and the AST checker's
  function-calling comparison.
* ``underscore_to_dot=True`` because ``convert_to_tool`` rewrites ``.`` to ``_`` in function
  names for every OpenAI-style handler (the OpenAI API rejects dots); the checker then needs to
  know to map them back. The same flag is set on the gpt-* FC entries.
"""

import os
import sys

from bfcl_eval.__main__ import cli
from bfcl_eval.constants import model_config
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

REGISTRY_NAME = os.environ.get("BFCL_MODEL_NAME", "olmoe3-kda-sft-FC")
SERVED_MODEL_NAME = os.environ.get("BFCL_SERVED_MODEL_NAME", REGISTRY_NAME.removesuffix("-FC"))


def register() -> None:
    if REGISTRY_NAME in model_config.MODEL_CONFIG_MAPPING:
        return
    # Mutated in place: every consumer imports this dict object by name, so an in-place insert is
    # visible to the generation and evaluation code without touching the installed package.
    model_config.MODEL_CONFIG_MAPPING[REGISTRY_NAME] = model_config.ModelConfig(
        model_name=SERVED_MODEL_NAME,
        display_name=f"{REGISTRY_NAME} (FC, vLLM endpoint)",
        url="https://github.com/allenai/open-instruct",
        org="Ai2",
        license="apache-2.0",
        model_handler=OpenAICompletionsHandler,
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,
    )


def main() -> None:
    register()
    cli(sys.argv[1:])


if __name__ == "__main__":
    main()
