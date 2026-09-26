"""Populate checkpoint Python modules before SGLang spawns concurrent importers."""

import argparse

from transformers import AutoConfig, AutoTokenizer

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def prime_serving_modules(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model-path", "--model", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--tokenizer-mode", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-tokenizer-init", action="store_true")
    args, _ = parser.parse_known_args(argv)
    # Each engine has its own HF_MODULES_CACHE. Complete the parent's copies
    # before its tokenizer, scheduler and detokenizer processes can import them.
    # No model weights are loaded here.
    AutoConfig.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code)
    if not args.skip_tokenizer_init and args.tokenizer_mode in {"auto", "slow"}:
        AutoTokenizer.from_pretrained(
            args.tokenizer_path or args.model_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=args.tokenizer_mode != "slow",
        )
    logger.info("Checkpoint Python module cache primed before serving subprocess startup")
