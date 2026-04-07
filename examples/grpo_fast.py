"""GRPO-fast CLI with example verifier fields on ``StreamingDataLoaderConfig``.

Uses :class:`ExamplesGRPOStreamingConfig` (core streaming config + Manufactoria/Ballsim
fields) so HuggingFace parses one dataclass group—same internal path as ``code_*`` flags.

Use ``python -m examples.grpo_fast`` instead of ``python -m open_instruct.grpo_fast`` when
you need typed ``--help`` for those fields. Stock ``grpo_fast`` still accepts
``--manufactoria_*`` / ``--ballsim_*`` as trailing argv (see ``parse_extra_verifier_cli_args``).

This entrypoint imports the example ``register`` modules so verifiers are registered
before training.
"""

from __future__ import annotations

import examples.ballsim.register  # noqa: F401
import examples.manufactoria.register  # noqa: F401
from examples.grpo_streaming_config import ExamplesGRPOStreamingConfig
from open_instruct import data_loader as data_loader_lib
from open_instruct import grpo_utils
from open_instruct import utils as open_instruct_utils
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.environments.tools.utils import EnvsConfig
from open_instruct.grpo_fast import main as grpo_fast_main
from open_instruct.model_utils import ModelConfig
from open_instruct.utils import ArgumentParserPlus

if __name__ == "__main__":
    open_instruct_utils.check_oe_eval_internal()

    parser = ArgumentParserPlus(
        (
            grpo_utils.GRPOExperimentConfig,
            TokenizerConfig,
            ModelConfig,
            ExamplesGRPOStreamingConfig,
            data_loader_lib.VLLMConfig,
            EnvsConfig,
        )
    )
    parser.set_defaults(exp_name="grpo", warmup_ratio=0.0, max_grad_norm=1.0, per_device_train_batch_size=1)
    (args, tokenizer_config, model_config, streaming_config, vllm_config, tools_config) = (
        parser.parse_args_into_dataclasses()
    )

    grpo_fast_main(args, tokenizer_config, model_config, streaming_config, vllm_config, tools_config)
