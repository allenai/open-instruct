"""CPU tests for the multimodal SFT entry point's argument surface."""

from open_instruct import (
    olmo_core_multimodal_finetune,
    olmo_core_multimodal_utils,
    olmo_core_utils,
    sft_mixture,
    utils,
)


def _parser() -> utils.ArgumentParserPlus:
    parser = utils.ArgumentParserPlus(
        (  # ty: ignore[invalid-argument-type]
            olmo_core_utils.ExperimentConfig,
            olmo_core_multimodal_utils.MultimodalModelConfig,
            olmo_core_multimodal_utils.MultimodalTrainingConfig,
            sft_mixture.MixtureConfig,
            olmo_core_utils.LoggingConfig,
            olmo_core_utils.CheckpointConfig,
        )
    )
    parser.set_defaults(
        exp_name="mm_sft",
        ephemeral_save_interval=olmo_core_multimodal_finetune._DEFAULT_EPHEMERAL_SAVE_INTERVAL,
        checkpointing_steps=2000,
    )
    return parser


def test_defaults_are_stage2_parity():
    tracking, model, training, mixture, _, checkpoint = _parser().parse_args_into_dataclasses(args=[])
    assert tracking.exp_name == "mm_sft"
    assert model.base_hf_model_id == "allenai/Molmo2-4B"
    assert training.max_seq_length == 16384
    assert training.learning_rate == 1e-5
    assert mixture.mixture == "debug"
    assert mixture.nlp_source == "tulu4"
    assert checkpoint.checkpointing_steps == 2000
    assert checkpoint.ephemeral_save_interval == 250


def test_merged_stage_cli_round_trip():
    argv = [
        "--mixture",
        "image-only-v9",
        "--nlp_source",
        "open_instruct",
        "--mixer_list",
        "allenai/Dolci-Instruct-SFT",
        "1.0",
        "--nlp_rate",
        "0.166",
        "--freeze_params",
        "vision.*",
        "--model_preset",
        "molmo2_8B",
        "--max_train_steps",
        "20000",
    ]
    _, model, training, mixture, _, _ = _parser().parse_args_into_dataclasses(args=argv)
    assert model.model_preset == "molmo2_8B"
    assert training.max_train_steps == 20000
    assert training.freeze_params == ["vision.*"]
    assert mixture.nlp_source == "open_instruct"
    assert mixture.nlp_rate == 0.166
    specs = mixture.source_specs()
    assert specs[0].type == sft_mixture.OPEN_INSTRUCT_SFT_TYPE
    assert specs[0].args["mixer_list"] == ["allenai/Dolci-Instruct-SFT", "1.0"]
