from open_instruct.dataset_transformation import (
    TokenizerConfig,
    get_cached_dataset_tulu,
    visualize_token,
)
from open_instruct.finetune import FlatArguments
from open_instruct.utils import ArgumentParserPlus


def main(args: FlatArguments, tc: TokenizerConfig) -> None:
    transform_fn_args = [
        {"max_seq_length": args.max_seq_length},
        {},
    ]
    train_dataset = get_cached_dataset_tulu(
        dataset_mixer_list=args.dataset_mixer_list,
        dataset_mixer_list_splits=args.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        target_columns=args.dataset_target_columns,
        dataset_cache_mode=args.dataset_cache_mode,
        dataset_config_hash=args.dataset_config_hash,
        hf_entity=args.hf_entity,
        dataset_local_cache_dir=args.dataset_local_cache_dir,
        dataset_skip_cache=args.dataset_skip_cache,
    )
    print(train_dataset)
    visualize_token(train_dataset[0]["input_ids"], tc.tokenizer)


if __name__ == "__main__":
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    main(args, tc)
