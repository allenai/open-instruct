from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu, visualize_token

tc = TokenizerConfig(
    tokenizer_name_or_path="meta-llama/Llama-3.1-8B",
    tokenizer_revision="main",
    use_fast=True,
    chat_template_name="tulu",
)
dataset_mixer_list = [
    "allenai/tulu-3-sft-personas-instruction-following",
    "1.0",
    "allenai/tulu-3-sft-personas-code",
    "1.0",
]
dataset_mixer_list_splits = ["train"]
dataset_transform_fn = ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]
transform_fn_args = [{"max_seq_length": 4096}, {}]
train_dataset = get_cached_dataset_tulu(
    dataset_mixer_list,
    dataset_mixer_list_splits,
    tc,
    dataset_transform_fn,
    transform_fn_args,
    target_columns=["input_ids", "attention_mask", "labels"],
    dataset_cache_mode="local",
)
print(train_dataset)
visualize_token(train_dataset[0]["input_ids"], tc.tokenizer)
