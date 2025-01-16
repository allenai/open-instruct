# this file deals with dataset pre-processing before training

# 1. PPO (prompt)
# 2. SFT (prompt + demonstration), there is also packing.
# 3. ✅ RM / DPO (chosen and rejected)
# 4. ✅ Visualization of length distributions?
# 5. ✅ Filter?
#   * Smart truncation?
# 6. ✅ dataset_num_proc
# 7. ✅ check EOS token
# 8. dataset mixer?
# 9. ✅ pretty print that show tokenization?
# 10. ✅ hashable tokneization?
# 11. inputs / labels / attention_mask
# 12. ✅ always set a `tokenizer.pad_token_id`?
# 13. a new DataCollatorForLanguageModeling?
# 14. ✅ `add_bos_token` and `add_eos_token`? E.g., LLAMA models
# 15. ✅ generate properties: has eos_token, bos_token (through chat template)

# ✅ get tokenizer revision
# ✅ get dataset revision
# create a cached tokenized dataset, with tokenized revision, dataset revision, tokenization function name.

# too many names related to "maximum length":
# * `max_seq_length` in SFT
# * `max_length`, `max_target_length` in RM / DPO,
# * `max_prompt_length` in DPO
"""
This file contains the utility to transform and cache datasets with different configurations.
The main things we are looking for are:
* handle dataset mixing
* handle different tokenization functions
* **cache** the tokenized dataset so we don't have to re-tokenize every time
    * This is especially important when we have 405B SFT models: 32 nodes are just spending like
    5 minutes to tokenize the dataset. This translates to 32 * 5 * 8 = 1280 minutes = 21 hours of
    wasted H100 time.
    * Sometimes we also launch on places that don't have a shared cache (e.g., GCP), so we would
    download individual datasets 32 times, and wait for concatenation and tokenization (actually 
    twice because the `with accelerator.main_process_first()` function assumes a shared cache)
"""

import copy
import hashlib
import json
from dataclasses import dataclass, field, asdict
import multiprocessing
import os
from typing import Any, Dict, List, Optional
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
)
import transformers
from transformers.utils.hub import cached_file, extract_commit_hash
from datasets import Dataset, load_dataset, concatenate_datasets
from huggingface_hub import HfApi, revision_exists

from open_instruct.dataset_processor import CHAT_TEMPLATES


# ----------------------------------------------------------------------------
# Utilities
def get_commit_hash(model_name_or_path: str, revision: str, filename: str = "config.json", repo_type: str = "model"):
    file = cached_file(model_name_or_path, revision=revision, filename=filename, repo_type=repo_type)
    commit_hash = extract_commit_hash(file, None)
    return commit_hash

# Performance tuning. Some rough numbers:
APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU = 400
FILTER_EXAMPLE_PER_SECOND_PER_CPU = 1130

def get_num_proc(dataset_len: int, num_available_cpus: int, example_per_second_per_cpu) -> int:
    num_required_cpus = max(1, dataset_len // example_per_second_per_cpu)
    return min(num_required_cpus, num_available_cpus)


# ----------------------------------------------------------------------------
# Tokenization
@dataclass
class TokenizerConfig:
    model_name_or_path: str
    revision: str
    trust_remote_code: bool = True
    use_fast: bool = True
    chat_template_name: Optional[str] = None
    add_bos: bool = False
    get_tokenizer_fn: str = "get_tokenizer_v1"
    
    # for tracking purposes
    tokenizer_commit_hash: Optional[str] = None
    
    def __post_init__(self):
        self.tokenizer_commit_hash = get_commit_hash(self.model_name_or_path, self.revision)


def get_tokenizer_v1(tc: TokenizerConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        tc.model_name_or_path,
        revision=tc.revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        # OLMo newer models use this tokenizer
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert (
                tc.add_bos
            ), "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        # else, pythia / other models
        else:
            num_added_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": "<pad>",
                }
            )
            assert (
                num_added_tokens <= 1
            ), "GPTNeoXTokenizer should only add one special token - the pad_token (or no tokens if already set in SFT)."
    # NOTE: (Costa) I just commented the `OPTForCausalLM` because we are not likely to use it.
    # elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
    #     num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."
        
    # set the tokenizer chat template to the training format
    # this will be used for encoding the training examples
    # and saved together with the tokenizer to be used later.
    if tc.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[tc.chat_template_name]
    else:
        try:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(tc.chat_template_name).chat_template
        except Exception:
            raise ValueError(f"Could not find chat template for {tc.chat_template_name}.")

    if tc.add_bos:
        if tokenizer.chat_template.startswith("{{ bos_token }}") or (
            tokenizer.bos_token is not None and tokenizer.chat_template.startswith(tokenizer.bos_token)
        ):
            raise ValueError(
                "You specified add_bos=True, but the chat template already has a bos_token at the beginning."
            )
        # also add bos in the chat template if not already there
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template
        
        
    # TODO: test it out: PPO should have the same tokenizer as SFT / DPO.
    # # create a tokenizer (pad from right)
    # config = AutoConfig.from_pretrained(model_config.model_name_or_path, revision=model_config.model_revision)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_config.model_name_or_path, revision=model_config.model_revision, padding_side="right"
    # )
    # if config.architectures == "LlamaForCausalLM" and config.bos_token_id == 128000:
    #     tokenizer.pad_token_id = 128002  # <|reserved_special_token_0|>
    # else:
    #     tokenizer.add_special_tokens({"pad_token": "[PAD]"})  # NOTE: we do not resize the embedding
    # if dataset_config.chat_template is not None:
    #     tokenizer.chat_template = CHAT_TEMPLATES[dataset_config.chat_template]
    return tokenizer

GET_TOKENIZER_FNS = {
    "get_tokenizer_v1": get_tokenizer_v1,
}

def get_tokenizer(tc: 'TokenizerConfig'):
    return GET_TOKENIZER_FNS[tc.get_tokenizer_fn](tc)


# TODO: for testing, we should load the tokenizer from the sft / dpo / rl and make sure they are all the same.


# ----------------------------------------------------------------------------
# Dataset Transformation
COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]
# Preference dataset
INPUT_IDS_CHOSEN_KEY = "input_ids_chosen"
ATTENTION_MASK_CHOSEN_KEY = "attention_mask_chosen"
INPUT_IDS_REJECTED_KEY = "input_ids_rejected"
ATTENTION_MASK_REJECTED_KEY = "attention_mask_rejected"
INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_PROMPT_KEY = "attention_mask_prompt"
GROUND_TRUTHS_KEY = "ground_truth"
DATASET_SOURCE_KEY = "dataset"

# NOTE (Costa): the `INPUT_IDS_PROMPT_KEY` is just for visualization purposes only
# also we don't really need `ATTENTION_MASK_CHOSEN_KEY` and `ATTENTION_MASK_REJECTED_KEY`
# since we are always padding from the right with a collator; however they might become
# more useful if we want to do some sort of packing in the future. The nice thing is
# that the tokenization logic would work for both DPO and RM training.
TOKENIZED_PREFERENCE_DATASET_KEYS = [
    INPUT_IDS_CHOSEN_KEY,
    INPUT_IDS_REJECTED_KEY,
    # ATTENTION_MASK_CHOSEN_KEY,
    # ATTENTION_MASK_REJECTED_KEY,
    # INPUT_IDS_PROMPT_KEY,
    # ATTENTION_MASK_PROMPT_KEY,
]

# SFT dataset
SFT_MESSAGE_KEY = "messages"
INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"

# Binary dataset
BINARY_LABEL_KEY = "binary_labels"
BINARY_DATASET_KEYS = [
    INPUT_IDS_KEY,
    LABELS_KEY,
    BINARY_LABEL_KEY,
]


def sft_tokenize_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    if len(row[SFT_MESSAGE_KEY]) == 1:
        prompt = row[SFT_MESSAGE_KEY]
    else:
        prompt = row[SFT_MESSAGE_KEY][:-1]
    
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[SFT_MESSAGE_KEY])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels
    return row
    

def sft_tokenize_mask_out_prompt_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    """mask out the prompt tokens by manipulating labels"""
    if len(row[SFT_MESSAGE_KEY]) == 1:
        prompt = row[SFT_MESSAGE_KEY]
    else:
        prompt = row[SFT_MESSAGE_KEY][:-1]
    
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[SFT_MESSAGE_KEY])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [-100] * len(row[INPUT_IDS_PROMPT_KEY])
    row[LABELS_KEY] = labels
    return row


def sft_filter_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_prompt_token_length: Optional[int] = None, max_token_length: Optional[int] = None, need_contain_labels: bool = True):
    max_prompt_token_length_ok = True
    if max_prompt_token_length is not None:
        max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= max_prompt_token_length

    max_token_length_ok = True
    if max_token_length is not None:
        max_token_length_ok = len(row[INPUT_IDS_KEY]) <= max_token_length

    contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
    return (
        max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
    )
    
def preference_tokenize_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    # Extract prompt (all messages except the last one)
    prompt = row["chosen"][:-1]
    
    # Tokenize prompt
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[ATTENTION_MASK_PROMPT_KEY] = [1] * len(row[INPUT_IDS_PROMPT_KEY])
    
    # Tokenize chosen completion
    row[INPUT_IDS_CHOSEN_KEY] = tokenizer.apply_chat_template(row["chosen"])
    row[ATTENTION_MASK_CHOSEN_KEY] = [1] * len(row[INPUT_IDS_CHOSEN_KEY])
    
    # Tokenize rejected completion
    row[INPUT_IDS_REJECTED_KEY] = tokenizer.apply_chat_template(row["rejected"])
    row[ATTENTION_MASK_REJECTED_KEY] = [1] * len(row[INPUT_IDS_REJECTED_KEY])
    
    return row

def preference_filter_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_prompt_token_length: Optional[int] = None, max_token_length: Optional[int] = None):
    # Check prompt length if specified
    if max_prompt_token_length is not None:
        if len(row[INPUT_IDS_PROMPT_KEY]) > max_prompt_token_length:
            return False
            
    # Check total sequence lengths if specified
    if max_token_length is not None:
        if len(row[INPUT_IDS_CHOSEN_KEY]) > max_token_length:
            return False
        if len(row[INPUT_IDS_REJECTED_KEY]) > max_token_length:
            return False
            
    return True

TRANSFORM_FNS = {
    "sft_tokenize_v1": (sft_tokenize_v1, "map"),
    "sft_tokenize_mask_out_prompt_v1": (sft_tokenize_mask_out_prompt_v1, "map"),
    "sft_filter_v1": (sft_filter_v1, "filter"),
    "preference_tokenize_v1": (preference_tokenize_v1, "map"),
    "preference_filter_v1": (preference_filter_v1, "filter"),
}

# ----------------------------------------------------------------------------
# Dataset Configuration and Caching
@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_split: str
    dataset_revision: str
    transform_fn: List[str] = field(default_factory=list)
    transform_fn_args: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    get_dataset_fn: str = "get_dataset_v1"
    
    # for tracking purposes
    dataset_commit_hash: Optional[str] = None
    
    def __post_init__(self):
        self.dataset_commit_hash = get_commit_hash(self.dataset_name, self.dataset_revision, "README.md", "dataset")

def get_dataset_v1(dc: DatasetConfig, tc: TokenizerConfig):
    # beaker specific logic; we may get assigned 15.5 CPU, so we convert it to float then int
    num_proc = int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", multiprocessing.cpu_count())))
    
    tokenizer = get_tokenizer(tc)
    dataset = load_dataset(
        dc.dataset_name,
        split=dc.dataset_split,
        revision=dc.dataset_revision,
    )
    
    for fn_name in dc.transform_fn:
        fn, fn_type = TRANSFORM_FNS[fn_name]
        # always pass in tokenizer and other args if needed
        fn_kwargs = {"tokenizer": tokenizer}
        if fn_name in dc.transform_fn_args:
            fn_kwargs.update(dc.transform_fn_args[fn_name])

        # perform the transformation
        if fn_type == "map":
            dataset = dataset.map(
                fn,
                fn_kwargs=fn_kwargs,
                num_proc=get_num_proc(len(dataset), num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            )
        elif fn_type == "filter":
            dataset = dataset.filter(
                fn, 
                fn_kwargs=fn_kwargs,
                num_proc=get_num_proc(len(dataset), num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            )
        # NOTE: elif we can implement packing here to create a packed SFT dataset. Low priority for now.
        else:
            raise ValueError(f"Unknown transform function type: {fn_type}")

    return dataset

class DatasetTransformationCache:
    def __init__(self, hf_entity: Optional[str] = None):
        self.hf_entity = hf_entity or HfApi().whoami()["name"]
        
    def compute_config_hash(self, dcs: List[DatasetConfig], tc: TokenizerConfig) -> str:
        """Compute a deterministic hash of both configs for caching."""
        dc_dicts = [
            {k: v for k, v in asdict(dc).items() if v is not None}
            for dc in dcs
        ]
        tc_dict = {k: v for k, v in asdict(tc).items() if v is not None}
        combined_dict = {
            "dataset_configs": dc_dicts,
            "tokenizer_config": tc_dict
        }
        config_str = json.dumps(combined_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:10]

    def load_or_transform_dataset(self, dcs: List[DatasetConfig], tc: TokenizerConfig) -> Dataset:
        """Load dataset from cache if it exists, otherwise transform and cache it."""
        config_hash = self.compute_config_hash(dcs, tc)
        repo_name = f"{self.hf_entity}/dataset-mix-cached"
        
        # Check if the revision exists
        if revision_exists(repo_name, config_hash, repo_type="dataset"):
            print(f"Found cached dataset at {repo_name}@{config_hash}")
            # Use the split from the first dataset config as default
            return load_dataset(
                repo_name,
                split=dcs[0].dataset_split,
                revision=config_hash
            )
        
        print(f"Cache not found, transforming datasets...")
        
        # Transform each dataset
        transformed_datasets = []
        for dc in dcs:
            dataset = get_dataset_v1(dc, tc)
            # Add id column if not present
            if "id" not in dataset.column_names:
                base_name = dc.dataset_name.split("/")[-1]
                id_col = [f"{base_name}-{i}" for i in range(len(dataset))]
                dataset = dataset.add_column("id", id_col)
            transformed_datasets.append(dataset)
        
        # Combine datasets
        combined_dataset = concatenate_datasets(transformed_datasets)
        
        # Push to hub with config hash as revision
        combined_dataset.push_to_hub(
            repo_name,
            private=True,
            revision=config_hash,
            commit_message=f"Cache combined dataset with configs hash: {config_hash}"
        )
        print(f"Pushed transformed dataset to {repo_name}@{config_hash}")

        # NOTE: Load the dataset again to make sure it's downloaded to the HF cache
        print(f"Found cached dataset at {repo_name}@{config_hash}")
        return load_dataset(
            repo_name,
            split=dc.dataset_split,
            revision=config_hash
        )


def get_cached_dataset(dcs: List[DatasetConfig], tc: TokenizerConfig, hf_entity: Optional[str] = None) -> Dataset:
    """Get transformed and cached dataset from multiple dataset configs."""
    cache = DatasetTransformationCache(hf_entity=hf_entity)
    return cache.load_or_transform_dataset(dcs, tc)

def test_config_hash_different():
    """Test that different configurations produce different hashes."""
    tc = TokenizerConfig(
        model_name_or_path="meta-llama/Llama-3.1-8B",
        revision="main",
        chat_template_name="tulu"
    )
    
    dcs1 = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_v1"],
            transform_fn_args={}
        )
    ]
    
    dcs2 = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_mask_out_prompt_v1"],
            transform_fn_args={}
        )
    ]
    
    cache = DatasetTransformationCache()
    hash1 = cache.compute_config_hash(dcs1, tc)
    hash2 = cache.compute_config_hash(dcs2, tc)
    assert hash1 != hash2, "Different configs should have different hashes"

def test_sft_dataset_caching():
    """Test caching functionality for SFT datasets."""
    tc = TokenizerConfig(
        model_name_or_path="meta-llama/Llama-3.1-8B",
        revision="main",
        chat_template_name="tulu"
    )
    
    dcs = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_v1"],
            transform_fn_args={}
        ),
        DatasetConfig(
            dataset_name="allenai/tulu-3-hard-coded-10x",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_v1"],
            transform_fn_args={}
        )
    ]
    
    # First transformation should cache
    dataset1 = get_cached_dataset(dcs, tc)
    
    # Second load should use cache
    dataset1_cached = get_cached_dataset(dcs, tc)
    
    # Verify the datasets are the same
    assert len(dataset1) == len(dataset1_cached), "Cached dataset should have same length"

def test_sft_different_transform():
    """Test different transform functions produce different cached datasets."""
    tc = TokenizerConfig(
        model_name_or_path="meta-llama/Llama-3.1-8B",
        revision="main",
        chat_template_name="tulu"
    )
    
    dcs = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_mask_out_prompt_v1"],
            transform_fn_args={}
        ),
        DatasetConfig(
            dataset_name="allenai/tulu-3-hard-coded-10x",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_mask_out_prompt_v1"],
            transform_fn_args={}
        )
    ]
    
    dataset = get_cached_dataset(dcs, tc)
    assert dataset is not None, "Should successfully create dataset with different transform"


def test_sft_filter():
    """Test different transform functions produce different cached datasets."""
    tc = TokenizerConfig(
        model_name_or_path="meta-llama/Llama-3.1-8B",
        revision="main",
        chat_template_name="tulu"
    )
    
    ARBITRARY_MAX_LENGTH = 1000
    dcs = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_v1", "sft_filter_v1"],  # First tokenize, then filter
            transform_fn_args={
                "sft_filter_v1": {
                    "max_token_length": ARBITRARY_MAX_LENGTH  # Filter to sequences <= ARBITRARY_MAX_LENGTH tokens
                }
            }
        )
    ]
    
    filtered_dataset = get_cached_dataset(dcs, tc)
    # Verify that all sequences are <= ARBITRARY_MAX_LENGTH tokens
    max_length = max(len(example[INPUT_IDS_KEY]) for example in filtered_dataset)
    assert max_length <= ARBITRARY_MAX_LENGTH, f"Found sequence with length {max_length} > {ARBITRARY_MAX_LENGTH}"
    
    print("Filter test passed! Max sequence length:", max_length)
    print("All tests passed!")
    assert filtered_dataset is not None, "Should successfully create dataset with different transform"


def test_preference_dataset():
    """Test caching functionality for preference datasets."""
    tc = TokenizerConfig(
        model_name_or_path="meta-llama/Llama-3.1-8B",
        revision="main",
        chat_template_name="tulu"
    )
    
    dcs_pref = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-pref-personas-instruction-following",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["preference_tokenize_v1"],
            transform_fn_args={}
        ),
        DatasetConfig(
            dataset_name="allenai/tulu-3-wildchat-reused-on-policy-70b",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["preference_tokenize_v1"],
            transform_fn_args={}
        )
    ]
    
    dataset_pref = get_cached_dataset(dcs_pref, tc)
    assert dataset_pref is not None, "Should successfully create preference dataset"


if __name__ == "__main__":
    test_config_hash_different()
    test_sft_dataset_caching()
    test_sft_different_transform()
    test_preference_dataset()
    test_sft_filter()
    print("All tests passed!")