import gc
import os
import shutil
import tempfile
import unittest

import datasets
from parameterized import parameterized

import open_instruct.dataset_transformation

HAS_CACHE = (
    "HF_HOME" in os.environ
    or "HF_DATASETS_CACHE" in os.environ
    or os.path.exists(os.path.expanduser("~/.cache/huggingface/datasets"))
)


class TestTokenizerEquality(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "llama",
                "meta-llama/Llama-3.1-8B",
                "allenai/Llama-3.1-Tulu-3-8B-SFT",
                "allenai/Llama-3.1-Tulu-3-8B-DPO",
                False,
            ),
            ("olmo", "allenai/OLMo-2-1124-7B", "allenai/OLMo-2-1124-7B-SFT", "allenai/OLMo-2-1124-7B-DPO", True),
        ]
    )
    def test_sft_dpo_same_tokenizer(self, name, base_model, sft_model, dpo_model, add_bos):
        base_to_sft_tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=base_model, tokenizer_revision="main", chat_template_name="tulu", add_bos=add_bos
        )
        sft_to_dpo_tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=sft_model, tokenizer_revision="main", chat_template_name="tulu", add_bos=add_bos
        )
        dpo_to_rl_tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=dpo_model, tokenizer_revision="main", chat_template_name="tulu", add_bos=add_bos
        )

        self._assert_tokenizers_equal(base_to_sft_tc, sft_to_dpo_tc)
        self._assert_tokenizers_equal(sft_to_dpo_tc, dpo_to_rl_tc)
        self._assert_tokenizers_equal(base_to_sft_tc, dpo_to_rl_tc)

    def _assert_tokenizers_equal(self, tc1, tc2):
        tok1 = tc1.tokenizer
        tok2 = tc2.tokenizer
        self.assertEqual(tok1.vocab_size, tok2.vocab_size, "Vocab size should be the same")
        self.assertEqual(tok1.model_max_length, tok2.model_max_length, "Model max length should be the same")
        self.assertEqual(tok1.is_fast, tok2.is_fast, "is_fast should be the same")
        self.assertEqual(tok1.padding_side, tok2.padding_side, "padding_side should be the same")
        self.assertEqual(tok1.truncation_side, tok2.truncation_side, "truncation_side should be the same")
        self.assertEqual(
            tok1.clean_up_tokenization_spaces,
            tok2.clean_up_tokenization_spaces,
            "clean_up_tokenization_spaces should be the same",
        )
        self.assertEqual(
            tok1.added_tokens_decoder, tok2.added_tokens_decoder, "added_tokens_decoder should be the same"
        )


class TestConfigHash(unittest.TestCase):
    def test_config_hash_different(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path="meta-llama/Llama-3.1-8B", tokenizer_revision="main", chat_template_name="tulu"
        )

        dcs1 = [
            open_instruct.dataset_transformation.DatasetConfig(
                dataset_name="allenai/tulu-3-sft-personas-algebra",
                dataset_split="train",
                dataset_revision="main",
                transform_fn=["sft_tokenize_v1"],
                transform_fn_args=[{}],
            )
        ]

        dcs2 = [
            open_instruct.dataset_transformation.DatasetConfig(
                dataset_name="allenai/tulu-3-sft-personas-algebra",
                dataset_split="train",
                dataset_revision="main",
                transform_fn=["sft_tokenize_mask_out_prompt_v1"],
                transform_fn_args=[{}],
            )
        ]
        hash1 = open_instruct.dataset_transformation.compute_config_hash(dcs1, tc)
        hash2 = open_instruct.dataset_transformation.compute_config_hash(dcs2, tc)
        self.assertNotEqual(hash1, hash2, "Different configs should have different hashes")


class TestCachedDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.original_hf_home = os.environ.get("HF_HOME")
        self.original_hf_datasets_cache = os.environ.get("HF_DATASETS_CACHE")
        self.original_transformers_cache = os.environ.get("TRANSFORMERS_CACHE")

        os.environ["HF_HOME"] = self.temp_dir.name
        os.environ["HF_DATASETS_CACHE"] = os.path.join(self.temp_dir.name, "datasets")
        os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.temp_dir.name, "transformers")

    def tearDown(self):
        if self.original_hf_home is not None:
            os.environ["HF_HOME"] = self.original_hf_home
        else:
            os.environ.pop("HF_HOME", None)

        if self.original_hf_datasets_cache is not None:
            os.environ["HF_DATASETS_CACHE"] = self.original_hf_datasets_cache
        else:
            os.environ.pop("HF_DATASETS_CACHE", None)

        if self.original_transformers_cache is not None:
            os.environ["TRANSFORMERS_CACHE"] = self.original_transformers_cache
        else:
            os.environ.pop("TRANSFORMERS_CACHE", None)

        self.temp_dir.cleanup()
        if os.path.exists(self.temp_dir.name):
            shutil.rmtree(self.temp_dir.name, ignore_errors=True)
        gc.collect()

    def test_get_cached_dataset_tulu_sft(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path="meta-llama/Llama-3.1-8B",
            tokenizer_revision="main",
            use_fast=True,
            chat_template_name="tulu",
            add_bos=False,
        )
        dataset_mixer_list = ["allenai/tulu-3-sft-mixture", "1.0"]
        dataset_mixer_list_splits = ["train[:1%]"]
        dataset_transform_fn = ["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"]

        transform_fn_args = [{"max_seq_length": 4096}, {}]
        dataset = open_instruct.dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list,
            dataset_mixer_list_splits,
            tc,
            dataset_transform_fn,
            transform_fn_args,
            open_instruct.dataset_transformation.TOKENIZED_SFT_DATASET_KEYS,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
        )
        gold_tokenized_dataset = datasets.load_dataset(
            "allenai/dataset-mix-cached", split="train", revision="4c47c491c0"
        )
        self.assertEqual(len(dataset), len(gold_tokenized_dataset))
        for i in range(len(dataset)):
            self.assertEqual(dataset[i]["input_ids"], gold_tokenized_dataset[i]["input_ids"])

    def test_get_cached_dataset_tulu_preference(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path="allenai/Llama-3.1-Tulu-3-8B-SFT",
            tokenizer_revision="main",
            use_fast=False,
            chat_template_name="tulu",
            add_bos=False,
        )
        dataset_mixer_list = ["allenai/llama-3.1-tulu-3-8b-preference-mixture", "1.0"]
        dataset_mixer_list_splits = ["train[:1%]"]
        dataset_transform_fn = ["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"]
        transform_fn_args = [{"max_seq_length": 2048}, {}]
        dataset = open_instruct.dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list,
            dataset_mixer_list_splits,
            tc,
            dataset_transform_fn,
            transform_fn_args,
            open_instruct.dataset_transformation.TOKENIZED_PREFERENCE_DATASET_KEYS,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
        )
        gold_tokenized_dataset = datasets.load_dataset(
            "allenai/dataset-mix-cached", split="train", revision="7c4f2bb6cf"
        )
        self.assertEqual(len(dataset), len(gold_tokenized_dataset))
        for i in range(len(dataset)):
            self.assertEqual(dataset[i]["chosen_input_ids"], gold_tokenized_dataset[i]["chosen_input_ids"])

    def test_get_cached_dataset_tulu_rlvr(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path="allenai/Llama-3.1-Tulu-3-8B-DPO",
            tokenizer_revision="main",
            use_fast=False,
            chat_template_name="tulu",
            add_bos=False,
        )
        dataset_mixer_list = ["allenai/RLVR-GSM-MATH-IF-Mixed-Constraints", "1.0"]
        dataset_mixer_list_splits = ["train[:1%]"]
        dataset_transform_fn = ["rlvr_tokenize_v1", "rlvr_max_length_filter_v1"]
        transform_fn_args = [{}, {"max_prompt_token_length": 2048}]
        dataset = open_instruct.dataset_transformation.get_cached_dataset_tulu(
            dataset_mixer_list,
            dataset_mixer_list_splits,
            tc,
            dataset_transform_fn,
            transform_fn_args,
            dataset_skip_cache=True,
            dataset_local_cache_dir=self.temp_dir.name,
        )
        gold_tokenized_dataset = datasets.load_dataset(
            "allenai/dataset-mix-cached", split="train", revision="1f2adb3bb9"
        )
        self.assertEqual(len(dataset), len(gold_tokenized_dataset))
        for i in range(len(dataset)):
            self.assertEqual(
                dataset[i][open_instruct.dataset_transformation.INPUT_IDS_PROMPT_KEY],
                gold_tokenized_dataset[i][open_instruct.dataset_transformation.INPUT_IDS_PROMPT_KEY],
            )


if __name__ == "__main__":
    unittest.main()
