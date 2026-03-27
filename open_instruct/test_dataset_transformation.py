import gc
import hashlib
import os
import shutil
import tempfile
import unittest

import open_instruct.dataset_transformation

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
TOKENIZER_PATH = os.path.join(TEST_DATA_DIR, "tokenizer")

GOLD_SFT = {"count": 100, "hash": "3e745ff9615c9b0e3d8efe74f3f96cde01ac6a720535f0b4ef7175ebb2d1d6cf"}
GOLD_PREFERENCE = {"count": 97, "hash": "415d8c34ac25cf04d798f27a88c90df38826f71a404e5345563635778bdf9bb3"}
GOLD_RLVR = {"count": 100, "hash": "9ebada598693087c4cd4804d474fbbe07f41a7dffb38104ddee4e93ba0bfd3b1"}


class TestEnvConfigNormalization(unittest.TestCase):
    def test_normalize_single_dict_env_config(self):
        row = {"env_config": {"env_name": "guess_number", "number": "7"}}
        open_instruct.dataset_transformation._normalize_env_config_column(row)
        self.assertEqual(row["env_config"], {"env_configs": [{"env_name": "guess_number", "number": "7"}]})

    def test_normalize_list_env_config(self):
        row = {"env_config": [{"env_name": "counter", "target": "3"}]}
        open_instruct.dataset_transformation._normalize_env_config_column(row)
        self.assertEqual(row["env_config"], {"env_configs": [{"env_name": "counter", "target": "3"}]})

    def test_normalize_canonical_env_config(self):
        row = {"env_config": {"max_steps": 10, "env_configs": [{"env_name": "guess_number", "number": "5"}]}}
        open_instruct.dataset_transformation._normalize_env_config_column(row)
        self.assertEqual(
            row["env_config"], {"max_steps": 10, "env_configs": [{"env_name": "guess_number", "number": "5"}]}
        )


class TestConfigHash(unittest.TestCase):
    def test_config_hash_different(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH, tokenizer_revision="main", chat_template_name="tulu"
        )

        sft_data = os.path.join(TEST_DATA_DIR, "sft_sample.jsonl")
        dcs1 = [
            open_instruct.dataset_transformation.DatasetConfig(
                dataset_name=sft_data,
                dataset_split="train",
                dataset_revision="main",
                transform_fn=["sft_tokenize_v1"],
                transform_fn_args=[{}],
            )
        ]

        dcs2 = [
            open_instruct.dataset_transformation.DatasetConfig(
                dataset_name=sft_data,
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
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=True,
            chat_template_name="tulu",
            add_bos=False,
        )
        dataset_mixer_list = [os.path.join(TEST_DATA_DIR, "sft_sample.jsonl"), "1.0"]
        dataset_mixer_list_splits = ["train"]
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
        self.assertEqual(len(dataset), GOLD_SFT["count"])
        dataset_hash = hashlib.sha256()
        for row in dataset:
            dataset_hash.update(str(row["input_ids"]).encode())
        self.assertEqual(dataset_hash.hexdigest(), GOLD_SFT["hash"])

    def test_get_cached_dataset_tulu_preference(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=False,
            chat_template_name="tulu",
            add_bos=False,
        )
        dataset_mixer_list = [os.path.join(TEST_DATA_DIR, "preference_sample.jsonl"), "1.0"]
        dataset_mixer_list_splits = ["train"]
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
        self.assertEqual(len(dataset), GOLD_PREFERENCE["count"])
        dataset_hash = hashlib.sha256()
        for row in dataset:
            dataset_hash.update(str(row["chosen_input_ids"]).encode())
        self.assertEqual(dataset_hash.hexdigest(), GOLD_PREFERENCE["hash"])

    def test_get_cached_dataset_tulu_rlvr(self):
        tc = open_instruct.dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path=TOKENIZER_PATH,
            tokenizer_revision="main",
            use_fast=False,
            chat_template_name="tulu",
            add_bos=False,
        )
        dataset_mixer_list = [os.path.join(TEST_DATA_DIR, "rlvr_sample.jsonl"), "1.0"]
        dataset_mixer_list_splits = ["train"]
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
        self.assertEqual(len(dataset), GOLD_RLVR["count"])
        dataset_hash = hashlib.sha256()
        for row in dataset:
            dataset_hash.update(str(row[open_instruct.dataset_transformation.INPUT_IDS_PROMPT_KEY]).encode())
        self.assertEqual(dataset_hash.hexdigest(), GOLD_RLVR["hash"])


if __name__ == "__main__":
    unittest.main()
