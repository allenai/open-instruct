# coding=utf-8
# Copyright 2024 AllenAI Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copied from https://github.com/huggingface/alignment-handbook/blob/main/tests/test_data.py
import unittest

import pytest
from dateutil import parser

from open_instruct.utils import get_datasets


class GetDatasetsTest(unittest.TestCase):
    """Each of these test datasets has 100 examples"""

    def test_loading_data_args(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 0.5,
            "HuggingFaceH4/testing_self_instruct_small": 0.3,
            "HuggingFaceH4/testing_codealpaca_small": 0.2,
        }
        datasets = get_datasets(dataset_mixer, columns_to_keep=["prompt", "completion"])
        self.assertEqual(len(datasets["train"]), 100)
        self.assertEqual(len(datasets["test"]), 300)

    def test_loading_with_unit_fractions(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 1.0,
            "HuggingFaceH4/testing_self_instruct_small": 1.0,
            "HuggingFaceH4/testing_codealpaca_small": 1.0,
        }
        datasets = get_datasets(dataset_mixer, columns_to_keep=["prompt", "completion"])
        self.assertEqual(len(datasets["train"]), 300)
        self.assertEqual(len(datasets["test"]), 300)

    def test_loading_with_fractions_greater_than_unity(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 0.7,
            "HuggingFaceH4/testing_self_instruct_small": 0.4,
        }
        datasets = get_datasets(dataset_mixer, columns_to_keep=["prompt", "completion"])
        self.assertEqual(len(datasets["train"]), 70 + 40)
        self.assertEqual(len(datasets["test"]), 200)

    def test_loading_fails_with_negative_fractions(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 0.7,
            "HuggingFaceH4/testing_self_instruct_small": -0.3,
        }
        with pytest.raises(ValueError, match=r"Dataset fractions / lengths cannot be negative."):
            get_datasets(dataset_mixer, columns_to_keep=["prompt", "completion"])

    def test_loading_single_split_with_unit_fractions(self):
        dataset_mixer = {
            "HuggingFaceH4/testing_alpaca_small": 1.0,
        }
        datasets = get_datasets(dataset_mixer, splits=["test"], columns_to_keep=["prompt", "completion"])
        self.assertEqual(len(datasets["test"]), 100)
        self.assertRaises(KeyError, lambda: datasets["train"])

    def test_loading_preference_data(self):
        dataset_mixer = {
            "ai2-adapt-dev/ultrafeedback-small": 1000,
            "ai2-adapt-dev/summarize_from_feedback_small": 1000,
        }
        pref_datasets = get_datasets(dataset_mixer, splits=["train"], columns_to_keep=["chosen", "rejected"])
        self.assertEqual(len(pref_datasets["train"]), 2000)

    def test_time_parser_used_in_get_beaker_dataset_ids(self):
        # two special cases which beaker uses
        self.assertTrue(parser.parse("2024-09-16T19:03:02.31502Z"))
        self.assertTrue(parser.parse("0001-01-01T00:00:00Z"))


# useful for checking if public datasets are still available
# class CheckTuluDatasetsTest(unittest.TestCase):
#     """
#     Try to rebuild Tulu from public sources
#     """

#     def test_loading_tulu(self):
#         dataset_mixer = {
#             "natolambert/tulu-v2-sft-mixture-flan": 50000,
#             "natolambert/tulu-v2-sft-mixture-cot": 49747,
#             "allenai/openassistant-guanaco-reformatted": 7708,  # not exact subset
#             "Vtuber-plan/sharegpt-cleaned": 114046,
#             # original https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered
#             "vicgalle/alpaca-gpt4": 20000,
#             "HuggingFaceH4/CodeAlpaca_20K": 18000,  # original uses https://github.com/sahil280114/codealpaca
#             "natolambert/tulu-v2-sft-mixture-lima": 1018,  # original has 1030
#             "WizardLMTeam/WizardLM_evol_instruct_V2_196k": 30000,
#             "Open-Orca/OpenOrca": 30000,
#             "natolambert/tulu-v2-sft-mixture-science": 7468,  # original data slightly different
#         }
#         _ = get_datasets(dataset_mixer, splits=["train"], columns_to_keep=["messages"])
