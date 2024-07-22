# !/usr/bin/env python
# coding=utf-8
# Copyright 2024 AllenAI. All rights reserved.
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

# script for mixing and saving data
from .utils import ArgumentParserPlus, FlatArguments, get_datasets

# Run as module for local imports, e.g.:
# python -m open_instruct.mix_data configs/train_configs/sft/default.yaml --save_data_dir=output/tmp/


def main():
    parser = ArgumentParserPlus((FlatArguments))
    args = parser.parse()

    # assert that data_mixer is not none in config
    assert args.dataset_mixer is not None, "data_mixer is required in config"

    raw_datasets = get_datasets(
        args.dataset_mixer,
        configs=args.dataset_config_name,
        splits=["train"],
        save_data_dir=args.dataset_mix_dir,  # location where dataset is saved as json
        columns_to_keep=["messages"],
        need_columns=["messages"],
    )

    # print first 5 samples of dataset
    for i in range(5):
        print(raw_datasets["train"][i])


if __name__ == "__main__":
    main()
