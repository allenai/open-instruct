# !/usr/bin/env python
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

from open_instruct.finetune import FlatArguments

# script for mixing and saving data
from open_instruct.utils import ArgumentParserPlus, get_datasets

# Run as module for local imports, e.g.:
# python open_instruct/mix_data.py configs/train_configs/sft/tulu3_8b_preview_mix_v3.4.yaml --dataset_mix_dir=output/tmp/
# can pass --save_to_hub=allenai/tulu-v3.1-mix-preview-4096-OLMoE
# note that = is needed with our argparser


def main() -> None:
    parser = ArgumentParserPlus((FlatArguments,))  # type: ignore[arg-type]
    (args,) = parser.parse()  # type: ignore[misc]
    assert isinstance(args, FlatArguments)

    # assert that data_mixer is not none in config
    assert args.dataset_mixer is not None, "data_mixer is required in config"
    configs = [args.dataset_config_name] if args.dataset_config_name else None

    raw_datasets = get_datasets(
        args.dataset_mixer,
        configs=configs,
        splits=["train"],
        save_data_dir=args.dataset_mix_dir,  # location where dataset is saved as json
        columns_to_keep=["messages"],
        keep_ids=True,
        add_source_col=True,
    )

    # print first 5 samples of dataset
    for i in range(5):
        print(raw_datasets["train"][i])

    # if args.save_to_hub is not none, push dataset to hub
    if args.save_to_hub:
        raw_datasets["train"].push_to_hub(args.save_to_hub, private=True)


if __name__ == "__main__":
    main()
