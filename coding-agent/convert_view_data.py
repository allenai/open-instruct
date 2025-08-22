PATH_TO_DATA = "/weka/oe-adapt-default/saurabhs/repos/open-instruct-3/coding-agent/data/ft_hermes_search_swesmith_think_atk_ru_rc_SYSTEM_WITH_TOOL_FIND.jsonl"
PATH_TO_GOLD = "/weka/oe-adapt-default/saurabhs/repos/open-instruct-3/coding-agent/data/post_instances_final.yaml"
HF_OUTPUT_MULTI_STEP_TOOL = "saurabh5/rlvr-code-view-tool"
HF_OUTPUT_SINGLE_STEP = "saurabh5/rlvr-code-view-single-turn"

import json
import yaml
from datasets import Dataset


with open(PATH_TO_GOLD, "r") as f:
    gold_data = yaml.safe_load(f)


def main():
    with open(PATH_TO_GOLD, "r") as f:
        gold_data = yaml.safe_load(f)

    with open(PATH_TO_DATA, "r") as f:
        data = [json.loads(line) for line in f]
    


if __name__ == "__main__":
    main()