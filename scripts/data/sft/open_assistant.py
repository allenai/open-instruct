import argparse
import os

from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils
from scripts.data.sft.utils import convert_sft_dataset


def convert_oasst_dataset(ds: Dataset, top_k: int | None = None) -> Dataset:
    """
    For Open Assistant datasets, because it's in a tree structure, where every user input might get multiple replies,
    we have to save every path from the root node to the assistant node
    (including both leaf node and intemediate node).
    This results in some of the messages being duplicated among different paths (instances).
    You can set top_k to control how many replies to consider
    when traversing the tree, which will consider the replies with
    the highest human-reviewed quality scores when expanding the tree.
    """
    # convert the hf dataset to a list of examples
    ds = ds.to_list()

    # for messages, get all of their replies
    print("Pairing replies with parent messages...")
    parent_id_to_replies = {}
    for message in ds:
        if message["parent_id"]:
            if message["parent_id"] not in parent_id_to_replies:
                parent_id_to_replies[message["parent_id"]] = []
            parent_id_to_replies[message["parent_id"]].append(message)
    print("Done!")

    # extract the quality score from the labels. If none is found, set it to 0
    for message in ds:
        assert "labels" in message, "`labels` field should be in the Open Assistant dataset"
        if not message["labels"] or "quality" not in message["labels"]["name"]:
            message["quality_score"] = 0
        else:
            assert len(message["labels"]["name"]) == len(
                message["labels"]["value"]
            ), "Number of label names and values should be the same"
            message["quality_score"] = message["labels"]["value"][message["labels"]["name"].index("quality")]

    # tranvers the conversation tree from a given messagenode, and collect all valid sequences
    def dfs(node, stack, valid_sequences):
        if node["deleted"]:
            return
        replies = parent_id_to_replies.get(node["message_id"], [])
        if node["role"] == "assistant":
            stack.append({"role": "assistant", "content": node["text"], "quality_score": node["quality_score"]})
            if not replies:  # leaf node
                valid_sequences.append(stack[:])
            else:
                replies = [child for child in replies if not child["deleted"]]
                if top_k is not None:
                    replies = sorted(replies, key=lambda x: x["quality_score"], reverse=True)[:top_k]
                for child in replies:
                    dfs(child, stack, valid_sequences)
            stack.pop()
        elif node["role"] == "prompter":
            stack.append({"role": "user", "content": node["text"], "quality_score": node["quality_score"]})
            replies = [child for child in replies if not child["deleted"]]
            if top_k is not None:
                replies = sorted(replies, key=lambda x: x["quality_score"], reverse=True)[:top_k]
            for child in replies:
                dfs(child, stack, valid_sequences)
            stack.pop()
        else:
            raise ValueError(f"Unknown role: {node['role']}")

    # find all the root nodes
    root_messages = [d for d in ds if d["parent_id"] is None]
    valid_sequences = []
    for root in root_messages:
        dfs(root, [], valid_sequences)
    return valid_sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Open Assistant 1 and 2 datasets and optionally upload to Hugging Face Hub."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )
    parser.add_argument(
        "--converted_dataset_name", type=str, default=None, help="Name of the converted dataset on Hugging Face Hub."
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default=None,
        help="Local directory to save the dataset (if not provided, does not save locally)",
    )
    parser.add_argument(
        "--apply_keyword_filters",
        action="store_true",
        help=(
            "Apply keyword filters to the dataset. "
            "Currently filters out conversations with OpenAI, ChatGPT, BingChat, etc."
        ),
    )
    parser.add_argument(
        "--apply_empty_message_filters", action="store_true", help="Apply empty message filters to the dataset."
    )
    parser.add_argument(
        "--top_k", type=int, default=1, help="Number of children replies to consider when traversing the tree."
    )
    args = parser.parse_args()

    v1_readme_content = (
        "This is a converted version of the Open Assistant 1 dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/open_assistant.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- top_k: {args.top_k}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "Please refer to the [original dataset](https://huggingface.co/datasets/OpenAssistant/oasst1) "
        "for more information about this dataset and the license."
    )

    v1_ds = load_dataset("OpenAssistant/oasst1", num_proc=open_instruct_utils.max_num_processes())["train"]
    v1_sequences = convert_oasst_dataset(v1_ds, top_k=args.top_k)
    v1_instances = []
    for i, sequence in enumerate(v1_sequences):
        quality_scores = [m["quality_score"] for m in sequence]
        avg_quality_score = sum(quality_scores) / len(quality_scores)
        sequence = [{"role": m["role"], "content": m["content"]} for m in sequence]
        v1_instances.append(
            {
                "dataset": "oasst1",
                "id": f"oasst1_{i}",
                "messages": sequence,
                "quality_scores": quality_scores,
                "avg_quality_score": avg_quality_score,
            }
        )
    v1_ds = Dataset.from_list(v1_instances)
    convert_sft_dataset(
        ds=v1_ds,
        hf_dataset_id=None,
        convert_fn=None,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name="oasst1_converted"
        if args.converted_dataset_name is None
        else args.converted_dataset_name + "_v1",
        local_save_dir=os.path.join(args.local_save_dir, "oasst1"),
        readme_content=v1_readme_content,
    )

    v2_readme_content = (
        "This is a converted version of the Open Assistant 2 dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/open_assistant.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- top_k: {args.top_k}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "Please refer to the [original dataset](https://huggingface.co/datasets/OpenAssistant/oasst2) "
        "for more information about this dataset and the license."
    )

    v2_ds = load_dataset("OpenAssistant/oasst2", num_proc=open_instruct_utils.max_num_processes())["train"]
    v2_sequences = convert_oasst_dataset(v2_ds, top_k=args.top_k)
    v2_instances = []
    for i, sequence in enumerate(v2_sequences):
        quality_scores = [m["quality_score"] for m in sequence]
        avg_quality_score = sum(quality_scores) / len(quality_scores)
        sequence = [{"role": m["role"], "content": m["content"]} for m in sequence]
        v2_instances.append(
            {
                "dataset": "oasst2",
                "id": f"oasst2_{i}",
                "messages": sequence,
                "quality_scores": quality_scores,
                "avg_quality_score": avg_quality_score,
            }
        )
    v2_ds = Dataset.from_list(v2_instances)
    convert_sft_dataset(
        ds=v2_ds,
        hf_dataset_id=None,
        convert_fn=None,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name="oasst2_converted"
        if args.converted_dataset_name is None
        else args.converted_dataset_name + "_v2",
        local_save_dir=os.path.join(args.local_save_dir, "oasst2"),
        readme_content=v2_readme_content,
    )
