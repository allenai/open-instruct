from datasets import Dataset, DatasetDict, load_dataset

import open_instruct.utils as open_instruct_utils

# load dataset
dataset = load_dataset(
    "rulins/gpqa_extended_decontamination", split="train", num_proc=open_instruct_utils.max_num_processes()
)
# construct rlvr-style dataset
new_train_data = []
for data in dataset:
    choices = [
        data["Correct Answer"],
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    correct_answer = data["Correct Answer"]
    # shuffle the choices
    choices = sorted(choices, key=lambda x: x.lower())
    # find the letter of the correct answer
    letter = chr(65 + choices.index(correct_answer))
    # add the letter to the choices
    choices = [f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)]
    # construct the new data
    new_train_data.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Answer the following multiple choice question. The last line of your response "
                        "should be of the following format: `<answer>LETTER</answer>` where LETTER "
                        "is one of ABCD. Search the web by wrapping a query in query tags like so: <query></query>\n"
                        "Queries should comprise of a few keywords that are relevant to the question. "
                        "For example: <query>personalized dialogue generation, personalized language models, personalized dialogue</query>\n"
                        f"Question {data['Question']}\n"
                        f"Choices: {', '.join(choices)}"
                    ),
                }
            ],
            "ground_truth": letter,
            "dataset": "string_matcher",
        }
    )

# create eval set (gpqa diamond)
dataset = load_dataset(
    "Idavidrein/gpqa", "gpqa_diamond", split="train", num_proc=open_instruct_utils.max_num_processes()
)
# construct rlvr-style dataset
new_val_data = []
for data in dataset:
    choices = [
        data["Correct Answer"],
        data["Incorrect Answer 1"],
        data["Incorrect Answer 2"],
        data["Incorrect Answer 3"],
    ]
    correct_answer = data["Correct Answer"]
    # shuffle the choices
    choices = sorted(choices, key=lambda x: x.lower())
    # find the letter of the correct answer
    letter = chr(65 + choices.index(correct_answer))
    # add the letter to the choices
    choices = [f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices)]
    # construct the new data
    new_val_data.append(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Answer the following multiple choice question. The last line of your response "
                        "should be of the following format: `<answer>LETTER</answer>` where LETTER "
                        "is one of ABCD. Search the web by wrapping a query in query tags like so: <query></query>\n"
                        "Queries should comprise of a few keywords that are relevant to the question. "
                        "For example: <query>personalized dialogue generation, personalized language models, personalized dialogue</query>\n"
                        f"Question {data['Question']}\n"
                        f"Choices: {', '.join(choices)}"
                    ),
                }
            ],
            "ground_truth": letter,
            "dataset": "string_matcher",
        }
    )

ds_train = Dataset.from_list(new_train_data)
ds_train = ds_train.shuffle(seed=42)
ds_val = Dataset.from_list(new_val_data)
ds_val = ds_val.shuffle(seed=42)
ds = DatasetDict({"train": ds_train, "test": ds_val})
ds.push_to_hub("hamishivi/GPQA-RLVR")
