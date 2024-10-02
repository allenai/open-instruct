"""
Preference dataset manipulation utils.
"""


def convert_message_keys(example):
    converted_messages = []
    for message in example["chosen"]:
        converted_messages.append(
            {"role": "user" if message["from"] == "user" else "assistant", "content": message["value"]}
        )

    # Create a new dictionary with the converted messages
    converted_example = example.copy()
    converted_example["chosen"] = converted_messages

    # repeat for rejected
    converted_messages = []
    for message in example["rejected"]:
        converted_messages.append(
            {"role": "user" if message["from"] == "user" else "assistant", "content": message["value"]}
        )
    converted_example["rejected"] = converted_messages

    return converted_example
