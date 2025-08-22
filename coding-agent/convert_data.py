import json
from datasets import Dataset
"""
read the jsonl file and convert to rl dataset. 

each line in the jsonl file might create multiple rl dataset examples if its a multi-turn conversation. 

each rl example corresponds to an assistant turn, where the user turn is a concat of all previous turns.

the rl example is a dict with the following keys:
- "messages": list of dicts, each dict has "role" and "content" keys.
    - this will only have 1 message actually, which is the "user" turn. the content is the concat of all previous turns.
- "ground_truth": str, the ground truth answer -> the assistant turn in the jsonl file.
- "dataset": "string_f1" for all the rl examples
"""

JSONL_PATH = "/weka/oe-adapt-default/ethans/datagen/datagen/data/data_expts/simple_bug_full_v2/data/ft_hermes_distill_swesmith_think_atk_ru_rc_SYSTEM_WITH_EDIT.jsonl"
HF_DATASET_NAME = "saurabh5/rlvr-coding-agent-f1"

def convert_conversation_to_rl_examples(conversation):
    """
    Convert a single conversation (list of messages) into multiple RL examples.
    Each assistant turn becomes a separate RL example.
    """
    rl_examples = []
    
    for i, message in enumerate(conversation):
        if message.get("role") == "assistant":
            # Collect all previous messages (system, user, and assistant turns up to this point)
            previous_messages = conversation[:i]
            
            # Concatenate all previous messages into a single user prompt
            concatenated_content = ""
            for prev_msg in previous_messages:
                role = prev_msg.get("role", "")
                content = prev_msg.get("content", "")
                concatenated_content += f"{role}: {content}\n"
            
            # Remove trailing newline
            concatenated_content = concatenated_content.rstrip()
            
            # Create RL example
            rl_example = {
                "messages": [
                    {
                        "role": "user",
                        "content": concatenated_content
                    }
                ],
                "ground_truth": message.get("content", ""),
                "dataset": "string_f1"
            }
            
            rl_examples.append(rl_example)
    
    return rl_examples

def main():
    print(f"Reading JSONL file: {JSONL_PATH}")
    
    all_rl_examples = []
    
    # Read JSONL file line by line
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Check if the data has messages (assuming standard conversation format)
                if "messages" in data:
                    conversation = data["messages"]
                elif isinstance(data, list):
                    # If the line itself is a list of messages
                    conversation = data
                else:
                    print(f"Warning: Skipping line {line_num}, unexpected format")
                    continue
                
                # Convert conversation to RL examples
                rl_examples = convert_conversation_to_rl_examples(conversation)
                all_rl_examples.extend(rl_examples)
                
                if line_num % 100 == 0:
                    print(f"Processed {line_num} conversations, generated {len(all_rl_examples)} RL examples so far")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"Total RL examples generated: {len(all_rl_examples)}")
    
    if len(all_rl_examples) == 0:
        print("No examples generated. Please check the input file format.")
        return
    
    # Create Hugging Face dataset
    print("Creating Hugging Face dataset...")
    dataset = Dataset.from_list(all_rl_examples)
    
    print(f"Dataset created with {len(dataset)} examples")
    print("Sample example:")
    print(dataset[0])
    
    # Upload to Hugging Face Hub
    print(f"Uploading to Hugging Face Hub: {HF_DATASET_NAME}")
    try:
        dataset.push_to_hub(HF_DATASET_NAME)
        print("Successfully uploaded to Hugging Face Hub!")
    except Exception as e:
        print(f"Error uploading to Hugging Face Hub: {e}")
        print("You may need to login first with: huggingface-cli login")


if __name__ == "__main__":
    main()
