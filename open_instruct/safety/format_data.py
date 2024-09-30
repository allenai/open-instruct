import json

def reformat_wildguard_responses(input_file, output_file):
    # Read the original JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Initialize the new format
    new_format = []

    # Process each item in the original data
    for item in data:
        prompt = item['prompt']
        response = item['responses'].get('gpt_response') or item['responses'].get('anthropic_response')

        if response:
            # Create a separate block for each prompt-response pair
            conversation_block = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "assistant",
                        "content": response
                    }
                ]
            }
            new_format.append(conversation_block)

    # Write the new format to a JSON file
    with open(output_file, 'w') as f:
        json.dump(new_format, f, indent=2)

    print(f"Reformatted data saved to {output_file}")

if __name__ == "__main__":
    input_file = 'wildguard_responses.json'
    output_file = 'wildguard_responses_reformatted.json'
    reformat_wildguard_responses(input_file, output_file)