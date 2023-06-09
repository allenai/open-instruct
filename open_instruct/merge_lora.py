import argparse
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_name_or_path", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str)
    parser.add_argument("--output_dir", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    peft_config = PeftConfig.from_pretrained(args.lora_model_name_or_path)
    print("Loading the base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path if args.base_model_name_or_path else peft_config.base_model_name_or_path,
    )
    print("Loading the lora model...")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_model_name_or_path)
    print("Merging the lora modules...")
    merged_model = lora_model.base_model.merge_and_unload()
    output_dir = args.output_dir if args.output_dir else args.lora_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(args.lora_model_name_or_path)
    embedding_size = merged_model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        print(f"The vocabulary size of the tokenizer in the lora model folder contains {len(tokenizer)-embedding_size} more tokens than the base model.")
        print("Resizing the token embeddings of the merged model...")
        merged_model.resize_token_embeddings(len(tokenizer))
    print(f"Saving to {output_dir}...")
    merged_model.save_pretrained(output_dir)



    