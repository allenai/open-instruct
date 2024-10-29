import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from datasets import load_dataset
from typing import List, Dict, Tuple, Optional
import tiktoken

choices = ["A", "B", "C", "D"]


def load_mmlu_data_from_hf(subject: str, split: str, n_instances: Optional[int] = None) -> pd.DataFrame:
    """Load MMLU data for a specific subject and split from Hugging Face."""
    dataset = load_dataset('cais/mmlu', subject)[split]

    # Convert the dataset to pandas DataFrame
    df = pd.DataFrame({
        0: dataset['question'],
        1: [choices[0] for choices in dataset['choices']],  # A
        2: [choices[1] for choices in dataset['choices']],  # B
        3: [choices[2] for choices in dataset['choices']],  # C
        4: [choices[3] for choices in dataset['choices']],  # D
        5: [choices[dataset['answer'][i]] for i in range(len(dataset))]  # Convert answer index to letter
    })

    if n_instances and n_instances < len(df):
        df = df.sample(n_instances, random_state=42)

    return df


def get_available_subjects() -> List[str]:
    """Get list of available subjects in MMLU."""
    dataset = load_dataset('cais/mmlu')
    return sorted(dataset.keys())


def format_subject(subject: str) -> str:
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df: pd.DataFrame, idx: int, include_answer: bool = True) -> str:
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt


def gen_prompt(train_df: pd.DataFrame, subject: str, k: int = -1) -> str:
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()
def eval_hf_model(args, subject: str, model, tokenizer, dev_df: pd.DataFrame, test_df: pd.DataFrame,
                  batch_size: int = 1) -> Tuple[np.ndarray, float, np.ndarray]:
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            prompt += " The answer is:" if prompt[-1] not in ["\n", " "] else "The answer is:"

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids

        # Ensure prompt is within token limit
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                prompt += " The answer is:" if prompt[-1] not in ["\n", " "] else "The answer is:"

            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        prompts.append(prompt)

    # Get predictions
    answer_choice_ids = [tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1] for answer_choice in
                         choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids,
        return_token_predictions=False, batch_size=batch_size
    )

    # Calculate metrics
    cors = []
    ground_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = ground_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def eval_openai_chat_engine(args, subject: str, engine: str, dev_df: pd.DataFrame, test_df: pd.DataFrame,
                            batch_size: int = 1) -> Tuple[np.ndarray, float, np.ndarray]:
    gpt_tokenizer = tiktoken.get_encoding("cl100k_base")
    answer_choice_ids = [gpt_tokenizer.encode(" " + x)[0] for x in choices]

    prompts = []
    for i in range(0, test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        prompts.append(prompt)

    instances = [{"id": prompt, "prompt": prompt} for _, prompt in enumerate(prompts)]
    results = query_openai_chat_model(
        engine=args.openai_engine,
        instances=instances,
        batch_size=args.eval_batch_size if args.eval_batch_size else 10,
        output_path=os.path.join(args.save_dir, f"{subject}_openai_results.jsonl"),
        logit_bias={token_id: 100 for token_id in answer_choice_ids},
        max_tokens=1,
    )

    cors = []
    ground_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = ground_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array([[0.25, 0.25, 0.25, 0.25] for _ in range(len(test_df))])

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def main(args):
    if args.model_name_or_path:
        print("Loading model and tokenizer...")
        tokenizer = load_hf_tokenizer(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
        model = load_hf_lm(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq,
        )

        from transformers import GPTNeoXForCausalLM, OPTForCausalLM
        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print(f"Set tokenizer.model_max_length to {model.config.max_position_embeddings}")

    # Get subjects from HF dataset
    subjects = get_available_subjects()
    if args.subjects:
        assert all(subj in subjects for subj in args.subjects), f"Invalid subjects: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors = []
    subcat_cors = {subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists}
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects, desc="Evaluating subjects"):
        # Load data from HF
        dev_df = load_mmlu_data_from_hf(subject, 'dev')[:args.ntrain]
        test_df = load_mmlu_data_from_hf(subject, 'test', args.n_instances)

        if args.model_name_or_path:
            cors, acc, probs = eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size)
        else:
            cors, acc, probs = eval_openai_chat_engine(args, subject, args.openai_engine, dev_df, test_df,
                                                       args.eval_batch_size)

        # Update metrics
        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # Save results
        test_df["correct"] = cors
        for j in range(probs.shape[1]):
            test_df[f"choice{choices[j]}_probs"] = probs[:, j]
        test_df.to_csv(os.path.join(args.save_dir, f"{subject}.csv"), index=None)

    # Calculate and print metrics
    for subcat in subcat_cors:
        subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
        print(f"Average accuracy {subcat_acc:.3f} - {subcat}")

    for cat in cat_cors:
        cat_acc = np.mean(np.concatenate(cat_cors[cat]))
        print(f"Average accuracy {cat_acc:.3f} - {cat}")

    weighted_acc = np.mean(np.concatenate(all_cors))
    print(f"Average accuracy: {weighted_acc:.3f}")

    # Save metrics
    metrics = {
        "average_acc": weighted_acc,
        "subcat_acc": {
            subcat: np.mean(np.concatenate(subcat_cors[subcat]))
            for subcat in subcat_cors
        },
        "cat_acc": {
            cat: np.mean(np.concatenate(cat_cors[cat]))
            for cat in cat_cors
        },
    }

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)

    # Upload results if specified
    if args.upload_to_hf is not None:
        task_name = f"oi_mmlu_{args.ntrain}shots"
        upload_results_to_hf(
            metrics,
            args.upload_to_hf,
            args.hf_upload_name,
            task_name=task_name,
            primary_score=metrics["average_acc"],
            prepend_timestamp=True,
        )
        check_and_upload_model_metadata(
            args.model_name_or_path, args.upload_to_hf, args.hf_upload_name, hf_revision=args.hf_revision
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="results/mmlu/")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--hf_revision", type=str, default=None,
                        help="revision of the model to load from the hub")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="if specified, we will load the tokenizer from here.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If given, we will use the slow tokenizer.")
    parser.add_argument("--openai_engine", type=str, default=None,
                        help="if specified, we will use the OpenAI API")
    parser.add_argument("--subjects", nargs="*",
                        help="which subjects to evaluate. If not specified, all subjects will be evaluated.")
    parser.add_argument("--n_instances", type=int,
                        help="if specified, maximum number of instances per subject")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="load model in 8bit mode")
    parser.add_argument("--gptq", action="store_true",
                        help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true",
                        help="If given, we will use the chat format for the prompts.")
    parser.add_argument("--chat_formatting_function", type=str,
                        default="eval.templates.create_prompt_with_tulu_chat_format",
                        help="The function to use to create the chat format.")
    parser.add_argument("--upload_to_hf", type=str, default=None,
                        help="If specified, upload results to Hugging Face Datasets.")
    parser.add_argument("--hf_upload_name", type=str, default=None,
                        help="If uploading to hf, this is the model name")

    args = parser.parse_args()

    assert (args.model_name_or_path is None) != (args.openai_engine is None), \
        "Either model_name_or_path or openai_engine should be specified."

    main(args)