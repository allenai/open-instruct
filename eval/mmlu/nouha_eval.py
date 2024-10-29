import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from datasets import load_dataset
from eval.mmlu.categories import subcategories, categories
from eval.utils import get_next_word_predictions, load_hf_tokenizer, load_hf_lm, query_openai_chat_model, \
    dynamic_import_function, upload_results_to_hf, check_and_upload_model_metadata

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=True):
    """Format a single example with proper indexing for choices."""
    prompt = df.iloc[idx, 0]  # question
    for j, choice in enumerate(choices):  # use choices directly
        prompt += "\n{}. {}".format(choice, df.iloc[idx, j + 1])

    # # Conditional chain-of-thought instruction
    # prompt += (
    #     "\n\nIf this question involves math or requires logical reasoning, please solve it step-by-step to ensure accuracy. "
    #     "Otherwise, select the answer based on your knowledge."
    # )
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, 5])  # correct answer is at index 5
    return prompt


def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


def load_mmlu_data(subject, split, n_instances=None):
    """Load MMLU data from Hugging Face datasets."""
    dataset = load_dataset('cais/mmlu', subject)[split]

    # Convert to DataFrame format compatible with existing code
    df = pd.DataFrame({
        0: dataset['question'],
        1: [choices[0] for choices in dataset['choices']],
        2: [choices[1] for choices in dataset['choices']],
        3: [choices[2] for choices in dataset['choices']],
        4: [choices[3] for choices in dataset['choices']],
        5: [choices[dataset['answer'][i]] for i in range(len(dataset))]
    })

    if n_instances and n_instances < len(df):
        df = df.sample(n_instances, random_state=42)

    return df


@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1):
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
            if prompt[-1] in ["\n", " "]:
                prompt += "The answer is:"
            else:
                prompt += " The answer is:"

        tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        while len(tokenized_prompt) > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

            if args.use_chat_format:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
                if prompt[-1] in ["\n", " "]:
                    prompt += "The answer is:"
                else:
                    prompt += " The answer is:"

            tokenized_prompt = tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids
        prompts.append(prompt)

    answer_choice_ids = [tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1] for answer_choice in
                         choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts,
        candidate_token_ids=answer_choice_ids,
        return_token_predictions=False,
        batch_size=batch_size
    )

    cors = []
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = groud_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)

    print("Average accuracy {:.3f} - {}".format(acc, subject))
    return cors, acc, all_probs


def eval_openai_chat_engine(args, subject, engine, dev_df, test_df, batch_size=1):
    import tiktoken
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
    groud_truths = test_df.iloc[:, -1].values
    for i in range(len(test_df)):
        prediction = results[i]["output"].strip()
        ground_truth = groud_truths[i]
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
            device_map="auto",
            gptq_model=args.gptq,
        )
        from transformers import GPTNeoXForCausalLM, OPTForCausalLM
        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
            tokenizer.model_max_length = model.config.max_position_embeddings
            print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(
                model.config.max_position_embeddings))

    # Get subjects from categories.py
    math_subjects = ['abstract_algebra', 'college_mathematics', 'elementary_mathematics',
                     'high_school_mathematics', 'high_school_statistics']

    if args.debug_math:
        subjects = math_subjects
        print(f"Evaluating math subjects: {subjects}")
    else:
        subjects = sorted(list(subcategories.keys()))

    if args.subjects:
        assert all(
            subj in subjects for subj in args.subjects), f"Some subjects specified are not valid: {args.subjects}"
        subjects = args.subjects

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }
    cat_cors = {cat: [] for cat in categories}

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        # Load data directly from HF
        dev_df = load_mmlu_data(subject, 'dev')[:args.ntrain]
        test_df = load_mmlu_data(subject, 'test', args.n_instances)

        if args.model_name_or_path:
            cors, acc, probs = eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size)
        else:
            cors, acc, probs = eval_openai_chat_engine(args, subject, args.openai_engine, dev_df, test_df,
                                                       args.eval_batch_size)

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # Save detailed results for this subject
        # Save detailed results for this subject
        test_df["correct"] = cors
        prompts = []
        for i in range(len(test_df)):
            prompt = test_df.iloc[i, 0]  # question
            choices_dict = {
                "A": test_df.iloc[i, 1],
                "B": test_df.iloc[i, 2],
                "C": test_df.iloc[i, 3],
                "D": test_df.iloc[i, 4],
            }
            formatted_prompt = prompt + "\n" + "\n".join([
                f"{choice}. {text}" for choice, text in choices_dict.items()
            ])
            prompts.append(formatted_prompt)

        test_df["prompt"] = prompts

        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df[f"choice{choice}_probs"] = probs[:, j]

        # Add model predictions
        test_df["prediction"] = [choices[pred_idx] for pred_idx in np.argmax(probs, axis=1)]

        # Save both CSV and detailed results
        test_df.to_csv(os.path.join(args.save_dir, f"{subject}.csv"), index=None)

        # Save detailed results in a more readable format
        detailed_results = []
        for i in range(len(test_df)):
            result = {
                "question": test_df.iloc[i, 0],
                "choices": {
                    "A": test_df.iloc[i, 1],
                    "B": test_df.iloc[i, 2],
                    "C": test_df.iloc[i, 3],
                    "D": test_df.iloc[i, 4]
                },
                "correct_answer": test_df.iloc[i, 5],
                "model_prediction": test_df["prediction"].iloc[i],
                "is_correct": bool(cors[i]),
                "probabilities": {
                    choice: float(probs[i][j])
                    for j, choice in enumerate(choices)
                }
            }
            detailed_results.append(result)

        with open(os.path.join(args.save_dir, f"{subject}_detailed.json"), "w") as f:
            json.dump(detailed_results, f, indent=2)

    # Calculate metrics only for subcategories with data
    for subcat in subcat_cors:
        if subcat_cors[subcat]:  # Only calculate if we have data
            subcat_acc = np.mean(np.concatenate(subcat_cors[subcat]))
            print("Average accuracy {:.3f} - {}".format(subcat_acc, subcat))

    # Calculate metrics only for categories with data
    for cat in cat_cors:
        if cat_cors[cat]:  # Only calculate if we have data
            cat_acc = np.mean(np.concatenate(cat_cors[cat]))
            print("Average accuracy {:.3f} - {}".format(cat_acc, cat))

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # Save metrics
    metrics = {
        "average_acc": float(weighted_acc),
        "subcat_acc": {
            subcat: float(np.mean(np.concatenate(subcat_cors[subcat])))
            for subcat in subcat_cors
            if subcat_cors[subcat]
        },
        "cat_acc": {
            cat: float(np.mean(np.concatenate(cat_cors[cat])))
            for cat in cat_cors
            if cat_cors[cat]
        },
    }

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if args.upload_to_hf is not None:
        task_name = f"oi_mmlu_{args.ntrain}shots"
        primary_score = metrics["average_acc"]
        upload_results_to_hf(
            metrics,
            args.upload_to_hf,
            args.hf_upload_name,
            task_name=task_name,
            primary_score=primary_score,
            prepend_timestamp=True,
        )
        check_and_upload_model_metadata(
            args.model_name_or_path, args.upload_to_hf, args.hf_upload_name, hf_revision=args.hf_revision
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="results/mmlu_Llama-3.1-8B/")
    parser.add_argument("--model_name_or_path", type=str, default=None,
                        help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--hf_revision", type=str, default=None,
                        help="if specified, we will load the model from a revision of the model in the hub")
    parser.add_argument("--debug_math", action="store_true",
                        help="Only evaluate math-related subjects")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None,
                        help="if specified, we will load the tokenizer from here.")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="If given, we will use the slow tokenizer.")
    parser.add_argument("--openai_engine", type=str, default=None,
                        help="if specified, we will use the OpenAI API to generate the predictions.")
    parser.add_argument("--subjects", nargs="*",
                        help="which subjects to evaluate. If not specified, all subjects will be evaluated.")
    parser.add_argument("--n_instances", type=int,
                        help="if specified, a maximum of n_instances per subject will be used for evaluation.")
    parser.add_argument("--eval_batch_size", type=int, default=1,
                        help="batch size for evaluation.")
    parser.add_argument("--load_in_8bit", action="store_true",
                        help="load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true",
                        help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--use_chat_format", action="store_true",
                        help="If given, we will use the chat format for the prompts.")
    parser.add_argument("--chat_formatting_function", type=str,
                        default="eval.templates.create_prompt_with_tulu_chat_format",
                        help="The function to use to create the chat format.")
    parser.add_argument("--upload_to_hf", type=str, default=None,
                        help="If specified, we will upload the results to Hugging Face Datasets.")
    parser.add_argument("--hf_upload_name", type=str, default=None,
                        help="If uploading to hf, this is the model name")

    args = parser.parse_args()

    assert (args.model_name_or_path is None) != (args.openai_engine is None), \
        "Either model_name_or_path or openai_engine should be specified."

    main(args)