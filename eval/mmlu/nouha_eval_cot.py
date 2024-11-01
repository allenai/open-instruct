'''This is CoT for MMLU'''

import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from typing import Dict, List, Optional
from datasets import load_dataset
from dataclasses import asdict, dataclass
from eval.mmlu.categories import subcategories, categories
from eval.utils import get_next_word_predictions, load_hf_tokenizer, load_hf_lm, query_openai_chat_model, \
    dynamic_import_function, upload_results_to_hf, check_and_upload_model_metadata
from vllm import LLM, SamplingParams



choices = ["A", "B", "C", "D"]

@dataclass
class GenerationArgs:
    num_completions: int = 3
    temperature: float = 0.8
    response_length: int = 2048
    top_p: float = 0.9
    tensor_parallel_size: int = 1


COT_SUBJECTS = {
    'abstract_algebra',
    'college_mathematics',
    'elementary_mathematics',
    'high_school_mathematics',
    'high_school_statistics',
    'formal_logic',
    'conceptual_physics',
    'high_school_physics',
    'college_physics'
}

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s


def format_example(df, idx, include_answer=False, subject=None):
    """Format a single example with or without CoT based on subject type."""
    is_cot_subject = subject in COT_SUBJECTS

    if is_cot_subject:
        # CoT prompt for math/logic subjects
        instruction = ("You're a helpful assistant, answer the following question by choosing an option. "
                       "Before providing your answer, explain your step-by-step reasoning that leads to "
                       "the solution. End your response with 'Answer: X' where X is one of A, B, C, or D.\n\n")
    else:
        # Direct prompt for other subjects
        instruction = ("You're a helpful assistant. Choose the correct option for this question."
                       " Provide your answer in the format 'Answer: X' where X is one of A, B, C, or D.\n\n")

    prompt = df.iloc[idx, 0]
    prompt = instruction + "Question: " + prompt

    # Add choices
    for j, choice in enumerate(choices):
        prompt += "\n{}. {}".format(choice, df.iloc[idx, j + 1])

    # Add final instruction
    if is_cot_subject:
        prompt += ("\n\nExplain your reasoning step by step, then provide your final answer."
                   " End your response with 'Answer: X' where X is one of A, B, C, or D.")
    else:
        prompt += "\n\nProvide your answer:"
    return prompt


def gen_prompt(train_df, subject, k=-1):
    """Generate prompt with structured CoT formatting."""
    prompt = "The following are multiple choice questions about {}. For each question, provide your step-by-step reasoning, then give your answer in the format 'Answer: X' where X is one of A, B, C, or D.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

import re


def parse_cot_response(response):
    """Parse the model's CoT response to extract the final answer."""
    # First try to find the explicit "Answer: X" format
    answer_pattern = r"Answer:\s*([ABCD])"
    match = re.search(answer_pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback patterns if the model didn't follow the format
    backup_patterns = [
        r"therefore,?\s*the\s*answer\s*is:?\s*([ABCD])",
        r"([ABCD])\s*is\s*correct",
        r".*[^A-D]*([ABCD])[^A-D]*$"
    ]

    for pattern in backup_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    return None


def generate_with_vllm(model_name_or_path: str, revision: str, prompt_token_ids: List[int], gen_args: GenerationArgs):
    from vllm.engine.arg_utils import EngineArgs

    # Use only the essential parameters
    engine_args = EngineArgs(
        model=model_name_or_path,
        trust_remote_code=True,
        tensor_parallel_size=gen_args.tensor_parallel_size
    )

    llm = LLM(engine_args=engine_args)

    sampling_params = SamplingParams(
        n=gen_args.num_completions,
        temperature=gen_args.temperature,
        top_p=gen_args.top_p,
        max_tokens=gen_args.response_length
    )

    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
    )

    return [
        {
            "outputs": [
                {
                    "text": out.text,
                    "token_ids": out.token_ids,
                } for out in output.outputs
            ],
            "prompt": output.prompt,
        }
        for output in outputs
    ]
# def generate_with_vllm(model_name_or_path: str, revision: str, prompt_token_ids: List[int], gen_args: GenerationArgs):
#     llm = LLM(
#         model=model_name_or_path,
#         revision=revision,
#         tokenizer_revision=revision,
#         tensor_parallel_size=gen_args.tensor_parallel_size,
#         max_model_len=gen_args.response_length,
#     )
#     # filter out prompts which are beyond the model's max token length
#     max_model_len = llm.llm_engine.scheduler_config.max_model_len
#     prompt_token_ids_len = len(prompt_token_ids)
#     prompt_token_ids = [item for item in prompt_token_ids if len(item) < max_model_len]
#     if len(prompt_token_ids) != prompt_token_ids_len:
#         print(f"Filtered out {prompt_token_ids_len - len(prompt_token_ids)} prompts which exceeds max token length")
#
#     outputs = llm.generate(
#         prompt_token_ids=prompt_token_ids,
#         sampling_params=SamplingParams(
#             n=gen_args.num_completions,
#             temperature=gen_args.temperature,
#             top_p=1.0,
#             max_tokens=gen_args.response_length,
#             include_stop_str_in_output=True,
#         ),
#     )
#     return [
#         {
#             "outputs": [asdict(out) for out in output.outputs],
#             "prompt": output.prompt,
#             "prompt_logprobs": output.prompt_logprobs,
#             "metrics": output.metrics,
#         }
#         for output in outputs
#     ]


@torch.no_grad()
def eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, batch_size=1):
    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None

    # Prepare all prompts
    for i in range(0, test_df.shape[0]):
        prompt_end = format_example(test_df, i, include_answer=False, subject=subject)
        prompt = prompt_end

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
        prompts.append(prompt)

    # Set up vLLM generation arguments
    gen_args = GenerationArgs(
        num_completions=1,
        temperature=0.9,
        response_length=2048,
        top_p=0.9,
        tensor_parallel_size=1
    )

    try:
        # Generate responses using vLLM
        outputs = generate_with_vllm(
            model_name_or_path=args.model_name_or_path,
            revision=args.hf_revision if args.hf_revision else None,
            prompt_token_ids=[tokenizer.encode(p) for p in prompts],
            gen_args=gen_args
        )

        # Process vLLM outputs
        full_responses = []
        for output in outputs:
            response_text = output["outputs"][0]["text"]  # Get first completion
            if response_text.startswith(output["prompt"]):
                response_text = response_text[len(output["prompt"]):]
            full_responses.append(response_text)

    except Exception as e:
        print(f"Error during vLLM generation: {e}")
        raise

    # Rest of the function remains the same
    parsed_answers = []
    for response in full_responses:
        # breakpoint()
        answer = parse_cot_response(response)
        parsed_answers.append(answer if answer else "A")  # Default to A if parsing fails

    pred_indices = [choices.index(ans) for ans in parsed_answers]
    all_probs = np.zeros((len(pred_indices), len(choices)))  # Placeholder for probabilities

    cors = []
    ground_truths = test_df.iloc[:, -1].values
    for i in range(len(pred_indices)):
        prediction = choices[pred_indices[i]]
        ground_truth = ground_truths[i]
        cors.append(prediction == ground_truth)

    acc = np.mean(cors)
    cors = np.array(cors)

    print("Average accuracy {:.3f} - {}".format(acc, subject))

    # Save the detailed responses including CoT reasoning
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
            "model_reasoning": full_responses[i],
            "parsed_answer": parsed_answers[i],
            "is_correct": bool(cors[i])
        }
        detailed_results.append(result)

    # Save the detailed results
    with open(os.path.join(args.save_dir, f"{subject}_cot_results.json"), "w") as f:
        json.dump(detailed_results, f, indent=2)

    return cors, acc, all_probs



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
        # Add these lines to properly set up the tokenizer
        # if tokenizer.pad_token is None:
        #     tokenizer.pad_token = tokenizer.eos_token
        #     tokenizer.pad_token_id = tokenizer.eos_token_id

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
    math_subjects = ['global_facts', 'high_school_physics', 'moral_scenarios', 'abstract_algebra', 'college_physics',
                     'college_chemistry', 'machine_learning', 'econometrics', 'professional_law', 'professional_accounting',
                     'college_mathematics', 'elementary_mathematics', 'high_school_chemistry', 'formal_logic',
                     'high_school_mathematics', 'high_school_statistics',
                     'college_medicine', 'college_computer_science', 'conceptual_physics']

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
    subject_results = {}  # Store results for each subject

    for subject in tqdm(subjects, desc=f"Evaluating subjects: "):
        # Load data directly from HF
        dev_df = load_mmlu_data(subject, 'dev')[:args.ntrain]
        test_df = load_mmlu_data(subject, 'test', args.n_instances)

        if args.model_name_or_path:
            cors, acc, probs = eval_hf_model(args, subject, model, tokenizer, dev_df, test_df, args.eval_batch_size)
        else:
            cors, acc, probs = eval_openai_chat_engine(args, subject, args.openai_engine, dev_df, test_df,
                                                       args.eval_batch_size)

        subject_results[subject] = (cors, acc, probs)

        subcats = subcategories[subject]
        for subcat in subcats:
            subcat_cors[subcat].append(cors)
            for key in categories.keys():
                if subcat in categories[key]:
                    cat_cors[key].append(cors)
        all_cors.append(cors)

        # Save detailed results for this subject
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
                "context": format_example(test_df, i, include_answer=False),
                "correct_answer": test_df.iloc[i, 5],
                "model_prediction": choices[np.argmax(probs[i])],
                "is_correct": bool(cors[i]),
                "probabilities": {
                    choice: float(probs[i][j])
                    for j, choice in enumerate(choices)
                }
            }
            detailed_results.append(result)

        # Save detailed results for this subject
        with open(os.path.join(args.save_dir, f"{subject}_results.json"), "w") as f:
            json.dump(detailed_results, f, indent=2)

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

    # Prepare the final metrics
    metrics = {
        "average_acc": float(weighted_acc),
        "results_by_subject": {
            subject: {
                "accuracy": float(acc),
                "num_examples": len(cors),
                "num_correct": int(np.sum(cors))
            }
            for subject, (cors, acc, _) in subject_results.items()
        },
        "subcat_acc": {
            subcat: float(np.mean(np.concatenate(subcat_cors[subcat])))
            for subcat in subcat_cors
            if subcat_cors[subcat]
        },
        "cat_acc": {
            cat: float(np.mean(np.concatenate(cat_cors[cat])))
            for cat in cat_cors
            if cat_cors[cat]
        }
    }

    # Save metrics json
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Create and save summary CSV
    model_name = args.model_name_or_path.split('/')[-1] if args.model_name_or_path else args.openai_engine

    csv_rows = []

    # Add overall accuracy
    csv_rows.append({
        'Category_Type': 'Overall',
        'Category_Name': 'Average',
        'Score': f"{weighted_acc:.3f}",
        'Model': model_name
    })

    # Add category accuracies
    for cat in categories:
        if cat in cat_cors and cat_cors[cat]:
            score = np.mean(np.concatenate(cat_cors[cat]))
            csv_rows.append({
                'Category_Type': 'Category',
                'Category_Name': cat,
                'Score': f"{score:.3f}",
                'Model': model_name
            })

    # Add subject accuracies
    for subject, (cors, acc, _) in subject_results.items():
        csv_rows.append({
            'Category_Type': 'Subject',
            'Category_Name': subject,
            'Score': f"{acc:.3f}",
            'Model': model_name
        })

    # Save to CSV
    summary_df = pd.DataFrame(csv_rows)
    summary_path = os.path.join(args.save_dir, "summary_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary results saved to: {summary_path}")

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
                        help="Only evaluate math-related subjects", default=False)
    parser.add_argument("--use_cot", action="store_true",
                        help="Use Chain-of-Thought prompting for reasoning tasks")
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