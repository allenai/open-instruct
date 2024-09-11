import argparse
import os
import json
import torch
import pandas as pd

import warnings
from eval.utils import (
    load_hf_lm,
    load_hf_tokenizer,
    load_hf_lm_and_tokenizer,
    query_openai_chat_model,
    query_openai_model,
    generate_completions,
    score_completions,
    dynamic_import_function,
    upload_results_to_hf,
    check_and_upload_model_metadata,
)
from eval.truthfulqa.utilities import (
    format_prompt,
    split_multi_answer,
    format_best,
    set_columns,
)
from eval.truthfulqa.metrics import run_gpt_classifier_eval, run_hf_classifier_eval, MC_calcs
from eval.truthfulqa.configs import BEST_COL, ANSWER_COL, INCORRECT_COL


def trim_answer(answer):
    # remove spaces at the beginning and end
    answer = answer.strip()
    # remove the "A:" prefix if it exists
    if answer.startswith('A:'):
        answer = answer[2:].strip()
    # remove everything after "Q:" if it exists
    if 'Q:' in answer:
        answer = answer.split('Q:')[0].strip()
    # reformat line-breaks for long-form answers
    answer = answer.replace('\n\n', ' ')
    return answer


def run_chatgpt(questions, engine, tag, preset='qa', batch_size=1, cache_path=None, verbose=False):

    """Stores answers from ChatGPT / GPT4 models (requires an API key)"""

    if tag not in questions.columns:
        questions[tag] = ''

    questions[tag].fillna('', inplace=True)
    questions[tag] = questions[tag].astype(str)

    instances = [
        {"prompt": format_prompt(questions.loc[idx], preset, format='general'), "id": idx} for idx in questions.index
    ]

    responses = query_openai_chat_model(
        engine=engine,
        output_path=cache_path, 
        instances=instances, 
        batch_size=batch_size,
        temperature=0.0
    )
    assert len(responses) == len(instances)
    return questions


def run_gpt3(questions, engine, tag, preset='qa', batch_size=1, cache_path=None, verbose=False):
    """Stores answers from GPT-3 models (requires an API key)"""

    if tag not in questions.columns:
        questions[tag] = ''

    questions[tag].fillna('', inplace=True)
    questions[tag] = questions[tag].astype(str)

    instances = [
        {"prompt": format_prompt(questions.loc[idx], preset, format='general'), "id": idx} for idx in questions.index
    ]

    responses = query_openai_model(
        engine=engine, 
        instances=instances, 
        output_path=cache_path,
        batch_size=batch_size,
        temperature=0.0, 
        stop=None if preset == 'long' else '\n\n', 
        max_tokens=50
    )
    assert len(responses) == len(instances)

    for idx, response in zip(questions.index, responses):
        questions.loc[idx, tag] = trim_answer(response["output"])

    return questions


def run_gpt3_mc(questions, engine, tag, preset='qa', batch_size=1, cache_path=None, verbose=False):
    """Runs multiple-choice metrics for GPT-3 models (requires an API key)"""

    set_columns(tag, questions)
    examples = []
    for idx in questions.index:
        # check that candidate answer exists
        if pd.isnull(questions.loc[idx, INCORRECT_COL]):
            warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
            continue
        if not len(questions.loc[idx, INCORRECT_COL]):
            warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
            continue

        # reference answers
        ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
        ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])

        # prompt for all answers
        prompt = format_prompt(questions.loc[idx], preset, format='general') + "\nA:"

        # candidate completions
        examples.append({"prompt": prompt, "completions": ref_true + ref_false})

    instances, instance_id = [], 0
    for example in examples:
        for completion in example["completions"]:
            instances.append({"prompt": example["prompt"] + " " + completion, "id": instance_id})
            instance_id += 1
    responses = query_openai_model(
        engine=engine, 
        instances=instances,
        output_path=cache_path,
        batch_size=batch_size,
        temperature=0.0, 
        stop=["\n\n"], 
        max_tokens=0, 
        echo=True, 
        logprobs=1
    )
    assert len(responses) == len(instances)
    responses = {response["id"]: response for response in responses}
    
    all_scores, instance_id = {}, 0
    for example in examples:
        all_scores[example["prompt"]] = {}
        for completion in example["completions"]:
            response = responses[instance_id]
            logprobs = response["response_metadata"]['choices'][0]['logprobs']
            # iterate through response to find the indexes of the start / end tokens for the ref answer
            idx_start = 0
            while idx_start < len(logprobs['text_offset']) - 1:
                if (logprobs['text_offset'][idx_start] >= len(example["prompt"])):
                    break
                idx_start += 1
            idx_end = idx_start
            while idx_end < len(logprobs['text_offset']) - 1:
                if (logprobs['text_offset'][idx_end] >= len(example["prompt"] + " " + completion)):
                    break
                idx_end += 1
            logprob_vals = logprobs['token_logprobs'][idx_start: idx_end]
            text_vals = logprobs['tokens'][idx_start + 3:idx_end]
            if True:
                print("LOGPROBS AND ANSWER TOKENS")
                print(logprob_vals)
                print(text_vals)
            all_scores[example["prompt"]][completion] = sum(logprob_vals)
            instance_id += 1

    for idx, example in zip(questions.index, examples):
        ref_best = format_best(questions.loc[idx, BEST_COL])
        ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
        ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])
        completion_scores = all_scores[example["prompt"]]
        scores_true = [completion_scores[ref] for ref in ref_true]
        scores_false = [completion_scores[ref] for ref in ref_false]
        
        MC_calcs(tag, questions, idx, scores_true, scores_false, ref_true, ref_best) 
    return questions



def run_hf_model(questions, model, tokenizer, tag, preset="qa", batch_size=1, max_new_tokens=50, chat_formatting_function=None):
    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""

    if tag not in questions.columns:
        questions[tag] = ''
    questions[tag].fillna('', inplace=True)
    questions[tag] = questions[tag].astype(str)
    
    prompts = [
        format_prompt(questions.loc[idx], preset, format='general') for idx in questions.index
    ]
    if chat_formatting_function is not None:
        for idx, prompt in enumerate(prompts):
            messages = [{"role": "user", "content": prompt}]
            prompts[idx] = chat_formatting_function(messages, tokenizer, add_bos=False)
            prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"
    
    # get the last token because the tokenizer may add space tokens at the start.
    stop_sequence = tokenizer.encode("\n\n", add_special_tokens=False)[-2:] 
    completions = generate_completions(
        model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens,
        stop_id_sequences=[stop_sequence] if chat_formatting_function is None else None, 
        do_sample=False,
    )
    assert len(completions) == len(prompts)

    # if it's not a chat format, we will do some post-processing for the answer to make sure it's valid
    # otherwise, we will just store the completions as is
    for idx, completion in zip(questions.index, completions):
        questions.loc[idx, tag] = trim_answer(completion) if not chat_formatting_function else completion
    return questions


def run_hf_model_mc(questions, model, tokenizer, tag, batch_size=1, preset='qa', chat_formatting_function=None):
    """Runs multiple-choice metrics for autoregressive HuggingFace models (GPT-2, GPT-Neo)"""

    set_columns(tag, questions)
    
    examples = []
    for idx in questions.index:
        # check that candidate answer exists
        if pd.isnull(questions.loc[idx, INCORRECT_COL]):
            warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
            continue
        if not len(questions.loc[idx, INCORRECT_COL]):
            warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
            continue

        # reference answers
        ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
        ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])

        # prompt for all answers
        prompt = format_prompt(questions.loc[idx], preset, format='general')
        if chat_formatting_function is not None:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, tokenizer, add_bos=False)
            prompt += "A:" if prompt[-1] in ["\n", " "] else " A:"
        else:
            prompt += "\nA:"

        # candidate completions
        examples.append({"prompt": prompt, "completions": ref_true + ref_false})

    all_scores = score_completions(model, tokenizer, examples, batch_size=batch_size, aggregation="sum")
    assert len(all_scores) == len(examples)

    for idx, example in zip(questions.index, examples):
        ref_best = format_best(questions.loc[idx, BEST_COL])
        ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
        ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])

        completion_scores = all_scores[example["prompt"]]
        scores_true = [completion_scores[ref] for ref in ref_true]
        scores_false = [completion_scores[ref] for ref in ref_false]
        
        MC_calcs(tag, questions, idx, scores_true, scores_false, ref_true, ref_best)
    return questions


def format_frame(results):
    results = results[[x for x in results.columns if (x != 'Context') and (results[x].dtype != 'O')]]
    new_cols = []
    for col in results.columns:
        split = col.split(' ')
        new_cols.append((split[0], ' '.join(split[1:])))
    results.columns = pd.MultiIndex.from_tuples(new_cols)
    return results


def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    questions = pd.read_csv(os.path.join(args.data_dir, "TruthfulQA.csv"))

    if args.num_instances is not None:
        questions = questions.sample(args.num_instances, random_state=42)

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
            print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))
        if "truth" in args.metrics or "info" in args.metrics:
            print("Running generations!")
            run_hf_model(
                questions,
                model, 
                tokenizer, 
                tag=args.model_name_or_path, 
                preset=args.preset, 
                batch_size=args.eval_batch_size, 
                chat_formatting_function=dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
            )
        if "mc" in args.metrics:
            print("Running multiple-choice classification!")
            run_hf_model_mc(
                questions, 
                model, 
                tokenizer, 
                tag=args.model_name_or_path, 
                batch_size=args.eval_batch_size, 
                preset=args.preset,
                chat_formatting_function=dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
            )
    elif args.openai_engine:
        # gpt-3 language models
        cache_path = os.path.join(args.save_dir, "openai_query_cache.jsonl")
        if args.openai_engine in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002']:
            if "truth" in args.metrics or "info" in args.metrics:
                print("Running generations")
                run_gpt3(questions, args.openai_engine, args.openai_engine, cache_path=cache_path, batch_size=args.eval_batch_size, preset=args.preset)
            if 'mc' in args.metrics:
                print("Running multiple-choice classification!")
                run_gpt3_mc(questions, args.openai_engine, args.openai_engine, cache_path=cache_path, batch_size=args.eval_batch_size, preset=args.preset)
        # other openai engines
        else:
            if "truth" in args.metrics or "info" in args.metrics:
                print("Running generations")
                run_chatgpt(questions, args.openai_engine, args.openai_engine, cache_path=cache_path, batch_size=args.eval_batch_size, preset=args.preset)
            if "mc" in args.metrics:
                raise ValueError("OpenAI Chat engines does not support MC metrics.")

    del model
    torch.cuda.empty_cache()

    # run metrics
    print("Running metrics!")

    model_key = args.model_name_or_path if args.model_name_or_path else args.openai_engine
    for metric in args.metrics:
        if metric == 'mc':
            continue
        elif metric in ['truth', 'info']:
            if model_key not in questions.columns:
                raise ValueError("Answers missing for {0}!".format(model_key))
            try:
                if metric == 'truth':
                    if args.gpt_truth_model_name:
                        questions = run_gpt_classifier_eval(model_key, 'truth', args.gpt_truth_model_name, questions, info=False)
                    elif args.hf_truth_model_name_or_path:
                        truth_classifier, truth_tokenizer = load_hf_lm_and_tokenizer(
                            model_name_or_path=args.hf_truth_model_name_or_path,
                            tokenizer_name_or_path=args.hf_truth_model_name_or_path,
                            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                        )
                        questions = run_hf_classifier_eval(model_key, 'truth', truth_classifier, truth_tokenizer, questions, info=False)
                else:
                    if args.gpt_info_model_name:
                        questions = run_gpt_classifier_eval(model_key, 'info', args.gpt_info_model_name, questions, info=True)
                    elif args.hf_info_model_name_or_path:
                        info_classifier, info_tokenizer = load_hf_lm_and_tokenizer(
                            model_name_or_path=args.hf_info_model_name_or_path, 
                            tokenizer_name_or_path=args.hf_info_model_name_or_path,
                            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
                        )
                        questions = run_hf_classifier_eval(model_key, 'info', info_classifier, info_tokenizer, questions, info=True)
            except Exception as err:
                print(err)
        else:
            warnings.warn("Metric {0} not known, skipping!".format(metric), stacklevel=2)

    if "truth" in args.metrics and "info" in args.metrics:
        questions["{} truth-info acc".format(model_key)] = questions["{} truth acc".format(model_key)] * questions["{} info acc".format(model_key)]

    # save all
    questions.to_csv(os.path.join(args.save_dir, "predictions.csv"), index=False)

    # format and print basic results
    results = format_frame(questions)
    results = results.mean(axis=0)
    results = results.reset_index().rename(columns={'level_0': 'Model',
                                                    'level_1': 'Metric',
                                                    0: 'Value'})

    # filter to most informative metrics
    results = results[results['Metric'].isin(['MC1', 'MC2',
                                              'truth acc',
                                              'info acc',
                                              'truth-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')
    results.to_csv(os.path.join(args.save_dir, 'summary.csv'))

    print(results)
    
    results = results.loc[model_key].to_dict()
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)

    if args.upload_to_hf is not None:
        # upload metrics to HF. Main metric is the accuracy
        task_name = "oi_truthfulqa"
        primary_score = results["truth-info acc"]
        upload_results_to_hf(
            results,
            args.upload_to_hf,
            args.hf_upload_name,
            task_name=task_name,
            primary_score=primary_score,
            prepend_timestamp=True,
        )
        check_and_upload_model_metadata(
            args.model_name_or_path, args.upload_to_hf, args.hf_upload_name, hf_revision=args.hf_revision
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", 
        type=str, 
        help="The HuggingFace model to be evaluated."
    )
    parser.add_argument(
        "--hf_revision",
        type=str,
        default=None,
        help="if specified, we will load the model from a revision of the model in the hub"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", 
        type=str, 
        default=None, 
        help="If specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--openai_engine", 
        type=str, 
        default=None, 
        help="If specified, we will evaluate the OpenAI engine."
    )
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="data/eval/truthfulqa", 
        help="The directory containing the truthfulqa data. Download from https://github.com/sylinrl/TruthfulQA/tree/main/data."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="results/truthfulqa/", 
        help="The directory to save the results."
    )
    parser.add_argument(
        "--num_instances", 
        type=int, 
        default=None, 
        help="The number of instances to evaluate. If not given, we will evaluate all instances."
    )
    parser.add_argument(
        "--load_in_8bit", 
        action="store_true", 
        help="Load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--gptq", 
        action="store_true", 
        help="If given, we're evaluating a 4-bit quantized GPTQ model."
    )
    parser.add_argument(
        "--eval_batch_size", 
        type=int, 
        default=1, 
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--use_chat_format", 
        action="store_true", 
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function", 
        type=str, 
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        '--metrics', 
        nargs='+', 
        default=['truth', 'info', 'mc'], 
        choices=['truth', 'info', 'mc'], 
        help='Metrics to run'
    )
    parser.add_argument(
        '--preset', 
        type=str, 
        default='qa', 
        help='Preset to use for prompt generation. Please see presets.py for options.'
    )
    parser.add_argument(
        '--gpt_truth_model_name', 
        type=str, 
        help='A trained GPT judge model name to be used for computing the metrics for `truth` if it is specified.' \
             'Either `gpt_truth_model_name` or `hf_truth_model_name_or_path` should be specified for computing the metric.'
    )
    parser.add_argument(
        '--gpt_info_model_name', 
        type=str, 
        help='A trained GPT judge model name to be used for computing the metrics for `info` if it is specified.' \
            'Either `gpt_info_model_name` or `hf_info_model_name_or_path` should be specified for computing the metric.'
    )
    parser.add_argument(
        '--hf_truth_model_name_or_path',
        type=str,
        help='A trained HuggingFace judge model name to be used for computing the metrics for `truth` if it is specified.' \
             'Either `gpt_truth_model_name` or `hf_truth_model_name_or_path` should be specified for computing the metric.'
    )
    parser.add_argument(
        '--hf_info_model_name_or_path',
        type=str,
        help='A trained HuggingFace judge model name to be used for computing the metrics for `info` if it is specified.' \
            'Either `gpt_info_model_name` or `hf_info_model_name_or_path` should be specified for computing the metric.'
    )
    parser.add_argument(
        "--upload_to_hf",
        type=str,
        default=None,
        help="If specified, we will upload the results to Hugging Face Datasets. "
             "This should be the name of the dataset to upload to."
    )
    parser.add_argument(
        "--hf_upload_name",
        type=str,
        default=None,
        help="If uploading to hf, this is the model name"
    )
    args = parser.parse_args()
    main(args)