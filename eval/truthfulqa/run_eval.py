import argparse
import os
import json
import tqdm
import torch
import pandas as pd

import openai
import warnings
from eval.utils import load_hf_lm_and_tokenizer, query_openai_chat_model, query_openai_model, generate_completions, score_completions
from eval.truthfulqa.utilities import (
    format_prompt,
    format_prompt_with_answer_strings,
    split_multi_answer,
    format_best,
    set_columns,
)
from eval.truthfulqa.metrics import run_end2end_GPT3, MC_calcs
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


def run_chatgpt(questions, engine, tag, preset='qa', verbose=False):

    """Stores answers from ChatGPT / GPT4 models (requires an API key)"""

    if tag not in questions.columns:
        questions[tag] = ''

    questions[tag].fillna('', inplace=True)
    questions[tag] = questions[tag].astype(str)

    instances = [
        {"prompt": format_prompt(questions.loc[idx], preset, format='general'), "id": idx} for idx in questions.index
    ]

    responses = query_openai_chat_model(engine=engine, instances=instances, temperature=0.0)
    assert len(responses) == len(instances)

    for idx, response in zip(questions.index, responses):
        questions.loc[idx, tag] = trim_answer(response["output"])

    # for idx in tqdm.tqdm(questions.index):
    #     if pd.isnull(questions.loc[idx, tag]) or not len(questions.loc[idx, tag]):
    #         input_prompt = format_prompt(questions.loc[idx], preset, format='general')

    #         if input_prompt is not None:
    #             response = query_openai_chat_model(engine=engine, instances=[{"prompt": input_prompt, "id" : 1}], temperature=0.0)
    #             output_str = response[0]["output"]

    #             idx_start = find_start(list(output_str))
    #             output_str = output_str[idx_start:].strip()
    #             output_str = output_str.replace('\n\n', ' ')  # reformat line-breaks for long-form answers

    #             questions.loc[idx, tag] = output_str

    #             if verbose:
    #                 print("MODEL ANSWER: {0}".format(output_str))
    #         else:
    #             questions.loc[idx, tag] = ''
    return questions


def run_GPT3(questions, engine, tag, preset='qa', verbose=False):

    """Stores answers from GPT-3 models (requires an API key)"""

    if tag not in questions.columns:
        questions[tag] = ''

    questions[tag].fillna('', inplace=True)
    questions[tag] = questions[tag].astype(str)

    instances = [
        {"prompt": format_prompt(questions.loc[idx], preset, format='general'), "id": idx} for idx in questions.index
    ]

    responses = query_openai_model(engine=engine, instances=instances, temperature=0.0)
    assert len(responses) == len(instances)

    for idx, response in zip(questions.index, responses):
        questions.loc[idx, tag] = trim_answer(response["output"])

    # for idx in tqdm.tqdm(questions.index):
    #     if pd.isnull(questions.loc[idx, tag]) or not len(questions.loc[idx, tag]):
    #         input_prompt = format_prompt(questions.loc[idx], preset, format='general')

    #         if input_prompt is not None:
    #             if preset == 'long':
    #                 response = openai.Completion.create(engine=engine, prompt=input_prompt, temperature=0, max_tokens=50,
    #                                                     stop=None, logprobs=1)  # long-form answers often include line-breaks
    #             else:
    #                 response = openai.Completion.create(engine=engine, prompt=input_prompt, temperature=0, max_tokens=50,
    #                                                     stop='\n\n', logprobs=1)
    #             logprobs = response['choices'][0]['logprobs']
    #             output_str = response['choices'][0]['text']

    #             idx_start = find_start(logprobs['tokens'])

    #             drop_str = ''.join(logprobs['tokens'][:idx_start])
    #             output_str = output_str[len(drop_str):].strip()
    #             output_str = output_str.replace('\n\n', ' ')  # reformat line-breaks for long-form answers

    #             questions.loc[idx, tag] = output_str

    #             if verbose:
    #                 print("MODEL ANSWER: {0}".format(output_str))
    #         else:
    #             questions.loc[idx, tag] = ''
    return questions


def run_probs_GPT3(questions, engine, tag, preset='qa', verbose=False):

    """Runs multiple-choice metrics for GPT-3 models (requires an API key)"""

    set_columns(tag, questions)

    for idx in tqdm.tqdm(questions.index):
        if pd.isnull(questions.loc[idx, '{0} lprob diff'.format(tag)]):

            # check that answer exists
            if pd.isnull(questions.loc[idx, INCORRECT_COL]):
                warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
                continue
            if not len(questions.loc[idx, INCORRECT_COL]):
                warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
                continue

            # reference answers
            ref_best = format_best(questions.loc[idx, BEST_COL])
            ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
            ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])

            scores_true = []
            scores_false = []

            for temp_ans in ref_true:
                # input_prompt appends the current answer choice to the prompt
                query_prompt = format_prompt(questions.loc[idx], preset, format='general')
                input_prompt = format_prompt_with_answer_strings(questions.loc[idx, 'Question'], temp_ans, preset, format='general')

                if input_prompt is not None:
                    response = openai.Completion.create(engine=engine, prompt=input_prompt, temperature=0, max_tokens=50, stop='\n\n', echo=True, logprobs=1)
                    logprobs = response['choices'][0]['logprobs']
                    output_str = response['choices'][0]['text']

                    # iterate through response to find the indexes of the start / end tokens for the ref answer
                    idx_start = 0
                    while idx_start < len(logprobs['text_offset']) - 1:
                        if (logprobs['text_offset'][idx_start] >= len(query_prompt)):
                            break
                        idx_start += 1

                    idx_end = idx_start
                    while idx_end < len(logprobs['text_offset']) - 1:
                        if (logprobs['text_offset'][idx_end] >= len(input_prompt)):
                            break
                        idx_end += 1

                    # increment indexes by +3 to skip the "\nA:" tokens before the answer
                    logprob_vals = logprobs['token_logprobs'][idx_start + 3:idx_end]
                    text_vals = logprobs['tokens'][idx_start + 3:idx_end]

                    if verbose:
                        print("LOGPROBS AND ANSWER TOKENS")
                        print(logprob_vals)
                        print(text_vals)

                    scores_true.append(sum(logprob_vals))

            for temp_ans in ref_false:
                query_prompt = format_prompt(questions.loc[idx], preset, format='general')
                input_prompt = format_prompt_with_answer_strings(questions.loc[idx, 'Question'],
                                                                 temp_ans,
                                                                 preset,
                                                                 format='general')

                if input_prompt is not None:
                    response = openai.Completion.create(engine=engine, prompt=input_prompt, temperature=0, max_tokens=50, stop='\n\n', echo=True, logprobs=1)
                    logprobs = response['choices'][0]['logprobs']
                    output_str = response['choices'][0]['text']

                    # iterate through response to find the indexes of the start / end tokens for the ref answer
                    idx_start = 0
                    while idx_start < len(logprobs['text_offset']) - 1:
                        if (logprobs['text_offset'][idx_start] >= len(query_prompt)):
                            break
                        idx_start += 1

                    idx_end = idx_start
                    while idx_end < len(logprobs['text_offset']) - 1:
                        if (logprobs['text_offset'][idx_end] >= len(input_prompt)):
                            break
                        idx_end += 1

                    # increment indexes by +3 to skip the "\nA:" tokens before the answer
                    logprob_vals = logprobs['token_logprobs'][idx_start + 3:idx_end]
                    text_vals = logprobs['tokens'][idx_start + 3:idx_end]

                    if verbose:
                        print("LOGPROBS AND ANSWER TOKENS")
                        print(logprob_vals)
                        print(text_vals)

                    scores_false.append(sum(logprob_vals))

            MC_calcs(tag, questions, idx, scores_true, scores_false, ref_true, ref_best)
    return questions



def run_answers(questions, model, tokenizer, tag, preset="qa", batch_size=1, max_new_tokens=50):

    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""

    
    if tag not in questions.columns:
        questions[tag] = ''

    questions[tag].fillna('', inplace=True)
    questions[tag] = questions[tag].astype(str)

    prompts = [
        format_prompt(questions.loc[idx], preset, format='general') for idx in questions.index
    ]

    completions = generate_completions(model, tokenizer, prompts, batch_size=batch_size, max_new_tokens=max_new_tokens, top_k=1)
    assert len(completions) == len(prompts)

    for idx, prompt, completion in zip(questions.index, prompts, completions):
        questions.loc[idx, tag] = trim_answer(completion)
        # print("--------------------")
        # print("Index: {0}".format(idx))
        # print("Prompt: {0}".format(prompt))
        # print("Model Answer: {0}".format(questions.loc[idx, tag]))

    # # get tokens for ending sequence
    # seq_start = np.array(tokenizer('A:')['input_ids'])
    # seq_end = np.array(tokenizer('Q:')['input_ids'])

    # with torch.no_grad():
    #     for idx in tqdm.tqdm(questions.index):
    #         if pd.isnull(questions.loc[idx, tag]) or not len(questions.loc[idx, tag]):
    #             prompt = format_prompt(questions.loc[idx], preset, format='general')

    #             input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    #             if model.device.type == "cuda":
    #                 input_ids = input_ids.cuda()

    #             max_len = input_ids.shape[-1] + 50

    #             outputs = model.generate(
    #                 input_ids, 
    #                 return_dict_in_generate=True, 
    #                 top_k=1, 
    #                 max_length=max_len, 
    #                 num_return_sequences=1,
    #                 output_scores=True
    #             )
    #             sequences, scores = outputs.sequences, outputs.scores

    #             # skip the tokens in the input prompt
    #             gen_sequences = sequences[:, input_ids.shape[-1]:][0, :]
    #             gen_arr = gen_sequences.cpu().numpy()

    #             # find indexes of answer, skipping A: (start of answer) and Q: (end of answer)
    #             idx_start = find_subsequence(gen_arr, seq_start, start=True)
    #             idx_end = find_subsequence(gen_arr, seq_end, start=False)

    #             gen_sequences = gen_sequences[idx_start:idx_end]
    #             output_str = tokenizer.decode(gen_sequences, skip_special_tokens=True)
    #             output_str = output_str.strip()

    #             if verbose:
    #                 print("--------------------")
    #                 print("PROMPT: {0}".format(prompt))
    #                 print('MODEL OUTPUT: {0}'.format(output_str))

    #             scores = torch.stack(scores, dim=1)
    #             scores = scores.squeeze(0)  # skip if running on a batch!
    #             scores = scores[idx_start:idx_end, :]

    #             # convert logits to log probs
    #             log_probs = scores.log_softmax(-1)
    #             log_probs = log_probs[range(scores.shape[0]), gen_sequences]

    #             probs = scores.softmax(-1)
    #             probs = probs[range(scores.shape[0]), gen_sequences]

    #             questions.loc[idx, tag] = output_str
    return questions


def run_probs(questions, model, tokenizer, tag, preset='qa'):

    """Runs multiple-choice metrics for autoregressive HuggingFace models (GPT-2, GPT-Neo)"""

    set_columns(tag, questions)

    with torch.no_grad():
        for idx in tqdm.tqdm(questions.index):
            if pd.isnull(questions.loc[idx, '{0} lprob max'.format(tag)]):

                # check that answer exists
                if pd.isnull(questions.loc[idx, INCORRECT_COL]):
                    warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
                    continue
                if not len(questions.loc[idx, INCORRECT_COL]):
                    warnings.warn("References missing for {0}!".format(idx), stacklevel=2)
                    continue

                # reference answers
                ref_best = format_best(questions.loc[idx, BEST_COL])
                ref_true = split_multi_answer(questions.loc[idx, ANSWER_COL])
                ref_false = split_multi_answer(questions.loc[idx, INCORRECT_COL])

                scores_true = []
                scores_false = []

                input_prompt = format_prompt(questions.loc[idx], preset, format='general')

                for temp_ans in ref_true:
                    # append the current answer choice to the prompt
                    prompt = format_prompt_with_answer_strings(questions.loc[idx, 'Question'], temp_ans, preset, format='general')
                    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
                    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    if model.device.type == "cuda":
                        input_ids = input_ids.cuda()
                        prompt_ids = prompt_ids.cuda()

                    outputs = model(prompt_ids)[0].squeeze(0)
                    outputs = outputs.log_softmax(-1)  # logits to log probs

                    # skip tokens in the prompt -- we only care about the answer
                    outputs = outputs[input_ids.shape[-1] - 1: -1, :]
                    prompt_ids = prompt_ids[0, input_ids.shape[-1]:]

                    # get logprobs for each token in the answer
                    log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                    log_probs = log_probs[3:]  # drop the '\nA:' prefix

                    scores_true.append(log_probs.sum().item())

                for temp_ans in ref_false:
                    # append the current answer choice to the prompt
                    prompt = format_prompt_with_answer_strings(questions.loc[idx, 'Question'], temp_ans, preset, format='general')
                    input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
                    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids
                    if model.device.type == "cuda":
                        input_ids = input_ids.cuda()
                        prompt_ids = prompt_ids.cuda()

                    outputs = model(prompt_ids)[0].squeeze(0)
                    outputs = outputs.log_softmax(-1)  # logits to log probs

                    # skip tokens in the prompt -- we only care about the answer
                    outputs = outputs[input_ids.shape[-1] - 1: -1, :]
                    prompt_ids = prompt_ids[0, input_ids.shape[-1]:]

                    # get logprobs for each token in the answer
                    log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                    log_probs = log_probs[3:] # drop the '\nA:' prefix

                    scores_false.append(log_probs.sum().item())

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
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path, 
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit, 
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            gptq_model=args.gptq
        )
        print("Running generations")
        run_answers(questions, model, tokenizer, tag=args.model_name_or_path, preset=args.preset, batch_size=args.eval_batch_size)
        if 'mc' in args.metrics:
            print("Running multiple-choice classification!")
            run_probs(questions, model, tokenizer, tag=args.model_name_or_path, preset=args.preset)
    elif args.openai_engine:
        # gpt-3 language models
        if args.openai_engine in ['ada', 'babbage', 'curie', 'davinci', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002']:
            print("Running generations")
            run_GPT3(questions, args.openai_engine, args.openai_engine, args.preset)
            if 'mc' in args.metrics:
                print("Running multiple-choice classification!")
                run_probs_GPT3(questions, args.openai_engine, args.openai_engine, preset=args.preset)
        # other openai engines
        else:  
            if "mc" in args.metrics:
                raise ValueError("OpenAI Chat engines does not support MC metrics.")
            print("Running generations")
            run_chatgpt(questions, args.openai_engine, args.openai_engine, args.preset)

    # run metrics
    print("Running metrics!")

    model_key = args.model_name_or_path if args.model_name_or_path else args.openai_engine
    
    if model_key not in questions.columns:
        raise ValueError("Answers missing for {0}!".format(model_key))

    for metric in args.metrics:
        if metric == 'mc':
            continue
        # if metric == 'bleurt':
        #     try:
        #         questions = metrics.run_BLEURT(model_key, questions, cache_dir=args.cache_dir)
        #     except Exception as err:
        #         print(err)
        # elif metric in ['bleu', 'rouge']:
        #     try:
        #         questions = metrics.run_bleu_and_rouge(model_key, questions)
        #     except Exception as err:
        #         print(err)
        elif metric in ['judge', 'info']:
            try:
                if metric == 'judge':
                    questions = run_end2end_GPT3(model_key, 'GPT-judge', args.gpt_judge_model_name, questions, info=False)
                else:
                    questions = run_end2end_GPT3(model_key, 'GPT-info', args.gpt_info_model_name, questions, info=True)
            except Exception as err:
                print(err)
        else:
            warnings.warn("Metric {0} not known, skipping!".format(metric), stacklevel=2)

    if "judge" in args.metrics and "info" in args.metrics:
        questions["{} GPT-judge-info acc".format(model_key)] = questions["{} GPT-judge acc".format(model_key)] * questions["{} GPT-info acc".format(model_key)]

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
                                              'GPT-judge acc',
                                              'GPT-info acc',
                                              'GPT-judge-info acc'])]
    results = pd.pivot_table(results, 'Value', 'Model', 'Metric')
    results.to_csv(os.path.join(args.save_dir, 'summary.csv'))

    print(results)
    
    results = results.loc[model_key].to_dict()
    with open(os.path.join(args.save_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, help="The HuggingFace model to be evaluated.")
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None, help="If specified, we will load the tokenizer from here.")
    parser.add_argument("--openai_engine", type=str, default=None, help="If specified, we will evaluate the OpenAI engine.")
    parser.add_argument("--data_dir", type=str, default="data/eval/truthfulqa", help="The directory containing the truthfulqa data. Download from https://github.com/sylinrl/TruthfulQA/tree/main/data.")
    parser.add_argument("--save_dir", type=str, default="results/truthfulqa/", help="The directory to save the results.")
    parser.add_argument("--num_instances", type=int, default=None, help="The number of instances to evaluate. If not given, we will evaluate all instances.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Load model in 8bit mode, which will reduce memory and speed up inference.")
    parser.add_argument("--gptq", action="store_true", help="If given, we're evaluating a 4-bit quantized GPTQ model.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--use_chat_format", action="store_true", help="If given, the prompt will be encoded as a chat format with the roles in prompt.")
    parser.add_argument('--metrics', nargs='+', default=['judge', 'info', 'mc'], choices=['judge', 'info', 'mc'], help='Metrics to run')
    parser.add_argument('--preset', type=str, default='qa', help='Preset to use for prompt generation. Please see presets.py for options.')
    parser.add_argument('--gpt_judge_model_name', type=str, help='If `judge` metric is used, the trained GPT judge model name should be provided.')
    parser.add_argument('--gpt_info_model_name', type=str, help='If `info` metric is used, the trained GPT info model name should be provided.')
    args = parser.parse_args()
    main(args)