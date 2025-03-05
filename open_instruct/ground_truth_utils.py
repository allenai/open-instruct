"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
"""

import json
import re
import string

import numpy as np

from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)


def verify_gsm8k_sample(model_output, ground_truth_answer):
    # gsm is easy: extract numbers, and then just compare last number with answer.
    # matches how we do eval.
    predictions = None
    # replace numbers like `x,xxx` with `xxxx`
    response = re.sub(r"(\d),(\d)", r"\1\2", model_output)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    if numbers:
        predictions = numbers[-1]
    else:
        predictions = response
    return str(predictions).lower() == str(ground_truth_answer).lower()


def verify_math_sample(model_output, ground_truth_answer):
    raw_answer = model_output
    # for math, more complex. We will try a few different ways to extract the answer.
    # this roughly follows 'flex em' in oe-eval-internal
    all_answers = []
    # First, try find answer in \boxed{}.
    boxed_answer = last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)
    # Second, try to extract via minerva format.
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)
    # If nothing still, try to find the last latex-formatted answer
    if len(all_answers) == 0:
        dollars = [m.start() for m in re.finditer("\\$", raw_answer)]
        if len(dollars) > 1:
            # Add the answer between the second to last and last dollar sign
            answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)
    # otherwise, just take the full output. Probably wont work, bit of a yolo.
    if len(all_answers) == 0:
        all_answers.append(normalize_final_answer(model_output))
    # now, compare all answers to ground truth.
    matched = False
    for answer in all_answers:
        if is_equiv(answer, ground_truth_answer):
            matched = True
            break
        elif hendrycks_is_equiv(answer, ground_truth_answer):
            matched = True
            break
    # if we got any match, we are good.
    return matched


def verify_strict_math_sample(model_output, ground_truth_answer):
    raw_answer = model_output
    # just trying minerva format.
    all_answers = []
    # Second, try to extract via minerva format.
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)
    # otherwise, just take the full output. Probably wont work, bit of a yolo.
    if len(all_answers) == 0:
        all_answers.append(normalize_final_answer(model_output))
    # now, compare all answers to ground truth.
    matched = False
    for answer in all_answers:
        if is_equiv(answer, ground_truth_answer):
            matched = True
            break
        elif hendrycks_is_equiv(answer, ground_truth_answer):
            matched = True
            break
    # if we got any match, we are good.
    return matched


def verify_ifeval_sample(model_output, constraint):
    # TODO: just pass in final answer. this should be fine for other evals too.
    answer = model_output.split("<|assistant|>\n")[-1].strip()
    if isinstance(constraint, str):
        constraint = json.loads(constraint)
    if "func_name" not in constraint:
        print("WARNING: constraint missing func_name")
        print(constraint)
        return False
    # first, parse out the constraint string.
    func_name = constraint.pop("func_name")
    # get the function
    func = IF_FUNCTIONS_MAP[func_name]
    # now, run the function
    # pop out any none args
    non_none_args = {k: v for k, v in constraint.items() if v is not None}
    # sometimes we have extra args, sometimes not.
    if len(constraint) == 0:
        return func(model_output)
    return func(answer, **non_none_args)


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    From https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def verify_flan_sample(model_output, ground_truth_answer):
    # Flan! we will just use... exact match with some basic cleaning, after extracting the answer.
    answer_string = model_output.split("The answer is: ")[-1].strip()
    return normalize_answer(answer_string) == normalize_answer(ground_truth_answer)


def soft_format_reward_func(responses: list[str], reward_scale: float = 1.0) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r".*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [reward_scale if match else 0.0 for match in matches]


def verify_max_length_sample(tokenized_response: list[int], max_length: int) -> list[float]:
    """Reward function that checks if the completion has a specific length."""
    return 1.0 if len(tokenized_response) <= int(max_length) else 0.0


def verify_max_length_sample_cosine(tokenized_response: list[int], max_length: int, tolerance_ratio: float =0.1):
    """
    Compute a cosine-shaped reward for a tokenized response length x.
    
    The reward peaks at 1.0 when x equals max_length. For differences up to
    tolerance (tolerance_ratio * max_length), the reward decays as:
        reward = 0.5 * (cos(pi * diff / tolerance) + 1)
    For differences greater than tolerance, the reward is 0.
    
    Args:
        x (int or np.array): Tokenized response length(s).
        max_length (int): Target token length.
        tolerance_ratio (float): Fraction of max_length defining the non-zero reward band.
        
    Returns:
        np.array: Cosine-shaped reward, between 0 and 1.
    """
    max_length = int(max_length)
    x = len(tokenized_response)
    tolerance = max_length * tolerance_ratio
    diff = np.abs(x - max_length)
    reward = np.where(diff <= tolerance,
                      0.5 * (np.cos(np.pi * diff / tolerance) + 1),
                      0.0)
    return reward


def reward_up_to_max(tokenized_response: list[int], max_length: int) -> float:
    """
    Reward function that linearly rewards tokenized responses for going up to max_length.
    
    - If the tokenized response length is less than or equal to max_length,
      the reward is the fraction of max_length achieved (i.e. len(tokenized_response) / max_length).
    - If the response exceeds max_length, the reward is 0.
    
    Args:
        tokenized_response (list[int]): The tokenized output.
        max_length (int): The target token length.
        
    Returns:
        float: The reward value.
    """
    max_length = int(max_length)
    actual_length = len(tokenized_response)
    return actual_length / max_length if actual_length <= max_length else 0.0



# debug code
if __name__ == "__main__":
    from datasets import load_dataset

    ds = load_dataset("ai2-adapt-dev/eurus2_ground_truth_with_random_max_length")
    test_model_output = "<|assistant|>\nThe answer is $\\boxed{3.14}$"
    for sample in ds["train"]:
        print(sample)
        print(verify_max_length_sample(test_model_output, sample["ground_truth"].split("<sep>")[-1]))
