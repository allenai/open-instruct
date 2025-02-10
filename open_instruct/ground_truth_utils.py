"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
"""

import json
import re
import string

from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
from typing import List, Dict, Any, Optional


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


def verify_function_sample(model_output: str, ground_truth:str) -> bool:
    """
    Verify if model output matches ground truth function calls using multiple extraction methods.
    Similar to math verification, tries multiple approaches to extract function calls.
    
    Args:
        model_output: Raw model response string
        ground_truth: List of expected function call dictionaries
    
    Returns:
        bool: True if any extraction method matches ground truth
    """
    all_extracted_calls = []
    ground_truth = json.loads(ground_truth)
    # Method 1: Extract from code blocks with ```
    code_blocks = re.findall(r"```(?:json)?\n?(.*?)```", model_output, re.DOTALL)
    for block in code_blocks:
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, list):
                all_extracted_calls.append(parsed)
        except json.JSONDecodeError:
            continue
            
    # Method 2: Extract from XML-style function_calls tags
    function_calls_blocks = re.findall(
        r"<function_calls>\s*(.*?)\s*</function_calls>",
        model_output,
        re.DOTALL
    )
    for block in function_calls_blocks:
        # Remove any embedded code blocks if present
        block = re.sub(r"```(?:json)?\n?(.*?)```", r"\1", block, flags=re.DOTALL)
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, list):
                all_extracted_calls.append(parsed)
        except json.JSONDecodeError:
            continue
            
    # Method 3: Look for direct JSON array in the text
    try:
        # Find text that looks like a JSON array
        json_arrays = re.findall(r"\[\s*{.*?}\s*\]", model_output, re.DOTALL)
        for array in json_arrays:
            try:
                parsed = json.loads(array)
                if isinstance(parsed, list):
                    all_extracted_calls.append(parsed)
            except json.JSONDecodeError:
                continue
    except Exception:
        pass
        
    # Method 4: Try parsing the entire output as JSON (last resort)
    try:
        parsed = json.loads(model_output.strip())
        if isinstance(parsed, list):
            all_extracted_calls.append(parsed)
    except json.JSONDecodeError:
        pass
        
    # Compare each extracted call list with ground truth
    for calls in all_extracted_calls:
        if is_function_calls_equivalent(calls, ground_truth):
            return True
            
    return False

def is_function_calls_equivalent(
    actual_calls: List[Dict[str, Any]],
    expected_calls: List[Dict[str, Any]]
) -> bool:
    """
    Check if two lists of function calls are equivalent.
    
    Args:
        actual_calls: List of actual function calls
        expected_calls: List of expected function calls
    
    Returns:
        bool: True if calls are equivalent
    """
    if len(actual_calls) != len(expected_calls):
        return False
        
    for actual, expected in zip(actual_calls, expected_calls):
        # Check basic structure
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            return False
            
        # Check required fields
        if "name" not in actual or "arguments" not in actual:
            return False
            
        # Compare function names
        if actual["name"] != expected["name"]:
            return False
            
        # Compare arguments
        actual_args = actual["arguments"]
        expected_args = expected["arguments"]
        
        if not isinstance(actual_args, dict) or not isinstance(expected_args, dict):
            return False
            
        # Check if all expected arguments are present with correct values
        for key, value in expected_args.items():
            if key not in actual_args or actual_args[key] != value:
                return False
                
    return True

# Example usage
def test_verifier():
    ground_truth = """[
        {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
        {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}
    ]"""
    
    test_cases = [
        # Case 1: Perfect match with code blocks
        '''```json
        [
            {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
            {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}
        ]
        ```''',
        
        # Case 2: XML tags with nested code blocks
        '''<thinking>We need to check both beta and game giveaways.</thinking>
        <function_calls>
        ```
        [
            {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
            {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}
        ]
        ```
        </function_calls>''',
        
        # Case 3: Direct JSON in text
        '''Here are the function calls: [
            {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
            {"name": "live_giveaways_by_type", "arguments": {"type": "game"}}
        ]''',
        
        # Case 4: Invalid format
        '''No valid JSON here'''
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        result = verify_function_sample(test_case, ground_truth)
        print(f"\nTest case {i}:")
        print(f"Matched: {result}")
        print(f"Input: {test_case[:100]}...")




# debug code
if __name__ == "__main__":
    from datasets import load_dataset
    print(test_verifier())
    ds = load_dataset("ai2-adapt-dev/prompts_with_constraints_for_ground_truth")
    test_model_output = "<|assistant|>\nThe answer is $\\boxed{3.14}$"
    for sample in ds["train"]:
        print(sample)
        verify_ifeval_sample(test_model_output, sample["ground_truth"])
        


