"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
Add new verifiers by subclassing VerifierFunction and implementing the __call__ method.
They are then automatically added to the REWARD_FN_MAPPING.
"""

import asyncio
import json
import logging
import re
import string
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import aiohttp

from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)

logger = logging.getLogger(__name__)


class VerifierFunction(ABC):
    """
    Abstract base class for verifier functions.

    Each verifier is initialized with a name and a weight (default 1.0).
    The __call__ method must be implemented by subclasses.
    """

    def __init__(self, name: str, weight: float = 1.0) -> None:
        self.name = name
        self.weight = weight

    @abstractmethod
    def __call__(self, tokenized_prediction: List[int], prediction: str, label: Any) -> float:
        """
        Evaluate the given prediction against the ground truth (or constraint).

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.

        Returns:
            int: Reward score. Can be binary (0/1) or continuous.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight})"


class GSM8KVerifier(VerifierFunction):
    """
    Verifier for GSM8K tasks that extracts the last number from the prediction
    and compares it (case-insensitively) to the ground truth.
    """

    def __init__(self) -> None:
        super().__init__("gsm8k", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> float:
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        extracted = numbers[-1] if numbers else response
        return float(str(extracted).lower() == str(label).lower())


class MathVerifier(VerifierFunction):
    """
    Verifier for math problems.

    Attempts several extraction methods (boxed answers, Minerva format,
    last LaTeX answer) and compares the extracted answers to the ground truth.
    """

    def __init__(self) -> None:
        super().__init__("math", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> bool:
        raw_answer = prediction
        all_answers = []

        # Attempt extraction from \boxed{}.
        boxed_answer = last_boxed_only_string(raw_answer)
        if boxed_answer is not None:
            try:
                boxed_answer = remove_boxed(boxed_answer)
            except AssertionError:
                boxed_answer = None
        if boxed_answer is not None:
            all_answers.append(boxed_answer)

        # Attempt extraction via Minerva format.
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)

        # Attempt extraction from the last LaTeX-formatted answer.
        if not all_answers:
            dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
            if len(dollars) > 1:
                answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
                all_answers.append(answer)

        # Fallback to the full output.
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))

        # Compare each candidate answer to the ground truth.
        for answer in all_answers:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return 1.0
        return 0.0


class StrictMathVerifier(VerifierFunction):
    """
    Strict verifier for math problems using only the Minerva format extraction.
    """

    def __init__(self) -> None:
        super().__init__("strict_math", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> bool:
        raw_answer = prediction
        all_answers = []
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))
        for answer in all_answers:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return 1.0
        return 0.0


class IFEvalVerifier(VerifierFunction):
    """
    Verifier for ifeval tasks that delegates evaluation to a function
    specified in the constraint.

    The constraint may be a JSON string or a dictionary containing a key
    'func_name' used to lookup the evaluation function.
    """

    def __init__(self) -> None:
        super().__init__("ifeval", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: Union[str, Dict]) -> bool:
        constraint = label
        answer = prediction.split("<|assistant|>\n")[-1].strip()
        if isinstance(constraint, str):
            constraint = json.loads(constraint)
        if "func_name" not in constraint:
            logger.warning("Constraint missing 'func_name': %s", constraint)
            return 0.0
        func_name = constraint.pop("func_name")
        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in constraint.items() if v is not None}
        if not constraint:
            return func(prediction)
        return float(func(answer, **non_none_args))


def normalize_answer(s: str) -> str:
    """
    Normalize the answer by lowercasing, removing punctuation, articles,
    and extra whitespace.

    Based on:
    https://github.com/huggingface/evaluate/blob/main/metrics/squad/compute_score.py
    """

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        return "".join(ch for ch in text if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(s.lower())))


class FlanVerifier(VerifierFunction):
    """
    Verifier for Flan tasks that extracts the answer after "The answer is:"
    and compares it to the ground truth after normalization.
    """

    def __init__(self) -> None:
        super().__init__("flan", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> bool:
        answer_string = prediction.split("The answer is: ")[-1].strip()
        return float(normalize_answer(answer_string) == normalize_answer(label))


class MaxLenVerifier(VerifierFunction):
    """
    Verifier that checks if the length of the prediction is within the maximum allowed length.

    The ground truth (label) is interpreted as the maximum length.
    """

    def __init__(self) -> None:
        super().__init__("max_length", weight=0.5)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> bool:
        max_length = float(label)
        # linear func that hits 1 at max_length and 0 after
        return 0 if len(tokenized_prediction) > max_length else len(tokenized_prediction) / max_length



def extract_python_code(model_output: str) -> str:
    """Extract the first code block between ``` markers from the model output."""
    # Find content between ``` markers
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)
    
    if not matches:
        return model_output
        
    # Return the first match, stripped of whitespace
    return matches[0].strip()


async def _verify_ace_coder_sample_async(model_output: str, tests: List[str], api_url: str, max_execution_time: float = 1.0) -> float:
    """Async helper to verify a single code sample against test cases."""
    # Extract the python code from the model output
    python_code = extract_python_code(model_output)

    # Test data
    payload = {
        "program": python_code,
        "tests": tests,
        "max_execution_time": max_execution_time
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(api_url, json=payload) as response:
                result = await response.json()
                passes = result["results"]
                pass_rate = sum(passes) / len(passes) if passes else 0.0
                return pass_rate
    except Exception as e:
        logger.warning(f"Error verifying code sample: {e}")
        return 0.0

async def _verify_ace_coder_samples_async(model_outputs: List[str], tests_list: List[List[str]], api_url: str, max_execution_time: float = 1.0) -> List[float]:
    """Async function to verify multiple code samples against their respective test cases."""
    tasks = []
    for model_output, tests in zip(model_outputs, tests_list):
        tasks.append(_verify_ace_coder_sample_async(model_output, tests, api_url, max_execution_time))
    
    return await asyncio.gather(*tasks)

def verify_ace_coder_sample(model_outputs: List[str], tests: List[List[str]], api_url: str, max_execution_time: float = 1.0) -> List[float]:
    """First extract the python code from the model outputs, then run them against their respective test cases.
    
    Args:
        model_outputs: A list of model output strings
        tests: A list of lists of test cases, where each inner list corresponds to a model output
        api_url: URL of the API endpoint for testing code
        max_execution_time: Maximum execution time per test case in seconds
        
    Returns:
        A list of pass rates for each model output
    """
    # Handle case where tests is a single list of strings (apply to all model outputs)
    if tests and isinstance(tests[0], str):
        tests = [tests] * len(model_outputs)
    
    # Ensure tests is a list of lists
    if not all(isinstance(t, list) for t in tests):
        raise ValueError("tests must be a list of lists of strings")
    
    # Ensure lengths match
    if len(model_outputs) != len(tests):
        raise ValueError(f"Length mismatch: {len(model_outputs)} model outputs vs {len(tests)} test sets")
    
    # Run async verification
    return asyncio.run(_verify_ace_coder_samples_async(model_outputs, tests, api_url, max_execution_time))

def get_all_verifiers() -> Dict[str, VerifierFunction]:
    """
    Auto-generate a dictionary mapping verifier names to their instances.
    """
    verifiers: Dict[str, VerifierFunction] = {}
    for subclass in VerifierFunction.__subclasses__():
        instance = subclass()
        verifiers[instance.name.lower()] = instance
    return verifiers


# Auto-generate the mappings.
REWARD_FN_MAPPING: Dict[str, VerifierFunction] = get_all_verifiers()


# special case, we use this outside our general verifier loop.
def soft_format_reward_func(responses: List[str], reward_scale: float = 1.0) -> List[float]:
    """
    Check if the completion has a specific format defined by a pattern.

    Returns a list of rewards scaled by reward_scale.
    """
    pattern = r".*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [reward_scale if match else 0.0 for match in matches]


def test_verify_ace_coder_sample():
    model_outputs = [
        "```python\nprint('Hello, world!')\n```",
        "```python\nprint('Hello, world!')\n```"
    ]
    tests = [["print('Hello, world!')"], ["print('Hello, world!')"]]
    print(verify_ace_coder_sample(model_outputs, tests))