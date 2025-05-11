"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
Add new verifiers by subclassing VerifierFunction and implementing the __call__ method.
They are then automatically added to the REWARD_FN_MAPPING.
"""

import json
import logging
import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Union

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
            # also provide original string in case normalization fails
            all_answers.append(prediction)

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


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return {"f1": 0, "precision": 0, "recall": 0}
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return {"f1": f1, "precision": precision, "recall": recall}


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


class StringMatcherVerifier(VerifierFunction):
    """
    Verifier for tasks that require string matching.

    It checks if the model output matches the ground truth answer.
    """

    def __init__(self) -> None:
        super().__init__("string_matcher", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> bool:
        if "<answer>" not in prediction or "</answer>" not in prediction:
            return False
        # extract out of answer tag
        answer_string = prediction.split("<answer>")[-1].split("</answer>")[0]
        # normalize
        return float(normalize_answer(answer_string) == normalize_answer(label))


class F1Verifier(VerifierFunction):
    """
    Verifier that computes the string F1 score between the prediction and the label.
    """

    def __init__(self) -> None:
        super().__init__("string_f1", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> float:
        # remove thinking section from the prediction
        prediction = prediction.split("</think>")[-1]
        # remove answer tags from the prediction
        prediction = prediction.replace("<answer>", "").replace("</answer>", "")
        # return f1 score
        return f1_score(prediction, label)["f1"]


class ReSearchVerifierF1(VerifierFunction):
    """
    Verifier from ReSearch paper (https://arxiv.org/abs/2503.19470)
    Uses F1 score + format. If format is achieved but f1 is 0, returns 0.1. Otherwise returns F1.
    """

    def __init__(self) -> None:
        self.answer_start_tag = "<finish>"
        self.answer_end_tag = "</finish>"
        super().__init__("re_search_f1", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> float:
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            label = label.strip()
        print(f"label: {label}")
        # extract answer
        if self.answer_start_tag not in prediction and self.answer_end_tag not in prediction:
            return 0.0
        answer_string = prediction.split(self.answer_start_tag)[-1].split(self.answer_end_tag)[0]
        # check answer non-empty
        if not answer_string:
            return 0.0
        # if label is list, max over labels
        if isinstance(label, list):
            f1 = max(f1_score(answer_string, lab)["f1"] for lab in label)
        else:
            label = str(label)  # safety.
            f1 = f1_score(answer_string, label)["f1"]
        # if f1 is 0, but format is correct, return 0.1
        if f1 == 0:
            return 0.1
        # otherwise return f1
        return f1


class R1SearchVerifier(VerifierFunction):
    """
    Verifier based on the Search-R1 paper (https://github.com/PeterGriffinJin/Search-R1).
    Uses normalized exact match: returns 1.0 if answer matches any label, else 0.0.
    Answer extraction is done via a case-insensitive regex on <finish>...</finish> tags.
    """

    # Precompile a case-insensitive regex to extract answer text
    TAG_PATTERN = re.compile(r"<finish>(.*?)</finish>", re.IGNORECASE | re.DOTALL)

    def __init__(self) -> None:
        super().__init__(name="re_search", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: Union[str, List[str]]) -> float:
        # 1. Parse JSON label safely
        parsed_labels: Union[List, str]
        try:
            parsed = json.loads(label)
            parsed_labels = parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat label as raw string or list-of-strings
            if isinstance(label, list):
                parsed_labels = label
            else:
                parsed_labels = [str(label).strip()]

        # 2. Extract answer between tags
        match = self.TAG_PATTERN.search(prediction)
        if not match:
            logging.debug("No <finish> tags found in prediction")
            return 0.0

        answer_text = match.group(len(match.groups())).strip()
        if not answer_text:
            logging.debug("Extracted answer is empty after stripping whitespace")
            return 0.0

        # 3. Normalize once
        norm_answer = normalize_answer(answer_text)

        # 4. Compare against each label
        for lbl in parsed_labels:
            try:
                lbl_str = normalize_answer(str(lbl))
                if norm_answer == lbl_str:
                    return 1.0
            except Exception as e:
                logging.warning(f"Error normalizing label '{lbl}': {e}")

        # 5. No match found
        return 0.0


class MaxLenVerifier(VerifierFunction):
    """
    Verifier that checks if the length of the prediction is within the maximum allowed length.

    The ground truth (label) is interpreted as the maximum length.
    """

    def __init__(self) -> None:
        super().__init__("max_length", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> bool:
        desired_length = float(label)
        # return absolute difference between the length of the prediction and the max length
        # make sure to disallow negative rewards
        length_diff = abs(len(tokenized_prediction) - desired_length)
        return 1 - (length_diff / 8192)


class UpToMaxLenVerifier(VerifierFunction):
    """
    Verifier that checks if the length of the prediction is within the maximum allowed length.

    The ground truth (label) is interpreted as the maximum length.
    """

    def __init__(self) -> None:
        super().__init__("up_to_max_length", weight=1.0)

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str) -> bool:
        desired_length = float(label)
        length_diff = len(tokenized_prediction) - desired_length
        # if we were too short, its fine! return 1.0
        if length_diff < 0:
            return 1.0
        # if we were too long, return the difference
        # make sure to disallow negative rewards
        return 1 - (length_diff / 8192)


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
