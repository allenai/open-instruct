"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
Add new verifiers by subclassing VerifierFunction and implementing the __call__ method.
They are then automatically added to the REWARD_FN_MAPPING.
"""

import ast
import asyncio
import copy
import json
import logging
import os
import re
import shlex
import string
import warnings
import weakref
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Union

import requests
from litellm import acompletion

from open_instruct import logger_utils
from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.IFEvalG import instructions_registry
from open_instruct.tool_utils.swe_tool_parser import Argument as SWEArgument
from open_instruct.tool_utils.swe_tool_parser import Command as SWECommand
from open_instruct.tool_utils.swe_tool_parser import FunctionCallingParser
from open_instruct.judge_utils import EXTRACTOR_MAP, JUDGE_PROMPT_MAP, PRICE_PER_TOKEN, build_messages
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)
from open_instruct.utils import extract_final_answer

logger = logger_utils.setup_logger(__name__)

# remove excessive logging from liteLLM
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("litellm.cost_calculator").setLevel(logging.CRITICAL)
logging.getLogger("litellm._client").setLevel(logging.CRITICAL)
logging.getLogger("cost_calculator").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


@dataclass
class VerifierConfig:
    """For now this config exists to support LMJudgeVerifer, can be expanded to support other verifers"""

    @classmethod
    def from_args(cls, args) -> "VerifierConfig":
        """
        Create a VerifierConfig from an Args object by automatically matching field names.
        Only fields that exist in both Args and VerifierConfig will be passed through.
        """
        import dataclasses

        # Get all field names from VerifierConfig
        verifier_fields = {field.name for field in dataclasses.fields(cls)}

        # Get all attributes from args that match VerifierConfig field names
        matching_kwargs = {}
        for field_name in verifier_fields:
            if hasattr(args, field_name):
                matching_kwargs[field_name] = getattr(args, field_name)

        return cls(**matching_kwargs)


@dataclass
class LMJudgeVerifierConfig(VerifierConfig):
    # judge args
    llm_judge_model: str
    llm_judge_max_tokens: int
    llm_judge_max_context_length: int
    llm_judge_temperature: float
    llm_judge_timeout: int
    seed: int


@dataclass
class CodeVerifierConfig(VerifierConfig):
    code_api_url: str
    code_max_execution_time: float
    code_pass_rate_reward_threshold: float
    code_apply_perf_penalty: bool


@dataclass
class CodeAgentVerifierConfig(VerifierConfig):
    code_agent_url: str
    code_agent_max_line_span: int = 0  # only reward if the view calls span at most this many lines (0 disables)
    code_agent_turns_after_view: int = -1  # only reward if the 'correct' view call is followed by at most this many turns (-1 disables)
    code_agent_repitition_penalty: float = 0.0  # if the same tool call is repeated, penalize the final score by this amount


@dataclass
class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: Optional[str] = None


@dataclass
class MaxLengthVerifierConfig(VerifierConfig):
    max_length_verifier_max_length: int


class VerifierFunction(ABC):
    """
    Base class for all verifier functions that evaluate model predictions against ground truth.

    Each verifier function takes a prediction and compares it to a ground truth label,
    returning a VerificationResult with a score between 0.0 and 1.0.
    """

    def __init__(self, name: str, weight: float = 1.0, verifier_config: Optional[VerifierConfig] = None) -> None:
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return VerifierConfig

    @abstractmethod
    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: Any, query: Optional[str] = None
    ) -> VerificationResult:
        """
        Evaluate the given prediction against the ground truth (or constraint).

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query

        Returns:
            VerificationResult
        """

    async def async_call(
        self, tokenized_prediction: List[int], prediction: str, label: Any, query: Optional[str] = None
    ) -> VerificationResult:
        """
        Asynchronous version of __call__. By default, it runs the synchronous __call__ in a thread pool.
        Subclasses can override this method for truly asynchronous implementation.

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused by most verifiers).
            prediction (str): The model output.
            label (Any): The ground truth answer or evaluation constraint.
            query (Optional[str]): The original query.

        Returns:
            VerificationResult
        """
        # Run the synchronous __call__ in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.__call__(tokenized_prediction, prediction, label, query))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, weight={self.weight})"


# small helper to optionally remove thinking section + answer output.
# assumes a certain format, so might not always be useful.
# we don't always need this -- for example, math evaluations just extract a final
# number, so we don't need to remove the thinking section.
def remove_thinking_section(prediction: str) -> str:
    prediction = prediction.replace("<|assistant|>", "").strip()
    # remove thinking section from the prediction
    prediction = prediction.split("</think>")[-1]
    # remove answer tags from the prediction
    prediction = prediction.replace("<answer>", "").replace("</answer>", "")
    return prediction.strip()


class GSM8KVerifier(VerifierFunction):
    """
    Verifier for GSM8K tasks that extracts the last number from the prediction
    and compares it (case-insensitively) to the ground truth.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("gsm8k", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
        response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        extracted = numbers[-1] if numbers else response
        score = float(str(extracted).lower() == str(label).lower())
        return VerificationResult(score=score)


class MathVerifier(VerifierFunction):
    """
    Verifier for math problems.

    Attempts several extraction methods (boxed answers, Minerva format,
    last LaTeX answer) and compares the extracted answers to the ground truth.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
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
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)


class StrictMathVerifier(VerifierFunction):
    """
    Strict verifier for math problems using only the Minerva format extraction.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("strict_math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
        raw_answer = prediction
        all_answers = []
        minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
        if minerva_answer is not None and minerva_answer != "[invalidanswer]":
            all_answers.append(minerva_answer)
        if not all_answers:
            all_answers.append(normalize_final_answer(prediction))
        for answer in all_answers:
            if is_equiv(answer, label) or hendrycks_is_equiv(answer, label):
                return VerificationResult(score=1.0)
        return VerificationResult(score=0.0)


class IFEvalVerifier(VerifierFunction):
    """
    Verifier for ifeval tasks that delegates evaluation to a function
    specified in the constraint.

    The constraint(s) are a list of constraint ids.
    This list is found under the key "instruction_id" in the ground_truth dict.

    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("ifeval", weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: Union[str, Dict], query: Optional[str] = None
    ) -> VerificationResult:
        instruction_dict = instructions_registry.INSTRUCTION_DICT
        constraint_dict = ast.literal_eval(label)
        constraint_dict = constraint_dict[0]
        if isinstance(constraint_dict, str):
            constraint_dict = json.loads(constraint_dict)
        answer = remove_thinking_section(prediction)
        instruction_keys = constraint_dict["instruction_id"]
        args_list = constraint_dict["kwargs"]
        rewards = []
        if len(prediction) == 0 or len(answer) == 0:
            logger.warning("Empty prediction received for IFEvalVerifier.")
            return VerificationResult(score=0.0)
        for instruction_key, args in zip(instruction_keys, args_list):
            if args is None:
                args = {}
            args = {k: v for k, v in args.items() if v is not None}
            instruction_cls = instruction_dict[instruction_key]
            instruction_instance = instruction_cls(instruction_key)
            instruction_instance.build_description(**args)
            if prediction.strip() and instruction_instance.check_following(answer):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return VerificationResult(score=sum(rewards) / len(rewards))


class IFEvalVerifierOld(VerifierFunction):
    """
    Verifier for ifeval tasks that delegates evaluation to a function
    specified in the constraint.

    The constraint may be a JSON string or a dictionary containing a key
    'func_name' used to lookup the evaluation function.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("ifeval_old", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: Union[str, Dict], query: Optional[str] = None
    ) -> VerificationResult:
        constraint = label
        answer = remove_thinking_section(prediction)
        if isinstance(constraint, str):
            constraint = json.loads(constraint)
        if "func_name" not in constraint:
            logger.warning("Constraint missing 'func_name': %s", constraint)
            return VerificationResult(score=0.0)
        func_name = constraint.pop("func_name")
        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in constraint.items() if v is not None}
        if not constraint:
            return VerificationResult(score=float(func(answer)))
        return VerificationResult(score=float(func(answer, **non_none_args)))


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

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("flan", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
        answer_string = prediction.split("The answer is: ")[-1].strip()
        score = float(normalize_answer(answer_string) == normalize_answer(label))
        return VerificationResult(score=score)


class StringMatcherVerifier(VerifierFunction):
    """
    Verifier for tasks that require string matching.

    It checks if the model output matches the ground truth answer.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("string_matcher", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
        if "<answer>" not in prediction or "</answer>" not in prediction:
            return VerificationResult(score=0.0)
        # extract out of answer tag
        answer_string = prediction.split("<answer>")[-1].split("</answer>")[0]
        # normalize
        score = float(normalize_answer(answer_string) == normalize_answer(label))
        return VerificationResult(score=score)


class F1Verifier(VerifierFunction):
    """
    Verifier that computes the string F1 score between the prediction and the label.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("string_f1", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
        # remove thinking section from the prediction
        prediction = prediction.split("</think>")[-1]
        # remove answer tags from the prediction
        prediction = prediction.replace("<answer>", "").replace("</answer>", "")
        # return f1 score
        score = f1_score(prediction, label)["f1"]
        return VerificationResult(score=score)


class PuzzleMatcherVerifier(VerifierFunction):
    """
    Verifier for Puzzle tasks that require string matching (exact matching).

    It checks if the model output matches the ground truth answer.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("puzzle", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
        # remove answer tags from the prediction
        prediction = remove_thinking_section(prediction)
        score = float(normalize_answer(prediction) == normalize_answer(label))
        return VerificationResult(score=score)


class ReSearchVerifierF1(VerifierFunction):
    """
    Verifier from ReSearch paper (https://arxiv.org/abs/2503.19470)
    Uses F1 score + format. If format is achieved but f1 is 0, returns 0.1. Otherwise returns F1.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        self.answer_start_tag = "<finish>"
        self.answer_end_tag = "</finish>"
        super().__init__("re_search_f1", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
        try:
            label = json.loads(label)
        except json.JSONDecodeError:
            label = label.strip()
        # extract answer
        if self.answer_start_tag not in prediction and self.answer_end_tag not in prediction:
            return VerificationResult(score=0.0)
        answer_string = prediction.split(self.answer_start_tag)[-1].split(self.answer_end_tag)[0]
        # check answer non-empty
        if not answer_string:
            return VerificationResult(score=0.0)
        # if label is list, max over labels
        if isinstance(label, list):
            f1 = max(f1_score(answer_string, str(lab))["f1"] for lab in label)
        else:
            label = str(label)  # safety.
            f1 = f1_score(answer_string, label)["f1"]
        # if f1 is 0, but format is correct, return 0.1
        if f1 == 0:
            return VerificationResult(score=0.1)
        # otherwise return f1
        return VerificationResult(score=f1)


class R1SearchVerifier(VerifierFunction):
    """
    Verifier based on the Search-R1 paper (https://github.com/PeterGriffinJin/Search-R1).
    Uses normalized exact match: returns 1.0 if answer matches any label, else 0.0.
    Answer extraction is done via a case-insensitive regex on <finish>...</finish> tags.
    """

    # Precompile a case-insensitive regex to extract answer text
    TAG_PATTERN = re.compile(r"<finish>(.*?)</finish>", re.IGNORECASE | re.DOTALL)

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__(name="re_search", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self,
        tokenized_prediction: List[int],
        prediction: str,
        label: Union[str, List[str]],
        query: Optional[str] = None,
    ) -> VerificationResult:
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
            return VerificationResult(score=0.0)

        answer_text = match.group(len(match.groups())).strip()
        if not answer_text:
            logging.debug("Extracted answer is empty after stripping whitespace")
            return VerificationResult(score=0.0)

        # 3. Normalize once
        norm_answer = normalize_answer(answer_text)

        # 4. Compare against each label
        for lbl in parsed_labels:
            try:
                lbl_str = normalize_answer(str(lbl))
                if norm_answer == lbl_str:
                    return VerificationResult(score=1.0)
            except Exception as e:
                logging.warning(f"Error normalizing label '{lbl}': {e}")

        # 5. No match found
        return VerificationResult(score=0.0)


class MaxLenVerifier(VerifierFunction):
    """
    Verifier that checks if the length of the prediction is within the maximum allowed length.

    The ground truth (label) is interpreted as the maximum length.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("max_length", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
        desired_length = float(label)
        # return absolute difference between the length of the prediction and the max length
        # make sure to disallow negative rewards
        length_diff = abs(len(tokenized_prediction) - desired_length)
        score = 1 - (length_diff / self.verifier_config.max_length_verifier_max_length)
        return VerificationResult(score=score)

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.
        Returns:
            type: The VerifierConfig class or its subclass
        """
        return MaxLengthVerifierConfig


class UpToMaxLenVerifier(VerifierFunction):
    """
    Verifier that checks if the length of the prediction is within the maximum allowed length.

    The ground truth (label) is interpreted as the maximum length.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("up_to_max_length", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: Optional[str] = None
    ) -> VerificationResult:
        desired_length = float(label)
        length_diff = len(tokenized_prediction) - desired_length
        # if we were too short, its fine! return 1.0
        if length_diff < 0:
            return VerificationResult(score=1.0)
        # if we were too long, return the difference
        # make sure to disallow negative rewards
        score = 1 - (length_diff / self.verifier_config.max_length_verifier_max_length)
        return VerificationResult(score=score)

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.
        Returns:
            type: The VerifierConfig class or its subclass
        """
        return MaxLengthVerifierConfig


class LMJudgeVerifier(VerifierFunction):
    """
    Verifier that uses a language model's judgement to score a response.
    """

    # Use WeakKeyDictionary to automatically clean up clients when event loops are garbage collected
    _client_cache = weakref.WeakKeyDictionary()

    def __init__(self, judge_type: str, verifier_config: LMJudgeVerifierConfig) -> None:
        super().__init__(f"general-{judge_type}", verifier_config=verifier_config, weight=1.0)
        self.prompt_template = JUDGE_PROMPT_MAP[judge_type]
        self.extractor = EXTRACTOR_MAP[judge_type]
        os.environ["AZURE_API_VERSION"] = "2024-12-01-preview"

    def parse_completion(self, completion):
        """
        Extract reasoning and score from an OpenAI API completion response.

        Args:
            completion: The OpenAI API completion response object

        Returns:
            tuple: (reasoning, score) extracted from the response
        """
        reasoning = ""
        score = 0.0

        if not completion:
            print("No completion received from the model.")
            return reasoning, score

        try:
            # remove anything between <think> and </think> including the tags using regex
            pattern = r"<think>\s*.*?\s*</think>\s*"
            content = re.sub(pattern, "", completion.choices[0].message.content, flags=re.DOTALL)
            content = content.replace("<answer>", "").replace("</answer>", "")
            reasoning, score = self.extractor(content)

        except Exception as e:
            print(f"Error processing model response: {str(e)}")
            if hasattr(completion, "choices") and completion.choices is not None and len(completion.choices) > 0:
                print(f"Response content: {getattr(completion.choices[0].message, 'content', 'No content available')}")

        return reasoning, score

    def get_cost(self, response, model: str):
        """
        Get the cost of the response.
        """
        model_name = model.split("/")[-1]  # for litellm, discard the namespace
        model_name = model_name.replace("-standard", "")  # azure OAI models have -standard in the name
        return (
            PRICE_PER_TOKEN.get(model_name, {}).get("input", 0) * response.usage.prompt_tokens
            + PRICE_PER_TOKEN.get(model_name, {}).get("output", 0) * response.usage.completion_tokens
        )

    async def async_call(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: str
    ) -> VerificationResult:
        """
        Asynchronous version of __call__ that properly handles the async OpenAI client.
        """
        # client = self._get_client()
        final_answer = extract_final_answer(prediction)
        prompt = self.prompt_template.format(input=query, output=final_answer, label=label)

        max_retries = 3  # for rate limits
        retry_delay = 1.0

        for attempt in range(max_retries):
            # judges the quality of a response
            try:
                messages = build_messages(prompt)

                # Faeze: check if the request would exceed context window
                # Import the context window checker
                try:
                    from open_instruct.context_window_checker import (
                        check_context_window_limit,
                        truncate_messages_to_fit_context,
                    )

                    context_check_available = True
                except ImportError:
                    logger.warning("Context window checker not available. Proceeding without context checking.")
                    context_check_available = False

                # Check if the request would exceed context window
                if context_check_available:
                    if not check_context_window_limit(
                        messages=messages,
                        max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                        model_name=self.verifier_config.llm_judge_model,
                        max_context_length=self.verifier_config.llm_judge_max_context_length,  # Adjust based on your model
                        safety_margin=150,
                    ):
                        # Try to truncate messages to fit
                        messages = truncate_messages_to_fit_context(
                            messages=messages,
                            max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                            model_name=self.verifier_config.llm_judge_model,
                            max_context_length=self.verifier_config.llm_judge_max_context_length,
                            safety_margin=200,
                        )

                        # Check again after truncation
                        if not check_context_window_limit(
                            messages=messages,
                            max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                            model_name=self.verifier_config.llm_judge_model,
                            max_context_length=self.verifier_config.llm_judge_max_context_length,
                            safety_margin=150,
                        ):
                            logger.error("Cannot fit request within context window even after truncation.")
                            return VerificationResult(score=0.0, cost=0.0, reasoning="Error: Context window exceeded")
                # end of Faeze's context window check
                response = await acompletion(
                    model=self.verifier_config.llm_judge_model,
                    messages=messages,
                    temperature=self.verifier_config.llm_judge_temperature,
                    max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                    seed=self.verifier_config.seed,
                    timeout=self.verifier_config.llm_judge_timeout,
                )
                reasoning, score = self.parse_completion(response)
                cost = self.get_cost(response, self.verifier_config.llm_judge_model)
                # normalize score to be between 0 and 1
                return VerificationResult(score=score, cost=cost, reasoning=reasoning)

            except Exception as e:
                logger.warning(f"LLM judge attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"LLM judge failed after {max_retries} attempts. Returning default score of 0.0")
                    return VerificationResult(score=0.0, cost=0.0, reasoning=f"Error: {str(e)}")
                else:
                    await asyncio.sleep(retry_delay * (2**attempt))  # Exponential backoff
        return VerificationResult(score=0.0, cost=0.0, reasoning="Unknown error after all retries.")

    def __call__(self, tokenized_prediction: List[int], prediction: str, label: str, query: str) -> VerificationResult:
        """
        Evaluates the prediction based on an LLM's judgement.

        Args:
            tokenized_prediction (List[int]): Tokenized representation of the prediction (unused).
            prediction (str): The model output string that was judged.
            label (str): An optional reference for the judge. Can be a reference answer or a rubric.
        Returns:
            float: The calculated reward (parsed_rating)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous __call__ method from within an async context. Use async_call instead."
                )
            else:
                return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))
        except RuntimeError:
            return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))

    @classmethod
    async def cleanup_all_clients(cls):
        """
        Manually close all cached clients. Call this before shutting down to avoid cleanup warnings.
        """
        clients_to_close = list(cls._client_cache.values())
        cls._client_cache.clear()

        for client in clients_to_close:
            try:
                await client.close()
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {e}")
                # Suppress the error to avoid breaking shutdown

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return LMJudgeVerifierConfig


class CodeVerifier(VerifierFunction):
    """
    Verifier that executes Python code against test cases using an external API.

    The label should be a list of test cases or a JSON string representation of a list.
    The API URL should be provided during initialization.
    """

    # Class-level session cache to reuse connections
    _session_cache = weakref.WeakKeyDictionary()

    def __init__(self, verifier_config: CodeVerifierConfig) -> None:
        super().__init__("code", verifier_config=verifier_config, weight=1.0)
        self.pass_rate_reward_threshold = verifier_config.code_pass_rate_reward_threshold
        self.apply_perf_penalty = verifier_config.code_apply_perf_penalty

    def extract_python_code(self, model_output: str) -> str:
        """Extract the last code block between ``` markers from the model output."""
        # Find content between ``` markers
        pattern = r"```(?:python)?(.*?)```"
        matches = re.findall(pattern, model_output, re.DOTALL)

        if not matches:
            return model_output

        # Return the last match, stripped of whitespace
        return matches[-1].strip()

    # Create a session pool for better performance
    _session_pool = None

    @classmethod
    def _get_session(cls):
        if cls._session_pool is None:
            cls._session_pool = requests.Session()
            # Configure connection pooling
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=100,
                pool_maxsize=100,
                max_retries=requests.adapters.Retry(
                    total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504]
                ),
            )
            cls._session_pool.mount("http://", adapter)
            cls._session_pool.mount("https://", adapter)
        return cls._session_pool

    async def async_call(
        self, tokenized_prediction: List[int], prediction: str, label: Any, query: Optional[str] = None
    ) -> VerificationResult:
        """
        Asynchronously verify code execution against test cases.

        Args:
            tokenized_prediction: Unused tokenized representation
            prediction: The model output containing Python code
            label: List of test cases or JSON string representation of a list
            query: Unused original query

        Returns:
            VerificationResult with score as the pass rate of test cases
        """
        # Extract Python code from the model output
        python_code = self.extract_python_code(prediction)

        # Test data
        payload = {
            "program": python_code,
            "tests": label,
            "max_execution_time": self.verifier_config.code_max_execution_time,
        }

        try:
            # Use connection pooling session
            session = self._get_session()

            # Calculate timeout
            http_timeout = max(30, min(300, self.verifier_config.code_max_execution_time * 10))

            # Make request in thread pool to keep it async
            def make_request():
                response = session.post(
                    self.verifier_config.code_api_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=http_timeout,
                )
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(make_request)
            passes = result["results"]
            pass_rate = sum(passes) / len(passes) if passes else 0.0
            score = 0.0 if pass_rate < self.pass_rate_reward_threshold else pass_rate
            if self.apply_perf_penalty and score > 0.0:
                runtimes = result["runtimes"]
                # for each runtime, multiply the percent of the timeout that was used
                multipliers = [
                    (self.verifier_config.code_max_execution_time - runtime)
                    / self.verifier_config.code_max_execution_time
                    for runtime in runtimes
                ]
                penalized_passes = [passes[i] * multipliers[i] for i in range(len(passes))]
                score = sum(penalized_passes) / len(penalized_passes)
            return VerificationResult(score=score)
        except Exception as e:
            logger.warning(f"Error verifying code sample: {e}")
            return VerificationResult(score=0.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: Any, query: Optional[str] = None
    ) -> VerificationResult:
        """
        Synchronously verify code execution against test cases.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous __call__ method from within an async context. Use async_call instead."
                )
            else:
                return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))
        except RuntimeError:
            return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return CodeVerifierConfig
class CodeSearchVerifier(VerifierFunction):
    """
    Simple verifier checks in the model makes a call to the CodeSearchTool in tool_utils.tool_vllm.py
    Checks that the call looks are the correct file and spans the buggy lines
    """

    def __init__(self, verifier_config: CodeAgentVerifierConfig) -> None:
        super().__init__("code_search", verifier_config=verifier_config, weight=1.0)

    async def get_file_line_count(self, file_path: str, repo_name: Optional[str] = None) -> Optional[int]:
        """Resolve the number of lines in a file via the code agent API."""

        logger.warning(
            "CodeSearchVerifier: resolving line count for path=%s repo=%s",
            file_path,
            repo_name,
        )

        path_candidates = self._generate_path_candidates(file_path, repo_name)
        logger.warning(
            "CodeSearchVerifier: candidate paths for %s (repo=%s): %s",
            file_path,
            repo_name,
            path_candidates,
        )
        if not path_candidates:
            logger.debug("No path candidates available when checking line count for %s", file_path)
            return None

        api_url = self._get_run_bash_url()
        if not api_url:
            logger.debug("No code agent URL configured; cannot resolve line count for %s", file_path)
            return None

        for candidate in path_candidates:
            line_count = await self._fetch_line_count_via_api(api_url, candidate, repo_name)
            if line_count is not None:
                if candidate != file_path:
                    logger.warning(
                        "Resolved %s to %s when checking line count (repo=%s)",
                        file_path,
                        candidate,
                        repo_name,
                    )
                return line_count

        logger.warning("Unable to resolve file length for %s via code agent", file_path)
        return None

    def _get_run_bash_url(self) -> Optional[str]:
        base_url = getattr(self.verifier_config, "code_agent_url", None)
        if not base_url:
            return None

        base_url = base_url.rstrip("/")
        for suffix in ("/run_bash", "/view_file", "/edit_file", "/test_program", "/test_program_stdio"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        if not base_url:
            return None
        return f"{base_url}/run_bash"

    def _generate_path_candidates(self, file_path: str, repo_name: Optional[str]) -> List[str]:
        raw_path = (file_path or "").strip().replace("\\", "/")
        if not raw_path:
            return []

        normalized_repo = (repo_name or "").strip().strip("/")
        candidates: List[str] = []
        seen: Set[str] = set()

        def add(path: str) -> None:
            cleaned = (path or "").strip().replace("\\", "/")
            if not cleaned:
                return
            cleaned = re.sub(r"/{2,}", "/", cleaned)
            if cleaned.startswith("./"):
                cleaned = cleaned[2:]
            if cleaned in {"", ".", "/"}:
                return
            if cleaned.endswith("/") and len(cleaned) > 1:
                cleaned = cleaned[:-1]
            if cleaned.startswith("/") and len(cleaned) > 1:
                cleaned = "/" + cleaned.lstrip("/")
            if cleaned not in seen:
                seen.add(cleaned)
                candidates.append(cleaned)

        add(raw_path)
        add(raw_path.lstrip("/"))
        add("/" + raw_path.lstrip("/"))

        if normalized_repo:
            repo_prefix = normalized_repo + "/"
            snapshot = list(candidates)
            for candidate in snapshot:
                stripped = candidate.lstrip("/")
                if not stripped:
                    continue
                if stripped.startswith(repo_prefix):
                    remainder = stripped[len(repo_prefix) :]
                    if remainder:
                        add(remainder)
                        add("/" + remainder)
                    continue
                add(f"{normalized_repo}/{stripped}")
                add(f"/{normalized_repo}/{stripped}")
                add(f"repos/{normalized_repo}/{stripped}")
                add(f"/repos/{normalized_repo}/{stripped}")

        for candidate in list(candidates):
            stripped = candidate.lstrip("/")
            if not stripped:
                continue
            if stripped.startswith("testbed/"):
                remainder = stripped[len("testbed/") :]
                if remainder:
                    add(remainder)
                    add("/" + remainder)
            else:
                add(f"testbed/{stripped}")
                add(f"/testbed/{stripped}")

        return candidates

    async def _fetch_line_count_via_api(
        self, api_url: str, candidate_path: str, repo_name: Optional[str]
    ) -> Optional[int]:
        quoted_path = shlex.quote(candidate_path)
        cmd = f"wc -l {quoted_path} 2>/dev/null || echo 'ERROR'"
        payload = {
            "cmd": cmd,
            "repo_name": repo_name,
            "timeout_seconds": 15,
        }

        logger.warning(
            "CodeSearchVerifier: querying line count via %s cmd=%s repo=%s",
            api_url,
            cmd,
            repo_name,
        )

        try:
            def make_request() -> Dict[str, Any]:
                response = requests.post(api_url, json=payload, timeout=10)
                response.raise_for_status()
                return response.json()

            result = await asyncio.to_thread(make_request)
        except Exception as exc:
            logger.warning(
                "Error querying code agent for %s using %s: %s", candidate_path, api_url, exc
            )
            return None

        content = result.get("content", "")
        logger.warning(
            "CodeSearchVerifier: code agent response for %s (repo=%s): returncode=%s content-preview=%s",
            candidate_path,
            repo_name,
            result.get("returncode"),
            content[:200],
        )
        stripped_lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
        if any(line == "ERROR" for line in stripped_lines):
            logger.warning(
                "CodeSearchVerifier: detected ERROR sentinel in response for %s (repo=%s)",
                candidate_path,
                repo_name,
            )
            return None

        for line in stripped_lines:
            stripped = line.strip()
            if not stripped:
                continue
            first_char = stripped[0]
            if not first_char.isdigit():
                continue
            parts = stripped.split()
            if not parts:
                continue
            try:
                return int(parts[0])
            except (ValueError, IndexError):
                continue

        logger.warning(
            "Could not parse line count for %s from API response: %s", candidate_path, content
        )
        return None

    # Removed local filesystem fallbacks; rely solely on the code agent API for line counts.

    def parse_tool_calls(self, prediction: str) -> List[Dict[str, Any]]:
        """Parse and validate tool calls using FunctionCallingParser.
        Accepts <tool_call> {"name": ..., "arguments": {...}} </tool_call> blocks.
        Validates each call against a minimal schema for str_replace_editor (view only).
        Returns only validated calls as dictionaries identical to the model JSON.
        """
        tool_call_pattern = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

        # Minimal command schema for the code-view tool
        str_replace_editor_cmd = SWECommand(
            name="str_replace_editor",
            docstring="Custom tool for viewing files; only 'view' is supported here.",
            arguments=[
                SWEArgument(
                    name="command", type="string", description="The command to run", required=True, enum=["view"]
                ),
                SWEArgument(
                    name="path", type="string", description="Absolute path to file or directory", required=True
                ),
                SWEArgument(
                    name="view_range",
                    type="array",
                    description="[start, end] line numbers when viewing a file",
                    required=True,
                    items={"type": "integer"},
                ),
                SWEArgument(name="repo_name", type="string", description="Optional repository name", required=False),
            ],
        )
        parser = FunctionCallingParser()

        parsed_calls: List[Dict[str, Any]] = []

        def _fallback_parse(raw_str: str) -> Optional[Dict[str, Any]]:
            try:
                m_name = re.search(r'"name"\s*:\s*"([^"]+)"', raw_str)
                if not m_name:
                    return None
                name = m_name.group(1)
                m_args = re.search(r'("arguments"\s*:\s*\{)|(?:arguments\s*=\s*\{)', raw_str)
                if not m_args:
                    return {"name": name, "arguments": {}}
                start = m_args.end() - 1
                depth = 0
                end = None
                for i in range(start, len(raw_str)):
                    ch = raw_str[i]
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                if end is None:
                    return None
                args_str = raw_str[start:end]
                arguments = json.loads(args_str)
                return {"name": name, "arguments": arguments}
            except Exception:
                return None

        for match in tool_call_pattern.finditer(prediction):
            raw = match.group(1)
            response_start = prediction.find("<tool_response>", match.end())
            response_end = None
            if response_start != -1:
                tmp_response_end = prediction.find("</tool_response>", response_start)
                if tmp_response_end != -1:
                    response_end = tmp_response_end + len("</tool_response>")
            call_meta = {
                "call_start": match.start(),
                "call_end": match.end(),
                "response_start": None if response_start == -1 else response_start,
                "response_end": response_end,
            }

            data = None
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                data = _fallback_parse(raw)
                if data is None:
                    warnings.warn(
                        f"Failed to parse tool call JSON and fallback failed. Raw content: {raw}, Error: {e}"
                    )
                    continue
            try:
                # Build LiteLLM-style function call wrapper expected by the parser
                model_response = {
                    "message": prediction,
                    "tool_calls": [{"function": {"name": data.get("name"), "arguments": data.get("arguments", {})}}],
                }
                # Validate; throws on schema issues
                _thought, _action = parser(model_response, commands=[str_replace_editor_cmd], strict=True)
                data["_meta"] = call_meta
                parsed_calls.append(data)
            except Exception as e:
                warnings.warn(f"Skipping invalid tool call: {e}")

        return parsed_calls

    def _count_turns_after_call(
        self, prediction: str, tool_calls: List[Dict[str, Any]], call_index: int
    ) -> int:
        """
        Count assistant turns after a given tool call. Each subsequent tool call counts as one turn.
        Any remaining assistant text after removing tool calls and tool responses counts as an additional turn.
        """
        total_calls = len(tool_calls)
        remaining_calls = max(0, total_calls - call_index - 1)
        turns = remaining_calls

        meta = tool_calls[call_index].get("_meta", {}) if 0 <= call_index < total_calls else {}
        response_end = meta.get("response_end")
        if response_end is None:
            return turns

        remainder = prediction[response_end:]
        if not remainder:
            return turns

        remainder = remainder.replace("<endoftext>", " ")
        remainder = re.sub(r"<tool_call>.*?</tool_call>", " ", remainder, flags=re.DOTALL)
        remainder = re.sub(r"user\s*<tool_response>.*?</tool_response>", " ", remainder, flags=re.DOTALL)

        if remainder.strip():
            turns += 1

        return turns

    async def async_call(
        self, tokenized_prediction: List[int], prediction: str, label: Any, query: Optional[str] = None
    ) -> VerificationResult:
        if "<tool_response>\nMax tool calls exceeded.</tool_response>" in prediction:
            return VerificationResult(score=0.0)
        
        buggy_info = json.loads(label)
        bug_fn_file = buggy_info["bug_fn_file"]
        bug_fn_line_start = buggy_info["line_start"]
        bug_fn_line_end = buggy_info["line_end"]
        label_repo_name = buggy_info.get("repo_name")

        tool_calls = self.parse_tool_calls(prediction)
        score = 0.0
        max_line_span = getattr(self.verifier_config, "code_agent_max_line_span", None)
        max_turns_after_view = getattr(self.verifier_config, "code_agent_turns_after_view", None)
        for idx, tool_call in enumerate(tool_calls):
            arguments = tool_call.get("arguments", {}) or {}
            view_range = arguments.get("view_range")
            if not isinstance(view_range, (list, tuple)) or len(view_range) != 2:
                logger.debug("Tool call missing valid view_range: %s", tool_call)
                continue

            tool_file = arguments.get("path")
            if not tool_file:
                logger.debug("Tool call missing path argument: %s", tool_call)
                continue

            try:
                tool_line_start = int(view_range[0])
                tool_line_end = int(view_range[1])
            except (TypeError, ValueError):
                logger.debug("Tool call has non-integer view_range values: %s", tool_call)
                continue

            if tool_line_end < tool_line_start:
                logger.debug("Tool call has inverted view_range %s", view_range)
                continue

            if not (tool_file.endswith(bug_fn_file) or bug_fn_file.endswith(tool_file)):
                logger.debug(f"Tool call requested file {tool_file} but bug is in {bug_fn_file}")
                continue
            
            if not (tool_line_start <= bug_fn_line_start and tool_line_end >= bug_fn_line_end):
                logger.debug(f"Tool call requested lines {tool_line_start}-{tool_line_end} "
                             f"but bug is in {bug_fn_line_start}-{bug_fn_line_end}")
                continue

            if max_line_span is not None and max_line_span > 0:
                view_span = tool_line_end - tool_line_start + 1
                if view_span > max_line_span:
                    logger.debug(
                        "Tool call view_range %s exceeds max line span %s", view_range, max_line_span
                    )
                    continue
            
            repo_name = arguments.get("repo_name") or label_repo_name
            file_line_count = await self.get_file_line_count(tool_file, repo_name)
            if file_line_count is not None:
                if tool_line_end > file_line_count:
                    logger.debug(
                        f"Tool call requested lines {tool_line_start}-{tool_line_end} "
                        f"but file {tool_file} only has {file_line_count} lines"
                    )
                    continue
            else:
                logger.warning(
                    f"Could not verify file length for {tool_file}, not counting this tool call "
                )
                continue

            if max_turns_after_view is not None and max_turns_after_view >= 0:
                turns_after_view = self._count_turns_after_call(prediction, tool_calls, idx)
                if turns_after_view > max_turns_after_view:
                    logger.debug(
                        "Tool call at index %s followed by %s turns (limit %s)",
                        idx,
                        turns_after_view,
                        max_turns_after_view,
                    )
                    continue
            
            score = 1.0
            break
        penalty_per_repeat = getattr(self.verifier_config, "code_agent_repitition_penalty", 0.0) or 0.0
        if score > 0.0 and penalty_per_repeat > 0.0:
            call_counts: Counter = Counter()
            for call in tool_calls:
                args = call.get("arguments", {}) or {}
                path = args.get("path")
                command = args.get("command")
                repo = args.get("repo_name")
                view_range_value = args.get("view_range")
                if isinstance(view_range_value, (list, tuple)):
                    try:
                        view_range_key = tuple(int(v) for v in view_range_value)
                    except (TypeError, ValueError):
                        view_range_key = tuple(view_range_value)
                elif view_range_value is None:
                    view_range_key = ()
                else:
                    view_range_key = (view_range_value,)

                key = (call.get("name"), command, path, view_range_key, repo)
                call_counts[key] += 1

            repeats = sum(max(0, count - 1) for count in call_counts.values())
            if repeats > 0:
                total_penalty = penalty_per_repeat * repeats
                score = max(0.0, score - total_penalty)

        return VerificationResult(score=score)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: Any, query: Optional[str] = None
    ) -> VerificationResult:
        """
        Synchronously verify code search operations.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "Cannot call synchronous __call__ method from within an async context. Use async_call instead."
                )
            else:
                return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))
        except RuntimeError:
            return asyncio.run(self.async_call(tokenized_prediction, prediction, label, query))

    @classmethod
    def get_config_class(cls) -> type:
        """
        Return the configuration class for this verifier.

        Returns:
            type: The VerifierConfig class or its subclass
        """
        return CodeAgentVerifierConfig

def build_all_verifiers(args) -> Dict[str, VerifierFunction]:
    """
    Build all verifiers with the given judge config.
    """
    verifiers: Dict[str, VerifierFunction] = {}
    for subclass in VerifierFunction.__subclasses__():
        if subclass == LMJudgeVerifier:
            continue

        verifier_config = subclass.get_config_class().from_args(args)
        instance = subclass(verifier_config)
        verifiers[instance.name.lower()] = instance

        # add the code_stdio verifier
        if subclass == CodeVerifier:
            stdio_config = copy.deepcopy(verifier_config)
            stdio_config.code_api_url = stdio_config.code_api_url.replace("/test_program", "/test_program_stdio")
            instance = CodeVerifier(stdio_config)
            instance.name = "code_stdio"
            verifiers["code_stdio"] = instance

    for judge_type in JUDGE_PROMPT_MAP.keys():
        instance = LMJudgeVerifier(judge_type, LMJudgeVerifierConfig.from_args(args))
        verifiers[instance.name.lower()] = instance

    # if we have remap arg, remap!
    if args.remap_verifier:
        remap = args.remap_verifier.split("=")
        assert len(remap) == 2, "Remap must be in the format old_name=new_name"
        old_name, new_name = remap
        # map so that the old name calls the new verifier
        assert new_name.lower() in verifiers, f"{new_name} not found in verifiers during remapping"
        verifiers[old_name.lower()] = verifiers[new_name.lower()]

    return verifiers


# special case, we use this outside our general verifier loop.
def soft_format_reward_func(responses: List[str], reward_scale: float = 1.0) -> List[float]:
    """
    Check if the completion has a specific format defined by a pattern.

    Returns a list of rewards scaled by reward_scale.
    """
    pattern = r".*?</think>\s*<answer>.*?</answer>"
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [reward_scale if match else 0.0 for match in matches]


async def cleanup_all_llm_judge_clients():
    """
    Cleanup function to properly close all LLM judge clients before shutdown.
    """
    await LMJudgeVerifier.cleanup_all_clients()
