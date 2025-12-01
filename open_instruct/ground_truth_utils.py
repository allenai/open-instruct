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
import string
import weakref
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import requests
from litellm import acompletion

from open_instruct import logger_utils
from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.IFEvalG import instructions_registry
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
class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: str | None = None


@dataclass
class MaxLengthVerifierConfig(VerifierConfig):
    max_length_verifier_max_length: int


class VerifierFunction(ABC):
    """
    Base class for all verifier functions that evaluate model predictions against ground truth.

    Each verifier function takes a prediction and compares it to a ground truth label,
    returning a VerificationResult with a score between 0.0 and 1.0.
    """

    def __init__(self, name: str, weight: float = 1.0, verifier_config: VerifierConfig | None = None) -> None:
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
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
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
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("gsm8k", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("strict_math", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("ifeval", weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str | dict, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("ifeval_old", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str | dict, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("flan", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
    ) -> VerificationResult:
        answer_string = prediction.split("The answer is: ")[-1].strip()
        score = float(normalize_answer(answer_string) == normalize_answer(label))
        return VerificationResult(score=score)


class StringMatcherVerifier(VerifierFunction):
    """
    Verifier for tasks that require string matching.

    It checks if the model output matches the ground truth answer.
    """

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("string_matcher", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("string_f1", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("puzzle", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        self.answer_start_tag = "<finish>"
        self.answer_end_tag = "</finish>"
        super().__init__("re_search_f1", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__(name="re_search", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str | list[str], query: str | None = None
    ) -> VerificationResult:
        # 1. Parse JSON label safely
        parsed_labels: list | str
        try:
            parsed = json.loads(label)
            parsed_labels = parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat label as raw string or list-of-strings
            parsed_labels = label if isinstance(label, list) else [str(label).strip()]

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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("max_length", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
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

    def __init__(self, verifier_config: VerifierConfig | None = None) -> None:
        super().__init__("up_to_max_length", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str | None = None
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
        self, tokenized_prediction: list[int], prediction: str, label: str, query: str
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
                if context_check_available and not check_context_window_limit(
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

    def __call__(self, tokenized_prediction: list[int], prediction: str, label: str, query: str) -> VerificationResult:
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
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
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
        self, tokenized_prediction: list[int], prediction: str, label: Any, query: str | None = None
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


def build_all_verifiers(args) -> dict[str, VerifierFunction]:
    """
    Build all verifiers with the given judge config.
    """
    verifiers: dict[str, VerifierFunction] = {}
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

    for judge_type in JUDGE_PROMPT_MAP:
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
def soft_format_reward_func(responses: list[str], reward_scale: float = 1.0) -> list[float]:
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


async def apply_verifiable_reward(
    reward_fn_mapping: dict[str, VerifierFunction],
    responses: list,
    decoded_responses: list[str],
    ground_truths: list,
    datasets: list[str],
    reward_mult: int = 10,
    queries: list[str] | None = None,
):
    if queries is None:
        queries = [None] * len(responses)

    async_tasks = []
    task_metadata = []

    for i, (tok_prediction, prediction, ground_truth, dataset, query) in enumerate(
        zip(responses, decoded_responses, ground_truths, datasets, queries)
    ):
        ground_truth_list = [ground_truth] if isinstance(ground_truth, str) else ground_truth
        dataset_list = [dataset] if isinstance(dataset, str) else dataset
        assert len(ground_truth_list) == len(dataset_list), "Ground truth and dataset list lengths do not match."

        for gt, ds in zip(ground_truth_list, dataset_list):
            reward_func = reward_fn_mapping.get(ds.lower())
            if reward_func is None:
                logger.warning("No reward function found for dataset %s. Skipping reward.", ds)
                continue

            task = reward_func.async_call(
                tokenized_prediction=tok_prediction, prediction=prediction, label=gt, query=query
            )
            async_tasks.append(task)
            task_metadata.append(
                {
                    "response_idx": i,
                    "dataset": reward_func.name,
                    "reward_weight": reward_func.weight,
                    "reward_mult": reward_mult,
                }
            )

    if async_tasks:
        reward_results = await asyncio.gather(*async_tasks)
        logger.debug(f"Applied {len(reward_results)} ground truth rewards in parallel")
    else:
        reward_results = []

    response_rewards = [0] * len(responses)
    response_per_func_rewards = [{} for _ in range(len(responses))]

    for result, metadata in zip(reward_results, task_metadata):
        response_idx = metadata["response_idx"]
        dataset = metadata["dataset"]
        reward_weight = metadata["reward_weight"]
        reward_mult = metadata["reward_mult"]

        score = result.score if hasattr(result, "score") else result
        weighted_reward = reward_mult * score * reward_weight

        response_rewards[response_idx] += weighted_reward
        response_per_func_rewards[response_idx][dataset] = (
            response_per_func_rewards[response_idx].get(dataset, 0) + weighted_reward
        )

    return response_rewards, response_per_func_rewards


@dataclass
class RewardConfig:
    """Configuration for reward function computation."""

    apply_r1_style_format_reward: bool = False
    r1_style_format_reward: float = 1.0
    apply_verifiable_reward: bool = True
    verification_reward: int = 10
    non_stop_penalty: bool = False
    non_stop_penalty_value: float = -10.0
    only_reward_good_outputs: bool = False
    additive_format_reward: bool = False
    verifier_functions: dict[str, VerifierFunction] = field(default_factory=dict)

    def build(self) -> Callable:
        """Build and return the reward function."""

        async def reward_fn(
            responses: list,
            decoded_responses: list[str],
            ground_truths: list[Any],
            datasets: list[str],
            finish_reasons: list[str],
            infos,
            queries: list[str] | None = None,
        ) -> tuple[list[float], dict[str, Any]]:
            timeouts = infos.timeouts
            tool_errors = infos.tool_errors
            tool_outputs = infos.tool_outputs
            tool_calleds = infos.tool_calleds
            good_outputs = [
                len(tool_outputs[i]) > 0 and tool_calleds[i] and not timeouts[i] and not tool_errors[i]
                for i in range(len(tool_outputs))
            ]
            scores = [0.0] * len(decoded_responses)
            metrics: dict[str, Any] = {}
            format_scores: list[float] = []

            if self.apply_r1_style_format_reward:
                format_scores = soft_format_reward_func(decoded_responses, self.r1_style_format_reward)
                if len(format_scores) != len(scores):
                    raise ValueError(f"{len(format_scores)=} != {len(scores)=}")
                for i in range(len(format_scores)):
                    scores[i] = format_scores[i] + scores[i]
                metrics["val/format_scores"] = np.array(format_scores).mean()

            if self.apply_verifiable_reward:
                verifiable_rewards, per_func_rewards = await apply_verifiable_reward(
                    self.verifier_functions,
                    responses,
                    decoded_responses,
                    ground_truths,
                    datasets,
                    reward_mult=self.verification_reward,
                    queries=queries,
                )
                if len(verifiable_rewards) != len(scores):
                    raise ValueError(f"{len(verifiable_rewards)=} != {len(scores)=}")
                for i in range(len(verifiable_rewards)):
                    if not self.only_reward_good_outputs or (good_outputs[i] and self.only_reward_good_outputs):
                        if self.apply_r1_style_format_reward and self.additive_format_reward:
                            scores[i] = verifiable_rewards[i] + scores[i]
                        elif self.apply_r1_style_format_reward and not self.additive_format_reward:
                            scores[i] = verifiable_rewards[i] if format_scores[i] == 1 else 0
                        else:
                            scores[i] = verifiable_rewards[i]
                np_verifiable_rewards = np.array(verifiable_rewards)
                metrics["objective/verifiable_reward"] = np_verifiable_rewards.mean()
                metrics["objective/verifiable_correct_rate"] = (np_verifiable_rewards > 0.0).mean()
                per_func_lists: dict[str, list] = defaultdict(list)
                for reward_dict in per_func_rewards:
                    for key, value in reward_dict.items():
                        per_func_lists[key].append(value)
                for key, value in per_func_lists.items():
                    np_value = np.array(value)
                    metrics[f"objective/{key}_reward"] = np_value.mean()
                    metrics[f"objective/{key}_correct_rate"] = (np_value > 0.0).mean()

            if self.non_stop_penalty:
                assert len(finish_reasons) == len(scores)
                for i in range(len(finish_reasons)):
                    if finish_reasons[i] != "stop":
                        scores[i] = self.non_stop_penalty_value

            return scores, metrics

        return reward_fn
