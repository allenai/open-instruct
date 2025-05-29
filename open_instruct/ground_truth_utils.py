"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
Add new verifiers by subclassing VerifierFunction and implementing the __call__ method.
They are then automatically added to the REWARD_FN_MAPPING.
"""

import asyncio
import json
import logging
import os
import re
import string
import weakref
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from openai import AsyncAzureOpenAI

from open_instruct.if_functions import IF_FUNCTIONS_MAP
from open_instruct.judge_utils import (
    JUDGE_PROMPT_MAP,
    PRICE_PER_TOKEN,
    build_messages,
    extract_score_from_string,
)
from open_instruct.math_utils import (
    get_unnormalized_answer,
    hendrycks_is_equiv,
    is_equiv,
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)

logger = logging.getLogger(__name__)


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
    llm_judge_temperature: float
    seed: int


@dataclass
class VerificationResult:
    score: float
    cost: float = 0.0
    reasoning: Optional[str] = None


class VerifierFunction(ABC):
    """
    Abstract base class for verifier functions.

    Each verifier is initialized with a name and a weight (default 1.0).
    The __call__ method must be implemented by subclasses.
    """

    def __init__(self, name: str, weight: float = 1.0, verifier_config: Optional[VerifierConfig] = None) -> None:
        self.name = name
        self.weight = weight
        self.verifier_config = verifier_config

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

    The constraint may be a JSON string or a dictionary containing a key
    'func_name' used to lookup the evaluation function.
    """

    def __init__(self, verifier_config: Optional[VerifierConfig] = None) -> None:
        super().__init__("ifeval", verifier_config=verifier_config, weight=1.0)

    def __call__(
        self, tokenized_prediction: List[int], prediction: str, label: Union[str, Dict], query: Optional[str] = None
    ) -> VerificationResult:
        constraint = label
        answer = prediction.split("<|assistant|>\n")[-1].strip()
        if isinstance(constraint, str):
            constraint = json.loads(constraint)
        if "func_name" not in constraint:
            logger.warning("Constraint missing 'func_name': %s", constraint)
            return VerificationResult(score=0.0)
        func_name = constraint.pop("func_name")
        func = IF_FUNCTIONS_MAP[func_name]
        non_none_args = {k: v for k, v in constraint.items() if v is not None}
        if not constraint:
            return VerificationResult(score=float(func(prediction)))
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
        print(f"label: {label}")
        # extract answer
        if self.answer_start_tag not in prediction and self.answer_end_tag not in prediction:
            return VerificationResult(score=0.0)
        answer_string = prediction.split(self.answer_start_tag)[-1].split(self.answer_end_tag)[0]
        # check answer non-empty
        if not answer_string:
            return VerificationResult(score=0.0)
        # if label is list, max over labels
        if isinstance(label, list):
            f1 = max(f1_score(answer_string, lab)["f1"] for lab in label)
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
        score = 1 - (length_diff / 8192)
        return VerificationResult(score=score)


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
        score = 1 - (length_diff / 8192)
        return VerificationResult(score=score)


class LMJudgeVerifier(VerifierFunction):
    """
    Verifier that uses a language model's judgement to score a response.
    """

    # Use WeakKeyDictionary to automatically clean up clients when event loops are garbage collected
    _client_cache = weakref.WeakKeyDictionary()

    def __init__(self, judge_type: str, verifier_config: LMJudgeVerifierConfig) -> None:
        super().__init__(f"general-{judge_type}", verifier_config=verifier_config, weight=1.0)
        self.prompt_template = JUDGE_PROMPT_MAP[judge_type]

    def _get_client(self):
        """
        Get or create an AsyncAzureOpenAI client for the current event loop.
        Uses WeakKeyDictionary to avoid cleanup issues when event loops close.
        """
        try:
            loop = asyncio.get_running_loop()

            # Use the loop object itself as the key (WeakKeyDictionary will clean up automatically)
            if loop not in self._client_cache:
                self._client_cache[loop] = AsyncAzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version="2024-12-01-preview",
                )

            return self._client_cache[loop]

        except RuntimeError:
            # No event loop running, create a fresh client (this should be rare)
            return AsyncAzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-12-01-preview",
            )

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
            content = completion.choices[0].message.content

            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    reasoning = data.get("REASONING", "")
                    score = float(data.get("SCORE", 0.0))
                    return reasoning, score
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

            if isinstance(content, str):
                reasoning = content
                extracted_score = extract_score_from_string(content)
                if extracted_score is not None:
                    score = extracted_score

        except Exception as e:
            print(f"Error processing model response: {str(e)}")
            if hasattr(completion, "choices") and completion.choices is not None and len(completion.choices) > 0:
                print(f"Response content: {getattr(completion.choices[0].message, 'content', 'No content available')}")

        return reasoning, score

    def get_cost(self, response, model: str):
        """
        Get the cost of the response.
        """
        model_name = model.replace("-standard", "")  # azure OAI models have -standard in the name
        return (
            PRICE_PER_TOKEN[model_name]["input"] * response.usage.prompt_tokens
            + PRICE_PER_TOKEN[model_name]["output"] * response.usage.completion_tokens
        )

    async def async_call(
        self, tokenized_prediction: List[int], prediction: str, label: str, query: str
    ) -> VerificationResult:
        """
        Asynchronous version of __call__ that properly handles the async OpenAI client.
        """
        client = self._get_client()

        prompt = self.prompt_template.format(input=query, output=prediction, label=label)

        max_retries = 3 # for rate limits
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                messages = build_messages(prompt)
                response = await client.chat.completions.create(
                    model=self.verifier_config.llm_judge_model,
                    messages=messages,
                    temperature=self.verifier_config.llm_judge_temperature,
                    max_completion_tokens=self.verifier_config.llm_judge_max_tokens,
                    seed=self.verifier_config.seed,
                )
                reasoning, score = self.parse_completion(response)
                cost = self.get_cost(response, self.verifier_config.llm_judge_model)
                return VerificationResult(score=score, cost=cost, reasoning=reasoning)

            except Exception as e:
                logger.warning(f"LLM judge attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"LLM judge failed after {max_retries} attempts. Returning default score of 0.0")
                    return VerificationResult(score=0.0, cost=0.0, reasoning=f"Error: {str(e)}")
                else:
                    await asyncio.sleep(retry_delay * (2**attempt))  # Exponential backoff

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


def build_all_verifiers(args) -> Dict[str, VerifierFunction]:
    """
    Build all verifiers with the given judge config.
    """
    verifiers: Dict[str, VerifierFunction] = {}
    for subclass in VerifierFunction.__subclasses__():
        if subclass == LMJudgeVerifier:
            continue
        instance = subclass(VerifierConfig.from_args(args))
        verifiers[instance.name.lower()] = instance

    for judge_type in JUDGE_PROMPT_MAP.keys():
        instance = LMJudgeVerifier(judge_type, LMJudgeVerifierConfig.from_args(args))
        verifiers[instance.name.lower()] = instance

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
