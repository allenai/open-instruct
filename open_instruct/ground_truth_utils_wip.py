"""
Collection of 'ground truth rewards' for different datasets/tasks.
Used to give feedback to the model based on the ground truth answer.
Add new verifiers by subclassing VerifierFunction and implementing the __call__ method.
They are then automatically added to the REWARD_FN_MAPPING.
"""

import json
import logging
import re
import os
import string
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union, Optional, Tuple, Literal

from openai import OpenAI
from textwrap import dedent
from judges.base import BaseJudge, Judgment
from judge_prompts import JUDGE_PROMPT_MAP
from judge_utils import (
    JudgeQuality, 
    JudgeFactuality, 
    JudgeRelevance, 
    JudgeCorrectness, 
    JudgeGroundedness, 
    JudgeHarmfulness, 
    extract_final_answer
)
from judge_utils import JUDGE_CLASS_MAP, extract_score_from_string

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

# define openai client reading openai_api-key from env
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    

# class LMJudgeVerifier(VerifierFunction):
#     """
#     Verifier for general tasks that delegates evaluation to LLM Judges (classifier or grader)
#     specified in the label.

#     The rubrics may be a JSON string or a dictionary containing criteria based on which
#     the response can be evaluated.
#     """

#     def __init__(self) -> None:
#         super().__init__("general", weight=1.0)

#     def __call__(self, query: str, prediction: str, label: Union[str, List], subtask: str = None) -> bool:
#         """
#         Evaluate the given prediction against the ground truth (and an optional rubric).

#         Args:
#             query (str): The user query.
#             prediction (str): The model output.
#             label (Any): The ground truth answer or evaluation rubric.
#             subtask (str): Optional subtask name for the selecting evaluation prompt.

#         Returns:
#             int: Reward score. Can be binary (0/1) or continuous.
#         """
#         answer = prediction.split("<|assistant|>\n")[-1].strip()
#         if subtask is None:
#             subtask = "general"
#         if subtask not in JUDGE_PROMPT_MAP:
#             raise ValueError(f"Not found the subtask prompt: {subtask}")
        
#         user_prompt = JUDGE_PROMPT_MAP[subtask]
#         user_prompt = dedent(user_prompt).format(input=query, output=answer)
        

    
#     def get_judgement(
#         self,
#         judge_prompt: str,
#         input: str,
#         output: str = None,
#         expected: str = None,
#     ) -> Judgment:
#         """
#         Judge the input and return a verdict.
#         """
#         system_prompt = None
#         user_prompt = dedent(judge_prompt)
#         # select the judge class based on the subtask


#         reasoning, score, cost, response_time = self._judge(
#             user_prompt=user_prompt,
#             system_prompt=system_prompt,
#         )
#         return Judgment(reasoning=reasoning, score=score, cost=cost, response_time=response_time)

    
# class LMJudgeVerifierGrader(VerifierFunction):
#     """
#     Verifier for general tasks that delegates evaluation to LLM Judges (classifier or grader)
#     specified in the constraint.

#     The rubrics may be a JSON string or a dictionary containing criteria based on which
#     the response can be evaluated.
#     """

#     def __init__(self) -> None:
#         super().__init__("general", weight=1.0)

#     def __call__(self, tokenized_prediction: List[int], prediction: str, label: Union[str, Dict], rubric: Union[str, List] = None) -> bool:
#         ### TODO: raise notimplemented error


class LMJudgeVerifier(VerifierFunction):
    """
    Verifier for general tasks that delegates evaluation to LLM Judges (classifier or grader)
    specified in the constraint.

    The rubrics may be a JSON string or a dictionary containing criteria based on which
    the response can be evaluated.
    """

    def __init__(
        self,
        name: str = "general",
        judge_type: str = "quality",
        weight: float = 1.0,
        model_name: str = "gpt-4o-mini",
        threshold: float = 0.5,
    ) -> None:
        """
        Initialize LMJudgeVerifier with specified parameters.

        Args:
            judge_type (str): Type of judge to use (e.g., "quality", "factuality", "relevance")
            name (str): Name of the verifier
            weight (float): Weight for this verifier when combined with others
            model_name (str): LLM model to use for judging
            threshold (float): Threshold for binary decisions (0/1)
        """
        super().__init__(name, weight)
        self.judge_type = judge_type
        self.model_name = model_name
        self.threshold = threshold
        # self.judge_map = {
        #     "quality": JudgeQuality,
        #     # Add more judge types here as needed
        #     "factuality": JudgeFactuality,
        #     "relevance": JudgeRelevance,
        #     "correctness": JudgeCorrectness,
        #     "groundedness": JudgeGroundedness,
        #     "harmfulness": JudgeHarmfulness,
        # }
        
        if judge_type not in JUDGE_CLASS_MAP:
            raise ValueError(f"Unsupported judge type: {judge_type}. Available types: {list(JUDGE_CLASS_MAP.keys())}")
            
        self.judge = JUDGE_CLASS_MAP[judge_type](model=model_name, judge_type=judge_type)

    def __call__(
        self, 
        tokenized_prediction: List[int], 
        prediction: str, 
        label: Any,
        query: Optional[str] = None,
        # criteria: Optional[Union[str, List]] = None
    ) -> float:
        """
        Evaluate the given prediction against criteria using an LLM judge.

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused)
            prediction (str): The model output
            label (Any): Ground truth or evaluation criteria
            query (str, optional): Original user query
            subtask (str, optional): Specific subtask for evaluation
            # criteria (Union[str, Dict], optional): Additional evaluation criteria

        Returns:
            float: Score between 0 and 1
        """
        # Clean prediction if needed

        answer = extract_final_answer(prediction)
        # answer = prediction.split("<|assistant|>\n")[-1].strip() if "<|assistant|>" in prediction else prediction.strip()
        
        # # Determine proper prompt template
        # if self.judge_type is None:
        #     subtask = "general"
        if self.judge_type not in JUDGE_PROMPT_MAP:
            logger.warning(f"Subtask prompt not found: {self.judge_type}, falling back to general")
            self.judge_type = "general"
        
        # # Format judge prompt with appropriate inputs
        # judge_prompt = JUDGE_PROMPT_MAP[self.judge_type]
        
        # # Handle criteria if provided
        # if criteria:
        #     if isinstance(criteria, str):
        #         criteria_str = json.loads(criteria, indent=2)
        #     else:
        #         criteria_str = criteria
            
        #     # Add criteria to prompt if template supports it
        #     if "{rubric}" in judge_prompt:
        #         judge_prompt = dedent(judge_prompt).format(
        #             input=query, 
        #             output=answer,
        #             criteria=criteria_str
        #         )
        #     else:
        #         # Append criteria to prompt if no placeholder exists
        #         judge_prompt = dedent(judge_prompt).format(
        #             input=query, 
        #             output=answer
        #         )
        #         judge_prompt += f"\n\nPlease evaluate based on the following criteria: {criteria_str}"
        # else:
        #     judge_prompt = dedent(judge_prompt).format(
        #         input=query, 
        #         output=answer
        #     )

        
            
        # Get judgment from appropriate judge
        judgment = self.get_judgement(
            input=query,
            output=answer,
            expected=label if isinstance(label, str) else json.dumps(label)
        )
        
        # Convert score to binary if needed or return raw score
        if isinstance(judgment.score, (int, float)):
            score = judgment.score
        elif isinstance(judgment.score, str):
            score = extract_score_from_string(judgment.score)
        else:
            logger.warning(f"Unexpected score type: {type(judgment.score)}, defaulting to 0.0")
            score = 0.0

        return score, judgment.cost, judgment.response_time
    
    async def async_call(
        self, 
        tokenized_prediction: List[int], 
        prediction: str, 
        label: Any,
        query: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """
        Async version of __call__ that directly awaits _judge without using asyncio.run()

        Args:
            tokenized_prediction (List[int]): Tokenized representation (unused)
            prediction (str): The model output
            label (Any): Ground truth or evaluation criteria
            query (str, optional): Original user query

        Returns:
            Tuple[float, float, float]: (score, cost, response_time)
        """
        # Clean prediction if needed
        answer = extract_final_answer(prediction)

        # Ensure judge_type is valid
        if self.judge_type not in JUDGE_PROMPT_MAP:
            logger.warning(f"Subtask prompt not found: {self.judge_type}, falling back to general")
            self.judge_type = "general"
        
        # Get user prompt (this is what get_judgement would format)
        user_prompt = JUDGE_PROMPT_MAP[self.judge_type]
        user_prompt = dedent(user_prompt).format(
            input=query,
            output=answer,
            label=label if isinstance(label, str) else json.dumps(label)
        )
        
        # Directly await the _judge method instead of using asyncio.run
        reasoning, score, cost, response_time = await self.judge._judge(
            user_prompt=user_prompt,
            system_prompt=None
        )
        
        # Process the score same as in __call__
        if isinstance(score, (int, float)):
            final_score = score
        elif isinstance(score, str):
            final_score = extract_score_from_string(score)
        else:
            logger.warning(f"Unexpected score type: {type(score)}, defaulting to 0.0")
            final_score = 0.0
        
        return final_score, cost, response_time

        
    def get_judgement(
        self,
        input: str,
        output: str = None,
        expected: str = None,
        judge_prompt: str = None,
    ) -> Judgment:
        """
        Judge the input and return a verdict.
        
        Args:
            judge_prompt (str): The prompt template for the judge
            input (str): Original user input/query
            output (str, optional): Model's response to evaluate
            expected (str, optional): Ground truth or evaluation criteria
            
        Returns:
            Judgment: Object containing reasoning and score
        """
        system_prompt = None  # Using default system prompt from the judge
        
        # Pass all available information to the judge
        judgment = self.judge.judge(
            input=input,
            output=output,
            expected=expected,
            user_prompt=judge_prompt,
        )
        
        return judgment   
    
    def _extract_score_from_string(self, score_str: str) -> float:
        """Extract numerical score from string response."""
        if isinstance(score_str, bool):
            return float(score_str)
        # Try to handle percentage expressions
        percent_matches = re.findall(r'(\d+\.?\d*|\.\d+)%', score_str)
        if percent_matches:
            return float(percent_matches[0]) / 100.0
            
        # Handle rating formats like "4/5"
        ratio_matches = re.findall(r'(\d+)\/(\d+)', score_str)
        if ratio_matches:
            numerator, denominator = ratio_matches[0]
            return float(numerator) / float(denominator)
        
        # Try to find numerical values in the string
        matches = re.findall(r'(\d+\.?\d*|\.\d+)', score_str)
        if matches:
            return float(matches[0])
    
            
        raise ValueError(f"Could not extract score from string: {score_str}")     
        


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
