import json
import logging
import re

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union, Any

from judges.voting_methods import AVAILABLE_VOTING_METHODS

from judges._client import async_get_completion, llm_client
import easyapi
import asyncio
import openai

if TYPE_CHECKING:
    import pydantic

litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.disabled = True


@dataclass
class Judgment:
    """
    A dataclass that represents a judgment.

    Attributes:
    -----------
    score: bool | int | str
        The score assigned by the judge, indicating the evaluation result.
    reasoning: str
        The reasoning provided by the judge for the assigned score.
    """

    score: bool | int | str
    reasoning: str
    cost: float = None
    response_time: float = None

    def __post_init__(self):
        """
        Post-initialization to normalize score values for consistency.
        """
        if isinstance(self.score, str):
            if self.score.lower() in ["yes", "true", "1", "good"]:
                self.score = True
            elif self.score.lower() in ["no", "false", "0", "bad"]:
                self.score = False
        elif isinstance(self.score, int):
            self.score = bool(self.score)


@dataclass
class Verdict:
    """
    A dataclass that represents a jury's verdict.

    Attributes:
    -----------
    score: bool | int
        The aggregated score determined by the jury.
    judgments: list[Judgment]
        A list of individual judgments from all judges in the jury.
    """

    score: bool | int
    judgments: list[Judgment] = None


@dataclass
class BaseJudge:
    """
    Base class for all judges. All judges must implement a judge method which
    produces a `Verdict` object that contains a score and reasoning.

    Attributes:
    -----------
    model: str
        The model used by the judge for evaluation.
    """

    def __init__(
        self,
        model: str,
        local_model: bool
    ):
        """
        Initialize the judge with a specific model.

        Parameters:
        -----------
        model: str
            The model identifier to be used for evaluations.
        """
        self.model = model
        if local_model:
            self.api = easyapi.Api()
        else:
            self.api = None

    def _build_messages(self, user_prompt: str, system_prompt: Optional[str] = None):
        """
        Build the message payload for the model evaluation.

        Parameters:
        -----------
        user_prompt: str
            The input prompt for the user.
        system_prompt: Optional[str]
            The optional system-level prompt.

        Returns:
        --------
        list[dict]:
            The list of messages to be sent to the model.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # add json format expectation to the user prompt:
        user_prompt += (
            '\nRespond in JSON format. {"REASONING": "[...]", "SCORE": "<your-score>"}'
        )

        messages.append({"role": "user", "content": user_prompt})
        return messages

    async def _judge(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        client: Optional[Union[openai.AsyncOpenAI, Any]] = None
    ):
        """
        Perform the judgment process using the configured model.

        Parameters:
        -----------
        user_prompt: str
            The input prompt for the user.
        system_prompt: Optional[str]
            The optional system-level prompt.
        client: Optional[Union[openai.AsyncOpenAI, Any]]
            The client to use for the model evaluation.

        Returns:
        --------
        tuple:
            The reasoning and score extracted from the model's response.
        """
        messages = self._build_messages(user_prompt, system_prompt)

        completion = await async_get_completion(
            model=self.model,
            messages=messages,
            max_tokens=2048,
            temperature=1,
            seed=None,
            response_model=None,
            response_format={"type": "json_object"},
            easyapi=self.api,
            client=client
        )
        # breakpoint()
        if completion:
            try:
                data = json.loads(completion.choices[0].message.content)
                reasoning = data.get("REASONING", "")
                score = data.get("SCORE", 0.0)
            except json.JSONDecodeError:
                print(
                    f"Model did not return a valid JSON response. Response: {completion.choices[0].message.content}")
                reasoning = ""
                score = extract_score_from_string(completion.choices[0].message.content) if isinstance(completion.choices[0].message.content, str) else 0.0
            except AttributeError:
                print(f"Model returned JSON without REASONING or SCORE. Response: {completion.choices[0].message.content}")
                reasoning = ""
                score = extract_score_from_string(completion.choices[0].message.content) if isinstance(completion.choices[0].message.content, str) else 0.0
        else:
            print("No completion received from the model.")
            reasoning = ""
            score = 0.0

        # check if cost/response_time is available
        try:
            cost = completion.cost
            response_time = completion.response_time
        except AttributeError:
            # set to some average cost score
            print("no api cost or response time attribute found")
            cost = 0.0001
            response_time = 0.0
            pass

        return reasoning, score, cost, response_time

    @abstractmethod
    def judge(
        self,
        input: str,
        output: Optional[str] = None,
        expected: Optional[str] = None,
    ) -> Judgment:
        """
        Judge the input and return a Judgment.

        Parameters:
        -----------
        input: str
            The input provided to the model to be judged.
        output: str
            The output generated by the model.
        expected: str
            The output that the model was expected to generate.

        Returns:
        --------
        Judgment:
            The evaluation result containing the score and reasoning.
        """
        raise NotImplementedError("all judges must implement a judge method")


@dataclass
class Jury:
    r"""
    A jury is a set of judges that averages or takes the mode of all the scores.

    Attributes:
    -----------
    judges: list[BaseJudge]
        A list of judges that form the jury.
    voting_method: str
        The voting method used to aggregate scores, such as "majority".

    @misc{verga2024replacingjudgesjuriesevaluating,
        title={Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models},
        author={Pat Verga and Sebastian Hofstatter and Sophia Althammer and Yixuan Su and Aleksandra Piktus and Arkady Arkhangorodsky and Minjie Xu and Naomi White and Patrick Lewis},
        year={2024},
        eprint={2404.18796},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2404.18796},
    }
    """

    def __init__(self, judges: list[BaseJudge], voting_method: str = "majority"):
        """
        Initialize the jury with a set of judges and a voting method.

        Parameters:
        -----------
        judges: list[BaseJudge]
            A list of judges to include in the jury.
        voting_method: str
            The voting method to use for aggregating judgments (default is "majority").
        """
        self.judges = judges
        self.voting_method = AVAILABLE_VOTING_METHODS[voting_method]

    def vote(
        self,
        input: str,
        output: Optional[str] = None,
        expected: Optional[str] = None,
    ) -> Verdict:
        """
        Aggregate the judgments of all judges to produce a final verdict.

        Parameters:
        -----------
        input: str
            The input provided to the models for judgment.
        output: str
            The output generated by the model.
        expected: str
            The expected output for comparison.

        Returns:
        --------
        Verdict:
            The final verdict containing the aggregated score and individual judgments.
        """
        judgments = []
        for judge in self.judges:
            judgment = judge.judge(
                input=input,
                output=output,
                expected=expected,
            )

            judgments.append(judgment)

        scores = [judgment.score for judgment in judgments]
        score = self.voting_method(scores=scores)
        return Verdict(score=score, judgments=judgments)


def extract_score_from_string(score_str: str) -> float:
    """Extract numerical score from string response.""" 
    # Handle rating formats like "4/5"
    ratio_matches = re.findall(r'(\d+)\/(\d+)', score_str)
    if ratio_matches:
        numerator, denominator = ratio_matches[0]
        return min(float(numerator) / float(denominator), 10)
       
    # Try to handle percentage expressions
    percent_matches = re.findall(r'(\d+\.?\d*|\.\d+)%', score_str)
    if percent_matches:
        return min(float(percent_matches[0]) / 100.0, 10)

    
    # Try to find numerical values in the string
    matches = re.findall(r'(\d+\.?\d*|\.\d+)', score_str)
    if matches:
        return min(float(matches[0]), 10)
    
    # If parsing fails, check for binary indicators
    if any(word in score_str.lower() for word in ["yes", "correct", "good", "true", "pass"]):
        return 10.0
    elif any(word in score_str.lower() for word in ["no", "incorrect", "bad", "false", "fail"]):
        return 0.0
    else:
        print(f"Could not parse score from: {score_str}, defaulting to 0.0")
        return 0.0