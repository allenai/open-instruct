import csv
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from textwrap import dedent
from typing import Optional, Union, List, Dict

from pydantic import BaseModel

from judges.classifiers.auto._metrics import (
    confusion_matrix as cm_func,
    calculate_accuracy,
    calculate_precision,
    calculate_recall,
    calculate_f1,
    calculate_kappa,
)
from judges.classifiers.auto._prompts import (
    GENERATE_RUBRIC_USER_PROMPT,
    FORMAT_RUBRIC_USER_PROMPT,
    STRUCTURE_FEEDBACK_USER_PROMPT,
)
from judges.base import BaseJudge, Judgment
from judges._client import get_completion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("autojudge")


class GradingNote(BaseModel):
    SCORE: bool
    REASONING: str


class AutoJudge(BaseJudge):
    """
    A class to process AI-generated responses, generate grading notes using an LLM,
    evaluate responses, and compute evaluation metrics.
    """

    def __init__(
        self,
        model: str = "gpt-4-turbo-2024-04-09",
        max_workers: int = 2,
        system_prompt: Optional[str] = '',
        user_prompt: Optional[str] = None,
        save_to_disk: bool = True,
    ):
        """
        Initializes the AutoJudge with the specified model and directories.
        """
        super().__init__(model=model)
        self.max_workers = max_workers
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.save_to_disk = save_to_disk

    @staticmethod
    def load_data(
        dataset: Union[
            str,
            Path,
            List[Dict[str, Union[str, int]]],
        ],
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Loads the dataset from a file or a list of dictionaries and validates required columns.
        """
        logger.info("loading dataset...")
        required_columns = {"label", "feedback", "input", "output"}
        data = []
        try:
            if isinstance(dataset, (str, Path)):
                dataset_path = Path(dataset)
                with dataset_path.open("r", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    data = [row for row in reader]
            elif isinstance(dataset, list):
                data = dataset
            else:
                raise TypeError("Dataset must be a file path or a list of dictionaries")

            # Check for required columns
            if not data or not all(required_columns.issubset(row.keys()) for row in data):
                raise ValueError(f"Dataset is missing one or more required columns: {required_columns}")

            logger.info(f"data loaded successfully with {len(data)} record(s)")
        except Exception as e:
            logger.error(f"error loading dataset: {e}")
            raise

        return data


        return data

    @staticmethod
    def aggregate_feedback(data: List[Dict[str, Union[str, int]]]) -> str:
        """
        Aggregates feedback from bad data points into a single formatted string.
        """
        try:
            logger.info("aggregating feedback from bad data points")
            bad_data_points = [
                row
                for row in data
                if str(row.get("label", "0")).strip() == "0" and row.get("feedback")
            ]
            if not bad_data_points:
                logger.warning("no bad data points found with label=0.")
                return ""

            aggregated_feedback = " - " + "\n - ".join(
                row["feedback"] for row in bad_data_points
            )
            logger.debug(f"aggregated Feedback: {aggregated_feedback}")
        except Exception as e:
            logger.error(f"error aggregating feedback: {e}")
            raise
        
        return aggregated_feedback

    def generate_structured_feedback(self, task: str, feedback: str) -> str:
        try:
            logger.info(f"generating structured feedback using {self.model}...")

            formatted_prompt = STRUCTURE_FEEDBACK_USER_PROMPT.format(
                task=task,
                feedback=feedback,
            )
            logger.debug(f"Generated prompt: {formatted_prompt}")

            messages = [
                {"role": "system", "content": ''},
                {"role": "user", "content": formatted_prompt},
            ]
            
            completion = get_completion(
                messages=messages,
                model=self.model,
                temperature=0.1,
                seed=42,
                max_tokens=1024,
                response_model=None,
            )
            
            if not completion or not completion.choices:
                raise ValueError("Empty response from LLM.")

            print("-----structured feedback:\n\n", completion.choices[0].message.content.strip())

            structured_feedback = completion.choices[0].message.content.strip()
            logger.debug(f"structured Feedback: {structured_feedback}")
        except Exception as e:
            logger.error(f"error generating structured feedback: {e}")
            logger.error(f"Prompt causing error: {messages}")
            raise

        return structured_feedback


    def generate_grading_notes(
        self,
        structured_feedback: str,
        task: str,
    ) -> str:
        """
        Generates grading notes using the LLM based on structured feedback.
        """
        try:
            logger.info(f"generating grading notes using {self.model}...")

            formatted_prompt = GENERATE_RUBRIC_USER_PROMPT.format(
                feedback=structured_feedback,
                task=task,
            )

            messages = [
                {"role": "system", "content": ''},
                {"role": "user", "content": formatted_prompt},
            ]

            print("------rubric prompt:\n", formatted_prompt)
            # Generate the raw rubric
            completion = get_completion(
                messages=messages,
                model=self.model,
                temperature=0,
                max_tokens=1024,
                seed=42,
                response_model=None,
            )
            rubric = completion.choices[0].message.content.strip()
            print("------rubric:\n", rubric)

            # Append formatting instructions
            grading_notes = f"{rubric}\n\n{FORMAT_RUBRIC_USER_PROMPT}"

            logger.debug(f"generated Rubric: {rubric}")
            logger.debug(f"formatted Rubric: {grading_notes}")
        except Exception as e:
            logger.error(f"error generating grading notes: {e}")
            raise

        return grading_notes



    def evaluate(
        self,
        data: List[Dict[str, Union[str, int]]],
        grading_notes: str,
        max_workers: int,
    ) -> List[GradingNote]:
        errors = []
        evaluations = []

        def _evaluate(row):
            try:
                output = row.get("output", "")
                input = row.get("input", "")
                if not output:
                    raise ValueError("Empty output provided for evaluation.")

                formatted_grading_note = grading_notes.format(input=input, output=output)

                messages = [
                    {"role": "system", "content": ''},
                    {"role": "user", "content": formatted_grading_note},
                ]

                print("------evaluation prompt:\n", formatted_grading_note)
                response = get_completion(
                    messages=messages,
                    model=self.model,
                    temperature=0,
                    max_tokens=1024,
                    seed=42,
                    response_model=GradingNote,
                )

                print("------evaluation response:\n", response.choices[0].message.content.strip())

                if not response:
                    raise ValueError("Empty response from LLM.")

                return response
            except Exception as e:
                logger.error(f"Error evaluating row {row}: {e}")
                errors.append((row, e))
                grading_note = GradingNote(SCORE=False, REASONING=str(e))
                return grading_note

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_row = {executor.submit(_evaluate, row): row for row in data}
            for future in as_completed(future_to_row):
                result = future.result()
                evaluations.append(result)

        if errors:
            logger.warning(f"{len(errors)} rows failed evaluation.")
            for row, error in errors:
                logger.error(f"Error for row: {row}, Error: {error}")

        return evaluations




    @staticmethod
    def classify(evaluations: List[GradingNote]) -> List[int]:
        """
        Extracts classification results from the LLM evaluations.
        """
        logger.info("extracting classification results from LLM evaluations...")
        try:
            generated_grading_note_classifications: List[int] = []
            for eval_result in evaluations:
                classification_val = (
                    1 if getattr(eval_result, "SCORE", False) else 0
                )
                generated_grading_note_classifications.append(classification_val)
            logger.debug("classification extraction completed.")
        except Exception as e:
            logger.error(f"error extracting classifications: {e}")
            raise

        return generated_grading_note_classifications

    @staticmethod
    def compute_metrics(data, classifications: List[int]) -> Dict[str, float]:
        """
        Computes evaluation metrics comparing model predictions with human labels.
        """
        logger.info("computing evaluation metrics...")
        try:
            true_labels = [int(row["label"]) for row in data]

            logger.debug(f"true labels: {true_labels}")
            logger.debug(f"predicted labels: {classifications}")

            classes = sorted(list(set(true_labels).union(set(classifications))))
            # Call confusion_matrix function as cm_func to avoid overshadowing
            cm = cm_func(
                y_true=true_labels,
                y_pred=classifications,
                classes=classes,
            )

            # Assuming these metric functions rely on some global state or previously computed results
            metrics = {
                "cohens-kappa": calculate_kappa(
                    true_labels, classifications, classes=classes
                ),
                "accuracy": calculate_accuracy(true_labels, classifications),
                "precision": calculate_precision(
                    true_labels, classifications, confusion_matrix=cm, classes=classes
                ),
                "recall": calculate_recall(
                    true_labels, classifications, confusion_matrix=cm, classes=classes
                ),
                "f1": calculate_f1(
                    true_labels, classifications, confusion_matrix=cm, classes=classes
                ),
            }
            logger.debug(f"computed metrics: {metrics}")
        except Exception as e:
            logger.error(f"error computing metrics: {e}")
            raise

        return metrics

    @classmethod
    def from_dataset(
        cls,
        dataset: Union[
            str,
            Path,
            List[Dict[str, Union[str, int]]],
        ],
        task: str,
        model: str = "gpt-4-turbo-2024-04-09",
        max_workers: int = 2,
    ) -> "AutoJudge":
        """
        Create an instance of the AutoJudge class from a dataset and task description.

        Parameters:
        -----------
        dataset, str or Path:
            The path to the dataset file or a list of dictionaries.
        task, str:
            A description of the task to be accomplished.

        Returns:
        --------
        AutoJudge:
            An instance of the AutoJudge class.
        """
        if not dataset:
            raise ValueError("Please provide a dataset.")

        if not task:
            raise ValueError("Please describe the task you are trying to accomplish.")

        data = cls.load_data(dataset=dataset)

        if not data:
            logger.error("No data loaded. Cannot run pipeline.")
            raise ValueError("Dataset is empty.")

        aggregated_feedback = cls.aggregate_feedback(data)
        if not aggregated_feedback:
            logger.warning(
                "No feedback to process. Returning an instance without evaluation."
            )
            return cls()

        # Create an instance of AutoJudge
        autojudge = cls(
            model=model,
            max_workers=max_workers,
            system_prompt=None,
            user_prompt=None,
        )

        structured_feedback = autojudge.generate_structured_feedback(
            task=task,
            feedback=aggregated_feedback,
        )
        grading_notes = autojudge.generate_grading_notes(structured_feedback, task)

        evaluations = autojudge.evaluate(
            data=data,
            grading_notes=grading_notes,
            max_workers=autojudge.max_workers,
        )

        print("------evaluations:\n", evaluations)
        classifications = cls.classify(evaluations=evaluations)
        metrics = cls.compute_metrics(
            data=data,
            classifications=classifications,
        )

        print("------metrics:\n", metrics)

        logger.debug("sample llm outputs:")
        for eval_result in evaluations[:5]:
            logger.debug(eval_result)

        logger.debug("sample classification results (grading note):")
        classification_counts = {}
        for classification_val in classifications:
            classification_counts[classification_val] = (
                classification_counts.get(classification_val, 0) + 1
            )

        logger.debug(classification_counts)

        logger.debug("sample classification results (human labels):")
        human_labels = {}
        for row in data[:10]:
            label = int(row["label"])
            human_labels[label] = human_labels.get(label, 0) + 1

        logger.debug(human_labels)

        logger.info(f"final evaluation metrics: {metrics}")
        # Assign the grading_notes as the user_prompt to the instance for future judging
        autojudge.user_prompt = grading_notes
        autojudge._save_prompts(user_prompt=grading_notes, system_prompt="")
        return autojudge

    def _save_prompts(self, user_prompt: str, system_prompt: str):
        """
        Save the user and system prompts to disk for future use.

        Parameters:
        -----------
        user_prompt: str
            The user prompt to save.
        system_prompt: str
            The system prompt to save.
        """
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S")
        if self.save_to_disk:
            with open(f"autojudge-user-prompt-{current_time}.txt", "w") as f:
                f.write(user_prompt)
            with open(f"autojudge-system-prompt-{current_time}.txt", "w") as f:
                f.write(system_prompt)

            logger.info("judge prompts automatically saved to disk.")

    def judge(
        self,
        input: str,
        output: str = None,
        expected: str = None,
    ) -> Judgment:
        """
        Judge the input and return a Judgment.
        """
        if not hasattr(self, "user_prompt") or not self.user_prompt:
            raise RuntimeError(
                "autojudge instance must be instantiated with `judge = AutoJudge.from_dataset(<your-dataset>)`"
            )

        system_prompt = self.system_prompt
        user_prompt = dedent(
            f"""
            {self.user_prompt}

            Input: {input}
            Response: {output}

            Respond in JSON format. {{"REASONING": "[...]", "SCORE": "<your-score>"}}
            """
        )
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        judgment = Judgment(reasoning=reasoning, score=score)
        return judgment
