from abc import ABC, abstractmethod
from warnings import warn
import os

from datasets import load_dataset

from evaluation import compute_code_eval

class Task(ABC):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    # The name of the `Task` benchmark as denoted in the HuggingFace datasets Hub
    DATASET_PATH: str = None

    # The name of a subset within `DATASET_PATH`.
    DATASET_NAME: str = None

    def __init__(self, stop_words=None, requires_execution=True):
        """
        :param stop_words: list
            list of stop words if the generation uses a stopping criteria during generation
        :param requires_execution: bool
            wheter the task requires code execution during evaluation or not
        """
        self.stop_words = stop_words
        self.requires_execution = requires_execution
        try:
            self.dataset = load_dataset(path=self.DATASET_PATH, name=self.DATASET_NAME)
        except Exception as e:
            warn(
                f"Loading the dataset failed with {str(e)}. This task will use a locally downloaded dataset, not from the HF hub. \
                This is expected behavior for the DS-1000 benchmark but not for other benchmarks!"
            )

    @abstractmethod
    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        return []

    def fewshot_examples(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        pass

    @abstractmethod
    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def get_reference(self, doc):
        """Builds the reference solution for the doc.
        :param doc: dict[str: str]
            sample from the test dataset
        """
        pass

    @abstractmethod
    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        pass

    @abstractmethod
    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations as in {"metric_name": result}.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        :return: dict[str: float]
        """
        pass

    @staticmethod
    def _stop_at_stop_token(decoded_string, stop_tokens):
        """
        Produces the prefix of decoded_string that ends at the first occurrence of
        a stop_token.
        WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
        itself.
        """
        min_stop_index = len(decoded_string)
        for stop_token in stop_tokens:
            stop_index = decoded_string.find(stop_token)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
        return decoded_string[:min_stop_index]


class MBPP(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "mbpp"

    def __init__(self):
        super().__init__(
            stop_words=["\nclass", "\nassert", '\n"""', "\nprint", "\nif", "\n<|/", "\n```"],
            requires_execution=True,
        )

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        # the wrong split of mbpp can be loaded with old datasets cache
        assert (
            len(dataset) == 500
        ), "please ensure you have the latest version of MBPP dataset, try deleting its old cache"
        return dataset

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        description = doc["text"]
        test_example = doc["test_list"][0]
        prompt = f'"""\n{description}\n{test_example}\n"""\n'
        return prompt

    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        return "\n".join(doc["test_list"])


    def postprocess_generation(self, generation, idx):
        """Defines the postprocessing for a LM generation.
        :param generation: str
            code generation from LM
        :param idx: int
            index of doc in the dataset to which the generation belongs
        """
        prompt = self.get_prompt(self.dataset["test"][idx])
        generation = generation[len(prompt) :]
        return prompt + self._stop_at_stop_token(generation, self.stop_words)

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
        )
        return results


class MBPPPlus(MBPP):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """

    DATASET_PATH = "evalplus/mbppplus"

    def get_prompt(self, doc):
        """Builds the prompt for the LM to generate from.
        MBPP prompt is built following to InCoder (Fried et al.) approach
        prompt = docstring that includes one test
        """
        description = doc["prompt"]  # sanitized testset use "prompt" instead of "text"
        test_example = doc["test_list"][0]
        prompt = f'"""\n{description}\n{test_example}\n"""\n'
        return prompt

    # NOTE(@ganler): MBPP+ extends the original MBPP jsonl data with a "test" field which
    #                includes the testing code ready for execution. Note the "test" field
    #                is different from HumanEval(+) which further requires a `check` func
    def get_reference(self, doc):
        """Builds the reference solution for the doc (sample from the test dataset)."""
        use_mbpp_tests = os.getenv("MBBPPLUS_USE_MBPP_TESTS", "0")
        if use_mbpp_tests == "1":
            return "\n".join(doc["test_list"])
        return "\n" + doc["test"]

    def get_dataset(self):
        """Returns dataset for the task or an iterable of any object, that get_prompt can handle"""
        dataset = self.dataset["test"]
        assert (
            len(dataset) == 399
        ), "MBPP+ only has 399 problems. Please retry by deleting its old cache"
        return dataset

    def process_results(self, generations, references):
        """Takes the list of LM generations and evaluates them against ground truth references,
        returning the metric for the generations.
        :param generations: list(list(str))
            list of lists containing generations
        :param references: list(str)
            list of str containing refrences
        """
        results, _ = compute_code_eval(
            references=references,
            predictions=generations,
            timeout=10.0,  # 10s timeout
        )
        return results