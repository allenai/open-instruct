from textwrap import dedent

from judges.base import BaseJudge, Judgment


class FactAlignQueryQuality(BaseJudge):
    r"""
    A judge that evaluates the quality of a query based on clarity, specificity, and relevance.

    Citation:
    ---------
    @misc{huang2024factalignlongformfactualityalignment,
        title={FactAlign: Long-form Factuality Alignment of Large Language Models},
        author={Chao-Wei Huang and Yun-Nung Chen},
        year={2024},
        eprint={2410.01691},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2410.01691},
    }

    Model(s) used in paper:
    -----------------------
    GPT-4 Turbo, GPT-3.5 Turbo, Olmo-7B Instruct, Llama 2 70B Chat. Additionally, Gemma 2B-SFT, Phi3 Mini, and Llama 3 8B Chat fine-tuned on the FactAlign dataset.
    """

    def judge(
        self,
        input: str,
        output: str = None,
        expected: str = None,
    ) -> Judgment:
        """
        Judge the input and return a verdict.
        """
        system_prompt = None
        user_prompt = dedent(
            f"""
        Instructions:

        1. You have been given a STATEMENT and some KNOWLEDGE points.
        2. Determine whether the given STATEMENT is supported by the given KNOWLEDGE. The STATEMENT does not need to be explicitly supported by the KNOWLEDGE but should be strongly implied by the KNOWLEDGE.
        3. Before showing your answer, think step-by-step and show your specific reasoning. As part of your reasoning, summarize the main points of the KNOWLEDGE related to the STATEMENT.
        4. If the STATEMENT is supported by the KNOWLEDGE, be sure to show the supporting evidence.
        5. After stating your reasoning, restate the STATEMENT and then determine your final answer based on your reasoning and the STATEMENT.
        6. Your final answer should be either "Supported" or "Not Supported".  Wrap your final answer in square brackets.

        KNOWLEDGE:
        {{input}}

        STATEMENT:
        {{output}}

        ANSWER:
        """
        )
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)
