from textwrap import dedent
import json

from judges.base import BaseJudge, Judgment


class MTBenchChatBotResponseQuality(BaseJudge):
    r"""
    A judge that evaluates the quality of an AI assistant's response to a user's question.
    It assigns a numerical grade based on factors such as helpfulness, relevance, accuracy,
    depth, creativity, and level of detail.

    Citation:
    ---------
    @misc{zheng2023judgingllmasajudgemtbenchchatbot,
          title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena}, 
          author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric P. Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
          year={2023},
          eprint={2306.05685},
          archivePrefix={arXiv},
          primaryClass={cs.CL},
          url={https://arxiv.org/abs/2306.05685}, 
    }

    Model(s) used in paper:
    -----------------------
    LLaMA-13B, Vicuna-13B
    """

    def judge(
        self,
        input: str,
        output: str = None,
        expected: str = None,
    ) -> Judgment:
        """
        Judge the quality of the assistant's response to the user's question.

        Parameters:
        -----------
        input: str
            The user's question.
        output: str
            The AI assistant's response to the user's question.
        expected: str
            The expected or ideal response (optional, can be used for reference or future enhancements).

        Returns:
        --------
        Judgment
            An object containing the grade and explanation.
        """
        system_prompt = None

        # Construct the user prompt with the provided exact template
        user_prompt = dedent(
            f"""
            [System]
            Please act as an impartial judge and evaluate the quality of the response provided by an
            AI assistant to the user question displayed below. Your evaluation should consider factors
            such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of
            the response. Begin your evaluation by providing a short explanation. Be as objective as
            possible. After providing your explanation, please rate the response on a scale of 1 to 10.
            [Question]
            {input}
            [The Start of Assistant’s Answer]
            {output}
            [The End of Assistant’s Answer]
            """
        )

        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        return Judgment(reasoning=reasoning, score=score)




