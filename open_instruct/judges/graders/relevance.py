from textwrap import dedent

from judges.base import BaseJudge, Judgment


class ReliableCIRelevance(BaseJudge):
    r"""
    A judge that evaluates the relevance of a passage to a query based on a four-point scale.

    Citation:
    ---------
    @inproceedings{10.1145/3637528.3671883,
        author = {Oosterhuis, Harrie and Jagerman, Rolf and Qin, Zhen and Wang, Xuanhui and Bendersky, Michael},
        title = {Reliable Confidence Intervals for Information Retrieval Evaluation Using Generative A.I.},
        year = {2024},
        isbn = {9798400704901},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3637528.3671883},
        doi = {10.1145/3637528.3671883},
        abstract = {The traditional evaluation of information retrieval (IR) systems is generally very costly as it requires manual relevance annotation from human experts. Recent advancements in generative artificial intelligence -specifically large language models (LLMs)- can generate relevance annotations at an enormous scale with relatively small computational costs. Potentially, this could alleviate the costs traditionally associated with IR evaluation and make it applicable to numerous low-resource applications. However, generated relevance annotations are not immune to (systematic) errors, and as a result, directly using them for evaluation produces unreliable results.In this work, we propose two methods based on prediction-powered inference and conformal risk control that utilize computer-generated relevance annotations to place reliable confidence intervals (CIs) around IR evaluation metrics. Our proposed methods require a small number of reliable annotations from which the methods can statistically analyze the errors in the generated annotations. Using this information, we can place CIs around evaluation metrics with strong theoretical guarantees. Unlike existing approaches, our conformal risk control method is specifically designed for ranking metrics and can vary its CIs per query and document. Our experimental results show that our CIs accurately capture both the variance and bias in evaluation based on LLM annotations, better than the typical empirical bootstrapping estimates. We hope our contributions bring reliable evaluation to the many IR applications where this was traditionally infeasible.},
        booktitle = {Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
        pages = {2307–2317},
        numpages = {11},
        keywords = {confidence intervals, conformal prediction, generative A.I., information retrieval evaluation, large language models},
        location = {Barcelona, Spain},
        series = {KDD '24}
    }

    Model(s) used in paper:
    -----------------------


    NOTE:
    -----
    The authors create this prompt to evaluate the relevance of a passage to a query, and is based on the TREC-DL evaluation benchmark: https://microsoft.github.io/msmarco/TREC-Deep-Learning#datasets
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
        Assess the relevance of the PASSAGE to the QUERY on a four-point scale:
        [0] Irrelevant: The passage has nothing to do with the query.
        [1] Related: The passage seems related to the query but does not answer it.
        [2] Highly relevant: The passage has some answer for the query, but the answer may be a bit unclear or hidden amongst extraneous information.
        [3] Perfectly relevant: The passage is dedicated to the query and contains the exact answer.
        
        Query: {input}
        Passage: {context}
        Relevance:
        """)
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)
