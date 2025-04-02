from textwrap import dedent

from judges.base import BaseJudge, Judgment


class HaystackBulletPointCoverageCorrectness(BaseJudge):
    r"""
    A judge that evaluates whether a reference insight is covered in a list of bullet points.
    It determines if the coverage is full, partial, or nonexistent.

    Citation:
    ---------
    @misc{laban2024summaryhaystackchallengelongcontext,
          title={Summary of a Haystack: A Challenge to Long-Context LLMs and RAG Systems}, 
          author={Philippe Laban and Alexander R. Fabbri and Caiming Xiong and Chien-Sheng Wu},
          year={2024},
          eprint={2407.01370},
          archivePrefix={arXiv},
          primaryClass={cs.CL},
          url={https://arxiv.org/abs/2407.01370}, 
    }

    Link to Prompt in Repository:
    -----------------------------
    https://github.com/salesforce/summary-of-a-haystack/blob/master/prompts/eval_summhay.txt

    Model(s) used in paper:
    -----------------------
    Claude 3 | Gemini 1.5

    NOTE:
    -----
    We remove the authors' initial output standardization requirement i.e., "Rating: [[rating]]" 
    so we don't need to define a new methood to extract the rating for just this one judge.
    """

    def judge(
        self,
        input: str,
        output: str = None,
        expected: str = None,
    ) -> Judgment:
        """
        Judge whether the reference insight is covered in the output bullet points.

        Parameters:
        -----------
        input: str
            The context or question provided (can be unused depending on implementation).
        output: str
            The generated list of bullet points.
        expected: str
            The reference insight to be assessed for coverage.

        Returns:
        --------
        Judgment
            An object containing the coverage score and reasoning.
        """
        system_prompt = None

        # Construct the user prompt with the provided template
        user_prompt = dedent(
            f"""
            You are given a list of bullet points (each with a unique number), and a specific reference insight. Your objective is to determine whether the reference insight is covered in any of the bullet points. You must further determine if the insight is partially covered ("PARTIAL_COVERAGE") or fully covered ("FULL_COVERAGE") by the bullet points. If the insight is not covered at all, you must return "NO_COVERAGE". See examples below:

            Example Reference Insight 1: "The doctor asks the patient about their medical history".

            Example Bullet Points 1:
            {{"bullets": [
                {{"bullet_id": 1, "text": "The patient often mention that they are worried about medication side-effect[12][14]."}},
                {{"bullet_id": 2, "text": "The doctor and patient spend time going over symptoms [3][8], particularly the initial symptoms and the progression in the last few months[5]."}},
                {{"bullet_id": 3, "text": "The doctor and patient discuss medical history within the patient's family[1][5], with the patient often unaware that some of the conditions are hereditary[2]."}}
            ]
            }}

            Example Output 1:
            {{"coverage": "FULL_COVERAGE", "bullet_id": 3}}

            Example Reference Insight 2: "The doctor asks the patient about their medical history".

            Example Bullet Points 2:
            {{"bullets": [
                {{"bullet_id": 1, "text": "The patient often mention that they are worried about medication side-effect[12][14]."}},
                {{"bullet_id": 2, "text": "The doctor and patient spend time going over symptoms [3][8], particularly the initial symptoms and the progression in the last few months[5]."}}
            ]
            }}

            Example Output 2:
            {{"coverage": "NO_COVERAGE", "bullet_id": "NA"}}

            Example Reference Insight 3: "The doctor asks the patient about their medical history".

            Example Bullet Points 3:
            {{"bullets": [
                {{"bullet_id": 1, "text": "The patient often mention that they are worried about medication side-effect[12][14]."}},
                {{"bullet_id": 2, "text": "The doctor and patient catch up after a long time, with the patient mentioning feeling unwell for a while, and knowing of other family member's similar experiences[20]."}},
                {{"bullet_id": 3, "text": "The doctor and patient spend time going over symptoms [3][8], particularly the initial symptoms and the progression in the last few months[5]."}}
            ]
            }}

            Example Output 3:
            {{"coverage": "PARTIAL_COVERAGE", "bullet_id": 2}}

            Now complete the task for the following insight and bullet points:

            Reference Insight:
            {expected}

            Bullet Points:
            {output}


            Requirements:
            - Do not hallucinate that the insight is covered by the bullet points if it is not.
            - Your response should only be the JSON output in the format above, such that it can directly parsed by Python's json module. DO NOT OUTPUT ANY EXPLANATION OR ANYTHING THAT IS NOT THE JSON RESPONSE.
            """
        )

        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )

        return Judgment(reasoning=reasoning, score=score)
    




