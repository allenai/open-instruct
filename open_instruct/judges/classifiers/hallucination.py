from textwrap import dedent

from judges.base import BaseJudge, Judgment


class HaluEvalAnswerNonFactual(BaseJudge):
    r"""
    A judge that evaluates whether an answer to a question contains non-factual or hallucinated information.

    Citation:
    ---------
    @misc{li2023haluevallargescalehallucinationevaluation,
        title={HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models}, 
        author={Junyi Li and Xiaoxue Cheng and Wayne Xin Zhao and Jian-Yun Nie and Ji-Rong Wen},
        year={2023},
        eprint={2305.11747},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2305.11747}, 
    }


    Model(s) used in paper:
    -----------------------
    GPT-3, Claude 2, Claude, Llama 2, Falcon, Vicuna, Alpaca, Davinci002, Davinci003.
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
        user_prompt = dedent(f"""
        I want you act as an answer judge. Given a question and an answer, your objective is to determine if the provided answer contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

        You are trying to determine if the answer misunderstands the question context and intention.
        #Question#: What is a rare breed of dog that was derived as a variant of Rat Terrier, Shiloh Shepherd dog or American Hairless Terrier?
        #Answer#: American Hairless Terrier
        #Your Judgement#: No

        You are trying to determine if there is a factual contradiction between the answer and the world knowledge. Some information in the answer might be fabricated.
        #Question#: Are the New Orleans Outfall Canals the same length as the Augusta Canal?
        #Answer#: No, the New Orleans Outfall Canals and the Augusta Canal are not the same length. The Orleans Canal is approximately 3.6 miles (5.8 kilometers) long while the Augusta Canal is approximately 7 miles (11.3 kilometers) long.
        #Your Judgement#: Yes
        #Question#: What U.S Highway gives access to Zilpo Road, and is also known as Midland Trail?
        #Answer#: U.S Highway 70
        #Your Judgement#: Yes

        You are trying to determine if the answer is too general or too specific to answer the question at an appropriate level of specificity.
        #Question#: What genre do Superheaven and Oceansize belong to?
        #Answer#: Superheaven and Oceansize belong to the rock genre.
        #Your Judgement#: No
        #Question#: What profession do Kōbō Abe and Agatha Christie share?
        #Answer#: Playwright.
        #Your Judgement#: No

        You are trying to determine if the answer can be correctly inferred from the knowledge.
        #Question#: Which band has more members, Muse or The Raconteurs?
        #Answer#: Muse has more members than The Raconteurs.
        #Your Judgement#: Yes
        #Question#: Which is currently more valuable, Temagami-Lorrain Mine or Meadowbank Gold Mine?
        #Answer#: Meadowbank Gold Mine, since Meadowbank Gold Mine is still producing gold and the TemagamiLorrain Mine has been inactive for years.
        #Your Judgement#: No

        You should try your best to determine if the answer contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\"".

        #Question#: {input}
        #Answer#: {output}
        #Your Judgement#:
        """)
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)


class HaluEvalDialogueResponseNonFactual(BaseJudge):
    r"""
    A judge that evaluates whether a dialogue response contains non-factual information.

    Citation:
    ---------
    @misc{li2023haluevallargescalehallucinationevaluation,
        title={HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models}, 
        author={Junyi Li and Xiaoxue Cheng and Wayne Xin Zhao and Jian-Yun Nie and Ji-Rong Wen},
        year={2023},
        eprint={2305.11747},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2305.11747}, 
    }


    Model(s) used in paper:
    -----------------------
    GPT-3, Claude 2, Claude, Llama 2, Falcon, Vicuna, Alpaca, Davinci002, Davinci003.
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
        user_prompt = dedent(f"""
        I want you act as a response judge. Given a dialogue history and a response, your objective is to determine if the provided response contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

        You are trying to determine if the true entity in the response is replaced with a highly similar entity.
        #Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
        #Response#: Christopher Nolan was the director. He also directed insomnia and inception.
        #Your Judgement#: No
        #Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
        #Response#: Steven Spielberg was the director. He also directed insomnia and inception.
        #Your Judgement#: Yes

        You are trying to determine if the true entity in the response is replaced with a dissimilar entity.
        #Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
        #Response#: Christopher Nolan was the director. He also directed insomnia and inception.
        #Your Judgement#: No
        #Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
        #Response#: Batman Begins was the director. He also directed insomnia and inception.
        #Your Judgement#: Yes

        You are trying to determine if the true entity in the response is replaced with a dissimilar entity in a different entity type.
        #Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
        #Response#: Christopher Nolan was the director. He also directed insomnia and inception.
        #Your Judgement#: No
        #Dialogue History#: [Human]: Could you recommand movies similar to The Dark Knight? [Assistant]: The sequel to Batman Begins is The Dark Knight. [Human]: Okay. Who is the director of The Dark Knight and any other movies from him not related to Batman?
        #Response#: United States of America was the director. He also directed insomnia and inception.
        #Your Judgement#: Yes

        You should try your best to determine if the response contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\"".

        #Dialogue History#: {input}
        #Response#: {output}
        #Your Judgement#:
        """)
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)


class HaluEvalDocumentSummaryNonFactual(BaseJudge):
    r"""
    A judge that evaluates whether a summary contains non-factual or hallucinated information.

    Citation:
    ---------
    @misc{li2023haluevallargescalehallucinationevaluation,
        title={HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models}, 
        author={Junyi Li and Xiaoxue Cheng and Wayne Xin Zhao and Jian-Yun Nie and Ji-Rong Wen},
        year={2023},
        eprint={2305.11747},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2305.11747}, 
    }


    Model(s) used in paper:
    -----------------------
    GPT-3, Claude 2, Claude, Llama 2, Falcon, Vicuna, Alpaca, Davinci002, Davinci003.
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
        user_prompt = dedent(f"""
        I want you act as a summary judge. Given a document and a summary, your objective is to determine if the provided summary contains non-factual or hallucinated information. You SHOULD give your judgement based on the following hallucination types and the world knowledge.

        You are trying to determine if the summary is factual but some information cannot be directly inferred or entailed from the document.
        #Document#: The panther chameleon was found on Monday by a dog walker in the wooded area at Marl Park. It had to be put down after X-rays showed all of its legs were broken and it had a deformed spine. RSPCA Cymru said it was an "extremely sad example of an abandoned and neglected exotic pet". Inspector Selina Chan said: "It is a possibility that the owners took on this animal but were unable to provide the care he needs and decided to release him to the wild. "We are urging potential owners of exotic animals to thoroughly research what is required in the care of the particular species before taking one on. "Potential owners need to make sure they can give their animal the environment it needs and they have the facilities, time, financial means and long-term commitment to maintain a good standard of care, as required under the Animal Welfare Act 2006." She added it was illegal to release non-native species into the wild.
        #Summary#: A chameleon that was found in a Cardiff park has been put down after being abandoned and neglected by its owners.
        #Your Judgement#: Yes

        You are trying to determine if there exists some non-factual and incorrect information in the summary.  
        #Document#: The city was brought to a standstill on 15 December last year when a gunman held 18 hostages for 17 hours. Family members of victims Tori Johnson and Katrina Dawson were in attendance. Images of the floral tributes that filled the city centre in the wake of the siege were projected on to the cafe and surrounding buildings in an emotional twilight ceremony. Prime Minister Malcolm Turnbull gave an address saying a "whole nation resolved to answer hatred with love". "Testament to the spirit of Australians is that with such unnecessary, thoughtless tragedy, an amazing birth of mateship, unity and love occurs. Proud to be Australian," he said. How the Sydney siege unfolded New South Wales Premier Mike Baird has also announced plans for a permanent memorial to be built into the pavement in Martin Place. Clear cubes containing flowers will be embedded into the concrete and will shine with specialised lighting. It is a project inspired by the massive floral tributes that were left in the days after the siege. "Something remarkable happened here. As a city we were drawn to Martin Place. We came in shock and in sorrow but every step we took was with purpose," he said on Tuesday.
        #Summary#: Crowds have gathered in Sydney's Martin Place to honour the victims of the Lindt cafe siege, one year on.
        #Your Judgement#: No

        You are trying to determine if there is a factual contradiction between the summary and the document.
        #Document#: Christopher Huxtable, 34, from Swansea, had been missing since the collapse in February. His body was found on Wednesday and workers who carried out the search formed a guard of honour as it was driven from the site in the early hours of the morning. Ken Cresswell, 57, and John Shaw, 61, both from Rotherham, remain missing. The body of a fourth man, Michael Collings, 53, from Brotton, Teesside, was previously recovered from the site. Swansea East MP Carolyn Harris, who has been involved with the family since the incident, said they still did not know all the facts about the collapse. She said: "I feel very sad. My heart and my prayers go out to the family who have waited desperately for Christopher's body to be found. They can finally have closure, and say goodbye to him and grieve his loss. "But let's not forget that there's two other families who are still waiting for their loved ones to be returned." The building was due for demolition when it partially collapsed in February.
        #Summary#: The body of a man whose body was found at the site of the Swansea Bay Power Station collapse has been removed from the site.
        #Your Judgement#: Yes

        You should try your best to determine if the summary contains non-factual or hallucinated information according to the above hallucination types. The answer you give MUST be \"Yes\" or \"No\"".

        #Document#: {input}
        #Summary#: {output}
        #Your Judgement#:
        """)
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)