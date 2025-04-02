from textwrap import dedent

from judges.base import BaseJudge, Judgment


class PollMultihopCorrectness(BaseJudge):
    r"""
    A judge that evaluates the factual correctness of a generated response against a reference answer using few-shot learning.

    Citation:
    ---------
    @misc{verga2024replacingjudgesjuriesevaluating,
        title={Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models},
        author={Pat Verga and Sebastian Hofstatter and Sophia Althammer and Yixuan Su and Aleksandra Piktus and Arkady Arkhangorodsky and Minjie Xu and Naomi White and Patrick Lewis},
        year={2024},
        eprint={2404.18796},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2404.18796},
    }

    Model(s) used in paper:
    -----------------------
    Command R, Haiku, GPT3.5.
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
            You will be given a Question and a Provided Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".

            Question 1: "When did the president who set the precedent of a two term limit leave office?"
            Provided Answer: "George Washington set the precedent of a two-term limit when he decided not to seek a third term in 1796. He left office in 4 March, 1797."
            Reference Answer: "March 4, 1797"
            Correct: True

            Question 2: "Where does  ́Sivar ̄ama Swami conduct courses on Vaishnava Theology?"
            Provided Answer: " ́Sivar ̄ama Swami conducts courses on Vaishnava Theology at Bhaktivedanta Manor." Reference Answer: "Where does  ́Sivar ̄ama Swami conduct courses on Vaishnava Theology?"
            Correct: False

            Question 3: "The most populous city in Punjab is how large (area wise)?"
            Provided Answer: "Ludhiana, the most populous city in Punjab, covers an area of 3,767 square kilometres according to the website of Punjab’s Department of Agriculture & Farmer Welfare. Another source states it is 310 square kilometres, making it the most densely populated urban centre in the state." Reference Answer: "310 square kilometers"
            Correct: True

            Question 4: "Who was mayor of New York City when Fiorello H. La Guardia was born?" Provided Answer: "Frank Edson was mayor of New York City when Fiorello H. La Guardia was born on December 11, 1882. Edson served as mayor from 1881 to 1882, and La Guardia himself later became the 99th Mayor of New York City, serving from 1934 to 1946 or 1945."
            Reference Answer: "William Russell Grace"
            Correct: False

            Question 5: "What is the third letter of the top level domain of the military?"
            Provided Answer: "The third letter of the top-level domain for the military, ’.mil’, is ’l’.
            The domain name ’mil’ is derived from the word ’military’. It is the only top-level domain for a country’s military, a legacy of the US military’s role in the creation of the internet."
            Reference Answer: "l"
            Correct: True

            Question 6: "Ed Oxenbould plays Tyler in a film directed by which famous director?" Provided Answer: "Ed Oxenbould played Tyler in *The Visit*"
            Reference Answer: "M. Night Shyamalan"
            Correct: False

            Question 7: "Alice David is the voice of Lara Croft in a video game developed by which company ?"
            Provided Answer: "Alice David is the voice of Lara Croft in the French dub of the video game *Tomb Raider*." Reference Answer: "Crystal Dynamics"
            Correct: False

            Question 8: {input}
            Provided Answer: {output} 
            Reference Answer: {expected}
            Correct:
        """
        )
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)


class PollZeroShotCorrectness(BaseJudge):
    r"""
    A judge that evaluates the factual correctness of a generated response against a reference answer using few-shot learning.

    Citation:
    ---------
    @misc{verga2024replacingjudgesjuriesevaluating,
        title={Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models},
        author={Pat Verga and Sebastian Hofstatter and Sophia Althammer and Yixuan Su and Aleksandra Piktus and Arkady Arkhangorodsky and Minjie Xu and Naomi White and Patrick Lewis},
        year={2024},
        eprint={2404.18796},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2404.18796},
    }

    Model(s) used in paper:
    -----------------------
    Command R, Haiku, GPT3.5.
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
        You will be given a Question, a Provided Answer and a Reference Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".

        Question: {{input}}
        Provided Answer: {{output}}
        Reference Answer:{{expected}}
        Correct:
        """
        )
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)


class PollKiltNQCorrectness(BaseJudge):
    r"""
    A judge that evaluates factual correctness of a generated response against a reference answer using few-shot learning, based on KILT versions of Natural Questions.

    Citation:
    ---------
    @misc{verga2024replacingjudgesjuriesevaluating,
        title={Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models},
        author={Pat Verga and Sebastian Hofstatter and Sophia Althammer and Yixuan Su and Aleksandra Piktus and Arkady Arkhangorodsky and Minjie Xu and Naomi White and Patrick Lewis},
        year={2024},
        eprint={2404.18796},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2404.18796},
    }

    Model(s) used in paper:
    -----------------------
    Command R, Haiku, GPT3.5.
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
            You will be given a Question and a Provided Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".

            Question 1: "when did the nfl start playing in london"
            Provided Answer: "According to the provided search results, the NFL started playing regular season games in London as part of the NFL International Series in 2007. Specifically:

            Document [5] states: "The NFL International Series was inaugurated in 2007 to host NFL regular season games outside the United States. Played at the new Wembley Stadium in London (rebuilt and reopened in 2007), the series increased from one to two games for the 2013 season, to three games for the 2014 season, and then to four games from the 2017 season."

            Document [9] also mentions: "Since 2007, the league has held multiple regular season games in London each season as part of NFL London Games, allowing the league to test solutions to some of the challenges facing a hypothetical London franchise." 

            So 2007 marked the start of the NFL playing regular season games at the newly rebuilt Wembley Stadium in London as part of the International Series.  Prior to that, there were some preseason games NFL games played in London, but 2007 was the first year for regular season NFL games in the city."

            Reference Answer: "since 2007"
            Correct: True

            Question 2: "where is wind power used most in the united states"
            Provided Answer: "Iowa is the state that generates the highest percentage of energy from wind power, with more than 35% of its electric power coming from wind. Alternatively, North Dakota has the highest per capita wind generation.
            The state with the most installed wind power capacity is Texas, followed by Iowa. Texas’s capacity is around 20,000 MW, almost double that of Iowa’s 3,670 MW. The Roscoe Wind Farm in Texas is also the world’s largest wind farm at 780 MW.
            California has doubled its wind power capacity since 2002 and now gets about 5% of its energy from wind power. This is still lower than the twelve U.S. states that get more than 20% of their energy from wind."
            Reference Answer: "Texas"
            Correct: True

            Question 3: "who is the owner of phoenix mall in chennai"
            Provided Answer: "The owner of Phoenix Marketcity in Chennai is Phoenix Mills Limited, as mentioned Mills Limited and Crest Ventures Ltd."
            Reference Answer: "Phoenix Mills Limited"
            Correct: True

            Question 4: "number of degree of freedom for plane mechanism"
            Provided Answer: "In mechanics, the degree of freedom (DOF) of a mechanical system is the number of mechanism on a two-dimensional plane is three, which consists of two translations and one rotation.
            in the document titled "Phoenix Marketcity (Chennai)." This mall was jointly developed by Phoenix
            independent parameters that define its configuration. The number of degrees of freedom for a
            For a system of multiple bodies, the combined DOF is the sum of the DOFs of the individual bodies, minus the internal constraints on their relative motion. Therefore, a mechanism formed of multiple rigid bodies may have more than three DOFs."
            Reference Answer: "two"
            Correct: False

            Question 5: "who had the best strike rate among batsmen who scored over 200 runs in the world cup"
            Provided Answer: "The provided documents do not specify the strike rate of all batsmen who scored over 200 runs in the World Cup. However, among the batsmen mentioned who have scored over 200 runs in the World Cup, Jehan Mubarak has the highest ever strike rate in a T20 World Cup match for Sri Lanka, with a strike rate of 353.84 (Document [6]). But it’s important to note that this strike rate is not necessarily reflective of his overall performance in the World Cup, as it could have been achieved in a single match.
            For a more accurate comparison of strike rates among batsmen who scored over 200 runs in the World Cup, a comprehensive list of all such batsmen and their respective strike rates would be required." Reference Answer: "Rohit Sharma"
            Correct: False

            Question 6: "who played ben stone son on law and order"
            Provided Answer: "According to the provided search results, Ben Stone’s son Peter Stone was played by Philip Winchester. Some key details:
            - Peter Stone is the son of Benjamin "Ben" Stone, the former Executive Assistant District Attorney portrayed by Michael Moriarty on Law & Order.
            - Peter first appeared as a character on the Chicago P.D. episode "Justice", which served as a backdoor pilot for the short-lived spin-off Chicago Justice, where he was the lead character played by
            Philip Winchester.
            - After Chicago Justice was canceled, Philip Winchester reprised his role as Peter Stone in Season 19 of Law & Order: Special Victims Unit, where he became the Assistant District Attorney for the Sex
            Crimes Bureau.
            - In several SVU episodes, Peter describes having a strained relationship with his workaholic father Ben Stone, who neglected his family in favor of his prosecutorial duties.
            So in summary, Philip Winchester played the role of Peter Stone, the son of the iconic Law & Order character Ben Stone, across the shows Chicago Justice and Law & Order: SVU." Reference Answer: "Michael Moriarty"
            Correct: False

            Question 7: "who won so you think you can dance season 13"
            Provided Answer: "I’m sorry, I couldn’t find any information about the winner of season 13 of So You Think You Can Dance." Reference Answer: "Leon ‘‘ Kida ’’ Burns" Correct: False
            Question 8: "when did sweet caroline start at red sox games"
            Provided Answer: "The song "Sweet Caroline" started being played at Fenway Park, home of the Boston Red Sox, since at least 1997. However, it has been played consistently in the middle of the eighth inning at every game since 2002."
            Reference Answer: "at least 1997"
            Correct: True

            Question 9: "when did end of the road come out"
            Provided Answer: "I found multiple results for "End of the Road":
            ## End of the Road (1944 film)
            End of the Road is a 1944 American crime film directed by George Blair and written by Denison Clift, Gertrude Walker and Albert Beich. The film stars Edward Norris, John Abbott, June Storey, Jonathan Hale, Pierre Watkin and Ted Hecht.
            ## End of the Road (1970 film)
            End of the Road is a 1970 American comedy-drama film directed, co-written, and edited by Aram Avakian and adapted from a 1958 novel by John Barth. The film stars Stacy Keach, James Earl Jones and Harris Yulin.
            ## End of the Road (Boyz II Men song)
            End of the Road is a single recorded by American R&B group Boyz II Men for the "Boomerang" soundtrack. It was released on June 30, 1992.

            ## End of the Road (Jerry Lee Lewis song)
            End of the Road is a song by Jerry Lee Lewis. It appeared on the 1976 Sun Records compilation album "The Original Jerry Lee Lewis" which was released by the Sun Record Company. Which of these were you referring to?"
            Reference Answer: "June 30 , 1992"
            Correct: True

            Question 10: "who played john connor in the original terminator"
            Provided Answer: "John Connor is not physically portrayed by any actor in the original "Terminator" film released in 1984, as he is only mentioned and not seen. The character of John Connor is first referred to in this film as the future leader of the human resistance against the machines, but he does not make an appearance."
            Reference Answer: "Edward Furlong throughout the remainder of the film"
            Correct: False

            Question 11: {input}
            Provided Answer: {output}
            Reference Answer: {expected}
            Correct:
            """
        )
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)


class PollKiltHotpotCorrectness(BaseJudge):
    r"""
    A judge that evaluates factual correctness of a generated response against a reference answer using few-shot learning, based on KILT versions of HotpotQA.

    Citation:
    ---------
    @misc{verga2024replacingjudgesjuriesevaluating,
        title={Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models},
        author={Pat Verga and Sebastian Hofstatter and Sophia Althammer and Yixuan Su and Aleksandra Piktus and Arkady Arkhangorodsky and Minjie Xu and Naomi White and Patrick Lewis},
        year={2024},
        eprint={2404.18796},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2404.18796},
    }

    Model(s) used in paper:
    -----------------------
    Command R, Haiku, GPT3.5.
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
            You will be given a Question and a Provided Answer. Judge whether the Provided Answer is correct by comparing it to the Reference Answer. Differently formatted dates, people with missing middle names, and alternative spellings
            should all be considered the same. If the Provided Answer is correct say exactly "True", otherwise say "False".
            Question 1: "Who was also an actor, Serri or John Fogerty?"
            Provided Answer: "Based on the provided search results, there is no information indicating that John Fogerty has pursued acting as a profession or has been involved in acting projects. The search results focus on his music career,
            including his time with Creedence Clearwater Revival, his solo career, and various performances and recognitions he has received over the years. Therefore, without information on Serri in the provided documents, it cannot be
            determined if Serri was an actor. However, it is clear from the search results that John Fogerty is primarily known as a musician, singer, and songwriter, not as an actor."
            Reference Answer: "Serri"
            Correct: False

            Question 2: "The 37th Disney animated feature film Tarzan features You’ll be in my heart by which artist?" Provided Answer: "The 37th Disney animated feature film Tarzan’s soundtrack, Tarzan: An Original Walt Disney Records,
            includes the song You’ll Be in My Heart written and performed by Phil Collins. The song won the Academy Award for Best Original Song and the Golden Globe Award for Best Original Song. Collins was originally hired as a songwriter
            but ended up performing the song as well."
            Reference Answer: "Phil Collins"
            Correct: True

            Question 3: "Who is the current drummer of the band who did the song "What Lovers Do"?"
            Provided Answer: "Unfortunately, none of the provided search results directly mention the current drummer for the band Maroon 5 who performed the song "What Lovers Do". The results provide background information on the song
            itself, its music video, chart performance, and other details, but do not specify the band members or drummer.
            Maroon 5 is a well-known pop rock band fronted by Adam Levine, but their full current lineup and specific drummer for the "What Lovers Do" era is not stated in these search results. To definitively answer who the drummer was,
            additional information directly about Maroon 5´s band members around 2017 when "What Lovers Do" was released would be needed." Reference Answer: "Matt Flynn"
            Correct: False

            Question 4: "Who is older, Ferdi Taygan or Mahesh Bhupathi?"
            Provided Answer: "Unfortunately, none of the provided search results contain any information about Ferdi Taygan or their age to compare with Mahesh Bhupathi’s age. The search results are all focused on providing biographical
            details about Mahesh Bhupathi, an Indian professional tennis player, and some other individuals named Mahesh Bhatt who appear to be unrelated to the question. Without any information about Ferdi Taygan, it is impossible to
            determine who is older between them and Mahesh Bhupathi based solely on these search results."
            Reference Answer: "Ferdi Taygan"
            Correct: False

            Question 5: "What is the birthday of the actress who was the Duchess in "The Revengers Tragedy"?"
            Provided Answer: "Diana Quick, who played the Duchess in "The Revengers Tragedy", was born on 23rd September 1934."
            Reference Answer: "23 November 1946"
            Correct: False

            Question 6: "Rejuvelac is kind of grain water invented and promoted by a "holistic health" practitioner born in which year ?"
            Provided Answer: "Rejuvelac is a kind of grain water invented and promoted by Ann Wigmore, who was born in 1909."
            Reference Answer: "1909"
            Correct: True

            Question 7: "Who is the current drummer of the band who did the song "What Lovers Do"?"
            Provided Answer: "The song "What Lovers Do" is by the American pop rock band Maroon 5. The current drummer of Maroon 5 is Matt Flynn."
            Reference Answer: "Matt Flynn"
            Correct: True

            Question 8: "The 37th Disney animated feature film Tarzan features You’ll be in my heart by which artist?"
            Provided Answer: "The song "You´ll Be in My Heart" featured in the 1999 Disney animated film Tarzan was written and performed by Phil Collins.
            The key details that support this answer from the provided search results:
            1. Document [2] states: "You´ll Be in My Heart" is a song by Phil Collins, from the 1999 Disney animated feature "Tarzan"."
            2. Document [3] mentions: "For his contribution to the soundtrack, Collins received an American Music Award for Favorite Adult Contemporary Artist."
            3. Document [5] says: "Collins was originally hired by Disney as a songwriter, but was chosen because he was a drummer at first and given his history as a drummer for the rock band Genesis, he appealed to the Disney production
            crew, who "wanted a strong jungle beat to accompany Tarzan´s adventures". This "ballad" is one of five original songs he wrote for the film, and he ended up also performing this song as well."
            4. Document [7] states: "The song is about "how love is a bond that cannot be broken"."
            So in summary, the artist who wrote and performed the song "You´ll Be in My Heart" featured in the Disney animated film Tarzan is Phil Collins."
            Reference Answer: "Phil Collins"
            Correct: True

            Question 9: "How long is the river for which Frenchmans Creek is a tributary?"
            Provided Answer: "The length of the river for which Frenchmans Creek is a tributary cannot be directly determined from the provided documents as they only provide information about the Frenchmans Creek itself and not the river it
            flows into. However, we can see that Frenchmans Creek is a tributary of the Darling River in New South Wales, Australia (Document 2), the Mississippi River in Missouri, USA (Document 3), and the Milk River in Saskatchewan, Canada,
            and Montana, USA (Document 5). The length of these rivers is not provided in the documents."
            Reference Answer: "2844 km"
            Correct: False

            Question 10: "Conrad Anker located the body of a mountaineer who took part it the first three British expeditions to which mountain?"
            Provided Answer: "Conrad Anker located the body of George Mallory on Mount Everest. Mallory was a British climber who took part in the first three British expeditions to Mount Everest, including the 1924 expedition during which
            he and his climbing partner Andrew Irvine disappeared while attempting to reach the summit. Anker found Mallory’s body in 1999 as a member of a search team looking for the remains of the British climber.
            References: [1] Conrad Anker [2] 2007 Altitude Everest expedition [9] Timeline of Mount Everest expeditions"
            Reference Answer: "Mount Everest"
            Correct: True

            Question 11: "The city that contains the Yunnan Provincial Museum is also known by what nickname?"
            Provided Answer: "The city that contains the Yunnan Provincial Museum, Kunming, is also known by the nickname "Spring City.""
            Reference Answer: "Spring city"
            Correct: True

            Question 12: "The 1st Virginia Regiment saw action at a battle that led to the departure of what Royal Governor?" Provided Answer: "Based on the provided search results, the 1st Virginia Regiment saw action at the Battle of Great
            Bridge in December 1775, which led to the departure of Lord Dunmore, the Royal Governor of Virginia at the time.
            Document [1] mentions that the 1st Virginia Regiment participated in the Battle of Great Bridge. Document [4] states that after the Battle of Great Bridge, "Dunmore responded by bombarding Norfolk with his ships on January 1,
            1776." This implies that the Battle of Great Bridge, in which the 1st Virginia Regiment fought, led to Lord Dunmore´s departure from Virginia as the Royal Governor.
            So the 1st Virginia Regiment´s involvement in the Battle of Great Bridge resulted in the departure of Lord Dunmore, the Royal Governor of the Colony of Virginia during the American Revolutionary War."
            Reference Answer: "Royal Governor Lord Dunmore"
            Correct: True

            Question 13: "The voice of Homer Simpson also voices what alien character in the television series "The Simpsons"?"
            Provided Answer: "Dan Castellaneta voices Homer Simpson, Grampa Simpson, Krusty the Clown, Groundskeeper Willie, Mayor Quimby, Barney Gumble, and other adult characters."
            Reference Answer: "Kodos"
            Correct: False

            Question 14: "Who were the producers of the show for which Chris McKay made his feature film directorial debut?" Provided Answer: "Chris McKay made his feature film directorial debut with "The Lego Batman Movie" in 2017. The
            producers of "The Lego Batman Movie" were not explicitly mentioned in the provided documents. However, it is noted that Seth Grahame-Smith co-wrote the screenplay for the movie, and Warner Bros. was involved in its development,
            as mentioned in Document 8. Additionally, Document 9 mentions that McKay had signed a first look deal for being a producer with Warner Bros. around the time of the movie´s development. While this information does not directly
            answer the question regarding the specific producers of "The Lego Batman Movie," it does highlight the key individuals and entities involved in its creation and development." Reference Answer: "Dan Lin, Roy Lee, Phil Lord and
            Christopher Miller" Correct: False

            Question 15: "Marion Greene was a health policy analyst for St. Judt Medical company, which had how many principal operations worldwide?"
            Provided Answer: "The provided documents do not specify the number of principal operations St. Jude Medical company had worldwide. Document 1 mentions that Marion Greene was a health policy analyst for St. Jude Medical, but it
            does not provide details about the company’s operations. Document 2 also mentions Greene’s work for St. Jude Medical, but again, no information is given about the company’s operations. The other documents do not pertain to St.
            Jude Medical. Therefore, I cannot provide an answer to this question based on the given documents."
            Reference Answer: "20"
            Correct: False

            Question 16: "What is the name of the actress who has received a Drama Desk award and was part of the cast album for the 2007 Broadway production of the hit musical, "Grease"?"
            Provided Answer: "Laura Osnes, who played the role of Sandy Dumbrowski in the 2007 Broadway production of "Grease", is a Drama Desk award winner and is featured on the cast album for the production."
            Reference Answer: "Laura Ann Osnes"
            Correct: True

            Question 17: "{input}"
            Provided Answer: "{output}"
            Reference Answer: "{expected}"
            Correct:
            """
        )
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)


class RAFTCorrectness(BaseJudge):
    r"""
    A judge that evaluates factual correctness using LLAMA-based methods.

    Citation:
    ---------
    @misc{zhang2024raftadaptinglanguagemodel,
        title={RAFT: Adapting Language Model to Domain Specific RAG},
        author={Tianjun Zhang and Shishir G. Patil and Naman Jain and Sheng Shen and Matei Zaharia and Ion Stoica and Joseph E. Gonzalez},
        year={2024},
        eprint={2403.10131},
        archivePrefix={arXiv},
        primaryClass={cs.CL},
        url={https://arxiv.org/abs/2403.10131},
    }

    Model(s) used in paper:
    -----------------------
    Llama-3 70B Instruct.

    NOTE:
    ----
    The prompt used in this judge comes from an implementation of the RAFT setup created in collaboration between Meta and Tanjun Zhang,
    so the prompt is not directly from the paper. The judge prompt implementation can be found here:
    https://github.com/meta-llama/llama-recipes/blob/main/recipes/use_cases/end2end-recipes/RAFT-Chatbot/raft_eval_config.yaml
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
        system_prompt = dedent(
            """
            You have been provided with a question, a teacher's answer and a student's answer below.
            Given that question, you need to score the how good the student answer is compare to
            the teacher's answer. If the student's answer is correct based on the teacher's answer, then return YES, else return NO.
            Here are the grade criterias to follow:
            1. Review it carefully to make sure that the keywords and numerical vaules are exactly the same.
            2. Ensure that the student answer does not contain any conflicting statements.
            3. It is OK if the student answer contains more information than the ground truth answer, as long as it is factually accurate relative to the  ground truth answer.
            YES means that the student's answer meets all of the criteria.
            NO means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.
            Only respond with "YES" or "NO", do not respond with anything else."""
        )
        user_prompt = "Question: {input} \n Teacher's Answer: {expected} \n Student's Answer: {output}"
        reasoning, score = self._judge(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
        )
        return Judgment(reasoning=reasoning, score=score)
