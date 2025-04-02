GENERATE_RUBRIC_USER_PROMPT = """
You are an evaluator tasked with creating a **"grading note"** to assess the quality of outputs for a specific task. A grading note is a set of precise, actionable guidelines derived from human provided feedback on LLM outputs that were deemed bad. This grading note will serve as a framework to ensure future outputs are meticulously scrutinized with absolute accuracy, unwavering consistency, and strict alignment with human expectations. 
Given a task description for which the outputs were generated and an overview of the feedback on these outputs, craft a grading note that outlines the critical evaluation criteria and provides explicit guidelines for assessing future outputs. The goal of the grading note is to determine a binary classification (1 for good, 0 for bad) for an LLM output based on the established criteria.

#### Task:

{task}

#### Overview of Feedback:

{feedback}

Leverage the task description and feedback above to construct a grading note that encapsulates the critical areas for improvement and provides a definitive framework for evaluating future outputs with uncompromising effectiveness.
You must not provide any explanation as to why you crafted a grading note in a certain way. You must only return the grading note.

#### Grading Notes:
"""


FORMAT_RUBRIC_USER_PROMPT = """
Your response must be structured as a JSON object with the following format:
{{
  "SCORE": True | False, 
  "REASONING": "A brief explanation of the evaluation results, detailing why the output meets or does not meet the specified criteria."
}}

Now consider the following input to an LLM and the corresponding output. Evaluate the output based on the grading notes.

### Input ###
-------------

{input}

--------------
### Output ###
--------------

{output}

------------------
### Evaluation ###
------------------
"""

STRUCTURE_FEEDBACK_USER_PROMPT = """
You are a diligent analyst tasked with transforming unstructured human feedback on a set of "bad" data points into a simple, organized format. The goal is to categorize the feedback and identify the key high-level issues.
Given the task and the feedback listed below, organize and destill the described issues into high-level categories (no more than 10 categories). Provide the response as a numbered list for each category of issues with a brief description.

#### Task:

{task}

#### Feedback:

{feedback}

#### Overview of Feedback:
"""
