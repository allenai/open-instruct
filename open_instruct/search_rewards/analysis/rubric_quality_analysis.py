import json
from typing import Any, Dict, List, Tuple
from datasets import load_dataset


import os
from openai import OpenAI, OpenAIError


# Set API key (or via environment variable)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def ask_with_search(prompt: str):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini-search-preview",
            # Provide search options (can be empty dict or include context size, user location, etc.)
            web_search_options={},
            messages=[
                {"role": "system", "content": "You are a helpful assistant with internet access."},
                {"role": "user", "content": prompt}
            ],
            # note: avoid unsupported parameters like temperature or top_p if they cause errors
        )
        # Extract the assistant’s reply
        answer = resp.choices[0].message.content
        return answer
    except OpenAIError as e:
        print("OpenAI API error:", e)
        raise


def load_rubric(rubric_id: str, num_samples: int) -> List[str]:
    rubric_data = load_dataset("rl-rag/"+rubric_id, split="train")
    all_criteria = []
    for example in rubric_data:
        question = json.loads(example["ground_truth"])["Question"]
        criteria = []
        for key in ["Answer Critical"]: #, "Valuable", "Context"]:
            per_sample_criteria = json.loads(example["ground_truth"])[key]
            for criterion in per_sample_criteria:
                criteria.append(criterion["Ingredient"])
        all_criteria.append((question, criteria))

    print(f"Number of questions: {len(all_criteria)}")
    newline = "\n"
    print(f"Few examples of questions: {newline.join([question for question, _ in all_criteria[:5]])}")

    all_criteria = all_criteria[:num_samples]  # Remove duplicates
    return all_criteria


def score_factuality(
    all_criteria: List[Tuple[str, List[str]]],
    output_path: str,
) -> List[Dict[str, Any]]:
    FACTUALITY_EVAL_PROMPT_LIST_CRITERIA = (
        "You are a careful evaluator that checks whether a list of criteria makes a factual claim and, if so, whether that claim is true."
        "\n\nInstructions:"
        "\n1. Read the question and the list of criteria carefully."
        "\n2. If the criterion is about writing style, clarity, tone, or other subjective qualities, and does not involve any factual claim, return 'NA'."
        "\n3. If the criterion includes a factual claim (something that can be verified as true or false), check whether it is factually correct based on reliable information or reasoning."
        "\n4. Assign a factuality score:"
        "\n   - 1 → All criteria are factual and correct."
        "\n   - 0.5 → Some criteria are factual and correct, some are not."
        "\n   - 0 → All criteria are factual but incorrect or false."
        "\n   - 'NA' → The criteria do not involve any factual claim."
        "\n\nOutput Format:"
        "\nReturn your result strictly in JSON format as follows:"
        "\n{{\"factual_score\": <0_or_1_or_\"NA\">, \"explanation\": \"<short explanation>\"}}"
        "\n\nNow evaluate the following:"
        "\nQuestion: {question}"
        "\nCriteria: {criteria}"
    )
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results: List[Dict[str, Any]] = []

    with open(output_path, "w", encoding="utf-8") as outfile:
        for question, criteria in all_criteria:
            criteria_text = "\n".join(f"- {criterion}" for criterion in criteria)
            prompt = FACTUALITY_EVAL_PROMPT_LIST_CRITERIA.format(
                question=question,
                criteria=criteria_text,
            )
            answer = ask_with_search(prompt)
            record: Dict[str, Any] = {
                "question": question,
                "criteria": criteria,
                "response": answer,
            }
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)
            print(answer)

    return results
        


if __name__ == "__main__":
    num_samples = 500
    all_rubric_ids = [
        "rl_rag_train_sqa_1k_clean_search_rubric_longform_rubrics",
        "rl-rag/rl_rag_train_sqa_1k_clean_dr_rubric_longform_rubrics",
        "rl-rag/rl_rag_train_sqa_1k_clean_cb_rubric_longform_rubrics"
    ]
    for rubric_id in all_rubric_ids:
        output_file = os.path.join(
            "open_instruct",
            "search_rewards",
            "analysis",
            "outputs",
            f"rubric_quality_analysis_{rubric_id}_{num_samples}.jsonl",
        )

        all_criteria = load_rubric(rubric_id, num_samples)
        score_factuality(all_criteria, output_file)
        print(f"Saved results to {output_file}")
