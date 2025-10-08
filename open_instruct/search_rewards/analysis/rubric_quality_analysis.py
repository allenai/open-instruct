import re
import json
import random
from typing import Any, Dict, List, Optional, Tuple
from datasets import load_dataset

import os
from openai import OpenAI, OpenAIError


FACTUALITY_EVAL_PROMPT_LIST_CRITERIA = (
    "You are a careful evaluator that determines whether each criterion contains a factual claim, "
    "and if so, whether that claim is factually correct and verifiable."
    "\n\nInstructions:"
    "\n1. Read the question and the list of criteria carefully."
    "\n2. For each criterion, decide first whether it *makes a factual claim* — that is, whether it asserts "
    "something that can be verified as true or false in the real world."
    "\n\n   **Distinguishing referential vs. assertive phrasing:**"
    "\n   - Referential (→ NA): Criteria that only ask to *mention*, *explain*, *describe*, *discuss*, or *include information about* something, "
    "without specifying what that information should be. These refer to factual topics but do not assert any particular fact."
    "\n     - Example: 'Explain the principle of masked diffusion models.' → NA (requests explanation, not asserting the content)."
    "\n     - Example: 'Mention information about A.' → NA."
    "\n   - Assertive (→ factual claim): Criteria that *state or imply a specific fact*, relationship, or property that could be true or false. "
    "They assert content, not just reference it."
    "\n     - Example: 'Masked diffusion models use random masking during the denoising process.' → factual claim."
    "\n     - Example: 'A is located in B.' → factual claim."
    "\n3. If the criterion is about writing style, tone, clarity, structure, or formatting, or if it only requires mentioning or explaining topics "
    "without specifying factual assertions, return 'NA'."
    "\n4. For each factual claim, check whether it can be verified using reliable evidence or reasoning."
    "\n   - If evidence confirms it → factual and correct."
    "\n   - If reliable evidence contradicts it → factual but incorrect."
    "\n   - If no verifiable evidence is found (e.g., no data, no known sources) → factual but *unverified*."
    "\n5. Compute the factuality score as:"
    "\n   - 1 → All verifiable factual claims are correct."
    "\n   - Between 0 and 1 → Some verifiable factual claims are correct, others are incorrect (average them)."
    "\n   - 0 → All verifiable factual claims are incorrect."
    "\n   - 'NA' → None of the criteria make any factual claims."
    "\n6. Do *not* lower the score for claims that are unverified (i.e., lacking evidence) unless there is evidence showing they are *false*."
    "\n7. Also count how many criteria are assertive but unverified."
    "\n\nOutput Format:"
    "\nReturn your result strictly in JSON format as follows:"
    "\n{{"
    "\"factual_score\": <float_or_\"NA\">, "
    "\"explanation\": \"<short explanation>\", "
    "\"num_non_na_criteria\": <number>, "
    "\"num_na_criteria\": <number>, "
    "\"num_unverified_assertive_criteria\": <number>"
    "}}"
    "\n\nNow evaluate the following:"
    "\nQuestion: {question}"
    "\nCriteria: {criteria}"
)


# Set API key (or via environment variable)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def ask_with_search(prompt: str):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-search-preview",
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


def load_rubric(rubric_id: str, num_samples: int, format: str = "v1", verbose: bool = True) -> List[str]:
    rubric_data = load_dataset("rl-rag/"+rubric_id, split="train")
    all_criteria = []
    for example in rubric_data:
        if format == "v1":
            question = json.loads(example["ground_truth"])["Question"]
            criteria = []
            for key in ["Answer Critical"]: #, "Valuable", "Context"]:
                per_sample_criteria = json.loads(example["ground_truth"])[key]
                for criterion in per_sample_criteria:
                    criteria.append(criterion["Ingredient"])
        elif format == "v2":
            question = json.loads(example["ground_truth"])["query"]
            criteria = []
            for key in ["rubrics"]:
                per_sample_criteria = json.loads(example["ground_truth"])[key]
                for criterion in per_sample_criteria:
                    criteria.append(criterion["description"])
        all_criteria.append((question, criteria))

    if verbose:
        print(f"Number of questions: {len(all_criteria)}")
        newline = "\n"
        print(f"Few examples of questions: {newline.join([question for question, _ in all_criteria[:5]])}")

    random.seed(42)
    random.shuffle(all_criteria)
    all_criteria = all_criteria[:num_samples]
    return all_criteria


def score_factuality(
    all_criteria: List[Tuple[str, List[str]]],
    output_path: str,
) -> List[Dict[str, Any]]:
    
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
        

def check_existence(output_file: str, num_samples: int) -> bool:
    if not os.path.exists(output_file):
        return False
    else:
        with open(output_file, "r") as f:
            return len(f.readlines()) == num_samples


def parse_response(response: str) -> Dict[str, Any]:
    try:
        # First try to parse as JSON directly
        parsed = json.loads(response)
        return parsed
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from markdown code blocks
        try:
            # Look for JSON content within ```json ... ``` blocks
            json_match = re.search(r'```json\s*\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                parsed = json.loads(json_content)
                return parsed
            else:
                # If no markdown block found, try to extract factual_score using regex
                # Look for factual_score pattern in the response
                score_match = re.search(r'"factual_score":\s*(["\']?(?:0|0\.5|1|NA)["\']?)', response)
                if score_match:
                    score_value = score_match.group(1).strip('"\'')
                    # Try to extract explanation as well
                    explanation_match = re.search(r'"explanation":\s*"([^"]*)', response)
                    explanation = explanation_match.group(1) if explanation_match else "Partial response - explanation truncated"
                    num_non_na_match = re.search(r'"num_non_na_criteria":\s*(\d+)', response)
                    num_na_match = re.search(r'"num_na_criteria":\s*(\d+)', response)
                    num_unverified_match = re.search(r'"num_unverified_assertive_criteria":\s*(\d+)', response)
                    parsed_response: Dict[str, Any] = {"factual_score": score_value, "explanation": explanation}
                    if num_non_na_match and num_na_match:
                        parsed_response["num_non_na_criteria"] = int(num_non_na_match.group(1))
                        parsed_response["num_na_criteria"] = int(num_na_match.group(1))
                    if num_unverified_match:
                        parsed_response["num_unverified_assertive_criteria"] = int(num_unverified_match.group(1))
                    return parsed_response
                else:
                    print(f"Failed to parse response: {response}")
                    raise Exception(f"Failed to parse response: {response}")
        except json.JSONDecodeError:
            # Last resort: try to extract factual_score using regex
            score_match = re.search(r'"factual_score":\s*(["\']?(?:0|0\.5|1|NA)["\']?)', response)
            if score_match:
                score_value = score_match.group(1).strip('"\'')
                # Try to extract explanation as well
                explanation_match = re.search(r'"explanation":\s*"([^"]*)', response)
                explanation = explanation_match.group(1) if explanation_match else "Partial response - explanation truncated"
                num_non_na_match = re.search(r'"num_non_na_criteria":\s*(\d+)', response)
                num_na_match = re.search(r'"num_na_criteria":\s*(\d+)', response)
                num_unverified_match = re.search(r'"num_unverified_assertive_criteria":\s*(\d+)', response)
                parsed_response = {"factual_score": score_value, "explanation": explanation}
                if num_non_na_match and num_na_match:
                    parsed_response["num_non_na_criteria"] = int(num_non_na_match.group(1))
                    parsed_response["num_na_criteria"] = int(num_na_match.group(1))
                if num_unverified_match:
                    parsed_response["num_unverified_assertive_criteria"] = int(num_unverified_match.group(1))
                return parsed_response
            else:
                print(f"Failed to parse response even after extracting from markdown: {response}")
                raise Exception(f"Failed to parse response: {response}")
    

def compute_scores(output_file: str) -> Dict[str, Any]:
    with open(output_file, "r") as f:
        results = [json.loads(line) for line in f]
    
    num_na = 0
    num_failed_to_parse = 0
    non_na_factual_scores: List[float] = []
    total_num_non_na_criteria = 0
    total_num_na_criteria = 0
    total_num_unverified_assertive_criteria = 0
    num_missing_criteria_counts = 0
    num_missing_unverified_counts = 0
    for result in results:
        try:
            response = parse_response(result["response"])
        except Exception:
            num_failed_to_parse += 1
            continue

        factual_score_raw = response.get("factual_score")
        numeric_score: Optional[float] = None

        if factual_score_raw == "NA":
            num_na += 1
        else:
            try:
                numeric_score = float(factual_score_raw)
            except (TypeError, ValueError):
                num_failed_to_parse += 1

        if numeric_score is not None:
            non_na_factual_scores.append(numeric_score)

        non_na_count = response.get("num_non_na_criteria")
        na_count = response.get("num_na_criteria")
        unverified_count = response.get("num_unverified_assertive_criteria")
        if non_na_count is not None and na_count is not None:
            try:
                total_num_non_na_criteria += int(non_na_count)
                total_num_na_criteria += int(na_count)
            except (TypeError, ValueError):
                num_missing_criteria_counts += 1
        else:
            num_missing_criteria_counts += 1

        if unverified_count is not None:
            try:
                total_num_unverified_assertive_criteria += int(unverified_count)
            except (TypeError, ValueError):
                num_missing_unverified_counts += 1
        else:
            num_missing_unverified_counts += 1
    
    print(f"Num NA: {num_na}, fraction: {num_na / len(results)}")
    print(f"Num failed to parse: {num_failed_to_parse}, fraction: {num_failed_to_parse / len(results)}")
    if non_na_factual_scores:
        mean_non_na = sum(non_na_factual_scores) / len(non_na_factual_scores)
        print(f"Mean non-NA factual score: {mean_non_na}")
    else:
        print("Mean non-NA factual score: NA (no non-NA responses)")
    total_criteria = total_num_non_na_criteria + total_num_na_criteria
    if total_criteria > 0:
        print(
            f"Fraction assertive claims: {total_num_non_na_criteria / total_criteria} "
            f"({total_num_non_na_criteria}/{total_criteria})"
        )
    if total_num_non_na_criteria > 0:
        print(
            f"Fraction unverified assertive claims: "
            f"{total_num_unverified_assertive_criteria / total_num_non_na_criteria} "
            f"({total_num_unverified_assertive_criteria}/{total_num_non_na_criteria})"
        )
    if num_missing_criteria_counts:
        print(f"Missing criteria counts in {num_missing_criteria_counts} responses")
    if num_missing_unverified_counts:
        print(f"Missing unverified assertive counts in {num_missing_unverified_counts} responses")
    print("#"*30)
    return results
    

if __name__ == "__main__":
    num_samples = 300
    rubric_format = "v2"
    all_rubric_ids = [
        # "rl_rag_train_sqa_1k_clean_search_rubric_longform_rubrics",
        # "rl_rag_train_sqa_1k_clean_dr_rubric_longform_rubrics",
        # "rl_rag_train_sqa_1k_clean_cb_rubric_longform_rubrics"
        "RaR-Medicine-20k-o3-mini-converted"
    ]
    for rubric_id in all_rubric_ids:
        output_file = f"open_instruct/search_rewards/analysis/outputs_old4/rubric_quality_analysis_{rubric_id}_{num_samples}.jsonl"

        all_criteria = load_rubric(rubric_id, num_samples, format=rubric_format, verbose=False)
        if not check_existence(output_file, num_samples):
            score_factuality(all_criteria, output_file)
            print(f"Saved results to {output_file}")
        else:
            print(f"Results already exist for {rubric_id}, skipping...")
        
        compute_scores(output_file)
