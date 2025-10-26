import os
import json
import re


keywords = ["dataset", "benchmark", "evaluation", "leaderboard", "metric"]

# rubric_path = "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_16_adaptive_3_active__1__1761288686/adaptive_rubrics_toy_rag_survey_bs_1_rollout_16_adaptive_3_active__1__1761288686.jsonl"
# rubric_path = "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_16_1_sample_likert__1__1761295862"
rubric_path = "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810"
rubric_path = "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761279009/"
rubric_path = "/checkpoint/comem/rulin/open-instruct/output_toy_rag_survey_analysis/toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810/adaptive_rubrics_toy_rag_survey_bs_1_rollout_8_1_sample_search_based__1__1761283810.jsonl"

if "adaptive" in rubric_path:
    with open(rubric_path, "r") as f:
        data = [json.loads(line) for line in f]

    print(len(data))
    print(data[0].keys())
    print(data[0]["adaptive_rubric_scores"])

    positive_code_related_rubrics = []
    negative_code_related_rubrics = []
    for item in data:
        for rubric in item["adaptive_rubric_scores"][0]["positive_rubrics"]:
            if any(keyword in rubric["description"].lower() for keyword in keywords):
                positive_code_related_rubrics.append(rubric)
        for rubric in item["adaptive_rubric_scores"][0]["negative_rubrics"]:
            if any(keyword in rubric["description"].lower() for keyword in keywords):
                negative_code_related_rubrics.append(rubric)

    print("Positive keyword-related rubrics: ", len(positive_code_related_rubrics))
    print("Negative keyword-related rubrics: ", len(negative_code_related_rubrics))
    print("-"*100)
    print(positive_code_related_rubrics[:3])
    print("-"*100)
    print(negative_code_related_rubrics[:3])

    data_dir = os.path.dirname(rubric_path)
else:
    data_dir = rubric_path

all_responses_filenames = os.listdir(data_dir)
step_range = (1, 100)
num_keywords_responses = 0
total_responses = 0
for response_filename in all_responses_filenames:
    # filename: eval_step_i.json
    pattern = r"eval_step_(\d+).json"
    match = re.search(pattern, response_filename)
    if match:
        step = int(match.group(1))
        if step < step_range[0] or step > step_range[1]:
            continue
        response_path = os.path.join(data_dir, response_filename)
        with open(response_path, "r") as f:
            responses = json.load(f)
            responses = [item["response"] for item in responses["samples"]]
            for response in responses:
                if any(keyword in response.lower() for keyword in keywords):
                    num_keywords_responses += 1
                total_responses += 1

print(total_responses)
print(num_keywords_responses)
print("Ratio of keywords responses: {:.2f}%".format(num_keywords_responses / total_responses * 100))