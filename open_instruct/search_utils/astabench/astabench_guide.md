# Asta-Bench Evaluation Guide

## Setup

Requires Ai2 credentials.
Based on [this google doc](https://docs.google.com/document/u/1/d/1E1v4wLAsW56AsKVEsLJBLXFMN8AFMqKxAPb2vPoqWUw/edit?tab=t.0).

First, setup astabench:
```bash
git clone https://github.com/allenai/asta-bench.git
# (preferred) use uv to just manage everything
uv sync
# or (weka conda setup)
conda create -n astabench python=3.11
pip install -e .
pyenv local miniconda3-3.9-24.1.2-0/envs/asta-bench2
```

The Ai2 credentials you need (check google doc for how to get, or ask Hamish):
```bash
export S2_API_KEY=
export OPENAI_API_KEY=
export ANTHROPIC_API_KEY=
export MODAL_TOKEN=
export MODAL_TOKEN_SECRET=
export HF_TOKEN=
export GOOGLE_API_KEY=
export GEMINI_API_KEY=
```

Then, download the dev/test data:
```bash
wget https://huggingface.co/datasets/allenai/asta-bench/resolve/main/tasks/sqa/rubrics_v1_recomputed.json
wget https://huggingface.co/datasets/allenai/asta-bench/resolve/main/tasks/sqa/rubrics_v2_recomputed.json
```

`v1` is the dev data, `v2` is the test data.


Finally, make sure you have open instruct installed. Follow the instructions in open-instructs README.

## Running an eval

1. You need generations!
```bash
export S2_API_KEY=xxxx
python open_instruct/search_utils/toolvllm_search_generate.py \
    --json_path rubrics_v1_recomputed.json \
    --model_path <model_path> \
    --output_dir asta_env_<model_name> \
    --num_docs 3 \
    --search_api_endpoint https://api.semanticscholar.org/graph/v1/snippet/search \
    --use_astabench_format # optional -- do our own citation extraction (a bit buggy)
```

`num_docs` controls how many documents get returned for each query.

This command generates samples into `asta_env_<model_name>/predictions.jsonl`.
Alternatively, `asta_env_<model_name>/astabench_formatted_predictions.json` will have custom asta-bench formatted answers if you set `--use_astabench_format` (and the old predictions file will still exist).
To then run asta-bench eval over it, we run:

INSPECT_EVAL_LOG_FILE_PATTERN=${model_name} uv run --extra sqa inspect eval astabench/evals/sqa/task.py@sqa --display plain -T simplified_eval=true -T assess_jointly=true --max-samples 12 --max-connections 16 --solver astabench/solvers/sqa/debug/cached_solver.py@cache_solver -S path="{formatted_answer_file_path}" -T sentence_wise_cit_eval=false -T all_at_once=true -T split='test' -T scorer_model="google/gemini-2.5-flash"

```bash
path_to_predictions=asta_env_<model_name>/predictions.jsonl
inspect eval astabench/evals/sqa/task.py@sqa --solver astabench/solvers/sqa/general_memorized/memorized_solver.py@formatted_solver --display plain -T simplified_eval=true -T assess_jointly=true --max-samples 12 --max-connections 16 -S sys_name_or_path=${path_to_predictions} -T sentence_wise_cit_eval=false -T all_at_once=true -T split='test' -T scorer_model="google/gemini-2.5-flash"
```

If you want to use the astabench formatted samples (and so not rely on astabench extracting citations on its own), you can run:

```bash
path_to_predictions=asta_env_<model_name>/astabench_formatted_predictions.json
inspect eval astabench/evals/sqa/task.py@sqa --solver astabench/solvers/sqa/debug/cached_solver.py@cache_solver --display plain -T simplified_eval=true -T assess_jointly=true --max-samples 12 --max-connections 16 -S sys_name_or_path=${path_to_predictions} -T sentence_wise_cit_eval=false -T all_at_once=true -T split='test' -T scorer_model="google/gemini-2.5-flash"
```
