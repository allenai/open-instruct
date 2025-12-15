# Azure OpenAI Batch Processing Scripts

This directory contains scripts for processing datasets using Azure OpenAI's Batch API. These scripts help you regenerate completions for datasets, monitor batch jobs, and process the results.

## Prerequisites

- Python 3.x
- Azure OpenAI API access with the following environment variables set:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `HF_TOKEN` (for uploading to Hugging Face)

## Scripts Overview

Install dependencies by creating a virtual env and `pip` installing the `requirements.txt` file from the root of the project.
I like to use [`uv`](https://github.com/astral-sh/uv):

```bash
uv venv
uv pip install -r requirements.txt
```

Then, you can either activate the `venv` or run the scripts with `uv run python script.py` (this acts the same as `python script.py`).

### 1. `regenerate_dataset_completions.py`

Regenerates completions for a dataset using Azure OpenAI's Batch API.

**Usage:**

```bash
python regenerate_dataset_completions.py [options]
```

**Key Options:**
- `--input-dataset`: Source dataset name (default: "allenai/tulu-3-sft-personas-code")
- `--split`: Dataset split to use (default: "train")
- `--sample-limit`: Limit number of samples to process
- `--model`: Model to use (default: "o3-batch")
- `--max-completion-tokens`: Maximum completion tokens (default: 8192)
- `--dry-run`: Preview without making API calls

### 2. `check_azure_batch_status.py`

Monitors the status of a batch API submission.

**Usage:**
```bash
# Check status once
python check_azure_batch_status.py <batch_id>

# Watch until completion
python check_azure_batch_status.py <batch_id> --watch
```

### 3. `process_azure_batch_results.py`

Processes batch results and creates a new dataset with updated completions.

**Usage:**
```bash
python process_azure_batch_results.py <batch_id> \
    --input-dataset <source_dataset> \
    --output-dataset <target_dataset> \
    --split <split_name> \
    [--no-upload]
```

## Typical Workflow

1. Use `regenerate_dataset_completions.py` to submit a batch job:
   ```bash
   python regenerate_dataset_completions.py --input-dataset your-dataset --split train
   ```

2. Monitor the batch job status:
   ```bash
   ./check_azure_batch_status.py <batch_id> --watch
   ```

3. Process the results and create a new dataset:
   ```bash
   python process_azure_batch_results.py <batch_id> \
       --input-dataset your-dataset \
       --output-dataset your-username/new-dataset \
       --split train
   ```

## Notes

- Batch jobs have a 24-hour completion window
- Maximum of 95,000 prompts per batch file
- Token usage and costs are tracked and reported
- Error handling and reporting is included in all scripts

If you have a dataset with more than 95k prompts, `regenerate_dataset_completions.py` will automatically split it up into multiple batches, and `process_azure_batch_results.py` will be able to combine them.

## Credentials

There are two main docs with the credentials you need:

- [NAIRR Pilot Azure AI Foundry Access (General Post-training)](https://docs.google.com/document/d/12fZEjqfopzi6hDroXrtgSIo00kpgMTLFs71-Lc2kkNI/edit?tab=t.0#heading=h.5m6cr2r4j0m) for access to GPT 4o, 4o-mini, and 4.1.
- [NAIRR Pilot Azure AI Foundry Access for using OpenAI Models for OLMo](https://docs.google.com/document/d/1PKygtkH-JmvayUwXQ-QaI_wj58P1KAF4uGRSh6yrBqs/edit?tab=t.0)  for access to GPT 4o, 4o-mini, 4.1, o3, o4-mini.

Reach out to [finbarrt@](finbarrt@allenai.org) for access to these docs if you don't have it.

## Validation

I have a [Colab](https://colab.research.google.com/drive/1rGmHyjIwlpg7T81RR9HSCJtX8YG7OQEi?usp=sharing) with a bunch of sanity checks when creating a new dataset.
