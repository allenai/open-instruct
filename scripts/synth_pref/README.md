# Synthetic Preference Pipeline

This directory contains the implementation of the synthetic data pipeline for Tulu 3.
This pipeline is based on the Ultrafeedback pipeline ([Cui et al., 2023](https://arxiv.org/abs/2310.01377)) but with modifications such as the inclusion of on-policy data during data generation, and the use of GPT-4o for preference annotation.

Here's an overview of the pipeline (and how each script corresponds to each component):

![](scripts/synth_pref/assets/ufpp_pipeline_v2_normal.png)
![](scripts/synth_pref/assets/ufpp_pipeline_v2_code.png)

## Setup

You need to install specific dependencies for this pipeline:

```sh
python3 -m venv venv
pip install -r scripts/synth_pref/requirements.txt
```

We also use the open-source Batch Inference Runtime (birr) tool to handle all calls to VLLM.
It is currently included in this repository as a submodule.

## How-to-use

### Dataset preparation for prompts

First, you need to prepare your prompts in a JSONL file with the following schema:

```
{"text": "Your text", **metadata}
```

Ideally, it is preferable to have multiple JSONL files with at most 250-500 rows each in a single directory so that `birr` can manage the queue more effectively.

> [!TIP]
> You can filter models by using the `--ignore_model` (`-x`) or `--include_model` (`-y`) tags.

### Response generation

First, let's generate configurations for `birr` (assuming your JSONL files in a directory called `source`):

```sh
python3 -m scripts.synth_pref.generate_responses \
    --name myprompts \
    --source_file "path/to/myprompts/*.jsonl" \
    --target_dir "path/to/outputs/" \
    --batch_size 128
```

This command will generate configuration files for each model that you can send to `birr`.
To do so, run the following command:

```sh
python3 src/birr/batch_inference/runner.py --config-file path/to/config/file.yaml
```

After running this command, you'll see all the outputs in the directory you specified for `--target_dir`.
From there, you can create a preference annotation mix that samples four (4) responses for each model:

```sh
python3 -m scripts.synth_pref.create_annotation_mix \
    --name myprompts \
    --input_dir path/to/birr/output/ \
    --output_dir path/to/annotation/output \
    --prompt_template ultrafeedback
```

If you want to create a subset of on-policy data, you can pass a model name in `--one_side_model`.
This will ensure that one of the responses is from that on-policy checkpoint, while the other response will be sampled from the remaining.

### Preference annotation
