# Synthetic Preference Pipeline

This directory contains the implementation of the synthetic data pipeline for Tulu 3.
This pipeline is based on the Ultrafeedback pipeline ([Cui et al., 2023](https://arxiv.org/abs/2310.01377)) but with modifications such as the inclusion of on-policy data during data generation, and the use of GPT-4o for preference annotation.

Here's an overview of the pipeline (and how each script corresponds to each component):

![](https://github.com/allenai/open-instruct/blob/main/scripts/synth_pref/assets/ufpp_pipeline_v2_normal.png)
![](https://github.com/allenai/open-instruct/blob/main/scripts/synth_pref/assets/ufpp_pipeline_v2_code.png)


## Setup

You need to install specific dependencies for this pipeline:

```sh
python3 -m venv venv
source venv/bin/activate
pip install -r scripts/synth_pref/requirements.txt
```

We also use the open-source [Batch Inference Runtime (birr) tool](https://github.com/allenai/birr) to handle all calls to VLLM.
In your current directory, run the following:

```sh
git clone git@github.com:allenai/birr.git
git checkout 72e1c14
```

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
    --source_file "example/generate_responses_in/*.jsonl" \
    --target_dir "example/generate_responses_out/" \
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
    --input_dir example/generate_responses_out/ \
    --output_dir example/create_annotation_mix_out/ \
    --prompt_template ultrafeedback
```

If you want to create a subset of on-policy data, you can pass a model name in `--one_side_model`.
This will ensure that one of the responses is from that on-policy checkpoint, while the other response will be sampled from the remaining.

### Preference annotation

Once you've created the annotation mix, you can now perform LLM-as-a-judge!
We use OpenAI's Batch API to query large sets of prompts, so first set your token:

```sh
export OPENAI_API_KEY=<your key>
```

Then, let's convert the annotation mix to the format desired by the Batch API:

```sh
python3 -m scripts.synth_pref.annotate_preferences \
    --model gpt-4o-2024-08-06 \
    --input_path create_annotation_mix_out/
    --output_dir create_annotation_mix_out/batch_openai
    --rows_per_shard 10000
```

This command will shard our annotation mix to `n` files with `10000` rows to fit OpenAI's file limits.
Then, we can send the annotations to OpenAI now:

```sh
python3 -m scripts.synth_pref.annotate_preferences \
    --model gpt-4o-2024-08-06 \
    --input_dir create_annotation_mix_out/batch_openai/
    --output_dir target_dir
```

This part may take some time, in the typical OpenAI API, the maximum wait time is 24 hours.
In addition, this command will save a `batch_infer_openai_results.csv` that keeps track of which file and which batch was sent to the server.
You can poll the file to see if they're done and it will automatically download the results via:

```sh
python3 -m scripts.synth_pref.annotate_preferences \
    --model gpt-4o-2024-08-06 \
    --batch_report create_annotation_mix_out/batch_infer_openai_results.csv
    --output_dir create_annotation_mix_out/batch_results/
```

If all files are done and downloaded, you can start parsing the preferences to obtain the final preference dataset:

```sh
python3 -m scripts.synth_pref.parse_preferences \
    --input_dir create_annotation_mix_out/batch_results \
    --output_path final_preference_dataset.yaml
```
