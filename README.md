# Training Open Instruction-following Language Models

This is the repository for the paper [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources
](https://arxiv.org/abs/2306.04751).

We explore instruction-tuning popular base models on publicly available datasets. This repository contains:
1. Training code used for training all models.
2. Evaluation code for the evaluation done in the paper.
3. Script for merging and creating model diffs.

As part of this work we introduce Tülu, a suite of LLaMa models fully-finetuned on a strong mix of datasets!

<p align="center">
<img src="images/tulu_logo.png" width="200" />
</p>

**Tülu 65B is the strongest model we build and available [here](https://huggingface.co/allenai/tulu-65b)** - see below for how to make use of this model yourself!

## Setup

You can install the required packages by running the following command (after installing pytorch):

```bash
pip install -r requirements.txt
```

If you just want the dependencies for the weight diff script, use:
```bash
pip install -r weight-diff-requirements.txt
```

### Model preparation

To get LLaMa checkpoints, please acquire them via Meta [here](https://docs.google.com/forms/d/e/1FAIpQLSfqNECQnMkycAp2jP4Z9TFX0cGR4uf7b_fBxjY_OjhJILlKGA/viewform) and consult [the Hugging Face documentation](https://huggingface.co/docs/transformers/model_doc/llama) for converting them to a huggingface-compatible format.

Generally, most huggingface-compatible models should work fine, potentially with some adjusting for different tokenizers etc.


## Weight Diff Script

We use a slightly modified form of the [Alpaca weight diff script](https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py), which runs the same.

To merge a model:
1. Download the relevant LLaMa model and convert it to Hugging Face format (see above).
2. Download our repository and install the right dependencies (see above).
3. Download the model diff you want.
4. Run the command below:

```bash
python scripts/weight_diff.py recover --path_raw ${hf_llama_path} --path_tuned ${output_path} --path_diff ${diff_location}
```

## Training

### Dataset Preparation

To download and prepare the instruction datasets we explore, use:

```bash
./scripts/prepare_train_data.sh
```

Please check these datasets for licenses and restrictions around their use!

### Finetuning
To run instruction tuning, you can use the following command:

```bash
./scripts/finetune_with_accelerate.sh
```

Adjust `model_name_or_path`, `tokenizer_name`, `train_file`, and `output_dir` to your models / data / setting. By default, this uses `deepspeed` with `accelerate`.

## Model Checkpoints

We provide a number of model checkpoints as diffs. You can find them on Hugging Face [here](https://huggingface.co/models?other=arxiv:2306.04751). They are also all here:

| **Model**                | **7B**                                                                         | **13B**                                                                         | **30B**                                                            | **65B**                                                            |
|--------------------------|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------|
| SuperNI                  | [link](https://huggingface.co/allenai/open-instruct-sni-7b)                    | [link](https://huggingface.co/allenai/open-instruct-sni-13b)                    |                                                                    |                                                                    |
| CoT                      | [link](https://huggingface.co/allenai/open-instruct-cot-7b)                    | [link](https://huggingface.co/allenai/open-instruct-cot-13b)                    |                                                                    |                                                                    |
| Flan V2                  | [link](https://huggingface.co/allenai/open-instruct-flan-v2-7b)                | [link](https://huggingface.co/allenai/open-instruct-flan-v2-13b)                |                                                                    |                                                                    |
| Dolly                    | [link](https://huggingface.co/allenai/open-instruct-dolly-7b)                  | [link](https://huggingface.co/allenai/open-instruct-dolly-13b)                  |                                                                    |                                                                    |
| Open Assistant 1         | [link](https://huggingface.co/allenai/open-instruct-oasst1-7b)                 | [link](https://huggingface.co/allenai/open-instruct-oasst1-13b)                 |                                                                    |                                                                    |
| ShareGPT                 | [link](https://huggingface.co/allenai/open-instruct-sharegpt-7b)               | [link](https://huggingface.co/allenai/open-instruct-sharegpt-13b)               | [link](https://huggingface.co/allenai/open-instruct-sharegpt-30b)  | [link](https://huggingface.co/allenai/open-instruct-sharegpt-65b)  |
| Self-instruct (original) | [link](https://huggingface.co/allenai/open-instruct-self-instruct-7b)          | [link](https://huggingface.co/allenai/open-instruct-self-instruct-13b)          |                                                                    |                                                                    |
| Unnatural Instructions   | [link](https://huggingface.co/allenai/open-instruct-unnatural-instructions-7b) | [link](https://huggingface.co/allenai/open-instruct-unnatural-instructions-13b) |                                                                    |                                                                    |
| Alpaca                   | [link](https://huggingface.co/allenai/open-instruct-stanford-alpaca-7b)        | [link](https://huggingface.co/allenai/open-instruct-stanford-alpaca-13b)        |                                                                    |                                                                    |
| Code-Alpaca              | [link](https://huggingface.co/allenai/open-instruct-code-alpaca-7b)            | [link](https://huggingface.co/allenai/open-instruct-code-alpaca-13b)            |                                                                    |                                                                    |
| GPT4-Alpaca              | [link](https://huggingface.co/allenai/open-instruct-gpt4-alpaca-7b)            | [link](https://huggingface.co/allenai/open-instruct-gpt4-alpaca-13b)            |                                                                    |                                                                    |
| Baize                    | [link](https://huggingface.co/allenai/open-instruct-baize-7b)                  | [link](https://huggingface.co/allenai/open-instruct-baize-13b)                  |                                                                    |                                                                    |
| Human-Mix                | [link](https://huggingface.co/allenai/open-instruct-human-mix-7b)              | [link](https://huggingface.co/allenai/open-instruct-human-mix-13b)              | [link](https://huggingface.co/allenai/open-instruct-human-mix-30b) | [link](https://huggingface.co/allenai/open-instruct-human-mix-65b) |
| **Tulu**                 | [link](https://huggingface.co/allenai/tulu-7b)                                 | [link](https://huggingface.co/allenai/tulu-13b)                                 | [link](https://huggingface.co/allenai/tulu-30b)                    | [link](https://huggingface.co/allenai/tulu-65b)                    |

We also trained Pythia and OPT models on the Tulu mixture (aka the Human+GPT mixture), and they are available here:
- [Pythia 6.9B Tulu](https://huggingface.co/allenai/open-instruct-pythia-6.9b-tulu)
- [OPT 6.7B Tulu](https://huggingface.co/allenai/open-instruct-opt-6.7b-tulu)

## Evaluation

First, run the following script to download all the evaluation datasets:

```bash
./scripts/prepare_eval_data.sh
```

Evaluation scripts for different datasets are put under `./scripts`. For example, you can use the following command to run the MMLU evaluation script:

```bash
./scripts/eval/mmlu.sh
```

### AlpacaFarm

To run AlpacaFarm eval, please make sure you install our fork of AlpacaFarm (https://github.com/hamishivi/alpaca_farm) and use the following script:
```bash
python eval/alpaca_farm_eval.py --model <model> --batch_size 8
```

Please check the script for more details on the script itself!

### Human Evaluation Interface

Coming soon!

### Licensing

The is licensed under Apache 2.0 as given in `LICENSE`.

The license we use for the models released (along with the base model licenses) can be found in `model_licenses/tulu_license.txt` - just replace `<MODELNAME>` with the actual model name (i.e., the name on HuggingFace).

# Citation

If you used this repository or our models, please cite our work:
```
@misc{wang2023far,
      title={How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources}, 
      author={Yizhong Wang and Hamish Ivison and Pradeep Dasigi and Jack Hessel and Tushar Khot and Khyathi Raghavi Chandu and David Wadden and Kelsey MacMillan and Noah A. Smith and Iz Beltagy and Hannaneh Hajishirzi},
      year={2023},
      eprint={2306.04751},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

