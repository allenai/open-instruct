# Training Open Instruction-Following Language Models

This repo serves as an open effort on instruction-tuning popular pretrained language models on publicly available datasets. We release this repo and will keep updating it with:

1. Code for finetuning language models with latest techniques and instruction datasets in a unified format.
2. Code for running standard evaluation on a range of benchmarks, targeting for differnt capabilities of these language models.
3. Checkpoints or other useful artifacts that we build in our exploration.

Please see our first paper [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751) for more thoughts behind this project and our initial findings.

<p align="center" width="100%">
      <img src="images/tulu_logo.png" alt="Tülu (a hybrid camel) represents a suite of LLaMa models that we built by fully-finetuning them on a strong mix of datasets." style="width: 20%; min-width: 200px; display: block; margin: auto;">
</p>

## News

- [2023-09-26] We switched to use the official [alpaca-eval](https://github.com/tatsu-lab/alpaca_eval) library to run AlpacaFarm evaluation but use regenerated longer reference outputs. This will change our numbers reported in the paper. We will update the paper soon.
- [2023-09-25] Supported using [vLLM](https://github.com/vllm-project/vllm/) for our evaluations, which speeds up the evaluation by 10x.
- [2023-09-17] Supported [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314) finetuning. See [here](#parameter-efficient-finetuning) for more details.
- [2023-08-18] Added support for [ToxiGen](https://github.com/microsoft/TOXIGEN)/[TrutufulQA](https://github.com/sylinrl/TruthfulQA) evaluation. Check our `scripts/eval/` for examples of running them.
- [2023-08-08] Supported several new instruction dataset, including [LIMA](https://huggingface.co/datasets/GAIR/lima) / [WizardLM](https://github.com/nlpxucan/WizardLM) / [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca). See the [preparation script](./scripts/prepare_train_data.sh) for details. Performance hasn't been evaluated yet.
- [2023-08-06] Supported LLaMa 2 finetuning and FlashAttention-2 by bumping the version of transformers and many other dependencies.
- [2023-06-29] Added [licensing info](#licensing) for our released models.
- [2023-06-09] Released Tülu (a suite of LLaMa models fully-finetuned on a strong mix of datasets) and many other checkpoints on HuggingFace [[Links]](#released-checkpoints).
- [2023-06-09] Initial release of the codebase containing the training and evaluation code for our [arxiv paper](https://arxiv.org/abs/2306.04751).

## Setup

To run training, evaluation, or inference for our finetuned models, you need to install the required packages by running the following command (after installing pytorch):

```bash
pip install -r requirements.txt
```

If you just want the dependencies for the weight diff script, use:
```bash
pip install -r weight-diff-requirements.txt
```

## Training

### Dataset preparation

We include a collection of representative instruction datasets in our exploration and are adding new ones to our list. We unify them into the same chatting format. To download and prepare these datasets, simply run the following command:

```bash
./scripts/prepare_train_data.sh
```

Please check these datasets for licenses and restrictions around their use!

### Model preparation

Generally, most huggingface-compatible causal language models should work fine with our codebase, potentially with some adjusting for different tokenizers etc. Some models may require addtional requests to download. E.g., for LLaMa 1 and 2, please consult [the Hugging Face documentation](https://huggingface.co/docs/transformers/model_doc/llama) for requesting access and converting them to a huggingface-compatible format.

### Finetuning

You can use the following command to run instruction tuning (finetuning a pretrained model to follow instructions):

```bash
./scripts/finetune_with_accelerate.sh
```

Make sure to adjust `model_name_or_path`, `tokenizer_name`, `train_file`, and `output_dir` to your models / data / setting. By default, this uses `deepspeed` with `accelerate`.

### Parameter-Efficient Finetuning

We support [LoRA](https://arxiv.org/abs/2106.09685) finetuning, wherein only a small number of parameters are updated, resulting in faster and cheaper training. For even more efficiency, we also support [QLoRA](https://arxiv.org/abs/2305.14314) finetuning, wherein the non-trained (underlying) model parameters are quantised during 4-bit training. This means you can train a 70b Llama model on a single 80GB A100! Please refer to the respective papers for more details.

Please also note you cannot currently run QLoRA with model parallelism - only data-parallel training is supported, so you cannot train a model that does not fit on one GPU. For LoRA, you can use deepspeed + zero-3 to achieve model parallelism (and FSDP is not currently supported).

Please see `./scripts/finetune_lora_with_accelerate.sh` and `./scripts/finetune_qlora_with_accelerate.sh` for example hyperparameters. We found a larger rank (e.g. 256) and higher learning rate (e.g. 2e-4) worked best. Additionally, we found that QLoRA tended to always achieve similar results to LoRA, while LoRA itself sometimes fell behind full-finetuning, especially in long, complex generation tasks. However, for most purposes, LoRA training essentially matches full-finetuning performance. Curiously, we found that merging QLoRA modules back into the non-quantised model tended to result in slightly better performance.

## Released Checkpoints

We provide a number of model checkpoints that we trained. You can find them on Hugging Face [here](https://huggingface.co/models?other=arxiv:2306.04751). Here are some quick links to the checkpoints that are finetuned from LLaMa 1:

| **Datasets ↓ Model Sizes →**                | **7B**                                                                         | **13B**                                                                         | **30B**                                                            | **65B**                                                            |
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


### Weight diff script

Some of the checkpoints are released as weight diffs to the base model (mostly for LLaMa 1). We use a slightly modified form of the [Alpaca weight diff script](https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py), which runs the same.

To merge a model:
1. Download the relevant LLaMa model and convert it to Hugging Face format (see above).
2. Download our repository and install the right dependencies (see above).
3. Download the model diff you want.
4. Run the command below:

```bash
python scripts/weight_diff.py recover --path_raw ${hf_llama_path} --path_tuned ${output_path} --path_diff ${diff_location}
```

## Evaluation

### Benchmark-based eval

We provide the scripts for running evaluation of Huggingface/OpenAI models on a list of standard benchmarks targeting for the core capabilities of large language models. These benchmakrs include:

- [MMLU](https://github.com/hendrycks/test)
- [Grade School Math (GSM)](https://github.com/openai/grade-school-math)
- [Big-Bench Hard (BBH)](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main)
- [TydiQA](https://github.com/google-research-datasets/tydiqa)
- [Codex HumanEval](https://github.com/openai/human-eval/tree/master)
- [ToxiGen](https://github.com/microsoft/TOXIGEN)

We are working on including more promising benchmarks into this list. Please stay tuned!

You can use the following script to download all the evaluation data:

```bash
./scripts/prepare_eval_data.sh
```

Evaluation scripts for different datasets are put under `./scripts`. For example, you can use the following command to run the MMLU evaluation script:

```bash
./scripts/eval/mmlu.sh
```

### Model-based eval

We support using GPT4 to evaluate the quality of model's response following the GPT4 evaluation protocol proposed in [AlpacaFarm](https://arxiv.org/abs/2305.14387). To run this AlpacaFarm eval, please make sure you install our fork of AlpacaFarm (https://github.com/hamishivi/alpaca_farm) and use the following script:

```bash
python eval/alpaca_farm_eval.py --model <model> --batch_size 8
```

Please check the script for more details on the script itself!

### Human evaluation

We will release our human evaluation interface and data soon!

## Licensing

This codebase is licensed under Apache 2.0 as given in [LICENSE](./LICENSE).

The license we use for the models released (along with the base model licenses) can be found in [model_licenses/tulu_license.txt](./model_licenses/tulu_license.txt) - just replace `<MODELNAME>` with the actual model name (i.e., the name on HuggingFace).

## Citation

If you used this repository or our models, please cite our work:

```bibtex
@misc{wang2023far,
   title={How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources}, 
   author={Yizhong Wang and Hamish Ivison and Pradeep Dasigi and Jack Hessel and Tushar Khot and Khyathi Raghavi Chandu and David Wadden and Kelsey MacMillan and Noah A. Smith and Iz Beltagy and Hannaneh Hajishirzi},
   year={2023},
   eprint={2306.04751},
   archivePrefix={arXiv},
   primaryClass={cs.CL}
}
```
