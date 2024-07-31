# Training Open Instruction-Following Language Models

This repo serves as an open effort on instruction-tuning popular pretrained language models on publicly available datasets. We release this repo and will keep updating it with:

1. Code for finetuning language models with latest techniques and instruction datasets in a unified format.
2. Code for running standard evaluation on a range of benchmarks, targeting for differnt capabilities of these language models.
3. Checkpoints or other useful artifacts that we build in our exploration.

Please see our first paper [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751) for more thoughts behind this project and our initial findings. Please see our second paper [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702) for results using Llama-2 models and direct preference optimization. We are still working on more models. For more recent results involving PPO and DPO please see our third paper [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/abs/2406.09279).

<p align="center" width="100%">
      <img src="assets/images/tulu_logo.png" alt="Tülu (a hybrid camel) represents a suite of LLaMa models that we built by fully-finetuning them on a strong mix of datasets." style="width: 20%; min-width: 200px; display: block; margin: auto;">
</p>

*Note:* Previous versions of Open Instruct used a pinned version of Transformers for replicating Tulu 1/2 results. If this is your goal, refer to [this commit or older](https://github.com/allenai/open-instruct/commit/f3424591638ed63b31d5869abd867932c359c1ed).

## News

- [2024-07-01] We released [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/abs/2406.09279) and have majorly updated our codebase to support new models and package versions.
- [2023-11-27] We released [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702). Check out our models [here](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101). We have added a DPO finetuning script for replicating our results.
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

Installation is lightweight and assumes **one of two installation strategies**. 
First, installing in a *bare environment* (no Cuda image).

Before installing, if not in a Docker container with NVCC installed, you should run:
```
conda install cuda-nvcc=<ver> -c nvidia
```
Then, install `torch==2.3.0` from source.

To run training, evaluation, or inference for our finetuned models, you need to install the required packages by running the following command (after installing pytorch):

```bash
pip install -r requirements.txt
```
*Note:* Previous versions of Open Instruct used a pinned version of Transformers for replicating Tulu 2 results. If this is your goal, refer to [this commit or older](https://github.com/allenai/open-instruct/commit/f3424591638ed63b31d5869abd867932c359c1ed).

If you just want the dependencies for the weight diff script, use:
```bash
pip install -r weight-diff-requirements.txt
```

For a second installation strategy, if you'd like to *run experiments within a Docker environment*, you can create one using:

```bash
docker build --build-arg CUDA=12.1.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 --build-arg REQUIRE=requirements.txt . -t <your tag here>
```

If you are internally at AI2, you can use this pre-built beaker image `hamishivi/open-instruct-eval` (most recent version [here](https://beaker.org/im/01J2CKY81A6N1WG5QS08Y3WNM5/details)). For finetuning, you can use `hamishivi/open-instruct-public` (most recent version [here](https://beaker.org/im/01J2CQFX7076PDHZJR2GB0C3A9/details)). I will try to update these periodically.

**Important for OLMo users:** Note that due to version conflicts between deepspeed and vLLM, we cannot support OLMo inference and deepspeed within the same image (this will be fixed once deepspeed allows pydantic >= 2). To build a docker image suitable for inference/evaluation for OLMo, use:
```bash
docker build --build-arg CUDA=12.1.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 --build-arg REQUIRE=requirements-olmo.txt -f Dockerfile.olmo . -t <your tag here>
```

For training, you can use the previous image.

### Developing
When submitting a PR to this repo, we check the core code in `open_instruct/` for style with the following:
```
make style
make quality
```

### Repo structure
```
├── assets/                     <- Images, licenses, etc.
├── configs/                    
|     ├── beaker_configs/       <- AI2 Beaker configs
|     ├── ds_configs/           <- DeepSpeed configs
|     └── train_configs/        <- Training configs
├── eval/                       <- Evaluation suite for fine-tuned models
├── human_eval/                 <- Human evaluation interface (not maintained)
├── open_instruct/              <- Source code (flat)
├── quantize/                   <- Scripts for quantization
├── scripts/                    <- Core training and evaluation scripts
├── Dockerfile                  <- Main Dockerfile
└── Dockerfile.olmo             <- Dockerfile for OLMo users (version conflict currently.)
```

## Training

### Dataset preparation

We include a collection of representative instruction datasets in our exploration and are adding new ones to our list. We unify them into the same chatting format. To download and prepare these datasets, simply run the following command:

```bash
./scripts/prepare_train_data.sh
```

Please check these datasets for licenses and restrictions around their use!

You can also find the processed [Tulu v1](https://huggingface.co/datasets/allenai/tulu-v1-sft-mixture) and [Tulu v2](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture) SFT datasets on HuggingFace. Note that the train data preparation script will not precisely recreate the Tulu v2 mixture due to randomness in the generation and shifts in data availability - see [this PR](https://github.com/allenai/open-instruct/pull/156) for some details. If you need exactly yhe training data used, the HuggingFace mixture is exactly this - the exact same data used during model training.

### Model preparation

Generally, most huggingface-compatible causal language models should work fine with our codebase, potentially with some adjusting for different tokenizers etc. Some models may require addtional requests to download. E.g., for LLaMa 1 and 2, please consult [the Hugging Face documentation](https://huggingface.co/docs/transformers/model_doc/llama) for requesting access and converting them to a huggingface-compatible format.

### Finetuning

You can use the following command to run instruction tuning (finetuning a pretrained model to follow instructions):

```bash
./scripts/finetune_with_accelerate.sh
```

Make sure to adjust `model_name_or_path`, `tokenizer_name`, `train_file`, and `output_dir` to your models / data / setting. By default, this uses `deepspeed` with `accelerate`.

**Note:** If you are looking to replicate the released Tulu 2 models, it may be useful to swap the loss calculation to `--reduce_loss sum`. This uses a sum reduction instead of a mean reduction for loss calculations, and means we weight all tokens evenly when training, better mimicking the larger batch sizes used to train Tulu 2 models. See https://github.com/huggingface/transformers/issues/24725 for more discussion and details. Generally, *you may get better results using the sum reduction if you need to use a lot of gradient accumulation* (including for training Llama 3 models).

### Parameter-Efficient Finetuning

We support [LoRA](https://arxiv.org/abs/2106.09685) finetuning, wherein only a small number of parameters are updated, resulting in faster and cheaper training. For even more efficiency, we also support [QLoRA](https://arxiv.org/abs/2305.14314) finetuning, wherein the non-trained (underlying) model parameters are quantised during 4-bit training. This means you can train a 70b Llama model on a single 80GB A100! Please refer to the respective papers for more details.

Please also note you cannot currently run QLoRA with model parallelism - only data-parallel training is supported, so you cannot train a model that does not fit on one GPU. For LoRA, you can use deepspeed + zero-3 to achieve model parallelism (and FSDP is not currently supported).

Please see `./scripts/finetune_lora_with_accelerate.sh` and `./scripts/finetune_qlora_with_accelerate.sh` for example hyperparameters. We found a larger rank (e.g. 256) and higher learning rate (e.g. 2e-4) worked best. Additionally, we found that QLoRA tended to always achieve similar results to LoRA, while LoRA itself sometimes fell behind full-finetuning, especially in long, complex generation tasks. However, for most purposes, LoRA training essentially matches full-finetuning performance. We recommend merging modules learnt with QLoRA into a dequantised model (run our merge script with the `--qlora` flag).

## DPO Finetuning

For an example of how to fully finetune a model with DPO, see `scripts/dpo_train_with_accelerate.sh`. Note you will require at least 8 80GB A100s to be able to train a 7b size model, and will require more compute for anything larger. We have not tested multi-node training with this script, but it should work.

Our script also supports PEFT training with QLoRA. See `scripts/dpo_train_with_qlora.sh` for an example. We have not trained models with this, so it may require additional hyperparameter tuning to achieve reasonable results.

## Released Checkpoints

Our checkpoints can be found:

- [Here](https://huggingface.co/collections/hamishivi/tulu-v1-suite-655138c3743e6349aaa07d7d) for all Tulu v1 models.
- [Here](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101) for all Tulu v2 models.
- [OLMo 7B SFT](https://huggingface.co/allenai/OLMo-7B-SFT) and [Instruct](https://huggingface.co/allenai/OLMo-7B-Instruct), along with a [2048 sequence length version of Tulu 2](https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture-olmo-2048).


### Weight diff script

Our Tulu V1 models were released as weight diffs (due to LLaMa 1 license). We use a slightly modified form of the [Alpaca weight diff script](https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py), which runs the same.

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
- [MATH](https://github.com/hendrycks/math)
- [Big-Bench Hard (BBH)](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main)
- [TydiQA](https://github.com/google-research-datasets/tydiqa)
- [Codex HumanEval](https://github.com/openai/human-eval/tree/master)
- [HumanEval+ and MBPP+](https://github.com/evalplus/evalplus)
- [IFEval](https://github.com/google-research/google-research/tree/master/instruction_following_eval)
- [ToxiGen](https://github.com/microsoft/TOXIGEN)
- [XSTest](https://github.com/paul-rottger/exaggerated-safety/)
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [AlpacaEval 1 and 2](https://github.com/tatsu-lab/alpaca_eval)

We are working on including more promising benchmarks into this list. Please stay tuned!

You can use the following script to download all the evaluation data:

```bash
./scripts/prepare_eval_data.sh
```

Evaluation scripts for different datasets are put under `./scripts`. For example, you can use the following command to run the MMLU evaluation script:

```bash
./scripts/eval/mmlu.sh
```

### Human evaluation

We release our human evaluation interface and collected annotations in the `./human_eval` folder. Please see the corresponding [README](./human_eval/README.md) for more details.

## Licensing

This codebase is licensed under Apache 2.0 as given in [LICENSE](./LICENSE).

The license we use for V1 models released (along with the base model licenses) can be found in [assets/model_licenses/tulu_license.txt](./assets/model_licenses/tulu_license.txt) - just replace `<MODELNAME>` with the actual model name (i.e., the name on HuggingFace).

V2 models are licensed under the [low-risk AI2 ImpACT license](https://allenai.org/licenses/impact-lr). See [here](https://allenai.org/impact-license) for more details.

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

```bibtex
@misc{ivison2023camels,
      title={Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2}, 
      author={Hamish Ivison and Yizhong Wang and Valentina Pyatkin and Nathan Lambert and Matthew Peters and Pradeep Dasigi and Joel Jang and David Wadden and Noah A. Smith and Iz Beltagy and Hannaneh Hajishirzi},
      year={2023},
      eprint={2311.10702},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

```bibtex
@misc{ivison2024unpacking,
      title={Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback}, 
      author={Hamish Ivison and Yizhong Wang and Jiacheng Liu and Zeqiu Wu and Valentina Pyatkin and Nathan Lambert and Noah A. Smith and Yejin Choi and Hannaneh Hajishirzi},
      year={2024},
      eprint={2406.09279},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
}
```
