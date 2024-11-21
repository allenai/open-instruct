## Tulu 1 and Tulu 2 Documentation


Note Tulu 1/2 results used an ealier version of Open Instruct with a pinned version of Transformers. If you are looking to replicate these results, refer to [this commit or older](https://github.com/allenai/open-instruct/commit/f3424591638ed63b31d5869abd867932c359c1ed).


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
python scripts/weights/weight_diff.py recover --path_raw ${hf_llama_path} --path_tuned ${output_path} --path_diff ${diff_location}
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
./scripts/data/prepare_eval_data.sh
```

Evaluation scripts for different datasets are put under `./scripts`. For example, you can use the following command to run the MMLU evaluation script:

```bash
./scripts/eval/mmlu.sh
```


### Human evaluation

We release our human evaluation interface and collected annotations in the `./human_eval` folder. Please see the corresponding [README](./human_eval/README.md) for more details.


## Training

### Dataset preparation

We include a collection of representative instruction datasets in our exploration and are adding new ones to our list. We unify them into the same chatting format. To download and prepare these datasets, simply run the following command:

```bash
./scripts/data/prepare_train_data.sh
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
