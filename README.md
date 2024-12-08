# Training Open Instruction-Following Language Models

This repo serves as an open effort on instruction-tuning and post-training popular pretrained language models on publicly available datasets. We release this repo and will keep updating it with:

1. Code for finetuning language models with latest techniques and instruction datasets in a unified format.
2. Code for DPO, preference finetuning and reinforcement learning with verifiable rewards (RLVR).
3. Code for running standard evaluation on a range of benchmarks, targeting for differnt capabilities of these language models (now in conjunction with [OLMES](https://github.com/allenai/olmes)).
4. Checkpoints or other useful artifacts that we build in our exploration.

The lastest details on open post-training are found in [TÜLU 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124).

Please see our first paper [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751) for more thoughts behind this project and our initial findings. 
Please see our second paper [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702) for results using Llama-2 models and direct preference optimization. We are still working on more models. 
For more recent results involving PPO and DPO please see our third paper [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/abs/2406.09279).

<p align="center" width="100%">
      <img src="assets/images/tulu_logo.png" alt="Tülu (a hybrid camel) represents a suite of LLaMa models that we built by fully-finetuning them on a strong mix of datasets." style="width: 20%; min-width: 200px; display: block; margin: auto;">
</p>

Try some of the models we train with Open Instruct. There is a [free demo](https://playground.allenai.org/) or download them from HuggingFace:

| **Stage**           | **Llama 3.1 8B**                                                                                          | **Llama 3.1 70B**                                                                                         | **OLMo-2 7B**                                                                                          | **OLMo-2 13B**                                                                                         |
|----------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Base Model**       | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)                                | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B)                              | [allenai/OLMo2-7B-1124](https://huggingface.co/allenai/OLMo2-7B-1124)                                | [allenai/OLMo-2-13B-1124](https://huggingface.co/allenai/OLMo-2-13B-1124)                             |
| **SFT**              | [allenai/Llama-3.1-Tulu-3-8B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT)                | [allenai/Llama-3.1-Tulu-3-70B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-SFT)              | [allenai/OLMo-2-1124-7B-SFT](https://huggingface.co/allenai/OLMo-2-1124-7B-SFT)                | [allenai/OLMo-2-1124-13B-SFT](https://huggingface.co/allenai/OLMo-2-1124-13B-SFT)              |
| **DPO**              | [allenai/Llama-3.1-Tulu-3-8B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO)                | [allenai/Llama-3.1-Tulu-3-70B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-DPO)              | [allenai/OLMo-2-1124-7B-DPO](https://huggingface.co/allenai/OLMo-2-1124-7B-DPO)                | [allenai/OLMo-2-1124-13B-DPO](https://huggingface.co/allenai/OLMo-2-1124-13B-DPO)              |
| **Final Models (RLVR)** | [allenai/Llama-3.1-Tulu-3-8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B)                        | [allenai/Llama-3.1-Tulu-3-70B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B)                      | [allenai/OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)                        | [allenai/OLMo-2-1124-13B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct)                      |
| **Reward Model (RM)**| [allenai/Llama-3.1-Tulu-3-8B-RM](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM)                                                     | (Same as 8B)                                                     | [allenai/OLMo-2-1124-7B-RM](https://huggingface.co/allenai/OLMo-2-1124-7B-RM)                                                     | (Same as 7B)                                                     |

## News

- [2024-11-22] We released [TÜLU 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124) and updated our entire stack of open post-training recipes with both Llama 3.1 and OLMo 2.
- [2024-07-01] We released [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/abs/2406.09279) and have majorly updated our codebase to support new models and package versions.
- [2023-11-27] We released [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702). Check out our models [here](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101). We have added a DPO finetuning script for replicating our results.
- [2023-09-26] We switched to use the official [alpaca-eval](https://github.com/tatsu-lab/alpaca_eval) library to run AlpacaFarm evaluation but use regenerated longer reference outputs. This will change our numbers reported in the paper. We will update the paper soon.
- [2023-09-25] Supported using [vLLM](https://github.com/vllm-project/vllm/) for our evaluations, which speeds up the evaluation by 10x.
- [2023-09-17] Supported [LoRA](https://arxiv.org/abs/2106.09685) and [QLoRA](https://arxiv.org/abs/2305.14314) finetuning. See [here](#parameter-efficient-finetuning) for more details.
- [2023-08-18] Added support for [ToxiGen](https://github.com/microsoft/TOXIGEN)/[TruthfulQA](https://github.com/sylinrl/TruthfulQA) evaluation. Check our `scripts/eval/` for examples of running them.
- [2023-08-08] Supported several new instruction dataset, including [LIMA](https://huggingface.co/datasets/GAIR/lima) / [WizardLM](https://github.com/nlpxucan/WizardLM) / [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca). See the [preparation script](./scripts/data/prepare_train_data.sh) for details. Performance hasn't been evaluated yet.
- [2023-08-06] Supported LLaMa 2 finetuning and FlashAttention-2 by bumping the version of transformers and many other dependencies.
- [2023-06-29] Added [licensing info](#licensing) for our released models.
- [2023-06-09] Released Tülu (a suite of LLaMa models fully-finetuned on a strong mix of datasets) and many other checkpoints on HuggingFace [[Links]](#released-checkpoints).
- [2023-06-09] Initial release of the codebase containing the training and evaluation code for our [arxiv paper](https://arxiv.org/abs/2306.04751).

## Setup

Our setup mostly follows our [Dockerfile](./Dockerfile), which uses Python 3.10. *Note that Open Instruct is a research codebase and does not guarantee backward compatibility.* We offer two installation strategies:

* **Local installation**: This is the recommended way to install Open Instruct. You can install the dependencies by running the following commands:
```bash
pip install --upgrade pip "setuptools<70.0.0" wheel 
# TODO, unpin setuptools when this issue in flash attention is resolved
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install packaging
pip install flash-attn==2.6.3 --no-build-isolation
pip install -r requirements.txt
python -m nltk.downloader punkt
pip install -e .
```


* **Docker installation**: You can also use the Dockerfile to build a Docker image. You can build the image with the following command:

```bash
docker build --build-arg CUDA=12.1.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 . -t open_instruct_dev
# if you are interally at AI2, you can create an image like this:
beaker image delete $(whoami)/open_instruct_dev 
beaker image create open_instruct_dev -n open_instruct_dev -w ai2/$(whoami)
```

If you are internally at AI2, you may launch experiments using our always-up-to-date auto-built image `nathanl/open_instruct_auto`.


## Training

After having setup the environment, you are ready to launch some experiments. We provide a few examples below. To learn more about how to reproduce the Tulu 3 models, please refer to the [Tulu 3 README](./docs/tulu3.md). The instructions and documentations for Tulu 1 and Tulu 2 are in [Tulu 1 and 2 README](./docs/tulu1_tulu2.md).

### Finetuning

You can run the following commands for getting started:

```bash
# quick debugging run using 1 GPU
sh scripts/finetune_with_accelerate_config.sh 1 configs/train_configs/sft/mini.yaml
# train an 8B tulu3 model using 8 GPU
sh scripts/finetune_with_accelerate_config.sh 8 configs/train_configs/tulu3/tulu3_sft.yaml
```


### Preference Tuning

```bash
# quick debugging run using 1 GPU
sh scripts/dpo_train_with_accelerate_config.sh 1 configs/train_configs/dpo/mini.yaml
# train an 8B tulu3 model using 8 GPU
sh scripts/dpo_train_with_accelerate_config.sh 8 configs/train_configs/tulu3/tulu3_dpo_8b.yaml
```


### Reinforcement Learning with Verifiable Rewards (RLVR)

```bash
# quick debugging run using 2 GPU (1 for inference, 1 for training)
# here we are using `HuggingFaceTB/SmolLM2-360M-Instruct`; it's prob not
# gonna work, but it's easy to test run and print stuff.
python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer '{"ai2-adapt-dev/gsm8k_math_ifeval_ground_truth_mixed": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"ai2-adapt-dev/gsm8k_math_ground_truth": 1.0}' \
    --dataset_eval_splits test \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path HuggingFaceTB/SmolLM2-360M-Instruct \
    --reward_model_path HuggingFaceTB/SmolLM2-360M-Instruct \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 10000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 32 \
    --local_rollout_batch_size 32 \
    --num_epochs 1 \
    --actor_num_gpus_per_node 1 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir output/rlvr_1b \
    --seed 3 \
    --num_evals 3 \
    --save_freq 100 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking

# train an 8B tulu3 model using 8 GPU (1 for inference, 7 for training)
python open_instruct/ppo_vllm_thread_ray_gtrl.py \
    --dataset_mixer '{"ai2-adapt-dev/gsm8k_math_ifeval_ground_truth_mixed": 1.0}' \
    --dataset_train_splits train \
    --dataset_eval_mixer '{"ai2-adapt-dev/gsm8k_math_ground_truth": 1.0}' \
    --dataset_eval_splits test \
    --max_token_length 2048 \
    --max_prompt_token_length 2048 \
    --response_length 2048 \
    --model_name_or_path allenai/Llama-3.1-Tulu-3-8B-DPO \
    --reward_model_path allenai/Llama-3.1-Tulu-3-8B-RM \
    --non_stop_penalty \
    --stop_token eos \
    --temperature 1.0 \
    --ground_truths_key ground_truth \
    --chat_template tulu \
    --sft_messages_key messages \
    --learning_rate 3e-7 \
    --total_episodes 10000000 \
    --penalty_reward_value -10.0 \
    --deepspeed_stage 3 \
    --per_device_train_batch_size 2 \
    --local_rollout_forward_batch_size 2 \
    --local_mini_batch_size 32 \
    --local_rollout_batch_size 32 \
    --actor_num_gpus_per_node 7 \
    --vllm_tensor_parallel_size 1 \
    --beta 0.05 \
    --apply_verifiable_reward true \
    --output_dir output/rlvr_8b \
    --seed 3 \
    --num_evals 3 \
    --save_freq 100 \
    --reward_model_multiplier 0.0 \
    --gradient_checkpointing \
    --with_tracking
```


## Contamination checks

We release our scripts for measuring the overlap between instruction tuning datasets and evaluation datasets in `./decontamination`. See the [README](./decontamination/README.md) for more details.

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
├── decontamination/            <- Scripts for measuring train-eval overlap
├── eval/                       <- Evaluation suite for fine-tuned models
├── human_eval/                 <- Human evaluation interface (not maintained)
├── open_instruct/              <- Source code (flat)
├── quantize/                   <- Scripts for quantization
├── scripts/                    <- Core training and evaluation scripts
└── Dockerfile                  <- Dockerfile
```


## Licensing

This codebase is licensed under Apache 2.0 as given in [LICENSE](./LICENSE).

The license we use for V1 models released (along with the base model licenses) can be found in [assets/model_licenses/tulu_license.txt](./assets/model_licenses/tulu_license.txt) - just replace `<MODELNAME>` with the actual model name (i.e., the name on HuggingFace).

V2 models are licensed under the [low-risk AI2 ImpACT license](https://allenai.org/licenses/impact-lr). See [here](https://allenai.org/impact-license) for more details.


## Acknowledgements

Open Instruct is a project that benefitd from many open-source projects and libraries. We would like to particularly thank the folloiwng projects:

* [HuggingFace Transformers](https://github.com/huggingface/transformers): We adapted Hugging Face's Trainer for our finetuning scripts.
* [HuggingFace TRL](https://github.com/huggingface/trl) and [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization): our preference tuning code is adapted from TRL and from Eric Mitchell's DPO code.
* OpenAI's [lm-human-preferences](https://github.com/openai/lm-human-preferences), [summarize-from-feedback](https://github.com/openai/summarize-from-feedback), and [vwxyzjn/summarize_from_feedback_details](https://github.com/vwxyzjn/summarize_from_feedback_details): Our core PPO code is adapted from OpenAI's original RLHF code and [Huang et al (2024)'s reproduction work](https://openreview.net/forum?id=kHO2ZTa8e3) of OpenAI's summarize from feedback work.
* [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF): We adapted OpenRLHF's Ray + vLLM distributed code for scaling up PPO RLVR training into the 70B scale.

## Citation

If you used this repository or our models, please cite our work:

Tulu 1:
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

Tulu 2:
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

Tulu 2.5:
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

Tulu 3:
```
@article{lambert2024tulu3,
  title = {Tülu 3: Pushing Frontiers in Open Language Model Post-Training},
  author = {
    Nathan Lambert and Jacob Morrison and Valentina Pyatkin and Shengyi Huang and Hamish Ivison and Faeze Brahman and Lester James V. Miranda and Alisa Liu and Nouha Dziri and Shane Lyu and Yuling Gu and Saumya Malik and Victoria Graf and Jena D. Hwang and Jiangjiang Yang and Ronan Le Bras and Oyvind Tafjord and Chris Wilhelm and Luca Soldaini and Noah A. Smith and Yizhong Wang and Pradeep Dasigi and Hannaneh Hajishirzi
  },
  year = {2024},
  email = {tulu@allenai.org}
}
```
