# Welcome to Open Instruct

This repo serves as an open effort on instruction-tuning and post-training popular pretrained language models on publicly available datasets. We release this repo and will keep updating it with:

1. Code for finetuning language models with latest techniques and instruction datasets in a unified format.
2. Code for DPO, preference finetuning and reinforcement learning with verifiable rewards (RLVR).
3. Checkpoints or other useful artifacts that we build in our exploration.


We also support some evaluations natively in the codebase, but these are now unmaintained and instead we suggest using [OLMES](https://github.com/allenai/olmes), which we used for TÜLU 3. Below are some of our papers:

* [TÜLU 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)
    * Latest research on open post-training techniques and methodologies
    * Comprehensive details on our most recent model training approaches

* [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751)
    * Our first paper introducing the project's foundation and vision
    * Presents initial findings and exploration of instruction tuning with open resources

* [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702)
    * Second paper focusing on Llama-2 model adaptations
    * Details our work with direct preference optimization (DPO) techniques

* [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/abs/2406.09279)
    * Most recent research comparing reinforcement learning approaches
    * Analyzes best practices for both DPO and PPO methodologies

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
| **Final Models (RLVR)** | (🔥 New, trained with GRPO) [allenai/Llama-3.1-Tulu-3.1-8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B)                        |          |                       |                      |
| **Reward Model (RM)**| [allenai/Llama-3.1-Tulu-3-8B-RM](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM)                                                     | (Same as 8B)                                                     | [allenai/OLMo-2-1124-7B-RM](https://huggingface.co/allenai/OLMo-2-1124-7B-RM)                                                     | (Same as 7B)                                                     |

## News

- [2025-11-20] We released [Olmo 3](https://allenai.org/blog/olmo3)!
- [2025-02-12] We released the [`allenai/Llama-3.1-Tulu-3.1-8B` model](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B), which is trained with our GRPO recipe and outperforms the old [`allenai/Llama-3.1-Tulu-3-8B` model](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B) in almost all of our evals.
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
```bibtex
@article{lambert2024tulu3,
  title = {Tülu 3: Pushing Frontiers in Open Language Model Post-Training},
  author = {
    Nathan Lambert and Jacob Morrison and Valentina Pyatkin and Shengyi Huang and Hamish Ivison and Faeze Brahman and Lester James V. Miranda and Alisa Liu and Nouha Dziri and Shane Lyu and Yuling Gu and Saumya Malik and Victoria Graf and Jena D. Hwang and Jiangjiang Yang and Ronan Le Bras and Oyvind Tafjord and Chris Wilhelm and Luca Soldaini and Noah A. Smith and Yizhong Wang and Pradeep Dasigi and Hannaneh Hajishirzi
  },
  year = {2024},
  email = {tulu@allenai.org}
}
```
