[![Beaker Experiment Launch](https://github.com/allenai/open-instruct/actions/workflows/beaker-experiment.yml/badge.svg)](https://github.com/allenai/open-instruct/actions/workflows/beaker-experiment.yml)

# Open Instruct

Open Instruct is a research codebase for post-training language models with supervised finetuning (SFT), direct preference optimization (DPO), reward modeling, and reinforcement learning with verifiable rewards (RLVR), including tool use.

For evaluation, use [OLMES](https://github.com/allenai/olmes). The legacy evaluation code in this repository is unmaintained.

## Models and recipes

Training recipes, checkpoints, and reproduction details are documented by model family:

| Model family | Recipes and documentation |
| --- | --- |
| Olmo 3 and 3.1 | [Training scripts and run provenance](scripts/train/olmo3/README.md), [tokenizer settings](docs/olmo3.md) |
| Olmo Hybrid | [Training recipes](scripts/train/olmo-hybrid/README.md), [tokenizer settings](docs/olmo3.md) |
| OLMo 2 | [Training recipes and checkpoints](docs/olmo2.md) |
| TÜLU 3 and 3.1 | [Training recipes and checkpoints](docs/tulu3.md) |
| TÜLU 1 and 2 | [Historical recipes and checkpoints](docs/tulu1_tulu2.md) |

Example released models include [Olmo 3 7B Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) and [Olmo 3.1 32B Think](https://huggingface.co/allenai/Olmo-3.1-32B-Think). See the [changelog](CHANGELOG.md) for code changes.

## Setup

Training requires Linux, NVIDIA GPUs, and Python 3.12. We use [uv](https://docs.astral.sh/uv/) to manage dependencies. Open Instruct is a research codebase and does not guarantee backward compatibility.

```bash
git clone https://github.com/allenai/open-instruct.git
cd open-instruct
uv sync --frozen
```

On x86_64 Linux, the default dependency groups include development tools and CUDA 12.8 packages. For CUDA 13 instead, use:

```bash
uv sync --frozen --no-default-groups --group dev --group cuda13
```

Tests require [Git LFS](https://git-lfs.com/). Install it and run `git lfs install` before cloning; for an existing checkout, also run `git lfs pull`. See [CONTRIBUTING.md](CONTRIBUTING.md#git-lfs).

The [Dockerfile](Dockerfile) provides the container environment:

```bash
docker build --platform linux/amd64 . \
    --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
    --build-arg GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
    -t open_instruct_dev
```

The image defaults to CUDA 12.8. Add `--build-arg CUDA_VERSION=13` for CUDA 13; the target GPUs must have compatible drivers.

## Training

The current training entry points are:

| Method | Implementation | Guide |
| --- | --- | --- |
| SFT | [OLMo-core SFT](open_instruct/olmo_core_finetune.py), [Hugging Face SFT](open_instruct/finetune.py) | [SFT guide](docs/algorithms/sft_running_guide.md) |
| DPO | [OLMo-core DPO](open_instruct/dpo.py) | [DPO guide](docs/algorithms/dpo.md) |
| RLVR / GRPO | [GRPO trainer](open_instruct/grpo_fast.py) | [GRPO guide](docs/algorithms/grpo.md), [tool training](docs/algorithms/tool_training.md) |
| Reward modeling | [Reward model trainer](open_instruct/reward_modeling.py) | [Reward modeling guide](docs/algorithms/reward_modeling.md) |

For Ai2 contributors, [build_image_and_launch.sh](scripts/train/build_image_and_launch.sh) builds a Beaker image from the current commit and passes it to the chosen script. Commit all changes before launching. These examples each request one GPU:

```bash
# OLMo-core SFT.
./scripts/train/build_image_and_launch.sh scripts/train/debug/oc_sft.sh

# OLMo-core DPO.
./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/single_gpu.sh

# GRPO / RLVR.
./scripts/train/build_image_and_launch.sh scripts/train/debug/single_gpu_on_beaker.sh
```

Pass `--cuda-version 13` before the script path when targeting a CUDA 13-compatible cluster. Review the script's cluster, workspace, priority, timeout, and preemption settings before launching; scripts using `--preemptible` may wait for backfill capacity. See [CONTRIBUTING.md](CONTRIBUTING.md#launching-experiments-on-beaker) for more examples. On other infrastructure, adapt the training command after the `mason.py` argument separator to your launcher.

Released-model reproduction may require the original commit and environment; use the provenance recorded in the model recipes above.

## Development

```bash
make style
make quality
uv run pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for test requirements and PR checks. The development dependencies already include pre-commit; enable the hooks with `uv run pre-commit install`.

Build or preview the documentation with `uv run mkdocs build` or `uv run mkdocs serve`. Dataset overlap checks are documented in the [decontamination README](decontamination/README.md).

## Licensing

The code is licensed under [Apache 2.0](LICENSE). Model and dataset licenses vary; check their individual model cards and dataset cards before use. Historical TÜLU 1 model terms are in [assets/model_licenses/tulu_license.txt](assets/model_licenses/tulu_license.txt).

## Acknowledgements

Our training code builds on [Hugging Face Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl), [Eric Mitchell's DPO implementation](https://github.com/eric-mitchell/direct-preference-optimization), [OLMo-core](https://github.com/allenai/OLMo-core), and [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). Earlier PPO implementations drew on OpenAI's [lm-human-preferences](https://github.com/openai/lm-human-preferences) and [summarize-from-feedback](https://github.com/openai/summarize-from-feedback), and [Huang et al.'s reproduction](https://github.com/vwxyzjn/summarize_from_feedback_details).

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

Olmo 3:

```bibtex
@misc{olmo2025olmo3,
      title={OLMo 3},
      author={Team OLMo and Allyson Ettinger and Amanda Bertsch and Bailey Kuehl and David Graham and David Heineman and Dirk Groeneveld and Faeze Brahman and Finbarr Timbers and Hamish Ivison and Jacob Morrison and Jake Poznanski and Kyle Lo and Luca Soldaini and Matt Jordan and Mayee Chen and Michael Noukhovitch and Nathan Lambert and Pete Walsh and Pradeep Dasigi and Robert Berry and Saumya Malik and Saurabh Shah and Scott Geng and Shane Arora and Shashank Gupta and Taira Anderson and Teng Xiao and Tyler Murray and Tyler Romero and Victoria Graf and Akari Asai and Akshita Bhagia and Alexander Wettig and Alisa Liu and Aman Rangapur and Chloe Anastasiades and Costa Huang and Dustin Schwenk and Harsh Trivedi and Ian Magnusson and Jaron Lochner and Jiacheng Liu and Lester James V. Miranda and Maarten Sap and Malia Morgan and Michael Schmitz and Michal Guerquin and Michael Wilson and Regan Huff and Ronan Le Bras and Rui Xin and Rulin Shao and Sam Skjonsberg and Shannon Zejiang Shen and Shuyue Stella Li and Tucker Wilde and Valentina Pyatkin and Will Merrill and Yapei Chang and Yuling Gu and Zhiyuan Zeng and Ashish Sabharwal and Luke Zettlemoyer and Pang Wei Koh and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2025},
      eprint={2512.13961},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.13961},
}
```
