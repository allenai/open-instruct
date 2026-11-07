[![Beaker Experiment Launch](https://github.com/allenai/open-instruct/actions/workflows/beaker-experiment.yml/badge.svg)](https://github.com/allenai/open-instruct/actions/workflows/beaker-experiment.yml)

# 训练开源指令遵循语言模型 (Open Instruct)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

本仓库旨在构建开源的指令微调（Instruction Tuning）与后训练（Post-Training）生态，在公开数据集上对主流预训练大语言模型进行系统性适配。我们将持续更新并发布：

1. 采用前沿技术并在统一样式格式下对语言模型进行指令微调的代码实现。
2. 包含 DPO、偏好对齐微调以及具可验证奖励的强化学习（RLVR）代码实现。
3. 在技术探索过程中构建的模型检查点（Checkpoints）及其他重要衍生构建产物。

我们曾在代码库中原生支持部分基准评测，但这些代码目前已不再维护；推荐使用我们在 TÜLU 3 中采用的评测套件 [OLMES](https://github.com/allenai/olmes)。

有关开源后训练的最新前沿进展，请参阅论文 [TÜLU 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)。

关于本项目背后的设计思考与早期发现，请参阅第一篇论文 [How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources](https://arxiv.org/abs/2306.04751)；
关于使用 Llama-2 模型和直接偏好优化（DPO）的结果，请参阅第二篇论文 [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702)；
关于 PPO 与 DPO 的前沿系统性对比与最佳实践，请参阅第三篇论文 [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/abs/2406.09279)。

<p align="center" width="100%">
      <img src="assets/images/tulu_logo.png" alt="Tülu（混血骆驼）代表了我们通过在强混合数据集上全量微调构建的一系列 LLaMa 模型。" style="width: 20%; min-width: 200px; display: block; margin: auto;">
</p>

欢迎体验使用 Open Instruct 训练的模型。可以在线访问[免费在线 Demo](https://playground.allenai.org/)，或从 Hugging Face 直接下载：

| **阶段** | **Llama 3.1 8B** | **Llama 3.1 70B** | **OLMo-2 7B** | **OLMo-2 13B** |
|:---|:---|:---|:---|:---|
| **基座模型 (Base Model)** | [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) | [meta-llama/Llama-3.1-70B](https://huggingface.co/meta-llama/Llama-3.1-70B) | [allenai/OLMo2-7B-1124](https://huggingface.co/allenai/OLMo2-7B-1124) | [allenai/OLMo-2-13B-1124](https://huggingface.co/allenai/OLMo-2-13B-1124) |
| **SFT 监督微调** | [allenai/Llama-3.1-Tulu-3-8B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-SFT) | [allenai/Llama-3.1-Tulu-3-70B-SFT](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-SFT) | [allenai/OLMo-2-1124-7B-SFT](https://huggingface.co/allenai/OLMo-2-1124-7B-SFT) | [allenai/OLMo-2-1124-13B-SFT](https://huggingface.co/allenai/OLMo-2-1124-13B-SFT) |
| **DPO 偏好微调** | [allenai/Llama-3.1-Tulu-3-8B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-DPO) | [allenai/Llama-3.1-Tulu-3-70B-DPO](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B-DPO) | [allenai/OLMo-2-1124-7B-DPO](https://huggingface.co/allenai/OLMo-2-1124-7B-DPO) | [allenai/OLMo-2-1124-13B-DPO](https://huggingface.co/allenai/OLMo-2-1124-13B-DPO) |
| **最终模型 (RLVR)** | [allenai/Llama-3.1-Tulu-3-8B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B) | [allenai/Llama-3.1-Tulu-3-70B](https://huggingface.co/allenai/Llama-3.1-Tulu-3-70B) | [allenai/OLMo-2-1124-7B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct) | [allenai/OLMo-2-1124-13B-Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct) |
| **奖励模型 (RM)** | [allenai/Llama-3.1-Tulu-3-8B-RM](https://huggingface.co/allenai/Llama-3.1-Tulu-3-8B-RM) | （同 8B 版本） | [allenai/OLMo-2-1124-7B-RM](https://huggingface.co/allenai/OLMo-2-1124-7B-RM) | （同 7B 版本） |

## 最新动态

- [2024-11-22] 发布论文 [TÜLU 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/abs/2411.15124)，并更新了基于 Llama 3.1 和 OLMo 2 的全套开源后训练训练方案。
- [2024-07-01] 发布论文 [Unpacking DPO and PPO: Disentangling Best Practices for Learning from Preference Feedback](https://arxiv.org/abs/2406.09279)，并对代码库进行了重大升级以支持新模型及新版依赖库。
- [2023-11-27] 发布论文 [Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2](https://arxiv.org/abs/2311.10702)。可在 [Hugging Face Collection](https://huggingface.co/collections/allenai/tulu-v2-suite-6551b56e743e6349aab45101) 获取模型，并添加了复现结果的 DPO 微调脚本。
- [2023-09-26] 切换至官方 [alpaca-eval](https://github.com/tatsu-lab/alpaca_eval) 库执行 AlpacaFarm 评测，并采用重新生成的较长参考输出。
- [2023-09-25] 评测流程支持 [vLLM](https://github.com/vllm-project/vllm/) 推理加速，使评测吞吐量提升约 10 倍。
- [2023-09-17] 支持 [LoRA](https://arxiv.org/abs/2106.09685) 与 [QLoRA](https://arxiv.org/abs/2305.14314) 参数高效微调。
- [2023-08-18] 支持 [ToxiGen](https://github.com/microsoft/TOXIGEN) 与 [TruthfulQA](https://github.com/sylinrl/TruthfulQA) 评测。运行示例请参见 `scripts/eval/`。
- [2023-08-08] 新增支持多个指令数据集，包括 [LIMA](https://huggingface.co/datasets/GAIR/lima)、[WizardLM](https://github.com/nlpxucan/WizardLM) 与 [Open-Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca)。
- [2023-08-06] 通过升级 transformers 等依赖支持 LLaMa 2 微调与 FlashAttention-2 加速。
- [2023-06-29] 为已开源的模型添加了[许可信息](#开源许可证)。
- [2023-06-09] 开源 Tülu（一系列在高质量混合数据集上全量微调的 LLaMa 模型）及多个检查点权重。
- [2023-06-09] 首次开源代码库，发布对应的训练与评测代码。

## 环境配置

环境构建遵循 [Dockerfile](./Dockerfile)。*请注意，Open Instruct 是科研代码库，不保证绝对的向后兼容性。*

### 使用 uv 安装环境

推荐使用 [uv](https://docs.astral.sh/uv/) 安装并运行代码。通过 `uv sync` 一键同步安装环境。

* **Git LFS**（用于运行单元测试）：在克隆代码前安装 [Git LFS](https://git-lfs.com/) 并运行 `git lfs install`。详情请参阅 [CONTRIBUTING.md](CONTRIBUTING.md#git-lfs)。
* **Docker 容器构建**：亦可通过 Dockerfile 构建镜像：

```bash
docker build . \
    --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
	--build-arg GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD) \
	-t open_instruct_dev

# 如果您是 AI2 内部成员，可以如下创建 Beaker 镜像：
beaker_user=$(beaker account whoami --format json | jq -r '.[0].name')
beaker image delete $beaker_user/open_instruct_dev
beaker image create open_instruct_dev -n open_instruct_dev -w ai2/$beaker_user
```

AI2 内部成员可直接使用自动构建的最新镜像 `nathanl/open_instruct_auto` 提交实验。

## 模型训练

完成环境配置后即可启动实验。下面提供几个示例。关于复现 Tulu 3 模型的详细说明，请参阅 [Tulu 3 说明文档](./docs/tulu3.md)。Tulu 1 与 Tulu 2 的说明文档请参见 [Tulu 1 and 2 说明文档](./docs/tulu1_tulu2.md)。

### 监督微调 (SFT)

可以通过以下命令快速上手：

```bash
# 使用 8 张 GPU 训练 8B 规模的 tulu3 模型
bash scripts/train/tulu3/finetune_8b.sh
```

**OLMo-core SFT**：对于支持的模型（OLMo、OLMoE、Qwen3），我们推荐使用显存效率更高的 [OLMo-core SFT 实现](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train/sft)。支持的模型列表请查阅 `open_instruct/olmo_core_utils.py`。

### 偏好对齐微调 (DPO)

```bash
# 使用 8 张 GPU 运行 8B 规模的 tulu3 DPO 偏好对齐训练
bash scripts/train/tulu3/dpo_8b.sh
```

### 具可验证奖励的强化学习 (RLVR)

核心训练通过 `open_instruct/grpo_fast.py` 执行。通过 `scripts/train/build_image_and_launch.sh` 脚本基于当前提交构建 Beaker 镜像并运行指定脚本：

```bash
# 在 Beaker 上运行单 GPU 冒烟测试（轻量模型，快速完成验证）
./scripts/train/build_image_and_launch.sh scripts/train/debug/single_gpu_on_beaker.sh

# 双节点 8xGPU 训练运行（基于代码 RLVR 训练 Qwen2.5-7B）
./scripts/train/build_image_and_launch.sh scripts/train/debug/large_test_script.sh
```

## 数据污染检测 (Contamination checks)

我们在 `./decontamination` 下开源了用于检测指令微调训练集与评测基准数据集之间重叠污染的工具脚本。详情请参阅其[说明文档](./decontamination/README.md)。

### 参与开发

向本仓库提交 PR 时，我们将通过以下命令检查 `open_instruct/` 中的核心代码规范：

```bash
make style
make quality
```

运行测试套件：`uv run pytest`。

#### Pre-commit 钩子

在每次 Git Commit 时自动运行代码检查与格式化：

```bash
uv add pre-commit --dev
uv run pre-commit install
```

在所有文件上执行全量检查（建议在初次配置后运行）：

```bash
uv run pre-commit run --all-files
```

### 仓库目录结构

```
├── assets/                     <- 图像资源、许可文件等
├── configs/
|     ├── beaker_configs/       <- AI2 Beaker 任务配置文件
|     ├── ds_configs/           <- DeepSpeed 分布式配置文件
|     └── train_configs/        <- 训练超参数配置文件
├── decontamination/            <- 训练-评测数据污染重叠度检测脚本
├── eval/                       <- 微调模型评测套件
├── human_eval/                 <- 人工评测接口（已归档不维护）
├── open_instruct/              <- 核心源码实现
├── quantize/                   <- 模型量化脚本
├── scripts/                    <- 核心训练与评测执行脚本
└── Dockerfile                  <- 容器构建文件
```

## 开源许可证

本代码库基于 [LICENSE](./LICENSE) 中指定的 Apache 2.0 许可证开源。

发布的 V1 版本模型采用的许可（连同基座模型许可）请参见 [assets/model_licenses/tulu_license.txt](./assets/model_licenses/tulu_license.txt) —— 只需将 `<MODELNAME>` 替换为实际模型名称（即 Hugging Face 上的模型标识符）。

V2 版本模型遵循 [AI2 ImpACT 低风险许可证](https://allenai.org/licenses/impact-lr)。详情请参见[此处说明](https://allenai.org/impact-license)。

## 致谢

Open Instruct 项目受惠于众多开源项目与社区库。我们特别向以下项目致以感谢：

* [HuggingFace Transformers](https://github.com/huggingface/transformers)：我们的微调脚本借鉴并适配了 Hugging Face 的 Trainer。
* [HuggingFace TRL](https://github.com/huggingface/trl) 与 [eric-mitchell/direct-preference-optimization](https://github.com/eric-mitchell/direct-preference-optimization)：偏好微调代码改编自 TRL 及 Eric Mitchell 的 DPO 实现。
* OpenAI 的 [lm-human-preferences](https://github.com/openai/lm-human-preferences)、[summarize-from-feedback](https://github.com/openai/summarize-from-feedback) 与 [vwxyzjn/summarize_from_feedback_details](https://github.com/vwxyzjn/summarize_from_feedback_details)：核心 PPO 代码借鉴了 OpenAI 原版 RLHF 代码及 [Huang 等人 (2024)](https://openreview.net/forum?id=kHO2ZTa8e3) 的复现成果。
* [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)：借鉴了 OpenRLHF 基于 Ray + vLLM 的分布式架构，将 PPO RLVR 训练扩展至 70B 规模。

## 论文引用

如果您在研究中使用了本仓库或相关模型，请引用我们的论文：

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

OLMo 3:
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

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月13日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
