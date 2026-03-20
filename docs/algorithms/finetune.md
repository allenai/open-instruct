# Supervised finetuning (SFT)

We support Supervised finetuning (SFT) on a variety of datasets.

## Implemented Variants

- **OLMo-core SFT** (recommended): Uses OLMo-core's native training infrastructure. For supported models (OLMo, Qwen, and more), this is more GPU-efficient. See `open_instruct/olmo_core_utils.py` for the current list of supported models.
- `finetune.py` is the legacy SFT implementation using DeepSpeed/Accelerate.

## OLMo-core SFT

The recommended SFT implementation uses [OLMo-core's SFT training script](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train/sft). It runs via `python src/scripts/train/sft/OLMo-sft.py` from the OLMo-core submodule. There are no debug scripts for OLMo-core SFT in this repo; see the [OLMo-core documentation](https://github.com/allenai/OLMo-core) for testing instructions.

### Olmo 3 SFT Scripts

Production SFT scripts for Olmo 3 models are available in `scripts/train/olmo3/`:

```bash
# Olmo 3 7B Instruct SFT (4 nodes)
bash scripts/train/olmo3/7b_instruct_sft.sh

# Olmo 3 7B Think SFT (4 nodes)
bash scripts/train/olmo3/7b_think_sft.sh

# Olmo 3 32B Instruct SFT
bash scripts/train/olmo3/32b_instruct_sft.sh

# Olmo 3 32B Think SFT
bash scripts/train/olmo3/32b_think_sft.sh
```

---

## `finetune.py` (Legacy)


This implementation has the following key features:

- Auto save the trained checkpoint to HuggingFace Hub
- Supports LigerKernel for optimized training with fused operations



### Debug Scripts

**Single GPU integration test (runs locally or on Beaker):**

```bash
bash scripts/train/debug/sft_integration_test.sh
```

**Multi-node integration test (2 nodes) with sequence parallelism on Beaker:**

```bash
bash scripts/train/debug/sft_multinode_test.sh
```

**Quick local smoke test (single GPU, no Beaker):**

```bash
bash scripts/train/debug/finetune.sh
```

![finetune](finetune/finetune_debug.png)


### Reproduce `allenai/Llama-3.1-Tulu-3-8B-SFT` (8 Nodes)

You can reproduce our `allenai/Llama-3.1-Tulu-3-8B-SFT` model by running the following command:

```bash
bash scripts/train/tulu3/finetune_8b.sh
```

???+ info

    If you are an external user, `mason.py` will print out the actual command being executed on our internal server, so you can modify the command as needed.

    ![tulu3_8b](finetune/tulu3_8b.png)



![finetune_plot](finetune/tulu3_8b_sft.png)
![finetune_plot](finetune/tulu3_8b_sft-time.png)


??? note "👉 Tracked WandB Experiments (Click to expand)"

    <iframe loading="lazy" src="https://wandb.ai/ai2-llm/open_instruct_public/reports/Tulu3-8B-SFT--VmlldzoxMTk0OTY4MA" style="width:100%; height:500px" title="Tulu3-8B-SFT"></iframe>


???+ info


    Based on our internal evaluation, the SFT model is roughly on par with the original `allenai/Llama-3.1-Tulu-3-8B` model, though there are some slight differences. Note that your results may vary slightly due to the random seeds used in the training.

    ![finetune_plot](finetune/tulu3_8b_eval.png)

???+ info

    We haven't quite figured out how to make our internal evaluation toolchains more open yet. Stay tuned!




### Reproduce `allenai/OLMo-2-1124-7B-SFT` (8 Nodes)

You can reproduce our `allenai/OLMo-2-1124-7B-SFT` model by running the following command:

```bash
bash scripts/train/olmo2/finetune_7b.sh
```

![finetune_plot](finetune/olmo2_7b_sft.png)
![finetune_plot](finetune/olmo2_7b_sft-time.png)


??? note "👉 Tracked WandB Experiments (Click to expand)"

    <iframe loading="lazy" src="https://wandb.ai/ai2-llm/open_instruct_public/reports/OLMo-2-1124-7B-SFT--VmlldzoxMTg1NzIxMw" style="width:100%; height:500px" title="OLMo2-1124-7B-SFT"></iframe>

???+ info

    Based on our internal evaluation, the SFT model is roughly on par with the original `allenai/OLMo-2-1124-7B` model, though there are some slight differences. Note that your results may vary slightly due to the random seeds used in the training.

    ![finetune_plot](finetune/olmo2_7b_sft_eval.png)

???+ info

    We haven't quite figured out how to make our internal evaluation toolchains more open yet. Stay tuned!


### Reproduce `allenai/OLMo-2-1124-13B-SFT` (8 Nodes)

You can reproduce our `allenai/OLMo-2-1124-13B-SFT` model by running the following command:

```bash
bash scripts/train/olmo2/finetune_13b.sh
```

![finetune_plot](finetune/olmo2_13b_sft.png)
![finetune_plot](finetune/olmo2_13b_sft-time.png)


??? note "👉 Tracked WandB Experiments (Click to expand)"

    <iframe loading="lazy" src="https://wandb.ai/ai2-llm/open_instruct_public/reports/OLMo-2-13B-SFT--VmlldzoxMjA0MjUyNg" style="width:100%; height:500px" title="OLMo2-1124-7B-SFT"></iframe>

???+ info

    Based on our internal evaluation, the SFT model is roughly on par with the original `allenai/OLMo-2-1124-7B` model, though there are some slight differences. Note that your results may vary slightly due to the random seeds used in the training.

    ![finetune_plot](finetune/olmo2_13b_sft_eval.png)

???+ info

    We haven't quite figured out how to make our internal evaluation toolchains more open yet. Stay tuned!


### Reproduce `allenai/OLMo-2-1124-32B-SFT` (8 Nodes)

You can reproduce our `allenai/OLMo-2-1124-32B-SFT` model by running the following command:

```bash
bash scripts/train/olmo2/finetune_32b.sh
```

![finetune_plot](finetune/olmo2_32b_sft.png)
![finetune_plot](finetune/olmo2_32b_sft-time.png)


??? note "👉 Tracked WandB Experiments (Click to expand)"

    <iframe loading="lazy" src="https://wandb.ai/ai2-llm/open_instruct_public/reports/OLMo-2-32B-SFT--VmlldzoxMjA0MjUxOQ" style="width:100%; height:500px" title="OLMo2-1124-7B-SFT"></iframe>

???+ info

    Based on our internal evaluation, the SFT model is roughly on par with the original `allenai/OLMo-2-1124-7B` model, though there are some slight differences. Note that your results may vary slightly due to the random seeds used in the training.

    ![finetune_plot](finetune/olmo2_32b_sft_eval.png)

???+ info

    We haven't quite figured out how to make our internal evaluation toolchains more open yet. Stay tuned!




### Training Metrics

During training, the following metrics are logged:

- `learning_rate`: The current learning rate from the learning rate scheduler
- `train_loss`: The average training loss over the logged steps
- `total_tokens`: Total number of tokens processed (excluding padding)
- `per_device_tps`: Tokens per second processed per device (excluding padding)
- `total_tokens_including_padding`: Total number of tokens including padding tokens
- `per_device_tps_including_padding`: Tokens per second processed per device (including padding)

The metrics are logged every `logging_steps` steps (if specified) and provide insights into:
- Training progress (loss, learning rate)
- Training efficiency (tokens per second)
- Resource utilization (padding vs non-padding tokens)

## Acknowledgements

We would like to thank the following projects for general infrastructure:

- [DeepSpeedAI/DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [HuggingFace/Transformers](https://github.com/huggingface/transformers)
