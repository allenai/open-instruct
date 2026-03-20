# Direct Preference Optimization (DPO)

We support Direct Preference Optimization (DPO) training on a variety of datasets.

## Implemented Variants

- `dpo.py` is the recommended DPO implementation, built on OLMo-core's native training infrastructure (FSDP). It supports tensor parallelism, sequence packing, and `torch.compile`.
- `dpo_tune_cache.py` is the legacy DPO implementation using DeepSpeed.

## `dpo.py` (OLMo-core)

### Debug Scripts

| Script | Scale | Launch |
|--------|-------|--------|
| `scripts/train/debug/dpo/local.sh` | 1 GPU, local | `bash scripts/train/debug/dpo/local.sh` |
| `scripts/train/debug/dpo/single_gpu.sh` | 1 GPU, Beaker | `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/single_gpu.sh` |
| `scripts/train/debug/dpo/multi_node.sh` | 2 nodes (16 GPUs), Beaker | `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/multi_node.sh` |

### Key Flags

| Group | Flag | Description | Default |
|-------|------|-------------|---------|
| **Model** | `--model_name_or_path` | Model checkpoint for weight initialization | — |
| | `--attn_backend` | Attention backend: `flash_2`, `flash_3`, `auto` | `auto` |
| **DPO Algorithm** | `--beta` | Beta parameter for DPO loss | `0.1` |
| | `--loss_type` | Loss type: `dpo`, `dpo_norm`, `simpo`, `wpo` | `dpo` |
| | `--packing` | Use packing/padding-free collation | `False` |
| | `--label_smoothing` | Label smoothing for DPO/SimPO loss | `0.0` |
| **Training** | `--learning_rate` | Initial learning rate | `2e-5` |
| | `--num_epochs` | Total number of training epochs | `2` |
| | `--per_device_train_batch_size` | Batch size per GPU | `8` |
| | `--gradient_accumulation_steps` | Gradient accumulation steps | `1` |
| | `--max_seq_length` | Maximum sequence length after tokenization | `2048` |
| | `--warmup_ratio` | Linear warmup fraction of total steps | `0.03` |
| | `--lr_scheduler_type` | LR scheduler: `linear`, `cosine`, etc. | `linear` |
| | `--compile_model` | Apply `torch.compile` to model blocks | `True` |
| | `--activation_memory_budget` | Activation checkpointing budget (0.0–1.0) | `1.0` |
| **Parallelism** | `--tensor_parallel_degree` | Tensor parallelism degree | `1` |
| | `--fsdp_shard_degree` | FSDP shard degree (None = auto) | `None` |
| **Data** | `--mixer_list` | List of datasets (local or HF) to sample from | — |
| | `--chat_template_name` | Chat template to use | `None` |
| **Checkpointing** | `--output_dir` | Output directory for checkpoints | `output/` |
| | `--checkpointing_steps` | Save every N steps or `epoch` | `500` |
| | `--resume_from_checkpoint` | Resume from checkpoint folder | `None` |
| **Logging** | `--with_tracking` | Track experiment with Weights and Biases | `False` |

### Olmo 3 7B DPO

| Script | Scale | Launch |
|--------|-------|--------|
| `scripts/train/debug/dpo/7b_instruct_dpo_olmo_core.sh` | 2 nodes (16 GPUs), Beaker | `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo/7b_instruct_dpo_olmo_core.sh` |
| `scripts/train/olmo3/7b_instruct_dpo_olmocore.sh` | 4 nodes (32 GPUs), Beaker | `./scripts/train/build_image_and_launch.sh scripts/train/olmo3/7b_instruct_dpo_olmocore.sh` |

---

## `dpo_tune_cache.py` (Legacy)

This implementation has the following key features:

- Auto save the trained checkpoint to HuggingFace Hub
- Supports LigerKernel for optimized training with fused operations
- Implements the DPO algorithm for direct preference optimization


There are several relevant implementation details:

1. To save memory, we 1) cache the logprobs of the reference model on the dataset, 2) remove the reference model from the memory after the logprobs are computed. This means that you won't see the initial training losses for a while until the logprobs are computed.
2. We use the `dpo_norm` loss type by default, which is a length-normalized loss. See the [SimPO](https://arxiv.org/abs/2405.14734) paper for more details.




### Debug Scripts

| Script | Scale | Launch |
|--------|-------|--------|
| `scripts/train/debug/dpo_integration_test.sh` | 1 GPU, Beaker | `./scripts/train/build_image_and_launch.sh scripts/train/debug/dpo_integration_test.sh` |

### Key Flags

| Group | Flag | Description | Default |
|-------|------|-------------|---------|
| **Model** | `--model_name_or_path` | Model checkpoint for weight initialization | — |
| | `--use_flash_attn` | Use flash attention | `True` |
| **DPO Algorithm** | `--beta` | Beta parameter for DPO loss | `0.1` |
| | `--loss_type` | Loss type: `dpo`, `dpo_norm`, `simpo`, `wpo` | `dpo` |
| | `--packing` | Use packing/padding-free collation | `False` |
| | `--label_smoothing` | Label smoothing for DPO/SimPO loss | `0.0` |
| **Training** | `--learning_rate` | Initial learning rate | `2e-5` |
| | `--num_epochs` | Total number of training epochs | `2` |
| | `--per_device_train_batch_size` | Batch size per GPU | `8` |
| | `--gradient_accumulation_steps` | Gradient accumulation steps | `1` |
| | `--max_seq_length` | Maximum sequence length after tokenization | `2048` |
| | `--warmup_ratio` | Linear warmup fraction of total steps | `0.03` |
| | `--lr_scheduler_type` | LR scheduler: `linear`, `cosine`, etc. | `linear` |
| **DeepSpeed** | `--zero_stage` | DeepSpeed ZeRO stage (0, 1, 2, or 3) | — |
| | `--offload_optimizer` | Offload optimizer states to CPU | `False` |
| | `--offload_param` | Offload parameters to CPU | `False` |
| **Optimization** | `--use_liger_kernel` | Use LigerKernel for optimized training | `False` |
| **LoRA** | `--use_lora` | Use LoRA for parameter-efficient training | `False` |
| | `--lora_rank` | Rank of LoRA | `64` |
| **Data** | `--mixer_list` | List of datasets (local or HF) to sample from | — |
| | `--chat_template_name` | Chat template to use | `None` |
| **Checkpointing** | `--output_dir` | Output directory for checkpoints | `output/` |
| | `--checkpointing_steps` | Save every N steps or `epoch` | `500` |
| | `--resume_from_checkpoint` | Resume from checkpoint folder | `None` |
| **Logging** | `--with_tracking` | Track experiment with Weights and Biases | `False` |

### Reproduce `allenai/Llama-3.1-Tulu-3-8B-DPO` (4 Nodes)

You can reproduce our `allenai/Llama-3.1-Tulu-3-8B-DPO` model by running the following command:

```bash
bash scripts/train/tulu3/dpo_8b.sh
```

![dpo_plot](dpo/tulu3_8b_dpo.png)
![dpo_plot](dpo/tulu3_8b_dpo-time.png)


??? note "👉 Tracked WandB Experiments (Click to expand)"

    <iframe loading="lazy" src="https://wandb.ai/ai2-llm/open_instruct_public/reports/Tulu3-8B-DPO--VmlldzoxMTg3NjY4Nw" style="width:100%; height:500px" title="Tulu3-8B-DPO"></iframe>


???+ info


    Based on our internal evaluation, the DPO model is roughly on par with the original `allenai/Llama-3.1-Tulu-3-8B-DPO` model, though there are some slight differences. Note that your results may vary slightly due to the random seeds used in the training.

    ![dpo_plot](dpo/tulu3_8b_dpo_eval.png)

    For example, DROP is lower than the reference, but DROP can be quite brittle due to parsing issues (see below).

    ![dpo_plot](dpo/tulu3_8b_dpo_eval_drop.png)


???+ info

    We haven't quite figured out how to make our internal evaluation toolchains more open yet. Stay tuned!



### Reproduce `allenai/OLMo-2-1124-7B-DPO` (4 Nodes)

You can reproduce our `allenai/OLMo-2-1124-7B-DPO` model by running the following command:

```bash
bash scripts/train/olmo2/dpo_7b.sh
```

???+ info

    If you are an external user, `mason.py` will print out the actual command being executed on our internal server, so you can modify the command as needed.

![dpo_plot](dpo/olmo2_7b_dpo.png)
![dpo_plot](dpo/olmo2_7b_dpo-time.png)


??? note "👉 Tracked WandB Experiments (Click to expand)"

    <iframe loading="lazy" src="https://wandb.ai/ai2-llm/open_instruct_public/reports/OLMo-2-7B-DPO--VmlldzoxMTkyNzUyOA" style="width:100%; height:500px" title="OLMo2-7B-DPO"></iframe>

???+ info

    Based on our internal evaluation, the DPO model is roughly on par with the original `allenai/OLMo-2-1124-7B-DPO` model, though there are some slight differences. Note that your results may vary slightly due to the random seeds used in the training.

    ![dpo_plot](dpo/olmo2_7b_dpo_eval.png)

???+ info

    We haven't quite figured out how to make our internal evaluation toolchains more open yet. Stay tuned!




### Reproduce `allenai/OLMo-2-1124-13B-DPO` (4 Nodes)

You can reproduce our `allenai/OLMo-2-1124-13B-DPO` model by running the following command:

```bash
bash scripts/train/olmo2/dpo_13b.sh
```

![dpo_plot](dpo/olmo2_13b_dpo.png)
![dpo_plot](dpo/olmo2_13b_dpo-time.png)


??? note "👉 Tracked WandB Experiments (Click to expand)"

    <iframe loading="lazy" src="https://wandb.ai/ai2-llm/open_instruct_public/reports/OLMo-2-13B-DPO--VmlldzoxMTg3NjcyMQ" style="width:100%; height:500px" title="OLMo2-13B-DPO"></iframe>


???+ info

    Based on our internal evaluation, the DPO model is roughly on par with the original `allenai/OLMo-2-1124-13B-DPO` model, though there are some slight differences. Note that your results may vary slightly due to the random seeds used in the training.

    ![dpo_plot](dpo/olmo2_13b_dpo_eval.png)


???+ info

    We haven't quite figured out how to make our internal evaluation toolchains more open yet. Stay tuned!


### Training Metrics

During training, the following metrics are logged:

- `training_step`: Current training step
- `learning_rate`: The current learning rate from the learning rate scheduler
- `epoch`: Current epoch (as a fraction of total dataset)
- `train_loss`: The average training loss over the logged steps
- `logps/chosen`: Average log probabilities for chosen responses
- `logps/rejected`: Average log probabilities for rejected responses

For DPO and DPO-norm loss types, additional metrics are logged:

- `rewards/chosen`: Average rewards for chosen responses
- `rewards/rejected`: Average rewards for rejected responses
- `rewards/average`: Average of chosen and rejected rewards
- `rewards/accuracy`: Accuracy of preference prediction
- `rewards/margin`: Margin between chosen and rejected rewards

When using load balancing loss (for OLMoE), the following metric is also logged:

- `aux_loss`: Auxiliary loss for load balancing

The metrics are logged every `logging_steps` steps (if specified) and provide insights into:

- Training progress (loss, learning rate, epoch)
- Model behavior (log probabilities, rewards)
- Preference learning (accuracy, margin)
- Resource utilization (auxiliary losses)

## Acknowledgements

We would like to thank the following projects for general infrastructure:

- [DeepSpeedAI/DeepSpeed](https://github.com/deepspeedai/DeepSpeed)
- [HuggingFace/Transformers](https://github.com/huggingface/transformers)
