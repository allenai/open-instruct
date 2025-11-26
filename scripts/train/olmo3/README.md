# Overview

For our recent [Olmo3 paper](insert link), we used the following scripts to train our models:

| Model           | Script name           | Beaker Link | Wandb URL | Commit |
|-----------------|----------------------|---|---|--------|
| 7B Think DPO    | `7b_think_dpo.sh`     | https://beaker.org/ex/01K5SXG8YH7NZDT5JCWJSNFCKG | https://wandb.ai/ai2-llm/open_instruct_internal/runs/drm42by2 | [`68da0a1`](https://github.com/allenai/open-instruct/commit/68da0a1) |
| 32B Think DPO   | `32b_think_dpo.sh`    | https://beaker.org/ex/01K9VYQV2RFPS9ECP63JFQFVDN | https://wandb.ai/ai2-llm/open_instruct_internal/runs/te37gyey | [`2fd104e`](https://github.com/allenai/open-instruct/commit/2fd104e) |
| 7B Instruct DPO | `7b_instruct_dpo.sh`  | https://beaker.org/ex/01KA62AJW9P8AWA3YKWE4Y6XZD | https://wandb.ai/ai2-llm/open_instruct_internal/runs/kxc617kc | [`2fd104e`](https://github.com/allenai/open-instruct/commit/2fd104e) |
| 7B Instruct RL  | `7b_instruct_rl.sh`   | https://beaker.org/ex/01KA8BY8MMAQWENWY4087MAPFE | https://wandb.ai/ai2-llm/open_instruct_internal/runs/p0l9m3ri | [`9ade62d`](https://github.com/allenai/open-instruct/commit/9ade62d) |
| 7B Think RL     | `7b_think_rl.sh` | https://beaker.org/ex/01KADRVRYEPW4YPKNN0RRNS137 | https://wandb.ai/ai2-llm/open_instruct_internal/runs/buq6ny46 | [`42aa63c`](https://github.com/allenai/open-instruct/commit/42aa63c) |
| 7B Think RL (no pipeline) | `7b_think_rl_no_pipeline.sh` | https://beaker.org/ex/01K6JZVN4EN3VHTJ820BV23HGC | https://wandb.ai/ai2-llm/open_instruct_internal/runs/pvb181bq | [`42aa63c`](https://github.com/allenai/open-instruct/commit/42aa63c) |
| 32B RL          | `32b_think_rl.sh` | https://beaker.org/ex/01KA4ZXT7MCVK493Y2B3K0BC82 | https://wandb.ai/ai2-llm/open_instruct_internal/runs/29h723j6 | [`42aa63c`](https://github.com/allenai/open-instruct/commit/42aa63c) |
| 7B RL Zero Math | `7b_rlzero_math.sh`   | https://beaker.org/ex/01K8V8TSX5K8BGZPJATZEE1003/ | https://wandb.ai/ai2-llm/open_instruct_internal/runs/w0ql4f5r | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 7B RL Zero Code | `7b_rlzero_code.sh`   | https://beaker.org/ex/01K7FSWM4717FAR9KF6GE958CA/ | https://wandb.ai/ai2-llm/open_instruct_internal/runs/o40rwmu8 | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 7B RL Zero IF   | `7b_rlzero_instruction_following.sh` | https://beaker.org/ex/01K7MVRTNJNYB37GC8SDTYHKC1/ | https://wandb.ai/ai2-llm/open_instruct_internal/runs/hk80a60o | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 7B RL Zero General | `7b_rlzero_general.sh` | https://beaker.org/ex/01K7FSZ2Y16KAV56Q0KB7TSWN7/ | https://wandb.ai/ai2-llm/open_instruct_internal/runs/0tscl05k | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |


To reproduce these runs, if you are internal to Ai2, you can run

```
git checkout $COMMIT
./scripts/train/build_image_and_launch.sh $SCRIPT_NAME
```

This will build an image and launch it. You can also check out the beaker link to see the **exact run** that produced the model! If you are external to Ai2, we have many [fine job postings](https://allenai.org/careers), but, unfortunately, do not have great advice on how to launch these jobs. Preliminary steps to launch on your own infrastructure would involve:

1. Modifying the launch scripts to remove the stuff attached to the `mason.py` command
2. Setting up your own cluster with the requisite number of {H,A}100 nodes, connected together via Ray.
