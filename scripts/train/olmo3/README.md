# Overview

For our recent [Olmo3 paper](https://arxiv.org/abs/2512.13961), we used the following scripts to train our models.

> **Note**: OLMo 3 SFT uses the [OLMo-core SFT implementation](https://github.com/allenai/OLMo-core/tree/main/src/scripts/train/sft) for better GPU efficiency. DPO and RL training use open-instruct. The `build_image_and_launch.sh` script only works for open-instruct jobs (DPO, RL), not for SFT.

| Model           | Script name           | Beaker Link | Wandb URL | Commit |
|-----------------|----------------------|---|---|--------|
| 7B Instruct SFT | [`7b_instruct_sft.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_instruct_sft.sh) | https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KA38QQENAC3S8GEKVYHYT2MV | https://wandb.ai/ai2-llm/jacobm-7B-sft/runs/zfn667tc | [`9e97471`](https://github.com/allenai/OLMo-core/commit/9e97471057d7046f0ae7315e0225d117b54186f9) |
| 7B Instruct DPO | [`7b_instruct_dpo.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_instruct_dpo.sh)  | https://beaker.org/ex/01KA62AJW9P8AWA3YKWE4Y6XZD | https://wandb.ai/ai2-llm/open_instruct_internal/runs/p8p0vbd9 | [`2fd104e`](https://github.com/allenai/open-instruct/commit/2fd104e) |
| 7B Instruct RL  | [`7b_instruct_rl.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_instruct_rl.sh)   | https://beaker.org/ex/01KA8BY8MMAQWENWY4087MAPFE | https://wandb.ai/ai2-llm/open_instruct_internal/runs/p0l9m3ri | [`9ade62d`](https://github.com/allenai/open-instruct/commit/9ade62d) |
| 7B Think SFT | [`7b_think_sft.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_think_sft.sh) | https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01K5FBWEXJV35P4J9H5FJQ7WE6 | https://wandb.ai/ai2-llm/saumyam-7b-sft/runs/4yx5d5bk, https://wandb.ai/ai2-llm/saumyam-7b-sft/runs/3t0hzqap | [`38f6652`](https://github.com/allenai/OLMo-core/commit/38f66526c9d1ba6b97269ebfb429749a5feb528f) |
| 7B Think DPO    | [`7b_think_dpo.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_think_dpo.sh)     | https://beaker.org/ex/01K5SXG8YH7NZDT5JCWJSNFCKG | https://wandb.ai/ai2-llm/open_instruct_internal/runs/drm42by2 | [`68da0a1`](https://github.com/allenai/open-instruct/commit/68da0a1) |
| 7B Think RL     | [`7b_think_rl.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_think_rl.sh) | https://beaker.org/ex/01KADRVRYEPW4YPKNN0RRNS137 | https://wandb.ai/ai2-llm/open_instruct_internal/runs/buq6ny46 | [`42aa63c`](https://github.com/allenai/open-instruct/commit/42aa63c) |
| 7B Think RL (no pipeline) | [`7b_think_rl_no_pipeline.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_think_rl_no_pipeline.sh) | https://beaker.org/ex/01K6JZVN4EN3VHTJ820BV23HGC | https://wandb.ai/ai2-llm/open_instruct_internal/runs/pvb181bq | [`42aa63c`](https://github.com/allenai/open-instruct/commit/42aa63c) |
| 7B RL Zero Math | [`7b_rlzero_math.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_rlzero_math.sh)   | https://beaker.org/ex/01K8V8TSX5K8BGZPJATZEE1003/ | https://wandb.ai/ai2-llm/open_instruct_internal/runs/w0ql4f5r | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 7B RL Zero Code | [`7b_rlzero_code.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_rlzero_code.sh)   | https://beaker.org/ex/01K7FSWM4717FAR9KF6GE958CA/ | https://wandb.ai/ai2-llm/open_instruct_internal/runs/o40rwmu8 | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 7B RL Zero IF   | [`7b_rlzero_instruction_following.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_rlzero_instruction_following.sh) | https://beaker.org/ex/01K7MVRTNJNYB37GC8SDTYHKC1/ | https://wandb.ai/ai2-llm/open_instruct_internal/runs/hk80a60o | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 7B RL Zero General | [`7b_rlzero_general.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/7b_rlzero_general.sh) | https://beaker.org/ex/01K7FSZ2Y16KAV56Q0KB7TSWN7/ | https://wandb.ai/ai2-llm/open_instruct_internal/runs/0tscl05k | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 32B Instruct SFT | [`32b_instruct_sft.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/32b_instruct_sft.sh) | https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KA4TA2T20EPNG1ADPBQ3A3RZ | https://wandb.ai/ai2-llm/jacobm-7B-sft/runs/en7w8mj1 | [`abfa4ea`](https://github.com/allenai/OLMo-core/commit/abfa4eaaf6dfa2b77b1a9586ccca31013fc3e4ea) |
| 32B Instruct DPO | [`32b_instruct_dpo.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/32b_instruct_dpo.sh) | https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KA9G0T8C8Y4RVN691AEETNJD | https://wandb.ai/ai2-llm/open_instruct_internal/runs/07o8dec7 | [`2fd104e`](https://github.com/allenai/open-instruct/commit/2fd104e7) |
| 32B Instruct RL | [`32b_instruct_rl.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/32b_instruct_rl.sh) | https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KAJQH1X2PRZP5VYZ1F0Z96KK | https://wandb.ai/ai2-llm/open_instruct_internal/runs/9gzo45lx | [`8d8232c`](https://github.com/allenai/open-instruct/commit/8d8232cc) |
| 32B Think SFT | [`32b_think_sft.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/32b_think_sft.sh) | https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01K9HRTZQGZJV22MPDZDK6KG56, https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01K9HWJQS5N23K3W02YHKEG6BS | https://wandb.ai/ai2-llm/saumyam-7B-sft/runs/gn5kre41, https://wandb.ai/ai2-llm/saumyam-7B-sft/runs/twcn6j46 | [`79a184c`](https://github.com/allenai/OLMo-core/commit/79a184cf70d83df6bcb7fe6f5fadffbc717b6ce5) |
| 32B Think DPO   | [`32b_think_dpo.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/32b_think_dpo.sh)    | https://beaker.org/ex/01K9VYQV2RFPS9ECP63JFQFVDN | https://wandb.ai/ai2-llm/open_instruct_internal/runs/te37gyey | [`2fd104e`](https://github.com/allenai/open-instruct/commit/2fd104e) |
| 32B Think RL          | [`32b_think_rl.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/32b_think_rl.sh) | https://beaker.org/ex/01KA4ZXT7MCVK493Y2B3K0BC82 | https://wandb.ai/ai2-llm/open_instruct_internal/runs/29h723j6 | [`42aa63c`](https://github.com/allenai/open-instruct/commit/42aa63c) |

To reproduce these runs, if you are internal to Ai2, you can run [`./scripts/train/build_image_and_launch.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/build_image_and_launch.sh)

```
git checkout $COMMIT
./scripts/train/build_image_and_launch.sh $SCRIPT_NAME
```

This will build an image and launch it. You can also check out the beaker link to see the **exact run** that produced the model! If you are external to Ai2, we have many [fine job postings](https://allenai.org/careers), but, unfortunately, do not have great advice on how to launch these jobs. Preliminary steps to launch on your own infrastructure would involve:

1. Modifying the launch scripts to remove the stuff attached to the [`mason.py`](https://github.com/allenai/open-instruct/blob/main/mason.py) command
2. Setting up your own cluster with the requisite number of {H,A}100 nodes, connected together via Ray.

## Wandb reports

We also have a bunch of Wandb reports for each stage, which contains the same information as above, in a slightly different format: 

| experiment name | wandb report |
|---|---|
| Olmo 3 7B Think (SFT, DPO, RL) | https://wandb.ai/ai2-llm/Olmo-3-7B-Think/reports/Olmo-3-7B-Think-SFT-DPO-RL--VmlldzoxNTE3ODQzMA |
| Olmo 3 7B Instruct (SFT, DPO, RL) | https://wandb.ai/ai2-llm/Olmo-3-7B-Instruct/reports/Olmo-3-7B-Instruct-SFT-DPO-RL--VmlldzoxNTE3ODk3Mg |
| Olmo 3 7B RL Zero (General, Math, Code, IF) | https://wandb.ai/ai2-llm/Olmo-3-7B-RL-Zero/reports/Olmo-3-7B-RL-Zero--VmlldzoxNTM0OTI1Nw |
| Olmo 3 32B Think (SFT, DPO, RL) & 3.1 | https://wandb.ai/ai2-llm/Olmo-3-32B-Think/reports/Olmo-3-32B-Think-SFT-DPO-RL--VmlldzoxNTE3OTA5Mg |
| Olmo 3 32B Instruct (SFT, DPO, RL) | https://wandb.ai/ai2-llm/Olmo-3-32B-Instruct/reports/Olmo-3-32B-Instruct-SFT-DPO-RL--VmlldzoxNTM0OTIzNw |
