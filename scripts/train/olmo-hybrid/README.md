# Overview

Olmo Hybrid is an experimental hybrid model built on top of the Olmo 3 recipe with a hybrid, gated delta net (GDN) architecture. We used the following scripts to train our models.

> **Note**: SFT runs (Think SFT and Instruct SFT) use the [OLMo-core](https://github.com/allenai/OLMo-core) SFT implementation and are not run from this repository. DPO is run with DeepSpeed via open-instruct's standalone DPO trainer, not the in-process OLMo-Core primitives version.

| Model           | Script name           | Beaker Link | Wandb URL | Commit |
|-----------------|----------------------|---|---|--------|
| 7B Think SFT | (OLMo-core) | [link](https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KG3YAYHXHNTRRDV8DJS902TJ) | [link](https://wandb.ai/ai2-llm/nathanl-Hybrid-7B-sft/runs/ued8kbuj) | TODO |
| 7B Instruct SFT | (OLMo-core) | [link](https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KHS8BD4N6B6AWYSYHGY8DQWB) | [link](https://wandb.ai/ai2-llm/nathanl-Hybrid-7B-sft/runs/gdm1bg5b) | TODO |
| 7B Instruct DPO | [`7b_instruct_dpo.sh`](7b_instruct_dpo.sh) | [link](https://beaker.org/orgs/ai2/workspaces/olmo-instruct/work/01KHWBB0GAQ0V9YS2B3VMH8MNN) | [link](https://wandb.ai/ai2-llm/open_instruct_internal/runs/xoqrwhvw) | TODO |

To reproduce DPO, if you are internal to Ai2, you can run [`./scripts/train/build_image_and_launch.sh`](https://github.com/allenai/open-instruct/blob/main/scripts/train/build_image_and_launch.sh)

```
git checkout $COMMIT
./scripts/train/build_image_and_launch.sh $SCRIPT_NAME
```

This will build an image and launch it. You can also check out the beaker link to see the **exact run** that produced the model! If you are external to Ai2, we have many [fine job postings](https://allenai.org/careers), but, unfortunately, do not have great advice on how to launch these jobs. Preliminary steps to launch on your own infrastructure would involve:

1. Modifying the launch scripts to remove the stuff attached to the [`mason.py`](https://github.com/allenai/open-instruct/blob/main/mason.py) command
2. Setting up your own cluster with the requisite number of {H,A}100 nodes, connected together via Ray.
