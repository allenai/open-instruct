# Overview

For our recent [Olmo3 paper](insert link), we used the following scripts to train our models:

| Model           | Script name           | Commit |
|-----------------|-----------------------|--------|
| 7B Think DPO    | `7b_think_rl.sh`     | [`68da0a1`](https://github.com/allenai/open-instruct/commit/68da0a1) |
| 7B Think DPO (w/out pipelineRL)    | `7b_think_no_pipeline.sh`     | [`68da0a1`](https://github.com/allenai/open-instruct/commit/68da0a1) |
| 7B Think DPO    | `7b_think_dpo.sh`     | [`68da0a1`](https://github.com/allenai/open-instruct/commit/68da0a1) |
| 32B Think DPO   | `32b_think_dpo.sh`    | [`2fd104e`](https://github.com/allenai/open-instruct/commit/2fd104e) |
| 7B Instruct DPO | `7b_instruct_dpo.sh`  | [`2fd104e`](https://github.com/allenai/open-instruct/commit/2fd104e) |
| 7B Instruct RL  | `7b_instruct_rl.sh`   | [`9ade62d`](https://github.com/allenai/open-instruct/commit/9ade62d) |
| 7B RL           | `7b_rl.sh`            | [`42aa63c`](https://github.com/allenai/open-instruct/commit/42aa63c) |
| 32B RL          | `32b_rl.sh`           | [`42aa63c`](https://github.com/allenai/open-instruct/commit/42aa63c) |
| 7B RL Zero      | `7b_rlzero.sh`        | [`f3ddfe1c`](https://github.com/allenai/open-instruct/commit/f3ddfe1c) |
| 32B RL Zero     | `32b_rlzero.sh`       | [`f3ddfe1c`](https://github.com/allenai/open-instruct/commit/f3ddfe1c) |
| 7B RL Zero Math | `7b_rlzero_math.sh`   | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 7B RL Zero Code | `7b_rlzero_code.sh`   | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 7B RL Zero IF   | `7b_rlzero_instruction_following.sh` | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |
| 7B RL Zero General | `7b_rlzero_general.sh` | [`d928a7c`](https://github.com/allenai/open-instruct/commit/d928a7c) |


To reproduce these runs, if you are internal to Ai2, you can run

```
git checkout $COMMIT
./scripts/train/build_image_and_launch.sh $SCRIPT_NAME
```

This will build an image and launch it. If you are external to Ai2, we have many [fine job postings](https://allenai.org/careers), but, unfortunately, do not have great advice on how to launch these jobs.
