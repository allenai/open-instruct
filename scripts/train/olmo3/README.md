# Overview

For our recent [Olmo3 paper](insert link), we used the following scripts to train our models:

Model           | Script name        | Commit |
-----------------------------------------------
7B Think DPO    | `7b_think_dpo.sh`     | 68da0a1
32B Think DPO   | `32b_think_dpo.sh`    | 2fd104e
7B Instruct DPO | `7b_instruct_dpo.sh`  | 2fd104e
7B RL           | `7b_rl.sh`            | 9ade62d
32B RL          | `32b_rl.sh`           | 42aa63c


To reproduce these runs, if you are internal to Ai2, you can run

```
git checkout $COMMIT
./scripts/train/build_image_and_launch.sh $SCRIPT_NAME
```

This will build an image and launch it. If you are external to Ai2, we have many [fine job postings](https://allenai.org/careers), but, unfortunately, do not have great advice on how to launch these jobs.
