The training image for Olmo 3.5 SFT: `../image/` (same Dockerfile and build.sh) built on Caleb's MaxText line for
Olmo 3.5 (AI-Hypercomputer/maxtext@51f8cc4b: Gagik's `olmo35-rebase` a7f34953 plus OLMo-core parity fixes, among
them forward-substitution KDA, `moe_aux_loss_reduction`, `warmup_step_offset`, and zeroed MoE overflow rows),
with this directory's MaxText patch: the SFT fixes cherry-picked onto it, KDA keeping the sequence whole
under context parallelism, Gagik's v4 performance commit (agagik-olmoe3-v4 dd9c9511: sub-block KDA, all-to-all
options), the `olmo35-ep-as-cp` mesh rule (KDA heads on the expert axis), the REQUIRED_RBF probe on the all-to-all
path, `max_num_checkpoints_to_keep` in SFT, and the block-doubling inverse indexing m directly (slicing a reshape of m
miscompiles on TPU v4). The patch is `git diff 51f8cc4b 9f55e7b6 -- src` on abhishekraok/maxtext (branch `olmo35-posttrain-perf`),
written by `fetch_patches.sh` from `maxtext_source`. It covers `src/` only: the base image has no top-level `scripts/`.

    cp ../image/build.sh . && ./build.sh 51f8cc4b7f4fc9976c3ab5eb2855ebc876c5e9d2   # -> maxtext-posttrain:51f8cc4b-p907d86f3
