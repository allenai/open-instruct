The training image for Olmo 3.5 SFT: `../image/` (same Dockerfile and build.sh) built on Caleb's MaxText line for
Olmo 3.5 (AI-Hypercomputer/maxtext@51f8cc4b: Gagik's `olmo35-rebase` a7f34953 plus OLMo-core parity fixes, among
them forward-substitution KDA, `moe_aux_loss_reduction`, `warmup_step_offset`, and zeroed MoE overflow rows),
with this directory's `patches/maxtext.patch` (the SFT fixes cherry-picked onto it, plus KDA keeping the sequence
whole under context parallelism).

    cp ../image/build.sh . && ./build.sh 51f8cc4b7f4fc9976c3ab5eb2855ebc876c5e9d2   # -> maxtext-posttrain:51f8cc4b-pf387f341
