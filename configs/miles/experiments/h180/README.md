# H180 corrected treatment launch

H180 studies whether rewarding only delivered final answers changes downstream accuracy and thinking closure. The first completed treatment accidentally discarded all instruction-following groups because its verifier required a closing-think marker after the reward adapter had stripped that marker. Robert authorized a treatment-only replacement from the original SFT checkpoint, reusing the completed control.

This branch starts from the original application revision `799ca13a13c2eb2d5985a54d50a95432f134066b` and keeps its runtime lock. The two reward files are exactly those in fix PR #2188 at `0483d5b730c940413db7cdc2a0015b1ddb3c3d5a`. The raw-response path delegates to the unchanged qualified IF verifier; the final-answer path grades the extracted suffix. All other reward entries and training settings are unchanged.

The evaluation runner is exactly the committed repair at `1c630e6e1e62b3e82e0cfd5b14f653cad1247435`, SHA-256 `d61993740a74c81816ef88aa321ed34ea85784906cf1703b1edbdd47f8d53c41`, already used for all three original SciCode checkpoint panels. The evaluator revision, image and metrics remain pinned.

Build `runtime/miles/Dockerfile.h180-if-repair` on the original qualified image `01M6G73QM7B1TRVDBSXGP5J2A8` / local Docker ID `sha256:9fc675b9e71598ad33ca9da25c6761d6c4981d404aa9c765d5e942eb67f8f8de`. Its small overlay carries only the two reward files, the matched evaluation runner and these experiment configs. Launch the committed recipe through `python -m open_instruct.miles run` with `MILES_EXISTING_IMAGE` set to the newly uploaded immutable ID.

The replacement uses one eight-B300 Holmes node: two trainer, five inference and one named judge, allocated urgent, four-hour protection, auto-resume and twelve-hour execution timeout. Local compiler caches remain enabled with shared persistence disabled. New output/record roots prevent resuming the bug-affected checkpoint. Save every 25 updates with native keep-last-one/keep-every-50 retention, preserving evaluation exports and final weights. Original resources remain intact.

The registered question, seed-17 design and endpoint panels remain unchanged. This is a nonconcurrent replacement comparison with the existing control, not a new seed replicate. Named-judge support is present. No production default changes or PR merges are requested.
