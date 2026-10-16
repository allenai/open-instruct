# H077 longer SFT continuation

This branch extends the original fixed 11768-update schedule from retained
short-run checkpoints. The original five-arm image remains pinned at f958fa1ba.
Do not launch a long run until its short-run audit and registered full-cache
validation have passed and the matched control/treatment decision is recorded.

The trainer uses the same historical Core, BF16 forward/backward, optimizer,
batching, attention implementation and compilation settings. It requires an
explicit registered full-cache manifest hash and stages that cache per node.
Before any update, `--atomic_verify_resume` compares the model, FP32 masters,
moments, optimizer steps, auxiliary rolling statistics, trainer counters, loader
position, callback state and RNG against the retained native checkpoint and its
training receipts. It checks the next scheduled LR without modifying the live
optimizer, then checks the first batch against the restored loader position.
These are runtime guards; CPU tests alone do not qualify a real 16-GPU restore.

The first `20261016b` continuation attempts stopped before any update because the
auxiliary-state verifier used PyTorch's generic filesystem reader on Core's
storage metadata (`_StorageInfo.transform_descriptors` missing). The verifier now
uses Core's paired reader. Its regression test writes with Core's actual writer;
the previous test's generic PyTorch writer failed to exercise this incompatibility.
No model, optimizer, schedule, tokenization or saved checkpoint is changed by this
reader correction. Replacement attempts must still pass every restoration guard.

Full checkpoints and state hashes are retained every 1024 updates and at 11768.
The first twelve resumed batches, checkpoint batches and final three batches are
audited. This avoids extending the short-run `step >= 510` audit to every long-run
update. No inference probes run inside the trainer. A replacement uses a fresh
output root and an explicit retained intermediate checkpoint.

`launch_long.sh IMAGE ARM RUN_ID MANIFEST_SHA256 RESUME_CHECKPOINT [MINUTES]`
renders two eight-GPU Holmes replicas, urgent allocated priority, protected
minimum runtime equal to the execution timeout, no automatic retry, and retained
checkpoints. Its maximum/default timeout is 420 minutes. The bound is based on
the short run's measured ~1.68 seconds/update on B300: about 5.3 hours for the
11256 remaining updates, plus measured startup and checkpoint/hash overhead.
Report actual startup, node-local staging, sustained throughput and checkpoint
cost separately. Stop and investigate a failed restore rather than falling back
to fresh initialization.

Build only with `scripts/train/atomic_think/Dockerfile`, passing both GIT_COMMIT
and GIT_BRANCH. It overlays project code on the immutable historical CUDA 13
runtime. Publish that image under the branch launcher alias before using
`build_image_and_launch.sh`; an ordinary Dockerfile rebuild changes the runtime.

Full-SFT quality, tokenizer/export behavior and early-exit usefulness still
require their separate evaluations. Successful continuation is not that result.
