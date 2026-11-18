# Core Ulysses context parallelism

The MoE trainer can opt into Ulysses with `trainer.context_parallel_size = 2`
or `4`. Core owns attention, recurrent computation and token sharding. MILES'
existing contiguous-output CP helpers select response logits and redistribute
response scores into its loss layout. Vocabulary logits are never gathered.
This is an initial mechanics implementation; qualify the intended topology and
sequence lengths before using it for a learning run.

Small's Scalable-Softmax requires the Ulysses support in
[OLMo-core #958](https://github.com/allenai/Olmo-core/pull/958), included in this
integration's Core pin. That change computes scaling from original document
positions before the attention exchange. The adapter is tracked in
[MILES #45](https://github.com/allenai/miles-olmo-core/pull/45), and configuration
and smoke tests in
[Open Instruct #2284](https://github.com/allenai/open-instruct/pull/2284).

```toml
[trainer]
gpus = 8
expert_parallel_size = 8
context_parallel_size = 2
sequence_packing = true
packing_max_tokens = 34816

[core]
router_aux_loss_weight = 0.0
router_z_loss_weight = 0.0
```

CP partners share a pack and process contiguous token slices. Each rank's
attention exchange reconstructs the full sequence for a subset of heads, then
exchanges outputs back to token slices. Core preserves document boundaries and
pads packs internally. The adapter shards replay IDs using the same balancer,
including synthetic final-token and padding routes.

Data parallelism has `trainer GPUs / CP` distinct input partitions. Core folds
EP over DP × CP, so EP8 plus CP2 can use eight trainer GPUs with four distinct
input partitions. The expert degree still divides the trainer world size. CP
does not change inference TP or allocate inference GPUs.

The adapter registers Core's groups in MILES and enables `miles.allgather_cp` for
contiguous Ulysses outputs. An explicit false value conflicts. Existing MILES
helpers distribute response probabilities, masks and advantages. Core averages
gradients over DP × CP; the adapter adjusts response-mean scaling and deduplicates
shared sample/token counts for token means. Full-response diagnostics gather
response vectors only and count each CP-shared sample once.

Initial scope is MoE GRPO policy loss, sequence packing and zero router auxiliary
losses. Dense trainers, trainer TP/PP, reduce-scatter, expert-balanced packing and
forced-exit training, custom loss overrides and global advantage whitening are
outside this qualification. CP degree must divide the
relevant head counts; Core validates model geometry. The full underlying Core
feature surface is broader than this adapter boundary.

Native checkpoints record CP degree and reject a changed degree on resume.
Older manifests without the field describe CP1. Same-topology resume still
requires its own GPU qualification.

`tests/miles/cp_contract.py` compares EP2/CP1 and EP2/CP2 Adam states for the same
hybrid MoE fixture, with response/token means and recomputation off/on. The first
moments catch uniform gradient-scaling errors that an Adam parameter update can
hide. The fixture includes Scalable-Softmax, nonunit per-head Q/K gains and
grouped-query attention. `scripts/miles/cp_contract.sh` runs this bounded two-GPU check in the
committed MILES image. `tests/miles/configs/cp2-smoke.toml` exercises a Tiny
EP2/CP2 trainer and a TP1 engine with synthetic rewards, four updates and no
checkpoint/export. It is not a Small throughput or learning-quality recipe.

The initial [two-GPU comparison](https://beaker.org/ex/01M7RFN4CWNQM1Z3BT1QWYQWEF)
passed all eight cases. Across dense, expert and router parameter groups, the
largest relative L2 differences were 0.242% for first Adam moments, 0.563% for
second moments and 0.0214% for master parameters. The
[Tiny integration smoke](https://beaker.org/ex/01M7RFN575RPX8JB55AXGQGTQN)
completed four updates and clean teardown with startup weight equality, routing
replay and zero difference between scoring and training forwards. Both ran on
Holmes B300s at high priority with zero minimum runtime (unallocated).
Full Small training, long-context capacity, CP4 model execution and checkpoint
resume remain unqualified by these checks.
