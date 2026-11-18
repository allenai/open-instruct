# Core Ulysses context parallelism

The MoE trainer can opt into Ulysses with `trainer.context_parallel_size = 2`
or `4`. Core owns attention, recurrent computation and token sharding. MILES'
existing contiguous-output CP helpers select response logits and redistribute
response scores into its loss layout. Vocabulary logits are never gathered.
This is an initial mechanics implementation; qualify the intended topology and
sequence lengths before using it for a learning run.

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
hide. `scripts/miles/cp_contract.sh` runs this bounded two-GPU check in the
committed MILES image. `tests/miles/configs/cp2-smoke.toml` exercises a Tiny
EP2/CP2 trainer and a TP1 engine with synthetic rewards, four updates and no
checkpoint/export. It is not a Small throughput or learning-quality recipe.
