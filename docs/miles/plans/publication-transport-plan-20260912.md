# Publication transport: plan

> Historical proposal. For current operating instructions, start at the [MILES guide](../index.md).

Written 2026-09-12. Target 6 from
[the optimization targets](core-optimization-targets-20260911.md).
Publication is the step after each optimizer update where the trainer pushes
new weights into the SGLang serving engine. It costs 3.6 to 3.9 s per update in
the 500-update runs, about 10 percent of Core's steady-state cycle.

## What the numbers say

Core's own publication timers, from `publication.jsonl` and the run logs:

| Stage | Seconds | What it covers |
|---|---:|---|
| pause and connect | 0.02 | pause generation, begin weight update |
| export and pack | 0.25 to 0.45 | Core native to HF conversion, bucket assembly |
| transport and load | 3.2 to 3.4 | NCCL broadcast plus the engine loading each bucket |
| finalize | 0.05 | flush cache, version bump, resume |

Each publication moves 29,669 tensors totaling 37.0 GB in 35 buckets of about
1 GiB. Megatron publishes the same tensors through the same engine and reports
its own split: gather 1.16 s, conversion 0.28 s, and 3.44 s waiting for the
engine to load. The engine-side wait is the same 3.3 to 3.4 s in both arms,
which points at the engine, not at either trainer.

Where those tensors come from: the model has 512 routed experts and 16 MoE
layers. Each expert contributes three HF tensors (gate, up, down), so 24,576 of
the 29,669 tensors are individual expert slices averaging about 1.25 MB. A
37 GB NCCL broadcast over intra-node NVLink should take well under half a
second. The remaining three seconds is per-tensor work.

## The transport chain

**Trainer side** (`open_instruct/miles/actor.py::update_weights`,
`open_instruct/miles/publication.py`). The Core exporter
(`olmo_core/nn/moe/v2/olmo3.py::iter_olmo3_moe_hf_state`) streams HF-named
tensors: for each MoE layer it all-gathers the expert slab across the two EP
ranks once, then splits it into per-expert gate, up, and down tensors. The actor
accumulates tensors into 1 GiB buckets. For each bucket it flattens the tensors
into one byte buffer, posts the names, dtypes, and shapes to the engine over
HTTP, broadcasts the buffer with one NCCL call, and waits for the engine's
response before packing the next bucket.

**Engine side** (`sglang/srt/model_executor/model_runner_components/weight_updater.py::_update_bucketed_weights_from_distributed`).
The engine allocates the bucket, receives the broadcast, reconstructs views
for each tensor, and calls the model's `load_weights` with the list. That
method lives in the olmo-sglang adapter
(`olmo_sglang/models/olmo3_moe.py::load_weights`). For every tensor it:

1. Checks the name against 8 stacked-parameter patterns by substring.
2. If unmatched, scans a list of 512 times 3 expert patterns by substring until
   one matches. For expert tensors this is on average about 770 string checks.
3. Calls `FusedMoE.weight_loader`, which maps the global expert to a local one,
   slices `param.data[expert_id]`, narrows the shard, and issues one
   device-to-device copy.

A local microbenchmark of step 2 alone, at 512 experts and 16 layers, takes
1.0 s per publication on the host CPU. That cost is real and measured. Whether
it and the 24,576 loader calls account for the whole 3.3 s is not established.
Three explanations fit the evidence so far, and phase 0 exists to choose
between them, not to confirm the first:

| Hypothesis | What it predicts | How to tell |
|---|---|---|
| Engine per-tensor loop (name scan plus 24,576 loader calls and small copies) | Engine `load_weights` time scales with tensor count, not bytes; broadcast is a small share | Engine-side timers split broadcast from load; a profile of one bucket |
| NCCL broadcast not on NVLink | 37 GB at 11 GB/s is close to NCCL's host-shared-memory fallback rate when peer access is unavailable; NVLink would move it in well under 0.5 s | `NCCL_DEBUG=INFO` on one publication shows the transport chosen for the update group; `nvidia-smi topo -m` on the node |
| Per-bucket control latency (Ray call, HTTP server, scheduler, worker, response) | Time scales with bucket count; smaller buckets are slower, larger are faster | Sweep 0.5, 1, 2, 4 GiB buckets with the same timers |

The run logs carry no NCCL debug output, so the transport cannot be read from
the completed runs. The controls exercise with smaller buckets was slightly
slower, which shows per-bucket overhead exists without sizing it.

## Phases

### Phase 0: attribute the engine-side wait (no behavior change)

Add timers inside the engine's bucketed update: broadcast wait, reconstruct,
and `load_weights`, returned in the HTTP response so the trainer can record
them next to its own stages. Run one short tiny job and one full-model
publication with `NCCL_DEBUG=INFO` on the update group, and sweep bucket size.
Profile `load_weights` on one full-model bucket, attributing time to the name
scan, the loader calls, and the copies.

Acceptance: a per-bucket breakdown that sums to the observed transport-and-load
time, plus the transport NCCL reports. The phases that follow assume the
per-tensor loop dominates. If the broadcast is the large share, fix the NCCL
path first (peer access, group placement, or CUMEM settings), and phases 1 and
2 become secondary. If bucket count dominates, phase 3 comes first.

Effort: a day. Repos: olmo-sglang (timers), open-instruct adapter (record them).

**Phase 0 result (2026-09-12, [full record](../measurements/publication-profile-20260912.md)).**
Done with trainer-side split timers only; no engine change was needed. On the
full model, the 37 GB broadcast takes about 25 ms over NVLink (`P2P/CUMEM`, 32
channels, 989 GB/s effective). The engine wait is 1.87 s at 1 GiB buckets and
regresses across 813 bucket observations as 40.6 µs per tensor plus 17.6 ms per
gigabyte plus 4.3 ms per bucket (R² 0.98): about 1.2 s of per-tensor work and
0.65 s of small-copy work per publication. Bucket size matters only at the
extremes (256 MiB costs 0.8 s more; 4 GiB adds allocation cost); 1 to 2 GiB is
flat. The equality check passed after 16 publications. The transport hypothesis
is closed and phase 3 is closed with it; phases 1 and 2 target the measured cost.
One caveat: the idle republish totals 2.6 s against 3.6 to 3.9 s during training;
the split timers are now permanent, so the next training run shows where that
second lives.

**Phases 1 and 2 result (2026-09-12, [full record](../measurements/publication-profile-20260912.md#phases-1-and-2-results)).**
Both landed and were measured with the same profile. Phase 1 took publication
at 1 GiB from 2.60 s to 2.21 s, halving the per-tensor coefficient. Phase 2
took it to 0.49 s at 1 GiB and 0.35 s at 2 GiB, with 523 tensors instead of
29,669 and the equality check passing after sixteen fused publications. The
full-SFT starter profiles now use fused publication with 2 GiB buckets. Phase 3
is unnecessary.

### Phase 1: constant-time name resolution in the engine loader

Replace the two substring scans in `load_weights` with a lookup table built
once per model: parse each incoming name with a regular expression for the
expert index and projection, and map directly to the fused parameter, shard,
and expert id. Everything downstream stays the same.

Expected: removes about 1 s per publication in both arms. Arithmetic untouched,
since it changes only how a tensor finds its destination.

Tests: the adapter's existing weight-load validation, plus the full
serving-weight equality check (`check_weight_update_equal`) at initial
publication in a tiny run. Add a unit test that the new lookup resolves every
name the old scan resolved, for a small and a 512-expert config, by comparing
the two implementations' outputs on the full HF name inventory.

Effort: a day. Repo: olmo-sglang only. Needs a runtime pin bump.

### Phase 2: publish experts as per-layer stacked tensors

Today the trainer gathers each layer's expert slab, splits it into 1,536
tensors, and the engine reassembles them into its own stacked storage with
1,536 copies. Both ends already hold the stacked form. Emit it directly:
one gate-and-up tensor and one down tensor per MoE layer, in the layout of
the engine's fused `w13_weight` and `w2_weight`, and load each with a single
copy.

SGLang already has a fused expert path (`make_expert_params_mapping_fused`,
`weight_loader_fused`) used for checkpoints that store experts stacked. The
work is:

- Core exporter: an option to yield per-layer stacked tensors under fused names
  instead of per-expert names. The gate and up halves must be concatenated in
  the engine's order, which differs from Core's `w_up_gate` slab order, so the
  exporter permutes once per layer as a view or a single concatenation.
- olmo-sglang loader: accept the fused names and route them to
  `weight_loader_fused`. Under serving EP greater than 1, the fused loader must
  slice the local expert range. Current serving is EP 1, so this can be
  asserted rather than implemented first.
- Adapter: a config switch, `core.expert_publication = "per_expert" | "fused"`,
  default per-expert until qualified, then flipped.

Expected: tensor count drops from 29,669 to about 5,100, engine loader calls
from 24,576 to 32 for the experts, and the trainer skips the per-expert split.
Combined with phase 1, transport-and-load should approach the broadcast time.
Target: publication under 1 s.

Tests, in order:

1. Export equivalence, no GPU engine needed: for a tiny MoE and for a
   512-expert small-hidden config, the fused export reassembled into per-expert
   tensors must equal the per-expert export bit for bit. Extend
   `models.export_state` coverage in `tests/miles/test_runtime.py`.
2. Toy MoE publication A/B (`tests/miles/local_moe.py`, launched by
   `scripts/miles/launch_trial.py --disaggregated`) gains a fused arm alongside
   the existing per-tensor and flattened arms. The audit already compares the
   engine's weights to the HF export after publication.
3. Full model: one short run with `check_weight_update_equal` and
   `diagnostic_interval = 1` so the full equality check runs after trained
   updates, not only at initial publication. Then the standard smoke.

Risk: a layout mismatch loads weights into the wrong expert slots. The full
equality check catches that deterministically, which is why it stays on for
every qualifying run. No numerical change is involved.

Effort: two to three days. Repos: OLMo-core, olmo-sglang, open-instruct.
Needs a runtime pin bump and image rebuild.

### Phase 3: bucket sizing and overlap

After phases 1 and 2, what remains is per-bucket fixed cost and the serial
pack, broadcast, load handshake. Options, in order of simplicity:

- Larger buckets. 1 GiB gives 35 round trips; 4 GiB gives 9. The controls
  exercise already tried smaller buckets without measuring an isolated
  effect, so measure 1, 2, and 4 GiB with phase 0's timers.
- Double-buffering. Pack and broadcast bucket N+1 while the engine loads
  bucket N. The trainer loop is easy to overlap; the engine's bucketed update
  is a blocking call, so this needs either two NCCL groups or an engine-side
  queue. MILES has a pipeline-depth option for the colocated IPC path only.
  Defer unless phase 0 shows the broadcast is a large share.

Expected: modest after phases 1 and 2, perhaps 0.2 to 0.3 s. Do not start here.

## Cross-cutting

- **Both arms benefit.** The engine loader is shared. Megatron's 3.4 s engine
  wait drops with phases 1 and 2 as well, which keeps the two arms comparable.
- **Runtime pins.** olmo-sglang and OLMo-core are pinned in
  `runtime/miles/runtime.lock.json`. Phases 1 and 2 each need a pin bump, an
  image rebuild, and the standard smoke. Batch them into one bump if the
  schedule allows.
- **The equality check is the safety net.** Every qualifying run keeps
  `check_weight_update_equal` on, and at least one run per phase uses
  `diagnostic_interval = 1` so the check covers trained weights.
- **Do not combine with other changes in the qualifying runs.** The phase
  breakdown script attributes the saving; anything else in the same run
  muddies it.

## Expected outcome

| State | Publication per update (idle republish) | Measured |
|---|---:|---|
| Baseline | 2.60 s | phase 0 |
| After phase 1 | 2.21 s | measured |
| After phase 2, 1 GiB buckets | 0.49 s | measured |
| After phase 2, 2 GiB buckets | 0.35 s | measured, now the profile default |

Training runs measured 3.6 to 3.9 s before these changes; the per-bucket timers
are permanent, so the next training run reports the in-loop figure directly.

The bigger consequence is structural: publication stops scaling with expert
count. At 512 experts the per-expert path is already the engine's slowest
step, and the hero model and any future larger MoE would make it worse.
