---
name: tpu
description: Run or debug post-training (SFT) on Ai2's TPU v4 cluster gke-tpu-v4 with MaxText + Tunix. Use before launching, resuming, watching, exporting or deleting anything on the TPUs, or when choosing a slice, queue, image or mesh for a TPU run.
---

# TPU post-training

The runbook is `docs/algorithms/tpu_sft_guide.md`; read it before acting. Scripts, overlays and images are in
`scripts/tpu/maxtext/`, and the cluster's own rules are in allenai/olmo-gcp-infra's README.

Rules an agent follows:

- **Queue `post-training` only.** It has nominal quota for the `8x8x8` and two `4x4x8`s, and Kueue never preempts a
  job running within its own queue's nominal quota; hero and dev reclaim only what we borrow (guide: What the cluster
  is). Never submit to `hero` or `dev`, and never hold a slice in a way that delays hero.
- **Delete nothing on GCS** (`gs://ai2-olmo-dev`): checkpoints, data and exports belong to the run's owner. Deleting
  your own failed JobSet object is fine; save its pod logs first.
- **Read loss from `LOSS-PARTS` lines only.** Per-pod `completed step … loss:` lines differ across pods. Compare `ce=`,
  never Tunix's `loss`.
- **Resume only on the restore-fix image** (`maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e`), never from step 0.
- **Use a `4x4x8` for the tiny SFT recipe (H045)** (guide: What the cluster is). Since 2026-11-03 the cluster is three
  `4x4x8`s, one `8x8x8` and hero's `8x8x48`. Check the live layout before planning (command in the guide); it changes
  when the cluster is resliced.
- **For `ici_context_parallelism` > 1, set `context_parallel_load_balance=False`** and require step-1 `LOSS-PARTS` CE
  to match an HF forward of the same batch within 1e-3 before training. With load balancing on, dense Olmo 3 7B's CE
  was off by +0.39 (cp4) and +1.98 (cp8); with it off, cp4 matched (guide: Traps).
- **Stop a run by deactivating its Kueue workload** (`spec.active=false`, command in the guide). Suspending the JobSet
  does nothing while it is admitted.
- **Pick the mesh from the guide's measured table** (Changing the recipe). On the `8x8x8`, a short sweep of the
  parallelism split, batch, vocab tiling and XLA collective flags found only data 8 x fsdp 8 faster for tiny (+8%).
  For small at 16 rows, the post-training MaxText notes
  ([doc](https://docs.google.com/document/d/1tqr-61buACTuGTBEQkVwb644o_QseqwpYipyE9aza4w/edit), 2026-11-17) measured
  drop-free 6.6 s/step on a `4x4x8` (fsdp 16 x expert 8) and 4.1 s/step on the `8x8x8` (data 4 x fsdp 4 x expert 32),
  plus host-loop and checkpoint fixes; copy its config but not its `LIBTPU_INIT_ARGS` (below). Don't set
  `--xla_tpu_enable_async_ragged_all_to_all`: it corrupts an MoE router gate's gradient (~1e33) while the forward loss
  stays normal, every few steps on the `8x8x8` (guide: Changing the recipe). Run any new slice well past step 20 before
  trusting it.
- **Size the MoE buffer for a trained router.** On SFT data, dropping nothing needs `ragged_buffer_factor` of about 7 at
  expert 8 and about 14 at expert 32; 3.0 drops tokens. Log `REQUIRED_RBF` and keep the factor above its maximum.
- **An inf grad norm can be finite garbage.** Without the async ragged flag, a patched small run on the `8x8x8` had
  routed-expert `wi_0` gradients of ~1e31 on 7 of 8 steps (every element finite, forward loss normal), so the fp32
  norm overflowed. When the norm is inf or huge, print each leaf's max |grad| before blaming the data. A new slice or mesh needs an xprof of one TPU step first (in-loop
  all-gathers cost 140x once), and its own same-slice control.
- **Stream `REQUIRED_RBF` from JAX process 0 from step 1** (`kubectl logs -f` on that pod) when a rule needs every step.
  Polling `kubectl logs` loses steps at kubelet's log rotation, and the agent account cannot read Cloud Logging.
- **Mount only the folders a job reads, never one that holds checkpoints.** gcsfuse's metadata prefetch lists
  everything under `only-dir` on every host at every pod start, so a mount of `post-training/` lists every run's
  checkpoints each time. Mount `post-training/data`
  and `post-training/tokenizers` separately at their old paths (`k8s/narrow_mounts.py`); read and write checkpoints
  over `gs://`. Jobs launched with a wide mount keep it until relaunched.
- **Stagger big-slice launches.** Two slices loading at once hit the GCS egress quota.
- **Keep one slice shape per experiment.** The data order depends on the host count, so arms and their control share
  a shape.
- **Set `CLOUDSDK_CORE_ACCOUNT`** to the agent service account in every process that calls kubectl or gcloud,
  background watchers included. The user login lapses daily and kubectl then fails silently.
- **Kill processes by PID, not `pkill -f`,** inside `bash -c`.

When the run answers a research question, the experiment ledger's rules apply too (olmo-post-training-ledger,
`ledger` and `storage` skills).
