---
name: tpu
description: Run or debug post-training (SFT) on Ai2's TPU v4 cluster gke-tpu-v4 with MaxText + Tunix. Use before launching, resuming, watching, exporting or deleting anything on the TPUs, or when choosing a slice, queue, image or mesh for a TPU run.
---

# TPU post-training

The runbook is `docs/algorithms/tpu_sft_guide.md`; read it before acting. Scripts, overlays and images are in
`scripts/tpu/maxtext/`, and the cluster's own rules are in allenai/olmo-gcp-infra's README.

Rules an agent follows:

- **Queue `post-training` only.** Hero owns the quota and preempts us. Never submit to `hero` or `dev`, and never hold
  a slice in a way that delays hero.
- **Delete nothing on GCS** (`gs://ai2-olmo-dev`): checkpoints, data and exports belong to the run's owner. Deleting
  your own failed JobSet object is fine; save its pod logs first.
- **Read loss from `LOSS-PARTS` lines only.** Per-pod `completed step … loss:` lines differ across pods. Compare `ce=`,
  never Tunix's `loss`.
- **Resume only on the restore-fix image** (`maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e`), never from step 0.
- **Use a `4x4x8`** (guide: What the cluster is). Since 2026-10-21 the cluster is seven `4x4x8`s and three `8x8x16`s;
  the `8x8x8` is gone. The `8x8x16`s are hero's, borrowed only. Check the live layout before planning (command in the
  guide); it changes when the cluster is resliced.
- **For `ici_context_parallelism` > 1, set `context_parallel_load_balance=False`** and require step-1 `LOSS-PARTS` CE
  to match an HF forward of the same batch within 1e-3 before training. With load balancing on, dense Olmo 3 7B's CE
  was off by +0.39 (cp4) and +1.98 (cp8); with it off, cp4 matched (guide: Traps).
- **Stop a run by deactivating its Kueue workload** (`spec.active=false`, command in the guide). Suspending the JobSet
  does nothing while it is admitted.
- **Pick the mesh from the guide's measured table** (Changing the recipe). None of them has been tuned. A new slice or
  mesh needs an xprof of one TPU step first (in-loop all-gathers cost 140x once), and its own same-slice control.
- **Stream `REQUIRED_RBF` from JAX process 0 from step 1** (`kubectl logs -f` on that pod) when a rule needs every step.
  Polling `kubectl logs` loses steps at kubelet's log rotation, and the agent account cannot read Cloud Logging.
- **Stagger big-slice launches.** Two slices loading at once hit the GCS egress quota.
- **Keep one slice shape per experiment.** The data order depends on the host count, so arms and their control share
  a shape.
- **Set `CLOUDSDK_CORE_ACCOUNT`** to the agent service account in every process that calls kubectl or gcloud,
  background watchers included. The user login lapses daily and kubectl then fails silently.
- **Kill processes by PID, not `pkill -f`,** inside `bash -c`.

When the run answers a research question, the experiment ledger's rules apply too (olmo-post-training-ledger,
`ledger` and `storage` skills).
