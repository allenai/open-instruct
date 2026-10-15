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
- **Keep one slice shape per experiment.** The data order depends on the host count, so arms and their control share
  a shape.
- **Set `CLOUDSDK_CORE_ACCOUNT`** to the agent service account in every process that calls kubectl or gcloud,
  background watchers included. The user login lapses daily and kubectl then fails silently.
- **Hammond prints UTC for `TZ=America/Los_Angeles`** (no tzdata). Convert times by hand, or use Python's `zoneinfo`
  with the `tzdata` package.
- **Kill processes by PID, not `pkill -f`,** inside `bash -c`.

When the run answers a research question, the experiment ledger's rules apply too (olmo-post-training-ledger,
`ledger` and `storage` skills).
