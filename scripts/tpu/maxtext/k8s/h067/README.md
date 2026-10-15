# H067 TPU runs

Six overlays, one per arm and data-order seed, for olmo-post-training-ledger H067. Each is `../olmo35-sft-4x4x8` (H045's TPU run r2) with only these changed:
- the names;
- `TRAIN_FILES`, set to `/mnt/post-training/data/h067/<arm>-s<seed>`;
- `STEPS` = `LR_SCHEDULE_STEPS`, the arm's own count (A0' 11,768, R 8,046, T 7,941; H067 Amendment 1);
- `NUM_EPOCH`, as H067's P2 measured it: 2 for A0', 1 for R and T. Every R/T data host reaches its cutoff in one pass.

Launch: `kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 apply -k scripts/tpu/maxtext/k8s/h067/<arm>-s<seed>`.

**Image for r-s1 and t-s1 (2026-10-15 PT).** They run on `maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e` (sha256:7d7f4938f98b4a70e918cc768d2109f6d1080d5a8944b829a5bceda49b24cabc). That is r2's image (`51f8cc4b-p907d86f3`, sha256:07355d6e91de3af20f685e796e98eba7a6e232f4d31cd0cd15d8f8a4c3b1cbc5) plus maxtext c469d980 (`restorefix-c469d980.patch`, from tpu-olmo-3).
- It restores the optimizer state into its live shardings; Tunix's `_fix_sharding` replicated the expert Adam moments and ran out of HBM.
- The rest of the diff is logging only (HBM lines, and an env-gated batch hash that is off here).
- Training numerics are unchanged, and the restore path runs only after a preemption. TPU test (H072 h072-rs-fix): CE at steps 301-310 bit-identical to an uninterrupted run, and 9,952 of 9,952 batch hashes equal.
- The other four runs started on r2's image. If one is preempted, it resumes on the fixed image.
