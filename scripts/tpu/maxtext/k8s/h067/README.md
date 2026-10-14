# H067 TPU runs

Six overlays, one per arm and data-order seed, for olmo-post-training-ledger H067. Each is `../olmo35-sft-4x4x8` (H045's TPU run r2) with only these changed:
- the names;
- `TRAIN_FILES`, set to `/mnt/post-training/data/h067/<arm>-s<seed>`;
- `STEPS` = `LR_SCHEDULE_STEPS`, the arm's own count (A0' 11,768, R 8,046, T 7,941; H067 Amendment 1);
- `NUM_EPOCH`, as H067's P2 measured it: 2 for A0', 1 for R and T. Every R/T data host reaches its cutoff in one pass.

Launch: `kubectl --context gke_ai2-olmo_us-central2-b_gke-tpu-v4 apply -k scripts/tpu/maxtext/k8s/h067/<arm>-s<seed>`.
