# H134 P4 x16 reference: optional diagnostic (not in prod6.sh's launch path)
Ledger-manager 2026-10-26: P4's x16 reference is a diagnostic. Run it on a free borrowed 8x8x16 under restore.lock (a 256-host
big restore: 30 min big-restore spacing, gr2 knobs), behind H134's production applies, each paired with its 4x4x8 candidate
when a 4x4x8 has a gap; it never holds a launch, and a missing one is reported as missing. Then release the slice.
Overlays (open-instruct h134-b0-overlays, checkout watch/tpu/h134/oi-h134/scripts/tpu/maxtext/k8s/h134), from mk_h134.py:
- pfd-ep8-<arm>-s<s> (4x4x8 candidate, the production mesh, lb weight 0, 20-step schedule, stopped after step 6)
- pfd-x16-<arm>-s<s> (8x8x16 reference, fsdp16 x ep64, factor 5.0, r2 emulation, lb weight 0, 6 steps; gcsfs restore patch
  plus H103's gr2 knobs: GCSFS_RETRIES 30, per-host stagger index mod 16 x 8 s, TENSORSTORE_GCS_REQUEST_CONCURRENCY 8)
  The registration's "ep16 on an 8x8x16" cannot hold 16 rows on 1,024 chips (fsdp 64 would need 64 rows); ep64 x fsdp16 is
  H123's x16 reference mesh (mk_x16.py), used here.
Driver: p4diag.sh (setsid nohup; pid in pfdiag/p4diag.pid; events tagged P4DIAG). Gate per pair: gate_p4diag2.py <arm> <seed>
(gate_p4diag.py is the B0-only original). Record the results in a dated H134 note; they gate nothing.
