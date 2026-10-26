# H134 P4 x16 reference: optional diagnostic (not in prod.sh's launch path)
Ledger-manager 2026-10-26: P4's x16 reference is a diagnostic. Run it only when a borrowed 8x8x16 is free, under
restore.lock (the x16 restore is a 256-host big restore: restore_lock_apply ... big, 30 min spacing, gr2 knobs apply).
Overlays (open-instruct h134-b0-overlays, checkout watch/tpu/h134/oi-h134/scripts/tpu/maxtext/k8s/h134):
- pfd-ep8-b0-s0 / -s1 (4x4x8 candidate, lb weight 0, 20 steps)
- pfd-x16-b0-s0 / -s1 (8x8x16 reference, lb weight 0, 6 steps, gcsfs restore patch from gcsfs_patch.py)
  NOTE: pfd-x16 carries H123's gcsfs retry patch (GCSFS_RETRIES 20), not gr2's knobs (GCSFS_RETRIES 30, stagger,
  TENSORSTORE_GCS_REQUEST_CONCURRENCY 8), which ledger-manager made default for 8x8x16 restores on 10-26. Add them
  (h103-gr2-overlay @bb1085a17 shows the diff) and re-render before using pfd-x16.
Save each run's pod logs into pfdiag/logs/<jobset>/ (all pods; LOSS-PARTS lines), then: python3 gate_p4diag.py <seed>.
Record the result in a dated H134 note; it does not gate b0's launch.
