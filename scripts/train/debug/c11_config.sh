# C11 values fixed by the registration (ledger H111 P3/P6/P7). TODO values block every training launch.
C11_DATA=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/data
C11_DOLCI=TODO   # after rule B/decontam: $C11_DATA/dolci/dolci-think-clean-v1-en-<sha8>.parquet
C11_NEMO=TODO    # $C11_DATA/nemo/nemotron-tool-oi-en-<sha8>.parquet
C11_SCI=TODO     # $C11_DATA/sci/nemotron-sci-v2-h083sel-<sha8>.parquet
C11_WN=TODO      # w_n (float), from per-source token counts
C11_WS=TODO      # w_s = 2 / phi11 (float)
C11_CACHE=TODO   # numpy cache dir name, ends in -rowaligned
C11_STEPS=TODO   # S = round(phi11 x I_C11 / 16)
C11_REPLAY_STEPS=TODO  # S' = round(S x (I_C + I_R) / I_C)
C11_REPLAY_DIR=TODO    # $C11_DATA/replay-v2
