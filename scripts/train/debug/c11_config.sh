# C11 values fixed by the registration (ledger H111 P3/P6/P7). TODO values block every training launch.
C11_DATA=/weka/oe-training-default/ai2-llm/checkpoints/abhishekr/c11/data
C11_DOLCI=$C11_DATA/dolci/dolci-think-clean-v1-en-dbv-5bd68837.parquet   # 2,052,645 rows; P2 English rule + P4 Amendment 1
C11_NEMO=$C11_DATA/nemo/nemotron-tool-oi-en-bd-825bdb83.parquet           # 687,118 rows; English rule, rule B, P4
C11_SCI=$C11_DATA/sci/nemotron-sci-v2-h083sel-bd-16f9aaaa.parquet         # 108,758 rows; rule B, P4
C11_WN=0.870525  # N_C10 / (phi11 x T_N); T_N = 3,482,394,648 (weight-1.0 tokenize), phi11 = 0.652611
C11_WS=3.064611  # 2 / phi11; phi11 = D_C10 / T_D, T_D = 18,904,838,154
C11_CACHE=f4702f53ad-6068a350-rowaligned
C11_STEPS=16304  # S = round(phi11 x I_C11 / 16); I_C11 = 399,714 packed instances (OLMo-core packing, DOCB)
C11_REPLAY_STEPS=18416  # S' = round(S x (I_C + I_R) / I_C); I_C + I_R = 451,489 (51,775 replay docs)
C11_REPLAY_DIR=$C11_DATA/replay-v2  # 51,775 docs, 2,372,036,476 loss tokens = 0.10 x the cache loss tokens
