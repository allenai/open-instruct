
## Memory at 64K: the pins change the compiled program (attempt 3, `-v24t`, identical settings in both images)

| image | `jit_train_step` HLO temporaries at 16 x 65536, fp32 | result |
|---|---|---|
| old (`sha256:7d7f4938…`) | not logged: no HBM figure after compile or step 1. Fits, so at most 31.75 GiB | compiled and completed step 1 (`h083-gc32-old-s64k-v24t`, 10:57 PT) |
| bs3ec936c3 (`sha256:c858123f…`) | **32.13G**, more than the available 31.75G | `RESOURCE_EXHAUSTED: Ran out of memory on HBM` at compile (`h083-gc32-bs-s64k-v24t`, 11:03 PT) |

The config, shape, placement, flags and tiles are identical, and only the image differs. So the bs layer's sharding pins
change the compiled program, at least in its HBM temporaries. Both images then step down to 32K, per the HBM rule.
