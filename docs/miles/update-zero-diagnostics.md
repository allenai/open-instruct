# Update-zero inference tracing

The completed 100-update comparison has only seven identical initial generations out of 128, despite identical prompt token IDs. These diagnostics use the original Core and Megatron runtimes to inspect the inference path before any optimizer update.

Each separate three-B300 Holmes job:

1. Starts the original SGLang engine with the original backend-specific settings and shared HF checkpoint.
2. Sends four frozen, identical token prefixes ending immediately before an originally differing next token. It makes uninstrumented requests, captured requests, and one repeated capture.
3. Snapshots and resets the serving weights using the original complete weight checker.
4. Initializes the actual native trainer and publishes its initial weights once. It compares the entire publication against the HF snapshot.
5. Repeats exactly the same requests and disposes the runtime. The diagnostic never calls the trainer's train method.

The requests bypass the router and chat templating and go sequentially to the same actual SGLang engine. They ask for one greedy next token and input log probabilities. Capture hooks record every prefill token's router logits, expert IDs, selected weights and top-k margins; activation rows are bounded to the first 16 and last 128 tokens. Embedding, layer inputs, attention/MLP outputs, layer outputs, final normalization, final logits, dtype inventory and source hashes are retained. All model text is treated as data.

The original prefill graph setting is disabled. Decode graphs are retained but not traced by Python hooks. These fixed-prefix prefill controls do not reproduce historical autoregressive decode batching; a matching result would motivate tracing the incremental decode path next. Traced timing is not a performance benchmark. Uninstrumented and repeated requests check observer effects and within-runtime repeatability.

The common driver delays the ordinary initial snapshot/reset until after the direct-HF probe. This is an explicit diagnostic ordering change. It preserves the complete check after the actual publication. Megatron's original source bootstrap at `0e648108c70c5d5256a9b93a88b4f8d610e44ea0` is necessary in addition to its frozen image; the launcher preserves both. Core imports and verifies the original runtime modules against retained SHA256 hashes.

Launch from a clean committed checkout:

```bash
MILES_EXISTING_IMAGE=01M26N80T0V9PREQTS87J849P8 ./scripts/train/build_image_and_launch.sh --miles scripts/train/debug/miles_update_zero.sh --backend core
MILES_EXISTING_IMAGE=01M26TT1EQTB84RV0MM650W0YE ./scripts/train/build_image_and_launch.sh --miles scripts/train/debug/miles_update_zero.sh --backend megatron
```

Jobs use urgent priority, `ai2/open-instruct-dev`, `ai2/holmes`, a one-hour minimum runtime and a 90-minute timeout. The default output is a new `update-zero-20260911-v1/{core,megatron}` directory under the original GSM8K campaign. Existing outputs are rejected. Raw tensor captures remain on WEKA; compact JSON manifests, request results, capture metadata and cleanup status are copied to Beaker results.

The frozen inputs include original source dump hashes and both original next-token IDs. The capture helper rejects nonmatching input tokens, missing layers/routes, and overwrites. Tests cover protocol ordering, zero optimizer calls, failure preservation/cleanup, import-hook behavior, full-prefix routes and bounded activation rows, exact source hashes, immutable images, original Megatron flags/bootstrap, and placement. The live scheduler capture remains an experimental gate until the short jobs complete.

The `--mode hf-matched` diagnostic is an independent inference-only control. It
uses the original image and checkpoint with a single rollout GPU, PyTorch sampling,
a 32,768-token pool, the `auto` Mamba cache strategy (radix caching remains disabled),
and explicitly fresh compiler cache directories. It performs the same fixed-prefix
controls, captures, and repeat, then exits before any trainer construction, tensor
reset, or publication. Its `hf-only-complete.json` explicitly records zero publications
and `full_protocol_complete=false`. Two separate invocations with distinct campaign
names can measure process-to-process variation in the same immutable image. Captures
include populated autotuner choices; these include prior warmup history and do not
by themselves prove which configuration every kernel invocation used.

`--mode trainer-routes` extends the original three-GPU protocol after its full
initial-weight comparison. Both native scorers process the same four prefixes and
the16 frozen responses from original Core rollout0; the retained artifact hash is
checked before a restricted `weights_only=True` load. SGLang also processes those
same16 sequences. These are rank-strided diagnostic cohorts, not a reconstruction
of historical rank assignment. Route positions refer to processed input tokens;
log-probability checks explicitly align target token t with vocabulary logits t−1.
The diagnostic performs no backward or optimizer call and preserves the original
Ray worker setup hook before adding its actor method.

`--mode hf-matched --pin-autotune-a` tests the seven observed tuner configurations
from the first matched control. Each is required to be a declared candidate in the
actual runtime before any tuner is changed. Local tests qualify that guard; B300
candidate membership is checked in the GPU diagnostic before warmup. The observer
records actual wrapped Python tuner calls during armed eager prefill, including
unpinned calls. This does not claim to trace all GPU kernels or CUDA graph replay.

### Native route comparison coordinates

`compare_trainer_routes.py` compares full response log probabilities with explicit target-token alignment and compares routes at identical captured input positions. Megatron captures use physical MLP layer indices: odd layer `2*i+1` maps to HF logical block `i`. Core indices already use logical blocks. The comparator rejects even physical MLP indices and missing routed layers, and sorts expert IDs with their weights before weight comparisons. A measured mismatch is reported separately from failed capture validity. `miles_trainer_route_compare.sh` runs both comparisons on CPU/Saturn after both capture completion markers exist.
