The training image for Olmo 3.5 multimodal SFT (`../k8s/olmo35-mm-sft-4x4x8`): MaxText from Jason's fork
(jason718/maxtext), built as MaxText builds its own TPU image, plus a layer (`Dockerfile`) with CPU torch, torchvision
and OLMo-core allenai/OLMo-core@fd9bba1ba3f12721ae1ff3493f0b266366debec6 (#942, the GPU recipe `olmo35_mm_sft`), which
`dataset_type=olmo_mm` runs in-process as the data loader. The recipe itself (`configs/olmo/olmo35_tiny_mm_sft_64k.yml`
and its OLMo-core data config) ships inside MaxText at that commit.

`maxtext_source` pins the images, one per line (`label repo maxtext-commit olmo-core-commit`):

| label | MaxText commit (branch) | what | tag |
|---|---|---|---|
| `run` | 217c8458 (`jasonr-olmo35-mmsft-perf`) | the reference run (ledger H271): `jasonr-olmo35-mmsft` 2b8446d3 (olmo35-sft c6c9fd1a5 + cross-CP EMo + `OLMO_MM_LOADER_WORLD` + the mm-SFT config) + balanced ViT crops + loader prefetch | `217c84582d2f985c9b2e9d3a7067e8423035c19e-olmocore-fd9bba1ba` (built) |
| `e4` | ed83783e (`jasonr-olmo35-mmsft-perf2`) | + the E4 performance flags (connector gather once, remat policy, K/V and LM-head gathers); 9.35 vs 10.73 s/step | `ed83783e2aea8715b07e397884660a26d6551825-olmocore-fd9bba1ba` (built) |
| `gates` | e3543ef2 (`jasonr-olmo35-mmsft-gates`) | perf2 + `olmo_mm_batch_hash_steps` (OLMo-core-comparable batch hashes) for the gate overlay | `e3543ef27552d042a54c74ea89cf4d6b06ca4520-olmocore-fd9bba1ba` (not built yet; the branch must be pushed first) |
| `small` | e162d68a (`jasonr-small-mm-sft`) | gates + text/image-row CE in LOSS-PARTS + the Small (A4B) multimodal SFT config `olmo35_small_mm_sft_64k.yml`, the prepacked-text + scheduled-image-row loader (`OLMO_MM_TEXT_NPZ_DIR`), `olmo_mm_keep_sources`, mid-training's producer loader; OLMo-core 7074cf40 (#942 head + #964, same loader) for `../k8s/olmo35-small-mm-sft-*` | `e162d68a82ff1fd29763283f8fba8ef03ba148b5-olmocore-7074cf408` (not built yet; branch pushed) |

    ./build.sh          # = ./build.sh run; prints IMAGE=... and exits at once when the tag exists already
    ./build.sh gates

The build is the one in allenai/olmo-gcp-infra@3575547e examples/olmo35-maxtext/scripts/vision-align-image
(`cloudbuild.yaml`, run with `MAXTEXT_REPO=https://github.com/jason718/maxtext.git MAXTEXT_REF=<commit>
OLMO_CORE_REF=fd9bba1ba...`), which pushed the published tags. A rebuild resolves the unpinned pip packages of the
layer (e.g. `datasets`) again, so it may not be byte-identical to a published image; existing tags are never rebuilt.
