# Olmo Hybrid checkpoint reference

Yasha's release checkpoints are prospective starting points for training on this
branch. The release covers **275m, 450m, 810m, 1.4b, and 2.7b**, at the
**pretrain, midtrain, and longcontext** stages, in HF and OLMo-core formats.
These stage names describe base-model training; the older `*-sft-think-*`
checkpoints used in this branch's GSM8K experiments are a separate set of inputs.

This reference combines Yasha's release handoff with a Weka inspection on
**October 21, 2026** and the pinned plugin/evaluation source below. No particular
size, stage, science arm, or checkpoint has been selected for the next run.

## Canonical GCS layouts

Primary artifacts:

```text
gs://ai2-llm/yashasbls/hybridish/<format>/v1/<size>/<stage>/step<NNNNNNN>/
```

Science ablations:

```text
gs://ai2-llm/yashasbls/hybridish-science/<format>/v1/<size>/<stage>/<arm>/step<NNNNNNN>/
```

- `<format>` is `olmo-core` or `hf`.
- `<size>` is exactly `275m`, `450m`, `810m`, `1.4b`, or `2.7b`.
- `<stage>` is `pretrain`, `midtrain`, or `longcontext`; consult the maps for
  supported combinations rather than assuming a complete Cartesian product.
- Steps have **seven digits**: step 4000 is `step0004000`.
- Science adds an arm such as `from-scratch`, `from-cx003.125`,
  `from-cx006.250`, `from-cx012.500`, `from-cx025.000`, `from-cx050.000`, or
  `from-cx075.000`. The current map also includes `from-cx001.5625` for 2.7b
  and `from-cx100.000` for some long-context trajectories.

`from-scratch` means Cx=0, with no pretraining anchor. Other arms branch from
the pretraining step recorded in `anchor_step`; that step is different across
sizes. Read `cx` and `anchor_step` from the map rather than parsing the arm name.
Its anchor reference is:

```text
gs://ai2-llm/yashasbls/hybridish/olmo-core/v1/<size>/pretrain/step<anchor_step:07d>/
```

These are internal staging locations. A release-tree name does not establish
public bucket access.

## Authoritative maps and final steps

The machine-readable location and selection specs are on Weka:

```text
/weka/oe-training-default/yashasbls/hybrid-small-suite/configs/release_map.yaml
/weka/oe-training-default/yashasbls/hybrid-small-suite/configs/science_map.yaml
```

The maps explicitly identify source runs. Their names are irregular, so do not
derive a legacy source-run path from a release destination. `release_map.yaml`
selects checkpoints on the stated cadence plus the final step and excludes
off-cadence rolling saves. Use each science arm's own selection fields as well.

Final primary checkpoint directory names from the inspected release map:

| Size | Pretrain | Midtrain | Longcontext |
| --- | --- | --- | --- |
| 275m | `step0161186` | `step0038147` | `step0047684` |
| 450m | `step0179814` | `step0038147` | `step0047684` |
| 810m | `step0269926` | `step0023842` | `step0023842` |
| 1.4b | `step0308433` | `step0011921` | `step0023842` |
| 2.7b | `step0604266` | `step0011921` | `step0023842` |

The primary map declares **1,707 native checkpoints**, consistent with the
handoff. The inspected science map declares **60 arms and 1,523 checkpoints**:
899 midtrain and 624 longcontext. This is broader than the handoff's **887 per
format** snapshot. Map declarations describe the intended selection, not proof
that every GCS object or mirror is currently present. GCS was not enumerated.

## Weka mirrors and available training inputs

Paths below are relative to `/weka/oe-training-default/yashasbls/`.
Counts are checkpoint directories observed on October 21, not weight-load tests.

| Mirror | Yasha's handoff | Observed on Weka |
| --- | --- | --- |
| `hybridish-local/hf/v1/` | 214 HF checkpoints | 214 directories; partial primary tree |
| `hybridish-local/olmo-core/v1/` | 57 directories, Cx anchors only | 57 directories; sparse primary mirror, including stage endpoints |
| `hybridish-science-local/hf/v1/` | 887, full science tree | 939 directories: 915 midtrain and 24 longcontext; not the full current map |
| `hybridish-science-local/olmo-core/v1/` | 887, full science tree | Path absent in this inspection |

The science staging root also contains `PENDING_SANITIZATION`; its marker says
the tree has not been sanitized and the `mainline_ladder` public name is still
pending. Treat the handoff counts as a historical snapshot and check the exact
directory before choosing an input. A mirror is not interchangeable with the
complete GCS tree or with the current map's declared inventory.

Examples of existing HF inputs whose metadata were read:

```text
# Primary 275m midtrain final
/weka/oe-training-default/yashasbls/hybridish-local/hf/v1/275m/midtrain/step0038147/

# Science 275m midtrain final, branched from the Cx25 anchor
/weka/oe-training-default/yashasbls/hybridish-science-local/hf/v1/275m/midtrain/from-cx025.000/step0038147/
```

For an experiment, record the tree, format, version, size, stage, arm if any,
exact step, full source URI/path, plugin commit, and tokenizer/prompt choice in
`experiment.md`. Record an intentional local copy's destination before moving
weights. No checkpoint weights were copied during this documentation update.

## Plugins and reference evaluation settings

Yasha's pinned source is
[`scaling-ladders/ladders/mainline` at `0fb485ce38733103f7da73973cdf0e74b0fc75db`](https://github.com/allenai/scaling-ladders/tree/0fb485ce38733103f7da73973cdf0e74b0fc75db/ladders/mainline).
Use both plugin packages at that same commit:

```text
transformers-plugin @ git+https://github.com/allenai/scaling-ladders.git@0fb485ce38733103f7da73973cdf0e74b0fc75db#subdirectory=ladders/mainline/transformers_plugin
vllm-plugin @ git+https://github.com/allenai/scaling-ladders.git@0fb485ce38733103f7da73973cdf0e74b0fc75db#subdirectory=ladders/mainline/vllm_plugin
```

The [transformers plugin](https://github.com/allenai/scaling-ladders/blob/0fb485ce38733103f7da73973cdf0e74b0fc75db/ladders/mainline/transformers_plugin/README.md)
registers `mainline_ladder` / `MainlineLadderForCausalLM` with the HF Auto classes.
Call `transformers_plugin.register()` before HF model loading, or
`register_config()` for config-only registration. Its package requires Python
`>=3.12,<3.14` and transformers `>=5.12` at this commit.

The [vLLM plugin](https://github.com/allenai/scaling-ladders/blob/0fb485ce38733103f7da73973cdf0e74b0fc75db/ladders/mainline/vllm_plugin/README.md)
registers automatically through `vllm.general_plugins`. It requires vLLM `>=0.18`
but does not declare a dependency on the transformers plugin; include both in
the runtime or evaluation provider dependencies.

The pinned [evaluation launcher](https://github.com/allenai/scaling-ladders/blob/0fb485ce38733103f7da73973cdf0e74b0fc75db/ladders/mainline/workloads/evals/eval.py)
uses `provider.kind=vllm`, `provider.package=wheel`, both plugin dependencies, and:

```yaml
provider.kwargs.enforce_eager: true
provider.kwargs.mamba_ssm_cache_dtype: float32
provider.kwargs.language_model_only: true
provider.kwargs.attention_backend: FLASH_ATTN
```

It also sets `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`. The cache dtype is an evaluation
serving setting, not a statement of the checkpoint weight or training dtype.

The [bundle definitions](https://github.com/allenai/scaling-ladders/blob/0fb485ce38733103f7da73973cdf0e74b0fc75db/ladders/mainline/workloads/evals/eval_list.py)
specify tasks, GPU counts, context limits, harnesses, and additional settings.
For example, `math` uses 8 GPUs; `easy` and `gen_mc` use 4; `lc` uses 4 and
`max_model_len=131072`, with RULER tasks from 4K to 128K. These are the source's
evaluation allocations, not allocations made for this work. Its launcher uses
the pinned `olmo-eval` submodule; use that source to reproduce the evaluation
protocol rather than assuming this branch's legacy launcher is equivalent.

## Compatibility with this branch

The sampled primary midtrain configs at all five sizes, and science
`from-scratch` midtrain configs at all five sizes, declare
`model_type=mainline_ladder` and `architectures=[MainlineLadderForCausalLM]`.
The sampled primary and science 275m midtrain tokenizers have no `chat_template`.
Do not carry over the older SFT checkpoints' `<think>` prompt convention without
an explicit training/prompt-format decision.

This branch still pins the older transformers/vLLM/OLMo-core forks in
`pyproject.toml`. Its hybrid weight-mapping dispatch in
`open_instruct/olmo_core_utils.py` recognizes `olmo_hybrid_small`, and native
config discovery expects a `stepN-hf` directory beside `stepN`. The release
layout instead separates `hf/v1/.../stepNNNNNNN` and
`olmo-core/v1/.../stepNNNNNNN`. **Changing only `MODEL_PATH` in the existing
GSM8K script does not establish support for these release artifacts.**

Before using them for training, align plugin/runtime versions and validate
registration, native config resolution, HF/native weight conversion, rollout
weight synchronization, tokenizer/prompt handling, and checkpoint export/reload
for the selected input. This update records the inputs and integration gap;
it does not install the plugins, change the training runtime, or launch a run.
