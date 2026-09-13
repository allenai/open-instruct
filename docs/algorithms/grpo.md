# Group Relative Policy Optimization (GRPO)

Use **`python -m open_instruct.miles`** for new GRPO work. MILES coordinates
rollouts through SGLang and trains through the OLMo-core adapter.

## Start a new run

1. Read the [MILES guide](../miles/index.md) and check
   [model/checkpoint support](../miles/models-and-checkpoints.md) and
   [topology limits](../miles/topology.md).
2. Copy a [structured TOML example](../miles/configuration-intro.md), then
   edit checkpoint, data, output and allocation settings.
3. Follow the [launch guide](../miles/launching.md) for the pinned runtime and
   committed-image workflow. Use the [configuration reference](../miles/configuration.md)
   for options and [operations guide](../miles/operations.md) for completion checks.

Planning and structured validation can run on a CPU host:

```bash
python -m open_instruct.miles plan /path/to/run.toml
python -m open_instruct.miles validate /path/to/run.toml
```

Validation does not establish model support, GPU memory fit or runtime qualification.
If MILES lacks a required capability, identify the gap before selecting a backend.

<a id="implemented-variants"></a>

## Deprecated entry points

`open_instruct/grpo.py` (OLMo-core/vLLM) and `open_instruct/grpo_fast.py`
(DeepSpeed/vLLM) are deprecated. They remain runnable for existing experiments and
historical reproduction and emit a startup notice. They are not starting points
for new GRPO recipes or features. The Core implementation in `grpo.py` is separate
from the MILES Core adapter.

See the [deprecated vLLM reference](legacy_grpo.md) for their CLI flags, debug
scripts and historical results. Old flags, dependencies and checkpoints are not
automatically interchangeable with MILES. Entry points do not silently redirect
or translate configurations; migration requires a reviewed MILES recipe.
