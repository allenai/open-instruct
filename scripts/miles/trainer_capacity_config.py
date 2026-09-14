"""CPU-safe, matched variants for a retained-batch trainer throughput screen."""

import dataclasses

from scripts.miles import throughput_basket

from open_instruct.miles.config import RunConfig

SOURCE = (
    "/weka/oe-training-default/robertb/open-instruct/throughput-profiles/steady-2t2i-c32-b128-graphs-986935b17569/run"
)
VARIANTS = {
    "baseline": {},
    "lean": {"scoring_pass_required": False, "replay_diagnostics": False},
    "no-recompute": {"activation_checkpointing": False},
    "optimizer-compile": {"compile_optimizer": True},
    "model-compile": {"compile_model": True},
    "reduce-scatter": {"use_reduce_scatter": True},
    "vector-grad-add": {"activation_checkpointing": False},
    "pairwise-swiglu": {"activation_checkpointing": False},
}


KERNEL_FLAGS = ("OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE", "OLMO_PROFILE_SWIGLU_PAIRWISE")


def environment(variant):
    enabled = {"vector-grad-add": KERNEL_FLAGS[0], "pairwise-swiglu": KERNEL_FLAGS[1]}.get(variant)
    return {key: "1" if key == enabled else "0" for key in KERNEL_FLAGS}


def configuration(variant, output):
    spec = throughput_basket.specification("packed-2t2i-c64-p512-b128", output)
    run = spec.compile()
    core = dataclasses.replace(
        run.core,
        row_specialization="dynamic",
        **({} if variant == "baseline" else VARIANTS["lean"] | VARIANTS[variant]),
    )
    miles = dict(run.miles)
    # Same 16 immutable collected batches and initial weights, with no inference
    # or publication transport. Recorded policy versions remain unchanged.
    miles.update(num_rollout=16, use_wandb=False, hf_checkpoint=str(throughput_basket.CAMPAIGN / "hf"))
    return RunConfig(core, miles)
