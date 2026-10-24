"""Compile the actual patched Tunix train step from abstract state, without training."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import optax
import routing_log_fix
import update_audit_patch
from flax import nnx
from flax.core import spmd
from maxtext.configs import pyconfig
from maxtext.optimizers import optimizers
from maxtext.trainers.pre_train import train, train_compile
from maxtext.utils import max_utils, maxtext_utils, model_creation_utils, sharding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", choices=("candidate", "reference-cp8"))
    parser.add_argument("--buffer-factor", type=float, default=2.5)
    parser.add_argument("--runtime-flags", action="store_true")
    parser.add_argument("--synchronous-collectives", action="store_true")
    options = parser.parse_args()
    args = json.loads(Path(f"/work/fp32-output512-{options.mesh}-aot-args.json").read_text())
    args = [a for a in args if not a.startswith("ragged_buffer_factor=")]
    args.append(f"ragged_buffer_factor={options.buffer_factor}")
    config = pyconfig.initialize(args)
    mesh = train_compile.get_topology_mesh(config)
    original = Path("/deps/src/maxtext/trainers/post_train/sft/train_sft.py").read_bytes()
    patched = update_audit_patch.patch(routing_log_fix.patch_source(original))
    namespace = {"__name__": "h127_aot_tunix"}
    exec(compile(patched, "<pinned-patched-train_sft>", "exec"), namespace)
    create_model, _ = model_creation_utils.create_nnx_abstract_model(config, mesh)

    # As in create_nnx_abstract_model, resolve logical metadata after eval_shape;
    # entering a physical mesh during eval_shape misreads logical names as axes.
    with spmd.logical_axis_rules(config.logical_axis_rules):

        def init():
            model = create_model()
            schedule = maxtext_utils.create_learning_rate_schedule(config)
            tx = optimizers.get_optimizer(config, schedule, model)
            tx = optax.chain(optax.clip_by_global_norm(config.gradient_clipping_threshold), tx)
            return model, nnx.Optimizer(model, tx, wrt=nnx.Param)

        model, optimizer = nnx.eval_shape(init)
        states = nnx.state((model, optimizer))
        named = sharding.nnx_construct_named_sharding(states, mesh)
        shaped = jax.tree_util.tree_map(lambda v, s: jax.ShapeDtypeStruct(v.shape, v.dtype, sharding=s), states, named)
        nnx.update((model, optimizer), shaped)

        def loss(model, **data):
            return train.loss_fn(model, config, data, dropout_rng=None, params=None, is_train=True)

        dummy = SimpleNamespace(
            model=model,
            loss_fn=loss,
            _has_aux=True,
            gen_model_input_fn=lambda x: x,
            _lora_enabled=False,
            config=SimpleNamespace(get_with_default=lambda key, default: 1, max_seq_token_per_tpu=None),
        )
        fn = namespace["MaxTextPeftTrainer"].create_train_step_fn(dummy)
        jitted = nnx.jit(fn, donate_argnames=("optimizer", "grad_accumulator"))
        batch_sharding = sharding.get_input_data_sharding(config, mesh)
        batch = maxtext_utils.get_shaped_batch(config, batch_sharding=batch_sharding)
        flag_label = "-runtime-flags" if options.runtime_flags else ""
        if options.synchronous_collectives:
            assert options.runtime_flags, "Synchronous override requires runtime flags"
            flag_label += "-sync"
        prefix = f"/work/tunix-{options.mesh}-rbf{options.buffer_factor:g}{flag_label}"
        compiler_options = max_utils.parse_libtpu_flags_to_dict(
            f"--xla_enable_async_all_gather={str(not options.synchronous_collectives).lower()} "
            f"--xla_tpu_enable_async_ragged_all_to_all={str(not options.synchronous_collectives).lower()} "
            "--xla_tpu_scoped_vmem_limit_kib=16384"
            if options.runtime_flags
            else ""
        )
        Path(prefix + "-args.json").write_text(json.dumps(args, indent=2) + "\n")
        print("TUNIX-AOT-LOWERING", options.mesh, options.buffer_factor, flush=True)
        try:
            with jax.set_mesh(mesh):
                compiled = jitted.lower(model, optimizer, inputs=batch).compile(compiler_options=compiler_options)
            memory = compiled.memory_analysis()
            fields = ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")
            result = {k: int(getattr(memory, k)) for k in fields}
            result["total_bytes"] = (
                result["argument_size_in_bytes"]
                + result["output_size_in_bytes"]
                - result["alias_size_in_bytes"]
                + result["temp_size_in_bytes"]
            )
            result["scope"] = (
                "Abstract-state compilation of patched Tunix step including update audit; no runtime or numerical qualification"
            )
            result["compiler_options"] = compiler_options
            Path(prefix + "-memory.json").write_text(json.dumps(result, indent=2) + "\n")
            print("TUNIX-AOT-MEMORY", json.dumps(result), flush=True)
        except Exception as error:
            Path(prefix + "-error.json").write_text(
                json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2) + "\n"
            )
            raise


if __name__ == "__main__":
    main()
