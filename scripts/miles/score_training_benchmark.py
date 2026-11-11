"""Fixed-length real rollouts and balanced synthetic rewards for trainer timing.

These hooks exercise actual model generation and optimizer updates, not reward
learning. Select via custom_generate_function_path and custom_rm_path in a
MILES RunSpec. Seeds and lengths are identical across benchmark configurations.
"""

from miles.rollout import base_types, sglang_rollout


async def generate(call):
    sampling = dict(call.sampling_params, ignore_eos=True, sampling_seed=call.args.rollout_seed + call.sample.index)
    sample = await sglang_rollout.generate(call.args, call.sample, sampling, evaluation=call.evaluation)
    return base_types.GenerateFnOutput(samples=sample)


async def reward(args, sample, **kwargs):
    """Alternating rewards keep each two-response group active during timing."""
    return float(sample.index % 2)
