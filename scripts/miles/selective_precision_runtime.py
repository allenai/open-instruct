"""Process-local selective precision and fixed-token sampling for diagnostics."""

import collections
import functools
import json
import os
from pathlib import Path

import torch
from fla.ops.kda import chunk_intra
from olmo_core.nn.attention import kda as core_kda
from olmo_sglang.kda import backend
from olmo_sglang.models import olmo3_moe
from scripts.miles import selective_kda_precision
from sglang.srt.layers import sampler
from sglang.srt.model_executor import model_runner
from sglang.srt.sampling import sampling_batch_info
from triton import language as tl


def fp32_output(module, serving, value):
    flat = value.reshape(-1, value.shape[-1])
    result = torch.mm(flat, module.weight.T, out_dtype=torch.float32)
    if module.bias is not None:
        result = result + module.bias.float()
    result = result.reshape(*value.shape[:-1], module.weight.shape[0])
    return (result, None) if serving else result


def set_linear_variant(model, variant, *, serving):
    selected = set(variant.split("+")) - {"none"}
    if selected - {"gate", "beta"}:
        raise ValueError(variant)
    count = collections.Counter()
    for module in model.modules():
        if not hasattr(module, "f_proj_2"):
            continue
        for name, attribute in (("gate", "f_proj_2"), ("beta", "beta_proj" if serving else "w_b")):
            linear = getattr(module, attribute)
            if not hasattr(linear, "_diagnostic_original_forward"):
                linear._diagnostic_original_forward = linear.forward
            linear.forward = (
                functools.partial(fp32_output, linear, serving)
                if name in selected
                else linear._diagnostic_original_forward
            )
            if name in selected:
                count[name] += 1
    if selected and set(count) != selected:
        raise ValueError(f"No matching projections: {variant}, {count}")
    return dict(count)


def forced_next_tokens(params, positions):
    result = []
    for config, position in zip(params, positions, strict=True):
        index = position - config["prefix_length"] + 1
        if not 0 <= index < len(config["forced_ids"]):
            raise ValueError(f"Forced-token position outside continuation: {position}, {index}")
        result.append(config["forced_ids"][index])
    return result


def strict_arithmetic():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if hasattr(chunk_intra, "SOLVE_TRIL_DOT_PRECISION"):
        chunk_intra.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


def install_chunk_fp32():
    """Widen only KDA kernel inputs; preserve projection and output dtypes."""
    strict_arithmetic()
    original_core = core_kda.dispatch_chunk_kda

    def core_chunk(**kwargs):
        dtype = kwargs["q"].dtype
        kwargs = {k: v.float() if k in {"q", "k", "v", "g"} else v for k, v in kwargs.items()}
        output, state = original_core(**kwargs)
        return output.to(dtype), state

    core_kda.dispatch_chunk_kda = core_chunk
    original_run = backend.OlmoFLAKDAKernel._run

    def run(self, q, k, v, raw_gate, raw_beta, **kwargs):
        result = original_run(self, q.float(), k.float(), v.float(), raw_gate.float(), raw_beta, **kwargs)
        if isinstance(result, tuple):
            return (result[0].to(q.dtype), *result[1:])
        return result.to(q.dtype)

    backend.OlmoFLAKDAKernel._run = run


def install():
    selective_kda_precision.install()
    if os.environ.get("OI_STRICT_ARITHMETIC") == "1":
        strict_arithmetic()
    if os.environ.get("OI_CHUNK_FP32") == "1":
        install_chunk_fp32()
    linear_variant = os.environ.get("OI_LINEAR_ABLATION", "none")
    original_init = olmo3_moe.Olmo3MoeForCausalLM.__init__

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        print("LINEAR_ABLATION", linear_variant, set_linear_variant(self, linear_variant, serving=True), flush=True)

    olmo3_moe.Olmo3MoeForCausalLM.__init__ = init
    if os.environ.get("OI_PREFILL_CORE") == "1":
        original_kernel_init = backend.OlmoFLAKDAKernel.__init__

        def kernel_init(self, *args, **kwargs):
            original_kernel_init(self, *args, **kwargs)
            self.core_compat = True

        backend.OlmoFLAKDAKernel.__init__ = kernel_init
    original_batch = sampling_batch_info.SamplingBatchInfo.from_schedule_batch.__func__

    @classmethod
    def batch(cls, scheduled, vocab_size):
        result = original_batch(cls, scheduled, vocab_size)
        params = [r.sampling_params.custom_params for r in scheduled.reqs]
        if any(p and "forced_ids" in p for p in params):
            if not all(p and "forced_ids" in p for p in params):
                raise ValueError("Mixed forced and ordinary requests")
            result.custom_params = params
        return result

    sampling_batch_info.SamplingBatchInfo.from_schedule_batch = batch
    original_sample = sampler.Sampler._sample_from_probs

    def sample(self, probs, sampling_info, positions, simple_sampling_case):
        params = sampling_info.custom_params
        if params and all(p and "forced_ids" in p for p in params):
            if not simple_sampling_case:
                raise ValueError("Forced diagnostic requires untruncated probabilities")
            ids = forced_next_tokens(params, positions.cpu().tolist())
            return torch.tensor(ids, device=probs.device, dtype=torch.int64)
        return original_sample(self, probs, sampling_info, positions, simple_sampling_case)

    sampler.Sampler._sample_from_probs = sample
    original_forward = model_runner.ModelRunner.forward
    counts = collections.Counter()
    trace_path = os.environ.get("OI_GRAPH_AUDIT")

    def forward(self, forward_batch, *args, **kwargs):
        result = original_forward(self, forward_batch, *args, **kwargs)
        if trace_path:
            key = f"{'decode' if forward_batch.forward_mode.is_decode() else 'prefill'}/graph={result.can_run_graph}"
            counts[key] += 1
            if counts[key] in (1, 2) or sum(counts.values()) % 256 == 0:
                Path(trace_path).write_text(json.dumps(dict(counts)))
        return result

    model_runner.ModelRunner.forward = forward
