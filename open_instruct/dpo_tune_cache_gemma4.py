"""Gemma-4 entry point for the HF/DeepSpeed DPO trainer (``dpo_tune_cache.py``): SafeComp post-ARR Phase 4, Gate 3.

Runs ``dpo_tune_cache.main`` unchanged, with three in-process additions from ``open_instruct/gemma4_dpo.py``:

* the ``preference_gemma4_tail_tokenize_v1`` transform is registered (exact vLLM-render tail labeling, segments packed
  per side, segment ids in ``*_attention_mask``);
* ``Gemma4ForCausalLM.forward`` gains a ``dpo_labels=`` path returning chunked, softcapped per-token log-probs (no
  full 32k x 262k logit tensor);
* ``forward_fn`` is ``gemma4_separate_forward`` (chosen then rejected; padding stripped; block-diagonal masks);
* the checkpoint's persistent buffers (Gemma-4 ``layer_scalar``) are copied in after loading: transformers 5.5.3
  skips buffers of parameter-less modules when loading under ZeRO-3 (``gemma4_dpo.load_persistent_buffers``);
* bookkeeping only: logged metrics are also printed (``GEMMA4_METRICS``), and with ``GEMMA4_EXPORT_PROBE`` set the
  trained model's log-probs on fixed sides are saved before the final HF export (export-vs-trained check).

Opt-in additions (2026-11-08, Gemma DPO stage 0; every default reproduces the stage-1 behaviour above):

* ``--chosen_nll_coef A`` (the OLMo-core trainer's flag and definition): loss = DPO + A * NLL(chosen), NLL = per-pair
  length-normalized negative log-likelihood of the chosen labeled tokens (``dpo_utils.chosen_nll_loss``), averaged
  over the micro-batch, as ``dpo_utils.add_chosen_nll`` does on the OLMo-core path. Implemented here by wrapping
  ``dpo_utils.compute_loss`` in-process; the stock ``dpo_tune_cache.main`` (which refuses a non-zero value, having no
  NLL code) receives 0.0. With A = 0 nothing is wrapped. The reference-cache hash never included the coefficient.
* ``--gemma4_length_grouped_batches``: ``gemma4_dpo.LengthGroupedMicrostepSampler`` replaces the training
  ``RandomSampler`` (pairs of similar cost run concurrently on the ranks; ``--gemma4_group_steps``, ``--gemma4_cost_l0``).
* ``--gemma4_hybrid_flash_attention``: sliding layers run FlashAttention-2 varlen, global layers SDPA
  (``gemma4_dpo.gemma4_hybrid_attention``); the reference cache then lives under
  ``$REFERENCE_LOGPROBS_CACHE_PATH/gemma4_hybrid`` so it never mixes with an SDPA-computed cache.
* ``--transform_fn preference_gemma4_tail_tokenize_v2 ...``: labeler policy v2 (``gemma4_dpo``, part 4).

Opt-in addition (2026-11-12, Gemma SFT-on-chosen control; the default reproduces every earlier run):

* ``--dpo_loss_coef C`` (the OLMo-core trainer's flag and definition, ``dpo_utils.apply_dpo_loss_coef``): loss =
  C * DPO + A * NLL(chosen). Applied inside the chosen-NLL wrapper, so it needs ``--chosen_nll_coef A`` != 0.
  ``--dpo_loss_coef 0 --chosen_nll_coef 1`` = SFT on the chosen side only (the Qwen ``RS-sftc`` control): the rejected
  forward, the reference cache and the reward / accuracy metrics still run and are logged, but the DPO term contributes
  zero gradient. With C = 1.0 (default) ``apply_dpo_loss_coef`` returns the same tensor, i.e. the loss is computed
  exactly as before. ``dpo_tune_cache.main`` (which refuses C != 1) receives 1.0. The reference-cache hash never
  included C.

Needs transformers >= 5.5 on the path BEFORE the venv's 5.4.0 (``agent-safety/scripts/cross_family/
gemma4_train_env.sh`` builds the overlay). Nothing here is imported by ``dpo.py`` / ``dpo_tune_cache.py``, so the
Qwen3.5 paths are unchanged. Launch with ``accelerate launch --use_deepspeed ... open_instruct/dpo_tune_cache_gemma4.py``
and the usual ``dpo_tune_cache.py`` flags, plus:
    --transform_fn preference_gemma4_tail_tokenize_v1 preference_tulu_filter_v1
    --concatenated_forward false --packing false --attn_implementation torch (= HF "sdpa")
    --activation_memory_budget 0.5 (any value < 1 turns on HF gradient checkpointing, non-reentrant in 5.5.3)
"""

import json
import os
from dataclasses import dataclass

import accelerate
import deepspeed
import torch
import torch.distributed as dist
import transformers

from open_instruct import dpo_tune_cache, dpo_utils, gemma4_dpo, logger_utils, model_utils
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.utils import ArgumentParserPlus

logger = logger_utils.setup_logger(__name__)
_ORIG_ACCEL_LOG = accelerate.Accelerator.log
_ORIG_SAVE = model_utils.save_with_accelerate
_ORIG_AUTO_MODEL = dpo_tune_cache.AutoModelForCausalLM
_ORIG_COMPUTE_LOSS = dpo_utils.compute_loss
_ORIG_RANDOM_SAMPLER = dpo_tune_cache.RandomSampler
# Run options of this process, set by main() before dpo_tune_cache.main runs.
_RUN = {"chosen_nll_coef": 0.0, "dpo_loss_coef": 1.0, "average_log_prob": True, "hybrid_attention": False,
        "logged_engine": False}


class _AutoModelWithBufferFix:
    """Stands in for ``AutoModelForCausalLM`` inside dpo_tune_cache (this process only): loads as usual, then copies
    the checkpoint's persistent buffers (Gemma-4 ``layer_scalar``), which transformers 5.5.3 skips under ZeRO-3 init
    (see ``gemma4_dpo.load_persistent_buffers``), and refuses to continue if any layer_scalar is still 1.0."""

    @staticmethod
    def from_pretrained(path, *args, **kwargs):
        model = _ORIG_AUTO_MODEL.from_pretrained(path, *args, **kwargs)
        if _RUN["hybrid_attention"]:
            gemma4_dpo.register_hybrid_attention()
            model.set_attn_implementation(gemma4_dpo.HYBRID_ATTN)
            logger.info("attention implementation: %s", model.config._attn_implementation)
        report = gemma4_dpo.load_persistent_buffers(model, path)
        report["rope"] = gemma4_dpo.recompute_rope_buffers(model)
        scal = [m.layer_scalar.float().item() for m in model.modules() if hasattr(m, "layer_scalar")]
        logger.info("persistent buffers reloaded from %s: %s; layer_scalar min %.4f max %.4f", path, report,
                    min(scal), max(scal))
        if not scal or max(scal) >= 0.999:
            raise RuntimeError(f"Gemma-4 layer_scalar buffers look unloaded (max {max(scal) if scal else None})")
        return model

    from_config = staticmethod(_ORIG_AUTO_MODEL.from_config)


def _log_and_print(self, values, step=None, log_kwargs=None):
    """Also print every logged metrics dict (rewards / margins / logps / perf) to the job log, main process only.

    With a chosen-NLL anchor the printed line adds ``chosen_nll`` = -logps/chosen (exact for average-log-prob losses
    such as dpo_norm: the per-pair NLL is minus the per-pair mean labeled log-prob, and both are averaged the same way)
    and ``chosen_nll_coef``, plus ``grad_norm`` (the DeepSpeed engine's global gradient norm of the last step). On the
    first call it also prints the engine's effective gradient clipping (accelerate fills a config value of "auto" with
    1.0: accelerate 1.12.0 accelerator.py:2171, :2300)."""
    if self.is_main_process:
        extra = {}
        if _RUN["chosen_nll_coef"] != 0.0:
            extra["chosen_nll_coef"] = _RUN["chosen_nll_coef"]
            if _RUN["average_log_prob"] and "logps/chosen" in values:
                extra["chosen_nll"] = -values["logps/chosen"]
        if _RUN["dpo_loss_coef"] != 1.0:
            extra["dpo_loss_coef"] = _RUN["dpo_loss_coef"]
        if not _RUN["logged_engine"]:
            _RUN["logged_engine"] = True
            ds_cfg = getattr(self, "deepspeed_config", None) or {}
            print("GEMMA4_ENGINE " + json.dumps({"gradient_clipping": ds_cfg.get("gradient_clipping"),
                                                  "gradient_accumulation_steps": ds_cfg.get("gradient_accumulation_steps"),
                                                  "train_micro_batch_size_per_gpu": ds_cfg.get("train_micro_batch_size_per_gpu"),
                                                  "attn_hybrid": _RUN["hybrid_attention"]}, default=str), flush=True)
        eng = getattr(self, "deepspeed_engine_wrapped", None)
        try:  # pre-clip global grad norm of the last optimizer step (ZeRO computes it every step); never fail a log
            gn = eng.get_global_grad_norm() if eng is not None else None
            extra["grad_norm"] = float(gn) if gn is not None else None
        except Exception:  # noqa: BLE001
            pass
        print("GEMMA4_METRICS " + json.dumps({"step": step, **{k: v for k, v in values.items()}, **extra}), flush=True)
    return _ORIG_ACCEL_LOG(self, values, step=step, log_kwargs=log_kwargs or {})


def make_compute_loss_with_chosen_nll(coef: float, dpo_loss_coef: float = 1.0):
    """``dpo_utils.compute_loss`` scaled by ``dpo_loss_coef``, plus coef * chosen NLL per pair (RPO; the OLMo-core
    definition).

    compute_loss returns PER-PAIR losses and the trainer takes their mean, so dpo_loss_coef * DPO_i + coef * NLL_i per
    pair gives dpo_loss_coef * mean(DPO) + coef * mean(NLL) =
    ``dpo_utils.add_chosen_nll(dpo_utils.apply_dpo_loss_coef(mean(DPO), dpo_loss_coef), ...)``, the OLMo-core
    micro-batch loss (``olmo_core_train_modules.DPOTrainModule._compute_microbatch_loss``). Unit-tested against it.
    With dpo_loss_coef 1.0 (default) ``apply_dpo_loss_coef`` returns the same tensor (the earlier loss, unchanged)."""
    if coef == 0.0:
        raise ValueError("coef 0 means no wrapper (the stock compute_loss is used unchanged)")
    if dpo_loss_coef < 0.0:
        raise ValueError(f"dpo_loss_coef must be >= 0, got {dpo_loss_coef}")

    def compute_loss(args, batch, policy_chosen_logps, policy_rejected_logps, reference_cache=None):
        losses, chosen_rewards, rejected_rewards = _ORIG_COMPUTE_LOSS(
            args, batch, policy_chosen_logps, policy_rejected_logps, reference_cache
        )
        losses = dpo_utils.apply_dpo_loss_coef(losses, dpo_loss_coef)
        nll = dpo_utils.chosen_nll_loss(policy_chosen_logps, batch.get("chosen_labels"), args.loss_type.is_average_loss)
        return losses + coef * nll, chosen_rewards, rejected_rewards

    compute_loss._gemma4_chosen_nll_coef = coef
    compute_loss._gemma4_dpo_loss_coef = dpo_loss_coef
    return compute_loss


def make_length_grouped_sampler_factory(per_device_batch: int, group_steps: int, seed: int, l0: float):
    """Stands in for ``RandomSampler`` inside dpo_tune_cache (this process only; the training dataloader is its only
    use): a ``LengthGroupedMicrostepSampler`` over the tokenized pairs' packed side lengths."""

    def factory(data_source, generator=None, **kwargs):
        del generator, kwargs  # the order is derived from (seed, epoch), identically on every rank
        lc = [len(x) for x in data_source["chosen_input_ids"]]
        lr = [len(x) for x in data_source["rejected_input_ids"]]
        costs = [gemma4_dpo.pair_cost(a, b, l0) for a, b in zip(lc, lr)]
        world = dist.get_world_size() if dist.is_available() and dist.is_initialized() else int(os.environ.get("WORLD_SIZE", "1"))
        logger.info("length-grouped sampler: %d pairs, %d ranks, %d per device, group_steps %d, L0 %.0f",
                    len(costs), world, per_device_batch, group_steps, l0)
        return gemma4_dpo.LengthGroupedMicrostepSampler(costs, world, per_device_batch, group_steps, seed)

    return factory


def _save_with_export_probe(accelerator, model, tokenizer, output_dir, *args, **kwargs):
    """Before the final save, record the TRAINED model's per-token log-probs on fixed sides (GEMMA4_EXPORT_PROBE =
    parity_inputs.json) so the exported checkpoint can be checked against the in-memory one. All ranks run the
    forward (ZeRO-3 gathers parameters collectively); the main process writes <output_dir>/export_probe.pt."""
    probe = os.environ.get("GEMMA4_EXPORT_PROBE")
    if probe:
        sides = json.load(open(probe))
        cfg = gemma4_dpo._unwrap_config(model)
        model.eval()
        out = {}
        with torch.no_grad():
            for s in sides:
                dev = accelerator.device
                ids = torch.tensor(s["input_ids"], device=dev)[None]
                seg = torch.tensor(s["segment_ids"], device=dev)
                masks = gemma4_dpo.gemma4_masks(seg, cfg.sliding_window, cfg._attn_implementation, cfg.dtype)
                lp = model(input_ids=ids, attention_mask=masks, position_ids=gemma4_dpo.segment_positions(seg)[None],
                           dpo_labels=ids)
                out[s["name"]] = lp[0].float().cpu()
        # exact export check: for every parameter as held in memory (bf16, gathered from the ZeRO-3 partitions), the
        # integer sum and sum of squares of its raw 16-bit patterns (order-independent, so exact on any device); the
        # exported file must reproduce them, i.e. the export is bit-identical to the trained weights.
        sums = {}
        unwrapped = accelerator.unwrap_model(model)
        for name, param in unwrapped.named_parameters():
            with deepspeed.zero.GatheredParameters([param], modifier_rank=None):
                bits = param.detach().contiguous().view(torch.int16).to(torch.int64)
                sums[name] = (int(bits.sum().item()), int((bits * bits).sum().item()), list(param.shape), str(param.dtype))
        out["param_sums"] = sums
        model.train()
        if accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(out, os.path.join(output_dir, "export_probe.pt"))
            logger.info("export probe written for %s", list(out))
        accelerator.wait_for_everyone()
    return _ORIG_SAVE(accelerator, model, tokenizer, output_dir, *args, **kwargs)


@dataclass
class Gemma4DPOExperimentConfig(dpo_utils.DPOExperimentConfig):
    gemma4_length_grouped_batches: bool = False
    """Replace the training RandomSampler by gemma4_dpo.LengthGroupedMicrostepSampler (pairs of similar cost run
    concurrently on the ranks). False (default) = the stock random order."""
    gemma4_group_steps: int = 8
    """Mega-batch size of the length-grouped sampler, in micro-steps (pairs are sorted by cost within it)."""
    gemma4_cost_l0: float = gemma4_dpo.COST_L0
    """Quadratic scale of the sampler's per-side cost model, cost(L) = L + L^2 / L0 (tokens)."""
    gemma4_hybrid_flash_attention: bool = False
    """Sliding-window layers use FlashAttention-2 varlen, global layers SDPA (gemma4_dpo.gemma4_hybrid_attention).
    False (default) = SDPA with explicit masks on every layer."""

    @property
    def forward_fn(self):
        return gemma4_dpo.gemma4_separate_forward


def check_args(args: Gemma4DPOExperimentConfig) -> None:
    if args.transform_fn[0] not in gemma4_dpo.GEMMA4_TAIL_TRANSFORMS:
        raise ValueError(f"first transform must be one of {gemma4_dpo.GEMMA4_TAIL_TRANSFORMS}, got {args.transform_fn}")
    if args.concatenated_forward or args.packing:
        raise ValueError("the Gemma-4 forward is separate-forward only: pass --concatenated_forward false --packing false")
    if str(args.attn_implementation) not in ("torch", "AttentionBackendName.torch"):
        raise ValueError(
            "Gemma-4 global layers have head_dim 512 (no FlashAttention): pass --attn_implementation torch (HF sdpa)"
        )
    if args.use_lora or args.use_qlora or args.load_balancing_loss:
        raise ValueError("LoRA / QLoRA / MoE aux loss are not supported on the Gemma-4 path")
    if not transformers.__version__.startswith("5.5."):
        logger.warning("transformers %s: the Gemma-4 path was validated with 5.5.3", transformers.__version__)
    if args.chosen_nll_coef != 0.0 and not args.loss_type.needs_reference_model:
        raise ValueError("--chosen_nll_coef on the Gemma-4 path is implemented for dpo / dpo_norm only")
    if args.dpo_loss_coef != 1.0 and args.chosen_nll_coef == 0.0:
        raise ValueError("--dpo_loss_coef != 1 on the Gemma-4 path is applied inside the chosen-NLL wrapper: it needs "
                         "--chosen_nll_coef != 0")
    if args.dpo_loss_coef < 0.0:
        raise ValueError(f"--dpo_loss_coef must be >= 0, got {args.dpo_loss_coef}")
    if args.gemma4_group_steps < 1:
        raise ValueError("--gemma4_group_steps must be >= 1")


def apply_run_options(args: Gemma4DPOExperimentConfig) -> None:
    """Install this run's opt-in features in-process (each one only when requested) and hand dpo_tune_cache.main a
    config it accepts. Records what was installed in ``_RUN``."""
    _RUN["average_log_prob"] = args.loss_type.is_average_loss
    if args.chosen_nll_coef != 0.0:
        dpo_coef = float(getattr(args, "dpo_loss_coef", 1.0))
        _RUN["chosen_nll_coef"] = float(args.chosen_nll_coef)
        _RUN["dpo_loss_coef"] = dpo_coef
        dpo_utils.compute_loss = make_compute_loss_with_chosen_nll(float(args.chosen_nll_coef), dpo_coef)
        args.chosen_nll_coef = 0.0  # implemented by the wrapper above; dpo_tune_cache.main refuses non-zero values
        if dpo_coef != 1.0:
            args.dpo_loss_coef = 1.0  # likewise: applied by the wrapper; dpo_tune_cache.main refuses values != 1
    if args.gemma4_length_grouped_batches:
        dpo_tune_cache.RandomSampler = make_length_grouped_sampler_factory(
            args.per_device_train_batch_size, args.gemma4_group_steps, args.seed, args.gemma4_cost_l0
        )
    if args.gemma4_hybrid_flash_attention:
        _RUN["hybrid_attention"] = True
        dpo_utils.REFERENCE_LOGPROBS_CACHE_PATH = os.path.join(dpo_utils.REFERENCE_LOGPROBS_CACHE_PATH, gemma4_dpo.HYBRID_ATTN)


def main() -> None:
    gemma4_dpo.install_forward_patch()
    gemma4_dpo.register_transform()
    # dpo_utils turns TF32 on for every fp32 matmul at import (module level). Gemma-4's RoPE angles are an fp32 matmul
    # of inverse frequencies by positions up to 32k; TF32 (10-bit mantissa) rounds them, so the training forward would
    # no longer match vLLM / the parity reference, which compute them in full fp32. bf16 matmuls are unaffected.
    torch.backends.cuda.matmul.allow_tf32 = False
    accelerate.Accelerator.log = _log_and_print
    model_utils.save_with_accelerate = _save_with_export_probe
    dpo_tune_cache.AutoModelForCausalLM = _AutoModelWithBufferFix
    parser = ArgumentParserPlus((Gemma4DPOExperimentConfig, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    check_args(args)
    logger.info(
        "Gemma-4 DPO: transformers %s, logprob chunks per sequence %d, transform %s, tf32 matmul %s, chosen_nll_coef %s, "
        "dpo_loss_coef %s, length-grouped batches %s (group_steps %d), hybrid flash attention %s",
        transformers.__version__,
        gemma4_dpo.LOGPROB_NUM_CHUNKS,
        args.transform_fn,
        torch.backends.cuda.matmul.allow_tf32,
        args.chosen_nll_coef,
        args.dpo_loss_coef,
        args.gemma4_length_grouped_batches,
        args.gemma4_group_steps,
        args.gemma4_hybrid_flash_attention,
    )
    apply_run_options(args)
    dpo_tune_cache.main(args, tc)


if __name__ == "__main__":
    main()
