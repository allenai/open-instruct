"""Unit tests for the tiled GRPO lm-head loss in grpo_utils.

These validate that the tiled lm-head loss produces the same scalar loss and the
same gradients (w.r.t. both the hidden states and the lm-head weight) as a dense
reference computation, across loss functions, shard counts, and temperatures.
"""

import unittest

import torch
import torch.nn as nn
from parameterized import parameterized

from open_instruct import grpo_utils, model_utils


def _dense_reference(
    lm_head,
    hidden,
    labels,
    mask,
    advantages,
    old_logprobs,
    ref_logprobs,
    temperature,
    beta,
    loss_fn,
    denom,
    scale,
    rho_weights=None,
    kl_estimator=2,
):
    logits = lm_head(hidden)
    if temperature != 1.0:
        logits = logits / temperature
    new_logprobs = torch.gather(logits, -1, labels.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
    ratio = torch.exp(new_logprobs - old_logprobs)
    if loss_fn == grpo_utils.GRPOLossType.dapo:
        pg1 = -advantages * ratio
        pg2 = -advantages * torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.272)
    else:
        pg1 = -advantages * torch.clamp(ratio.detach(), max=1.0 + 0.272) * new_logprobs
        pg2 = pg1
    pg = torch.max(pg1, pg2)
    if rho_weights is not None:
        pg = pg * rho_weights
    if ref_logprobs is not None:
        kl = model_utils.estimate_kl((new_logprobs - ref_logprobs).clamp(-40, 40), ratio)[kl_estimator]
    else:
        kl = torch.zeros_like(pg)
    per_token = pg + beta * kl
    loss = (per_token * mask).sum() / denom * scale
    clip = (pg2 > pg1).float() if loss_fn == grpo_utils.GRPOLossType.dapo else (ratio > 1.0 + 0.272).float()
    if rho_weights is not None:
        clip = clip * (rho_weights != 0).float()
    clipfrac = (clip * mask).sum() / mask.sum().clamp_min(1.0)
    return loss, clipfrac


class TestTiledGRPOLMHeadLoss(unittest.TestCase):
    @parameterized.expand(
        [
            (
                f"{loss_fn}_ref{int(has_ref)}_shards{shards}_temp{temperature}_rho{int(use_rho)}",
                loss_fn,
                has_ref,
                shards,
                temperature,
                use_rho,
            )
            for loss_fn in (grpo_utils.GRPOLossType.dapo, grpo_utils.GRPOLossType.cispo)
            for has_ref in (False, True)
            for shards in (1, 3, 100)
            for temperature in (1.0, 0.7)
            for use_rho in (False, True)
        ]
    )
    def test_matches_dense_loss_and_grads(self, _name, loss_fn, has_ref, shards, temperature, use_rho):
        torch.manual_seed(0)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 8, 17
        denom_val, scale_val = 7.0, 3.0
        beta = 0.1 if has_ref else 0.0

        base = nn.Linear(hidden_size, vocab_size, bias=False)
        hidden0 = torch.randn(batch_size, seq_len, hidden_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.rand(batch_size, seq_len) > 0.3
        advantages = torch.randn(batch_size, seq_len)
        # Very negative so some ratios exceed 1 + clip_higher, giving a non-zero cispo
        # clipfrac (guards against the clipfrac-always-zero bug).
        old_logprobs = -torch.rand(batch_size, seq_len) * 6 - 3
        ref = -torch.rand(batch_size, seq_len) * 2 if has_ref else None
        # Some rho weights clamped, some dropped to 0, to exercise the masking path.
        rho = None
        if use_rho:
            rho = (torch.rand(batch_size, seq_len) * 1.5).clamp(0.0, 1.2)
            rho = torch.where(torch.rand(batch_size, seq_len) > 0.2, rho, torch.zeros_like(rho))

        hidden_tiled = hidden0.clone().requires_grad_(True)
        lm_tiled = nn.Linear(hidden_size, vocab_size, bias=False)
        lm_tiled.load_state_dict(base.state_dict())
        loss_tiled, _pg, _kl, clipfrac_tiled, _ratio = grpo_utils.tiled_grpo_lm_head_loss(
            lm_head=lm_tiled,
            hidden_states=hidden_tiled,
            selected_token_ids=labels,
            response_mask=mask,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref,
            temperature=temperature,
            beta=beta,
            clip_lower=0.2,
            clip_higher=0.272,
            shards=shards,
            loss_scale=torch.tensor(scale_val),
            loss_denom=torch.tensor(denom_val),
            loss_fn=loss_fn,
            rho_weights=rho,
        )
        loss_tiled.backward()

        hidden_dense = hidden0.clone().requires_grad_(True)
        lm_dense = nn.Linear(hidden_size, vocab_size, bias=False)
        lm_dense.load_state_dict(base.state_dict())
        loss_dense, clipfrac_dense = _dense_reference(
            lm_dense,
            hidden_dense,
            labels,
            mask.float(),
            advantages,
            old_logprobs,
            ref,
            temperature,
            beta,
            loss_fn,
            denom_val,
            scale_val,
            rho,
        )
        loss_dense.backward()

        torch.testing.assert_close(loss_tiled, loss_dense, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(hidden_tiled.grad, hidden_dense.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(lm_tiled.weight.grad, lm_dense.weight.grad, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(clipfrac_tiled, clipfrac_dense, atol=1e-5, rtol=1e-5)

    def test_rejects_unsupported_loss_fn(self):
        lm_head = nn.Linear(4, 5, bias=False)
        hidden = torch.randn(1, 3, 4, requires_grad=True)
        with self.assertRaises(ValueError):
            grpo_utils.tiled_grpo_lm_head_loss(
                lm_head=lm_head,
                hidden_states=hidden,
                selected_token_ids=torch.zeros(1, 3, dtype=torch.long),
                response_mask=torch.ones(1, 3, dtype=torch.bool),
                advantages=torch.zeros(1, 3),
                old_logprobs=torch.zeros(1, 3),
                ref_logprobs=None,
                temperature=1.0,
                beta=0.0,
                clip_lower=0.2,
                clip_higher=0.272,
                shards=1,
                loss_scale=torch.tensor(1.0),
                loss_denom=torch.tensor(3.0),
                loss_fn="not_a_loss",
            )


class TestTiledTokenLogprobs(unittest.TestCase):
    @parameterized.expand([("s1_t1", 1, 1.0), ("s3_t07", 3, 0.7), ("sbig_t1", 100, 1.0)])
    def test_matches_dense_log_softmax_gather(self, _name, shards, temperature):
        torch.manual_seed(0)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 8, 17
        lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        hidden = torch.randn(batch_size, seq_len, hidden_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            tiled = grpo_utils.tiled_token_logprobs(lm_head, hidden, labels, temperature, shards)
            logits = lm_head(hidden)
            if temperature != 1.0:
                logits = logits / temperature
            dense = torch.gather(logits.log_softmax(dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        torch.testing.assert_close(tiled, dense, atol=1e-5, rtol=1e-5)


class _FakeCausalLM(nn.Module):
    base_model_prefix = "model"

    def __init__(self):
        super().__init__()
        self.model = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 7)


class _FakePeftModel(nn.Module):
    def __init__(self, base):
        super().__init__()
        self._base = base

    def get_base_model(self):
        return self._base


class _FakeDeepSpeedEngine(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


class TestGetCausalLMBackboneAndLMHead(unittest.TestCase):
    def _check(self, wrapped, hf):
        backbone, lm_head = grpo_utils.get_causal_lm_backbone_and_lm_head(wrapped)
        self.assertIs(backbone, hf.model)
        self.assertIs(lm_head, hf.lm_head)

    def test_plain_causal_lm(self):
        hf = _FakeCausalLM()
        self._check(hf, hf)

    def test_deepspeed_wrapped(self):
        hf = _FakeCausalLM()
        self._check(_FakeDeepSpeedEngine(hf), hf)

    def test_peft_wrapped(self):
        hf = _FakeCausalLM()
        self._check(_FakePeftModel(hf), hf)

    def test_deepspeed_wrapped_peft(self):
        hf = _FakeCausalLM()
        self._check(_FakeDeepSpeedEngine(_FakePeftModel(hf)), hf)


class TestLigerGRPOLossConfigValidation(unittest.TestCase):
    def test_rejects_record_entropy(self):
        with self.assertRaisesRegex(ValueError, "record_entropy"):
            grpo_utils.GRPOExperimentConfig(use_liger_grpo_loss=True, record_entropy=True)


class TestTiledGRPOKLEstimator(unittest.TestCase):
    @parameterized.expand([("est0", 0), ("est1", 1), ("est2", 2), ("est3", 3)])
    def test_kl_estimator_is_configurable(self, _name, kl_estimator):
        torch.manual_seed(0)
        batch_size, seq_len, hidden_size, vocab_size = 2, 5, 8, 17
        base = nn.Linear(hidden_size, vocab_size, bias=False)
        hidden0 = torch.randn(batch_size, seq_len, hidden_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        mask = torch.rand(batch_size, seq_len) > 0.3
        advantages = torch.randn(batch_size, seq_len)
        old_logprobs = -torch.rand(batch_size, seq_len) * 2
        ref = -torch.rand(batch_size, seq_len) * 2

        hidden_tiled = hidden0.clone().requires_grad_(True)
        lm_tiled = nn.Linear(hidden_size, vocab_size, bias=False)
        lm_tiled.load_state_dict(base.state_dict())
        loss_tiled, *_ = grpo_utils.tiled_grpo_lm_head_loss(
            lm_head=lm_tiled,
            hidden_states=hidden_tiled,
            selected_token_ids=labels,
            response_mask=mask,
            advantages=advantages,
            old_logprobs=old_logprobs,
            ref_logprobs=ref,
            temperature=1.0,
            beta=0.1,
            clip_lower=0.2,
            clip_higher=0.272,
            shards=2,
            loss_scale=torch.tensor(3.0),
            loss_denom=torch.tensor(7.0),
            loss_fn=grpo_utils.GRPOLossType.dapo,
            kl_estimator=kl_estimator,
        )

        hidden_dense = hidden0.clone().requires_grad_(True)
        lm_dense = nn.Linear(hidden_size, vocab_size, bias=False)
        lm_dense.load_state_dict(base.state_dict())
        loss_dense, _ = _dense_reference(
            lm_dense,
            hidden_dense,
            labels,
            mask.float(),
            advantages,
            old_logprobs,
            ref,
            1.0,
            0.1,
            grpo_utils.GRPOLossType.dapo,
            7.0,
            3.0,
            kl_estimator=kl_estimator,
        )
        torch.testing.assert_close(loss_tiled, loss_dense, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
