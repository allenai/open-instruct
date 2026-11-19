"""Tests for DPO utility functions."""

import copy
import inspect
import json
import logging
import multiprocessing
import os
import socket
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn.functional as F
from parameterized import parameterized
from torch import nn

from open_instruct import (
    dataset_transformation,
    dpo_utils,
    model_utils,
    olmo_core_train_modules,
    olmo_core_utils,
    padding_free_collator,
    template_tokens,
)
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.dpo_utils import DPOExperimentConfig, DPOLossType
from open_instruct.utils import ArgumentParserPlus

logging.basicConfig(level=logging.INFO)


def make_test_args(**overrides) -> DPOExperimentConfig:
    """Create a DPOExperimentConfig with test defaults."""
    defaults = {
        "model_name_or_path": "allenai/OLMo-2-1124-7B",
        "mixer_list": ["allenai/tulu-3-wildchat-reused-on-policy-8b", "1.0"],
        "config_hash": "test_dataset_config_hash",
    }
    defaults.update(overrides)
    return DPOExperimentConfig(**defaults)


class TestDPOLoss(unittest.TestCase):
    """Tests for dpo_loss function."""

    def test_basic_loss_computation(self):
        policy_chosen = torch.tensor([0.0, -1.0, -2.0])
        policy_rejected = torch.tensor([-1.0, -2.0, -3.0])
        ref_chosen = torch.tensor([0.0, -1.0, -2.0])
        ref_rejected = torch.tensor([-1.0, -2.0, -3.0])

        losses, chosen_rewards, rejected_rewards = dpo_utils.dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )

        self.assertEqual(losses.shape, (3,))
        self.assertEqual(chosen_rewards.shape, (3,))
        self.assertEqual(rejected_rewards.shape, (3,))

    def test_chosen_preferred_gives_lower_loss(self):
        policy_chosen = torch.tensor([0.0])
        policy_rejected = torch.tensor([-5.0])
        ref_chosen = torch.tensor([-1.0])
        ref_rejected = torch.tensor([-1.0])

        losses_good, _, _ = dpo_utils.dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1)

        policy_chosen_bad = torch.tensor([-5.0])
        policy_rejected_bad = torch.tensor([0.0])

        losses_bad, _, _ = dpo_utils.dpo_loss(
            policy_chosen_bad, policy_rejected_bad, ref_chosen, ref_rejected, beta=0.1
        )

        self.assertLess(losses_good.item(), losses_bad.item())

    def test_reference_free(self):
        policy_chosen = torch.tensor([0.0])
        policy_rejected = torch.tensor([-1.0])
        ref_chosen = torch.tensor([-5.0])
        ref_rejected = torch.tensor([-10.0])

        losses_ref, _, _ = dpo_utils.dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1, reference_free=False
        )

        losses_ref_free, _, _ = dpo_utils.dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1, reference_free=True
        )

        self.assertNotEqual(losses_ref.item(), losses_ref_free.item())

    def test_rewards_are_detached(self):
        policy_chosen = torch.tensor([0.0], requires_grad=True)
        policy_rejected = torch.tensor([-1.0], requires_grad=True)
        ref_chosen = torch.tensor([-1.0])
        ref_rejected = torch.tensor([-1.0])

        _, chosen_rewards, rejected_rewards = dpo_utils.dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
        )

        self.assertFalse(chosen_rewards.requires_grad)
        self.assertFalse(rejected_rewards.requires_grad)


class TestGetBatchLogps(unittest.TestCase):
    """Regression tests for PR #1625: division by zero when a sequence has
    every label masked (`-100`) and `average_log_prob=True`."""

    @parameterized.expand(
        [
            (
                "fully_masked_row_returns_zero_not_nan",
                [[-1.0, -2.0, -3.0, -4.0], [-1.0, -2.0, -3.0, -4.0]],
                [[0, 1, 2, 3], [-100, -100, -100, -100]],
                [-2.0, 0.0],
            ),
            ("partially_masked_row_matches_unclamped_average", [[-1.0, -2.0, -3.0, -4.0]], [[0, 1, 2, -100]], [-1.5]),
        ]
    )
    def test_average_log_prob(self, _name, per_token_logps, labels, expected):
        result = dpo_utils._get_batch_logps(torch.tensor(per_token_logps), torch.tensor(labels), average_log_prob=True)
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())
        torch.testing.assert_close(result, torch.tensor(expected))


class TestSimPOLoss(unittest.TestCase):
    """Tests for simpo_loss function."""

    def test_basic_loss_computation(self):
        policy_chosen = torch.tensor([0.0, -1.0, -2.0])
        policy_rejected = torch.tensor([-1.0, -2.0, -3.0])

        losses, chosen_rewards, rejected_rewards = dpo_utils.simpo_loss(
            policy_chosen, policy_rejected, beta=0.1, gamma_beta_ratio=0.3
        )

        self.assertEqual(losses.shape, (3,))
        self.assertEqual(chosen_rewards.shape, (3,))
        self.assertEqual(rejected_rewards.shape, (3,))

    def test_gamma_affects_loss(self):
        policy_chosen = torch.tensor([0.0])
        policy_rejected = torch.tensor([-1.0])

        losses_low_gamma, _, _ = dpo_utils.simpo_loss(policy_chosen, policy_rejected, beta=0.1, gamma_beta_ratio=0.1)

        losses_high_gamma, _, _ = dpo_utils.simpo_loss(policy_chosen, policy_rejected, beta=0.1, gamma_beta_ratio=1.0)

        self.assertNotEqual(losses_low_gamma.item(), losses_high_gamma.item())


class TestWPOLoss(unittest.TestCase):
    """Tests for wpo_loss function."""

    def test_basic_loss_computation(self):
        policy_chosen = torch.tensor([[0.0, -1.0, -2.0]])
        policy_rejected = torch.tensor([[-1.0, -2.0, -3.0]])
        ref_chosen = torch.tensor([[0.0, -1.0, -2.0]])
        ref_rejected = torch.tensor([[-1.0, -2.0, -3.0]])
        chosen_mask = torch.tensor([[True, True, True]])
        rejected_mask = torch.tensor([[True, True, True]])

        losses, chosen_rewards, rejected_rewards = dpo_utils.wpo_loss(
            policy_chosen,
            policy_rejected,
            ref_chosen,
            ref_rejected,
            beta=0.1,
            chosen_loss_mask=chosen_mask,
            rejected_loss_mask=rejected_mask,
        )

        self.assertEqual(losses.shape, (1, 3))
        self.assertEqual(chosen_rewards.shape, (1, 3))
        self.assertEqual(rejected_rewards.shape, (1, 3))


class TestComputeReferenceCacheHash(unittest.TestCase):
    """Tests for compute_reference_cache_hash function."""

    def test_deterministic_hash(self):
        args = make_test_args()
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")

        hash1 = dpo_utils.compute_reference_cache_hash(args, tc)
        hash2 = dpo_utils.compute_reference_cache_hash(args, tc)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)

    def test_different_model_different_hash(self):
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")

        args1 = make_test_args(model_name_or_path="allenai/OLMo-2-1124-7B")
        args2 = make_test_args(model_name_or_path="allenai/OLMo-2-1124-13B")

        hash1 = dpo_utils.compute_reference_cache_hash(args1, tc)
        hash2 = dpo_utils.compute_reference_cache_hash(args2, tc)

        self.assertNotEqual(hash1, hash2)

    def test_different_loss_type_different_hash(self):
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")

        args1 = make_test_args(loss_type=DPOLossType.dpo)
        args2 = make_test_args(loss_type=DPOLossType.simpo)

        hash1 = dpo_utils.compute_reference_cache_hash(args1, tc)
        hash2 = dpo_utils.compute_reference_cache_hash(args2, tc)

        self.assertNotEqual(hash1, hash2)

    def test_different_packing_different_hash(self):
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")

        args1 = make_test_args(packing=False)
        args2 = make_test_args(packing=True)

        hash1 = dpo_utils.compute_reference_cache_hash(args1, tc)
        hash2 = dpo_utils.compute_reference_cache_hash(args2, tc)

        self.assertNotEqual(hash1, hash2)


class TestChosenNLL(unittest.TestCase):
    """RPO-style NLL on the chosen response (--chosen_nll_coef, SafeComp recipe sweep 2026-10-26)."""

    def test_average_logps_is_negated(self):
        # dpo_norm / simpo pass the per-token MEAN log-prob: the length-normalized NLL is its negation.
        chosen = torch.tensor([-0.2, -1.5])
        nll = dpo_utils.chosen_nll_loss(chosen, None, average_log_prob=True)
        torch.testing.assert_close(nll, torch.tensor([0.2, 1.5]))

    def test_summed_logps_divided_by_labeled_tokens(self):
        # dpo / wpo pass the SUM; divide by the labeled-token count with _get_batch_logps' one-token shift.
        labels = torch.tensor([[-100, -100, 5, 6, 7], [-100, 3, 4, -100, -100]])  # 3 and 2 labeled targets
        per_token = torch.tensor([[0.0, -1.0, -2.0, -3.0, 0.0], [-0.5, -0.5, 0.0, 0.0, 0.0]])
        summed = dpo_utils._get_batch_logps(per_token, labels, average_log_prob=False)
        mean = dpo_utils._get_batch_logps(per_token, labels, average_log_prob=True)
        nll = dpo_utils.chosen_nll_loss(summed, labels, average_log_prob=False)
        torch.testing.assert_close(nll, -mean)
        torch.testing.assert_close(nll, torch.tensor([2.0, 0.5]))

    def test_summed_logps_need_labels(self):
        with self.assertRaises(ValueError):
            dpo_utils.chosen_nll_loss(torch.tensor([-3.0]), None, average_log_prob=False)

    def test_coef_zero_returns_same_loss_object(self):
        loss = torch.tensor(0.7, requires_grad=True)
        out, nll = dpo_utils.add_chosen_nll(loss, torch.tensor([-0.3]), {}, 0.0, average_log_prob=True)
        self.assertIs(out, loss)
        self.assertIsNone(nll)

    def test_default_is_off(self):
        self.assertEqual(make_test_args().chosen_nll_coef, 0.0)
        self.assertEqual(dpo_utils.DPOConfig().chosen_nll_coef, 0.0)

    def test_loss_value_and_gradient(self):
        # total = mean DPO loss + coef * mean NLL; d total / d chosen_logp_i = d DPO / d chosen_logp_i - coef / B.
        coef = 1.0
        ref_c, ref_r = torch.tensor([-0.2, -0.3, -0.25]), torch.tensor([-0.2, -0.1, -0.4])
        rej = torch.tensor([-0.5, -0.2, -0.6])

        chosen_a = torch.tensor([-0.1, -0.4, -0.3], requires_grad=True)
        losses_a, _, _ = dpo_utils.dpo_loss(chosen_a, rej, ref_c, ref_r, beta=5.0)
        total, nll = dpo_utils.add_chosen_nll(losses_a.mean(), chosen_a, {}, coef, average_log_prob=True)
        total.backward()

        chosen_b = chosen_a.detach().clone().requires_grad_(True)
        losses_b, _, _ = dpo_utils.dpo_loss(chosen_b, rej, ref_c, ref_r, beta=5.0)
        losses_b.mean().backward()

        torch.testing.assert_close(nll, torch.tensor(0.8 / 3))
        torch.testing.assert_close(total, losses_b.mean() + coef * nll)
        torch.testing.assert_close(chosen_a.grad, chosen_b.grad - coef / 3)

    def test_packing_with_summed_logps_refused(self):
        with self.assertRaises(NotImplementedError):
            dpo_utils.add_chosen_nll(
                torch.tensor(0.1), torch.tensor([-2.0]), {}, 1.0, average_log_prob=False, packing=True
            )

    def test_reference_cache_hash_ignores_coef(self):
        # Reference log-probs do not depend on the NLL term, so NLL runs reuse the plain-DPO reference cache.
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")
        self.assertEqual(
            dpo_utils.compute_reference_cache_hash(make_test_args(), tc),
            dpo_utils.compute_reference_cache_hash(make_test_args(chosen_nll_coef=1.0), tc),
        )

    def test_cli_flags_parse(self):
        parser = ArgumentParserPlus((DPOExperimentConfig, TokenizerConfig))
        base = ["--model_name_or_path", "allenai/OLMo-2-1124-7B", "--mixer_list", "x.jsonl", "1.0"]
        args, _ = parser.parse_args_into_dataclasses(base)
        self.assertEqual(args.chosen_nll_coef, 0.0)
        self.assertIsNone(args.hf_export_steps)
        self.assertEqual(args.hf_export_dtype, "float32")
        self.assertIsNone(args.checkpoint_save_folder)
        self.assertIsNone(args.pre_train_checkpoint)
        args, _ = parser.parse_args_into_dataclasses(
            base
            + ["--chosen_nll_coef", "1.0", "--hf_export_steps", "20", "40", "58", "--hf_export_dtype", "bfloat16"]
            + ["--checkpoint_save_folder", "/tmp/raw", "--pre_train_checkpoint", "false"]
        )
        self.assertEqual(args.chosen_nll_coef, 1.0)
        self.assertEqual(args.hf_export_steps, [20, 40, 58])
        self.assertEqual(args.hf_export_dtype, "bfloat16")
        self.assertEqual(args.checkpoint_save_folder, "/tmp/raw")
        self.assertIs(args.pre_train_checkpoint, False)


class TestCheckpointerPassthrough(unittest.TestCase):
    """--pre_train_checkpoint reaches OLMo-core's CheckpointerCallback; the default keeps OLMo-core's behaviour."""

    def test_default_none(self):
        cb = olmo_core_utils.build_checkpointer_callback(116, 20, save_async=False, max_checkpoints=-1)
        self.assertIsNone(cb.pre_train_checkpoint)
        self.assertIsNone(cb.max_checkpoints)
        self.assertEqual((cb.save_interval, cb.ephemeral_save_interval), (116, 20))

    def test_false_passed(self):
        cb = olmo_core_utils.build_checkpointer_callback(116, 20, save_async=False, pre_train_checkpoint=False)
        self.assertIs(cb.pre_train_checkpoint, False)


class TestDPOLossCoef(unittest.TestCase):
    """--dpo_loss_coef (SafeComp sftc control, 2026-11-03): scales the DPO term; 0 + chosen NLL = SFT on chosen."""

    ref_c, ref_r = torch.tensor([-0.2, -0.3, -0.25]), torch.tensor([-0.2, -0.1, -0.4])

    def _grads(self, dpo_coef, nll_coef):
        chosen = torch.tensor([-0.1, -0.4, -0.3], requires_grad=True)
        rejected = torch.tensor([-0.5, -0.2, -0.6], requires_grad=True)
        losses, _, _ = dpo_utils.dpo_loss(chosen, rejected, self.ref_c, self.ref_r, beta=5.0)
        loss = dpo_utils.apply_dpo_loss_coef(losses.mean(), dpo_coef)
        loss, _ = dpo_utils.add_chosen_nll(loss, chosen, {}, nll_coef, average_log_prob=True)
        loss.backward()
        return loss.detach(), chosen.grad, rejected.grad

    def test_default_is_one_and_same_object(self):
        self.assertEqual(make_test_args().dpo_loss_coef, 1.0)
        self.assertEqual(dpo_utils.DPOConfig().dpo_loss_coef, 1.0)
        loss = torch.tensor(0.7, requires_grad=True)
        self.assertIs(dpo_utils.apply_dpo_loss_coef(loss, 1.0), loss)

    def test_coef_zero_gives_nll_gradient_only(self):
        loss, g_chosen, g_rejected = self._grads(dpo_coef=0.0, nll_coef=1.0)
        # gradient of NLL alone: d/d chosen_i of mean_i(-chosen_i) = -1/B; the rejected side gets exactly zero
        chosen = torch.tensor([-0.1, -0.4, -0.3], requires_grad=True)
        nll = dpo_utils.chosen_nll_loss(chosen, None, average_log_prob=True).mean()
        nll.backward()
        torch.testing.assert_close(g_chosen, chosen.grad)
        torch.testing.assert_close(g_chosen, torch.full((3,), -1.0 / 3))
        torch.testing.assert_close(g_rejected, torch.zeros(3))
        torch.testing.assert_close(loss, nll.detach())

    def test_coef_scales_dpo_gradient(self):
        _, g1, r1 = self._grads(dpo_coef=1.0, nll_coef=0.0)
        _, gh, rh = self._grads(dpo_coef=0.5, nll_coef=0.0)
        torch.testing.assert_close(gh, 0.5 * g1)
        torch.testing.assert_close(rh, 0.5 * r1)

    def test_reference_cache_hash_ignores_new_flags(self):
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")
        self.assertEqual(
            dpo_utils.compute_reference_cache_hash(make_test_args(), tc),
            dpo_utils.compute_reference_cache_hash(
                make_test_args(
                    dpo_loss_coef=0.0, template_nll_coef=1.0, template_nll_tau=0.9, log_template_probs=True
                ),
                tc,
            ),
        )

    def test_cli_flags_parse(self):
        parser = ArgumentParserPlus((DPOExperimentConfig, TokenizerConfig))
        base = ["--model_name_or_path", "allenai/OLMo-2-1124-7B", "--mixer_list", "x.jsonl", "1.0"]
        args, _ = parser.parse_args_into_dataclasses(base)
        self.assertEqual(
            (args.dpo_loss_coef, args.template_nll_coef, args.template_nll_tau, args.log_template_probs),
            (1.0, 0.0, 0.99, False),
        )
        args, _ = parser.parse_args_into_dataclasses(
            base
            + ["--dpo_loss_coef", "0.0", "--template_nll_coef", "1.0", "--template_nll_tau", "0.99"]
            + ["--log_template_probs", "true"]
        )
        self.assertEqual(
            (args.dpo_loss_coef, args.template_nll_coef, args.template_nll_tau, args.log_template_probs),
            (0.0, 1.0, 0.99, True),
        )


class TestTemplateGatedNLL(unittest.TestCase):
    """Confidence-gated NLL on chosen-side template positions (SafeComp gtn arm, 2026-11-03)."""

    def test_zero_when_all_template_probs_at_or_above_tau(self):
        logps = torch.log(torch.tensor([[0.995, 0.3, 0.999, 0.99, 0.05]])).requires_grad_(True)
        mask = torch.tensor([[True, False, True, True, False]])  # template p = 0.995, 0.999, 0.99 (all >= tau)
        out = dpo_utils.template_gated_nll(logps, mask, tau=0.99)
        self.assertEqual(out.item(), 0.0)
        out.backward()
        torch.testing.assert_close(logps.grad, torch.zeros_like(logps))  # low-p NON-template tokens never count

    def test_positive_below_tau_normalized_by_template_count(self):
        p = torch.tensor([[0.995, 0.3, 0.9, 0.5, 0.05]])
        logps = torch.log(p).requires_grad_(True)
        mask = torch.tensor([[True, False, True, True, False]])
        out = dpo_utils.template_gated_nll(logps, mask, tau=0.99)
        want = -(torch.log(torch.tensor(0.9)) + torch.log(torch.tensor(0.5))) / 3  # 3 template positions, 2 gated
        self.assertGreater(out.item(), 0.0)
        torch.testing.assert_close(out, want)
        out.backward()
        torch.testing.assert_close(logps.grad, torch.tensor([[0.0, 0.0, -1 / 3, -1 / 3, 0.0]]))

    def test_floor_of_one_without_template_positions(self):
        logps = torch.log(torch.tensor([[0.2, 0.3]]))
        out = dpo_utils.template_gated_nll(logps, torch.zeros(1, 2, dtype=torch.bool), tau=0.99)
        self.assertEqual(out.item(), 0.0)

    def test_prob_sums(self):
        p = torch.tensor([[0.9, 0.5, 0.999, 0.2]])
        types = torch.tensor([[0, -1, 1, 0]])
        sums, mins = dpo_utils.template_prob_sums(torch.log(p), types, 2, 0.99, torch.tensor([False, True]))
        torch.testing.assert_close(sums[:3], torch.tensor([0.9 + 0.999 + 0.2, 3.0, 2.0]))
        torch.testing.assert_close(sums[3:5], torch.tensor([0.9 + 0.2, 0.999]))  # per-type sum p
        torch.testing.assert_close(sums[5:], torch.tensor([2.0, 1.0]))  # per-type count
        torch.testing.assert_close(mins, torch.tensor([0.2, 0.999]))  # min over all; min over deterministic types
        sums, mins = dpo_utils.template_prob_sums(torch.log(p), torch.full((1, 4), -1), 2, 0.99, torch.ones(2) > 0)
        self.assertEqual(sums.sum().item(), 0.0)
        self.assertTrue(torch.isinf(mins).all())


def _toy_patterns() -> template_tokens.TemplatePatterns:
    # think_close = [1, 2, 3], tc_open = [4, 5], tc_close = [6, 7, 8], end = [9, 1]; content_sep = (3, 4);
    # tokens 1 and 6 play the role of whitespace ('\n'), the only end tokens a trimmed match may drop.
    spans = (("think_close", (1, 2, 3)), ("tc_open", (4, 5)), ("tc_close", (6, 7, 8)), ("end", (9, 1)))
    names = tuple(f"{c}_{j}" for c, ids in spans for j in range(len(ids))) + ("content_sep_0",)
    return template_tokens.TemplatePatterns(
        spans=spans,
        content_sep=(3, 4),
        type_names=names,
        type_tokens=tuple(str(i) for i in range(len(names))),
        whitespace_ids=frozenset({1, 6}),
    )


class TestTemplateTypes(unittest.TestCase):
    """open_instruct/template_tokens.py position matching (shifted frame, labeled-only, priority, trimmed matches)."""

    def test_toy_row(self):
        P = _toy_patterns()
        # prompt (unlabeled) contains a full tc_open [4, 5] that must NOT match; then a labeled turn:
        # reasoning 50 51, think_close 1 2 3, prose 60, content_sep 3, tc_open 4 5, cmd 70, tc_close 6 7 8, end 9 1;
        # then a turn whose command token merges the '6' (tc_close trimmed match 7 8) and whose last '1' is unlabeled.
        ids = [4, 5, 0, 50, 51, 1, 2, 3, 60, 3, 4, 5, 70, 6, 7, 8, 9, 1, 0, 52, 1, 2, 3, 4, 5, 71, 7, 8, 9, 1]
        lab = [-100] * 3 + ids[3:18] + [-100] * 1 + ids[19:29] + [-100]
        types = template_tokens.template_types(torch.tensor([lab]), P)[0].tolist()
        got = {i + 1: P.type_names[t] for i, t in enumerate(types) if t >= 0}  # back to unshifted positions
        want = {
            5: "think_close_0",
            6: "think_close_1",
            7: "think_close_2",
            9: "content_sep_0",
            10: "tc_open_0",
            11: "tc_open_1",
            13: "tc_close_0",
            14: "tc_close_1",
            15: "tc_close_2",
            16: "end_0",
            17: "end_1",
            20: "think_close_0",
            21: "think_close_1",
            22: "think_close_2",
            23: "tc_open_0",
            24: "tc_open_1",
            26: "tc_close_1",
            27: "tc_close_2",
            28: "end_0",
        }
        self.assertEqual(got, want)
        self.assertTrue(all(lab[p] != -100 for p in got))
        self.assertEqual(template_tokens.template_mask(torch.tensor([lab]), P).sum().item(), len(want))

    def test_bare_whitespace_never_matches(self):
        P = _toy_patterns()
        lab = [-100, 1, 50, 6, 51, 1, 1, 6]  # whitespace-role tokens alone, not part of any template string
        self.assertEqual(template_tokens.template_mask(torch.tensor([lab]), P).sum().item(), 0)


QWEN35_TOKENIZER = os.environ.get("QWEN35_TOKENIZER", "/weka/oe-adapt-default/mingqianz/models/Qwen3.5-9B")
SAFECOMP_V21738 = (
    "/weka/oe-adapt-default/mingqianz/datasets/safecomp-test/final_run/exports_postarr_v2/budgets/eb1738/"
    "mixtures_alt/equal_budget/D-SC-27.jsonl"
)


@unittest.skipUnless(
    os.path.isdir(QWEN35_TOKENIZER) and os.path.isfile(SAFECOMP_V21738), "Qwen3.5 tokenizer / SafeComp data not found"
)
class TestTemplateMaskRenderedRow(unittest.TestCase):
    """The template mask on a real v2-1,738 training row rendered with the trainer's own transform."""

    @classmethod
    def setUpClass(cls):
        tok = dataset_transformation.TokenizerConfig(tokenizer_name_or_path=QWEN35_TOKENIZER).tokenizer
        cls.P = template_tokens.build_template_patterns(tok)
        with open(SAFECOMP_V21738) as f:
            rows = [json.loads(next(f)) for _ in range(3)]
        cls.encs = [dataset_transformation.preference_safecomp_tail_tokenize_v1(r, tok, 32768) for r in rows]

    def test_patterns_are_the_qwen35_template(self):
        self.assertEqual(
            dict(self.P.spans),
            {
                "think_close": (198, 248069, 271),
                "tc_open": (248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704, 28, 5454, 29, 198),
                "tc_close": (198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059),
                "end": (248046, 198),
            },
        )
        self.assertEqual(self.P.content_sep, (271, 248058))

    def test_mask_selects_template_positions(self):
        THINK, TCO, TCC, END, NL, NLNL = 248069, 248058, 248059, 248046, 198, 271
        for enc in self.encs:
            for side in ("chosen", "rejected"):
                ids = enc[f"{side}_input_ids"].tolist()
                lab = enc[f"{side}_labels"].tolist()
                types = template_tokens.template_types(torch.tensor([lab]), self.P)[0].tolist()
                got = {i + 1 for i, t in enumerate(types) if t >= 0}
                # independent reference from the special-token anchors of each labeled assistant span
                want, spans, st = set(), [], None
                for i, x in enumerate(lab + [-100]):
                    if x != -100 and st is None:
                        st = i
                    elif x == -100 and st is not None:
                        spans.append((st, i))
                        st = None
                self.assertGreater(len(spans), 0)
                for a, b in spans:
                    t = ids[a:b]
                    k, o, c, e = t.index(THINK), t.index(TCO), t.index(TCC), t.index(END)
                    want |= {a + k, a + k + 1} | ({a + k - 1} if t[k - 1] == NL else set())
                    if o > k + 2 and t[o - 1] == NLNL:
                        want.add(a + o - 1)  # prose then '\n\n' before the call
                    want |= {a + j for j in range(o, o + 14)}
                    first = c - 9 if t[c - 9] == NL else c - 8
                    want |= {a + j for j in range(first, c + 1)}
                    want |= {a + e} | ({a + e + 1} if e + 1 < len(t) else set())
                    self.assertEqual(ids[a + k], THINK)
                self.assertEqual(got, want, side)
                self.assertTrue(all(lab[p] != -100 for p in got))
                for i, t in enumerate(types):  # each position's type names the token that is actually there
                    if t >= 0:
                        name = self.P.type_names[t]
                        if name == "content_sep_0":
                            self.assertEqual(ids[i + 1], NLNL)
                        else:
                            cls_, j = name.rsplit("_", 1)
                            self.assertEqual(ids[i + 1], dict(self.P.spans)[cls_][int(j)])


# SafeComp 2026-11-07 (2026-11-05 DPO code review findings 1 and 2): --pair_avg_loss and --max_grad_norm, exercised
# through the real DPOTrainModule.train_batch / optim_step on a tiny CPU model.

_VOCAB = 13
_N_REF = 8  # reference-cache rows (pair indices 0.._N_REF-1)


class _TinyLM(nn.Module):
    """Policy stand-in: per-token logits from an embedding and a linear head. post_batch / post_optim_step are the
    OLMo-core Transformer hooks the train module calls (no-ops here)."""

    def __init__(self, dim: int = 6):
        super().__init__()
        self.emb = nn.Embedding(_VOCAB, dim)
        self.head = nn.Linear(dim, _VOCAB)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.tanh(self.emb(input_ids)))

    def post_batch(self, dry_run: bool = False) -> None:
        pass

    def post_optim_step(self) -> None:
        pass


def _tiny_forward(model, batch, average_log_prob, output_router_logits=False, **_kwargs):
    """separate_forward_olmo's contract on the tiny model: per-pair (chosen, rejected) log-probs and aux loss None."""
    logps = []
    for side in ("chosen", "rejected"):
        labels = batch[f"{side}_labels"]
        per_token = padding_free_collator.calculate_per_token_logps(model(batch[f"{side}_input_ids"]), labels)
        logps.append(dpo_utils._get_batch_logps(per_token, labels, average_log_prob))
    return logps[0], logps[1], None


def _make_pairs(lengths: list[tuple[int, int]], seed: int = 0) -> list[dict]:
    """One pair per (chosen_len, rejected_len); the first 2 tokens of each side are an unlabeled prompt."""
    gen = torch.Generator().manual_seed(seed)
    pairs = []
    for index, lens in enumerate(lengths):
        pair: dict = {"index": index}
        for side, n in zip(("chosen", "rejected"), lens):
            ids = torch.randint(1, _VOCAB, (n,), generator=gen)
            labels = ids.clone()
            labels[:2] = -100
            pair[f"{side}_input_ids"] = ids
            pair[f"{side}_attention_mask"] = torch.ones(n, dtype=torch.long)
            pair[f"{side}_labels"] = labels
        pairs.append(pair)
    return pairs


def _collate(pairs: list[dict]) -> dict[str, torch.Tensor]:
    """DataCollatorForSeq2SeqDPO's layout: chosen and rejected right-padded to one length, plus the pair index."""
    max_len = max(len(p[f"{s}_input_ids"]) for p in pairs for s in ("chosen", "rejected"))
    batch = {}
    for side in ("chosen", "rejected"):
        for key, pad in (("input_ids", 0), ("attention_mask", 0), ("labels", -100)):
            rows = [F.pad(p[f"{side}_{key}"], (0, max_len - len(p[f"{side}_{key}"])), value=pad) for p in pairs]
            batch[f"{side}_{key}"] = torch.stack(rows)
    batch["index"] = torch.tensor([p["index"] for p in pairs])
    return batch


def _dpo_module(model: nn.Module, pair_avg: bool, microbatch_size: int, chosen_nll_coef: float = 1.0):
    """A DPOTrainModule on the tiny model with the SafeComp recipe-(b) loss (dpo_norm, beta 5, separate forwards,
    chosen NLL 1.0), built without OLMo-core's parallel setup: train_batch(dry_run=True) needs only these attributes
    (world_mesh None -> dp_process_group None)."""
    tm = object.__new__(olmo_core_train_modules.DPOTrainModule)
    tm.model = model
    tm.world_mesh = None
    tm.scheduler = None
    tm.sample_microbatch_size = microbatch_size
    tm.dpo_config = dpo_utils.DPOConfig(
        beta=5.0,
        loss_type=DPOLossType.dpo_norm,
        concatenated_forward=False,
        chosen_nll_coef=chosen_nll_coef,
        pair_avg_loss=pair_avg,
    )
    gen = torch.Generator().manual_seed(1)
    tm.reference_cache = model_utils.TensorCache(
        tensors={
            "chosen_logps": -2.5 + 0.2 * torch.randn(_N_REF, generator=gen),
            "rejected_logps": -2.5 + 0.2 * torch.randn(_N_REF, generator=gen),
        }
    )
    names = ["loss", "chosen_logps", "rejected_logps", "chosen_rewards", "rejected_rewards", "accuracy"]
    tm._metrics = {k: torch.tensor(0.0) for k in names + (["chosen_nll"] if chosen_nll_coef != 0.0 else [])}
    tm._template_on, tm.template_patterns, tm._template_det_types = False, None, None
    tm._template_sums = tm._template_mins = None
    tm._forward_fn, tm._forward_kwargs = _tiny_forward, {}
    return tm


def _grads(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.grad.flatten() for p in model.parameters()])


def _run(base: nn.Module, pairs: list[dict], pair_avg: bool, microbatch_size: int):
    """One train_batch (dry run: forward + backward, no logging) on a fresh copy of ``base``; (gradient, module)."""
    model = copy.deepcopy(base)
    tm = _dpo_module(model, pair_avg, microbatch_size)
    tm.train_batch(_collate(pairs), dry_run=True)
    return _grads(model), tm


def _previous_train_batch(tm, batch: dict[str, torch.Tensor]) -> None:
    """The micro-batch loop of DPOTrainModule.train_batch before 2026-11-07 (token-share weighting), verbatim."""
    micro_batches = olmo_core_train_modules.split_batch_dpo(batch, tm.sample_microbatch_size)
    micro_token_counts = [padding_free_collator.get_num_tokens(mb) for mb in micro_batches]
    total_tokens = sum(micro_token_counts)
    for v in tm._metrics.values():
        v.zero_()
    for micro_batch_idx, micro_batch in enumerate(micro_batches):
        loss, step_metrics = tm._compute_microbatch_loss(micro_batch)
        micro_tokens = micro_token_counts[micro_batch_idx]
        weight = micro_tokens / total_tokens
        for k, v in step_metrics.items():
            tm._metrics[k] += v.detach() * micro_tokens
        (loss * weight).backward()


class _MetricSink:
    """Stands in for Trainer.record_metric; keeps the last value per '<namespace>/<name>'."""

    def __init__(self):
        self.metrics: dict[str, float] = {}

    def record_metric(self, name, value, reduce_type=None, namespace=None, merge_strategy=None):
        self.metrics[name if namespace is None else f"{namespace}/{name}"] = float(value)


# chosen/rejected lengths: pair 0 long (76 tokens), pair 1 short (9), pairs 2-3 in between
_LENS = [(40, 36), (4, 5), (12, 9), (25, 30)]


class TestPairAvgLoss(unittest.TestCase):
    """--pair_avg_loss: every preference pair weighs 1 / (global number of pairs) instead of its micro-batch's share
    of the rank's tokens. Gradients of the real DPOTrainModule.train_batch on a tiny model."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.base = _TinyLM()
        cls.pairs = _make_pairs(_LENS)
        cls.tokens = [c + r for c, r in _LENS]  # attention-mask tokens of each pair (both sides)
        # gradient of each pair's own loss (one pair, one micro-batch: weight 1 under either weighting)
        cls.g = [_run(cls.base, [p], pair_avg=True, microbatch_size=1)[0] for p in cls.pairs]

    def test_default_is_off(self):
        self.assertFalse(dpo_utils.DPOConfig().pair_avg_loss)
        self.assertFalse(make_test_args().pair_avg_loss)

    def test_microbatch_weights(self):
        loss_w, metric_w = dpo_utils.microbatch_weights([30000, 3000], [], pair_avg=False)
        self.assertEqual(loss_w, [30000 / 33000, 3000 / 33000])  # the previous expression, exactly
        self.assertEqual(metric_w, [30000, 3000])
        self.assertAlmostEqual(loss_w[0], 0.909, places=3)  # the review's example: the long pair gets ~91%
        loss_w, metric_w = dpo_utils.microbatch_weights(
            [30000, 3000], [1, 1], pair_avg=True, global_pairs=16, dp_world_size=8
        )
        self.assertEqual(loss_w, [0.5, 0.5])  # the SafeComp layout: 1 pair x 2 micro-batches x 8 ranks
        self.assertEqual(metric_w, [1, 1])
        with self.assertRaises(ValueError):
            dpo_utils.microbatch_weights([1], [1], pair_avg=True)

    def test_global_pair_count_without_distributed(self):
        self.assertFalse(dist.is_initialized())
        self.assertEqual(dpo_utils.global_pair_count(3, torch.device("cpu")), (3.0, 1))

    def test_flags_off_identical_to_previous_code(self):
        for mb in (1, 2, 4):
            new_model, old_model = copy.deepcopy(self.base), copy.deepcopy(self.base)
            new = _dpo_module(new_model, pair_avg=False, microbatch_size=mb)
            new.train_batch(_collate(self.pairs), dry_run=True)
            old = _dpo_module(old_model, pair_avg=False, microbatch_size=mb)
            _previous_train_batch(old, _collate(self.pairs))
            self.assertTrue(torch.equal(_grads(new_model), _grads(old_model)), f"microbatch size {mb}")
            for k in new._metrics:
                self.assertTrue(torch.equal(new._metrics[k], old._metrics[k]), f"{k}, microbatch size {mb}")

    def test_flags_off_weights_pairs_by_tokens(self):
        # long + short pair in separate micro-batches: the previous weighting is each pair's token share (finding 1)
        got, _ = _run(self.base, self.pairs[:2], pair_avg=False, microbatch_size=1)
        t0, t1 = self.tokens[:2]
        torch.testing.assert_close(got, (t0 * self.g[0] + t1 * self.g[1]) / (t0 + t1))
        self.assertGreater(t0 / (t0 + t1), 0.85)

    def test_short_and_long_pair_get_equal_weight(self):
        got, _ = _run(self.base, self.pairs[:2], pair_avg=True, microbatch_size=1)
        torch.testing.assert_close(got, (self.g[0] + self.g[1]) / 2)

    def test_accumulation_invariance(self):
        want = torch.stack(self.g).mean(0)
        for mb in (1, 2, 4):  # 4, 2 and 1 micro-batches
            got, _ = _run(self.base, self.pairs, pair_avg=True, microbatch_size=mb)
            torch.testing.assert_close(got, want, msg=f"microbatch size {mb}")
        # without the flag the split changes the gradient
        g_split, _ = _run(self.base, self.pairs, pair_avg=False, microbatch_size=1)
        g_whole, _ = _run(self.base, self.pairs, pair_avg=False, microbatch_size=4)
        self.assertFalse(torch.allclose(g_split, g_whole))

    def test_data_parallel_invariance_uneven_ranks(self):
        # DP ranks holding 1 and 3 pairs. global_pair_count returns what its all-reduce gives (checked over a real
        # gloo group in TestPairAvgLossGloo); the DP gradient reduction averages ranks (FSDP2 ReduceOp.AVG).
        want = torch.stack(self.g).mean(0)
        with mock.patch.object(dpo_utils, "global_pair_count", return_value=(4.0, 2)):
            r0, _ = _run(self.base, self.pairs[:1], pair_avg=True, microbatch_size=1)
            r1, _ = _run(self.base, self.pairs[1:], pair_avg=True, microbatch_size=1)
        torch.testing.assert_close((r0 + r1) / 2, want)

    def test_metric_numerators_are_pair_weighted(self):
        _, tm = _run(self.base, self.pairs, pair_avg=True, microbatch_size=1)
        singles = [_run(self.base, [p], pair_avg=True, microbatch_size=1)[1]._metrics for p in self.pairs]
        for k in tm._metrics:  # sum over pairs, each with weight 1 (divided by the global pair count when logged)
            torch.testing.assert_close(tm._metrics[k], sum(s[k] for s in singles), msg=k)

    def test_template_nll_needs_one_pair_per_microbatch(self):
        cfg = dpo_utils.DPOConfig(pair_avg_loss=True, template_nll_coef=1.0)
        tm = object.__new__(olmo_core_train_modules.DPOTrainModule)
        with self.assertRaises(NotImplementedError):
            olmo_core_train_modules.DPOTrainModule.__init__(
                tm,
                model=None,
                optim=None,
                sample_microbatch_size=2,
                max_sequence_length=8,
                dpo_config=cfg,
                attn_implementation=None,
                template_patterns=object(),
            )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestPairAvgLossGloo(unittest.TestCase):
    """--pair_avg_loss over a real 2-rank gloo process group (CPU, forked ranks): the all-reduced pair count, the
    DP-averaged gradient and the logged (non-dry-run) metrics, with 1 pair on rank 0 and 3 on rank 1."""

    def test_two_ranks_uneven(self):
        torch.manual_seed(0)
        base = _TinyLM()
        pairs = _make_pairs(_LENS)
        rank_pairs = [pairs[:1], pairs[1:]]
        port = _free_port()
        ctx = multiprocessing.get_context("fork")

        def worker(rank, conn):
            try:
                os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
                torch.set_num_threads(1)
                dist.init_process_group("gloo", rank=rank, world_size=2)
                model = copy.deepcopy(base)
                tm = _dpo_module(model, pair_avg=True, microbatch_size=1)
                sink = _MetricSink()
                tm._trainer = SimpleNamespace(
                    dp_process_group=None,
                    data_loader=SimpleNamespace(global_num_tokens_in_batch=lambda batch: 0),
                    global_step=1,
                    steps_per_epoch=1,
                    max_steps=None,
                    record_metric=sink.record_metric,
                )
                tm.train_batch(_collate(rank_pairs[rank]), dry_run=False)
                grad = _grads(model)
                dist.all_reduce(grad)  # gloo has no AVG: SUM / world size = FSDP2's ReduceOp.AVG over DP ranks
                grad /= 2
                dist.destroy_process_group()
                conn.send({"grad": grad.tolist(), "metrics": sink.metrics})
            except BaseException as e:  # report, never hang the parent
                conn.send({"error": repr(e)})
            finally:
                conn.close()
                os._exit(0)

        conns, procs = [], []
        for rank in range(2):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(rank, child_conn))
            proc.start()
            conns.append(parent_conn)
            procs.append(proc)
        results = []
        for rank, conn in enumerate(conns):
            self.assertTrue(conn.poll(180), f"rank {rank} timed out")
            results.append(conn.recv())
        for proc in procs:
            proc.join(30)
        for rank, res in enumerate(results):
            self.assertNotIn("error", res, f"rank {rank}: {res.get('error')}")

        singles = [_run(base, [p], pair_avg=True, microbatch_size=1) for p in pairs]
        want_grad = torch.stack([g for g, _ in singles]).mean(0)
        for res in results:
            torch.testing.assert_close(torch.tensor(res["grad"]), want_grad)
            m = res["metrics"]
            self.assertEqual(m["train/global_pairs_per_step"], 4.0)
            for name, key in (
                ("train_loss", "loss"),
                ("logps/chosen", "chosen_logps"),
                ("rewards/accuracy", "accuracy"),
                ("train/chosen_nll", "chosen_nll"),
            ):
                want = sum(tm._metrics[key].item() for _, tm in singles) / len(pairs)
                self.assertAlmostEqual(m[name], want, places=5, msg=name)


class TestGradClipping(unittest.TestCase):
    """--max_grad_norm: the EXISTING TrainingConfig flag (default None = no clipping). dpo.py passes it to
    DPOTrainModule, which passes it to OLMo-core's TransformerTrainModule; DPOTrainModule does not override
    optim_step, so OLMo-core's optim_step clips (and logs optim/total grad norm) before the optimizer step."""

    def _step(self, max_grad_norm: float | None, grad_norm: float):
        torch.manual_seed(0)
        model = _TinyLM()
        tm = object.__new__(olmo_core_train_modules.DPOTrainModule)
        tm.model, tm.max_grad_norm, tm.scheduler = model, max_grad_norm, None
        tm.optim = torch.optim.SGD(model.parameters(), lr=1.0)
        sink = _MetricSink()
        tm._trainer = SimpleNamespace(record_metric=sink.record_metric)
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        scale = grad_norm / _grads(model).norm()
        for p in model.parameters():
            p.grad.mul_(scale)
        grad = _grads(model).clone()
        before = torch.cat([p.detach().flatten() for p in model.parameters()])
        tm.optim_step()
        after = torch.cat([p.detach().flatten() for p in model.parameters()])
        return before - after, grad, sink.metrics  # SGD with lr 1: the update is the (clipped) gradient

    def test_existing_flag_default_none(self):
        self.assertIsNone(make_test_args().max_grad_norm)
        self.assertIsNone(olmo_core_utils.TrainingConfig().max_grad_norm)
        sig = inspect.signature(olmo_core_train_modules.DPOTrainModule.__init__)
        self.assertIsNone(sig.parameters["max_grad_norm"].default)
        self.assertNotIn("optim_step", vars(olmo_core_train_modules.DPOTrainModule))

    def test_off_does_not_clip(self):
        update, grad, metrics = self._step(None, 10.0)
        torch.testing.assert_close(update, grad)
        self.assertNotIn("optim/total grad norm", metrics)

    def test_on_clips_to_max_norm(self):
        update, grad, metrics = self._step(1.0, 10.0)
        self.assertAlmostEqual(update.norm().item(), 1.0, places=4)
        torch.testing.assert_close(update, grad / 10.0, rtol=1e-4, atol=1e-6)  # direction kept
        self.assertAlmostEqual(metrics["optim/total grad norm"], 10.0, places=3)  # pre-clip norm is logged

    def test_on_below_max_norm_unchanged(self):
        update, grad, metrics = self._step(1.0, 0.5)
        torch.testing.assert_close(update, grad)
        self.assertAlmostEqual(metrics["optim/total grad norm"], 0.5, places=4)


class TestPairAvgGradClipFlags(unittest.TestCase):
    """CLI parsing and reference-cache reuse for --pair_avg_loss / --max_grad_norm."""

    def test_cli_flags_parse(self):
        parser = ArgumentParserPlus((DPOExperimentConfig, TokenizerConfig))
        base = ["--model_name_or_path", "allenai/OLMo-2-1124-7B", "--mixer_list", "x.jsonl", "1.0"]
        args, _ = parser.parse_args_into_dataclasses(base)
        self.assertEqual((args.pair_avg_loss, args.max_grad_norm), (False, None))
        args, _ = parser.parse_args_into_dataclasses(base + ["--pair_avg_loss", "true", "--max_grad_norm", "1.0"])
        self.assertEqual((args.pair_avg_loss, args.max_grad_norm), (True, 1.0))

    def test_reference_cache_hash_ignores_new_flags(self):
        # reference log-probs do not depend on loss weighting or clipping: the RS-b-v21738 cache is reused
        tc = TokenizerConfig(tokenizer_name_or_path="allenai/OLMo-2-1124-7B")
        self.assertEqual(
            dpo_utils.compute_reference_cache_hash(make_test_args(), tc),
            dpo_utils.compute_reference_cache_hash(make_test_args(pair_avg_loss=True, max_grad_norm=1.0), tc),
        )


if __name__ == "__main__":
    unittest.main()
