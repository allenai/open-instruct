"""Tests for seeding promoted-token embedding rows, sharded and unsharded.

The olmo-core SFT path loads weights after `parallelize_model`, so the embedding is a DTensor
sharded on the vocabulary dimension by the time anything can write to it: the rows a promoted
token is seeded from live on other ranks, and the row being written lives on another again.
These run three gloo ranks on CPU with a vocabulary that does not divide evenly, which is the
case the offset arithmetic would get wrong.
"""

import os
import tempfile
import types
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import Replicate, Shard, distribute_tensor, init_device_mesh

from open_instruct import dataset_transformation, model_utils, olmo_core_utils

VOCAB, HIDDEN = 10, 4
# 10 rows over 3 ranks is deliberately uneven, and every target/source pair here crosses a
# shard boundary under that split: row 9 is on the last rank, its sources on the first two.
ROWS = [(9, (0, 1, 5)), (4, (2, 8))]


def reference_matrix() -> torch.Tensor:
    return torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN) / 10.0


def _sharded_worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        parameter = torch.nn.Parameter(distribute_tensor(reference_matrix(), mesh, [Shard(0)]))
        model_utils.seed_embedding_rows(parameter, ROWS)
        # full_tensor() on every rank: a write that only landed on the owning rank, or that
        # left ranks disagreeing, shows up as a difference between these files.
        torch.save(parameter.data.full_tensor(), os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _master_sync_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))
        for sharded_model in (False, True):
            original = reference_matrix() + 0.00013
            data = original.bfloat16()
            if sharded_model:
                data = distribute_tensor(data, mesh, [Shard(0)])
            param = torch.nn.Parameter(data)
            # Flat master shards have different boundaries from vocabulary shards.
            master = distribute_tensor(original.flatten().clone(), mesh, [Shard(0)])
            optimizer = types.SimpleNamespace(
                states={"embedding.main": master}, param_groups=[{"named_params": {"embedding": param}}]
            )
            model_utils.seed_embedding_rows(param, ROWS)
            olmo_core_utils.sync_promoted_token_optimizer_rows(optimizer, [param], [row for row, _ in ROWS])
            expected = original.clone()
            seeded = param.data.full_tensor() if sharded_model else param.data
            for row, _ in ROWS:
                expected[row] = seeded[row].float()
            torch.testing.assert_close(master.full_tensor().reshape_as(expected), expected, rtol=0, atol=0)
            # A copy-back must preserve the initialized model exactly.
            torch.testing.assert_close(master.full_tensor().reshape_as(expected).bfloat16(), seeded, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class TestSeedEmbeddingRowsUnsharded(unittest.TestCase):
    def test_target_row_is_the_mean_of_its_sources(self):
        matrix = reference_matrix()
        expected = {target: matrix[list(sources)].mean(dim=0).clone() for target, sources in ROWS}
        model_utils.seed_embedding_rows(matrix, ROWS)
        for target, mean in expected.items():
            torch.testing.assert_close(matrix[target], mean)

    def test_other_rows_are_untouched(self):
        matrix = reference_matrix()
        original = matrix.clone()
        model_utils.seed_embedding_rows(matrix, ROWS)
        targets = {target for target, _ in ROWS}
        for index in range(VOCAB):
            if index not in targets:
                torch.testing.assert_close(matrix[index], original[index])

    def test_sources_are_read_before_any_target_is_written(self):
        # A target that is also another target's source must contribute its ORIGINAL value.
        matrix = reference_matrix()
        rows = [(3, (0, 1)), (7, (3, 2))]
        expected = matrix[[3, 2]].mean(dim=0).clone()
        model_utils.seed_embedding_rows(matrix, rows)
        torch.testing.assert_close(matrix[7], expected)

    def test_row_out_of_range_raises(self):
        with self.assertRaisesRegex(ValueError, "only 10 rows"):
            model_utils.seed_embedding_rows(reference_matrix(), [(VOCAB, (0, 1))])

    def test_source_out_of_range_raises(self):
        with self.assertRaisesRegex(ValueError, "only 10 rows"):
            model_utils.seed_embedding_rows(reference_matrix(), [(0, (1, VOCAB))])


class TestSeedEmbeddingRowsSharded(unittest.TestCase):
    """Three gloo ranks on CPU; the sharded result must equal the single-process one."""

    world_size = 3

    def test_sharded_result_matches_unsharded(self):
        expected = reference_matrix()
        model_utils.seed_embedding_rows(expected, ROWS)

        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "rendezvous")
            # spawn, not fork: by this point the parent has torch loaded and is multi-threaded,
            # and forking such a process into a gloo rendezvous is a documented deadlock risk.
            # A spawned rank re-imports this module, which is slower but cannot wedge.
            mp.start_processes(
                _sharded_worker,
                args=(self.world_size, init_file, directory),
                nprocs=self.world_size,
                join=True,
                start_method="spawn",
            )
            for rank in range(self.world_size):
                actual = torch.load(os.path.join(directory, f"rank{rank}.pt"), weights_only=True)
                torch.testing.assert_close(actual, expected, msg=f"rank {rank} disagrees")

    def test_flat_master_shards_preserve_unrelated_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.start_processes(
                _master_sync_worker,
                args=(self.world_size, os.path.join(directory, "rendezvous")),
                nprocs=self.world_size,
                join=True,
                start_method="spawn",
            )


class _StubTokenizer:
    def __init__(self, rows):
        self.promoted_reserved_slot_tokens = [
            dataset_transformation.PromotedToken(content=f"<t{target}>", token_id=target, source_ids=tuple(sources))
            for target, sources in rows
        ]


class _StubTrainModule:
    """The parts of an olmo-core TransformerTrainModule the seeding reads."""

    def __init__(self, tie_word_embeddings: bool):
        self.model = types.SimpleNamespace(
            embeddings=types.SimpleNamespace(weight=reference_matrix()),
            lm_head=types.SimpleNamespace(w_out=types.SimpleNamespace(weight=reference_matrix())),
            tie_word_embeddings=tie_word_embeddings,
        )
        if tie_word_embeddings:
            self.model.lm_head.w_out.weight = self.model.embeddings.weight


class TestOlmoCoreSeeding(unittest.TestCase):
    # A target that is also a later target's source: seeding one matrix twice would feed the
    # row just written back in, so this pins the tie_word_embeddings guard.
    CHAINED = [(3, (0, 1)), (7, (3, 2))]

    def test_untied_head_is_seeded_as_well_as_the_embedding(self):
        train_module = _StubTrainModule(tie_word_embeddings=False)
        expected = reference_matrix()[[0, 1]].mean(dim=0)
        written = olmo_core_utils.initialize_promoted_token_embeddings(train_module, _StubTokenizer(self.CHAINED))
        self.assertEqual(written, len(self.CHAINED))
        torch.testing.assert_close(train_module.model.embeddings.weight[3], expected)
        torch.testing.assert_close(train_module.model.lm_head.w_out.weight[3], expected)

    def test_tied_weights_are_seeded_once(self):
        train_module = _StubTrainModule(tie_word_embeddings=True)
        # Row 7's mean must use row 3's ORIGINAL value, which a second pass would not see.
        expected = reference_matrix()[[3, 2]].mean(dim=0)
        olmo_core_utils.initialize_promoted_token_embeddings(train_module, _StubTokenizer(self.CHAINED))
        torch.testing.assert_close(train_module.model.embeddings.weight[7], expected)

    def test_nothing_promoted_is_a_noop(self):
        train_module = _StubTrainModule(tie_word_embeddings=False)
        original = train_module.model.embeddings.weight.clone()
        self.assertEqual(olmo_core_utils.initialize_promoted_token_embeddings(train_module, _StubTokenizer([])), 0)
        torch.testing.assert_close(train_module.model.embeddings.weight, original)


class TestPromotedOptimizerMasters(unittest.TestCase):
    def make_module(self, tied=False):
        module = _StubTrainModule(tie_word_embeddings=tied)
        matrices = {"embedding": module.model.embeddings, "head": module.model.lm_head.w_out}
        named, states = {}, {}
        for index, (name, holder) in enumerate(matrices.items()):
            if tied and name == "head":
                holder.weight = matrices["embedding"].weight
                continue
            original = reference_matrix() + 0.00013 + index * 0.0713
            holder.weight = torch.nn.Parameter(original.bfloat16())
            named[name] = holder.weight
            states[f"{name}.main"] = original.flatten().clone()
            states[f"{name}.exp_avg"] = torch.full_like(original.flatten(), 0.003)
            states[f"{name}.exp_avg_sq"] = torch.full_like(original.flatten(), 0.007)
        module.optim = types.SimpleNamespace(states=states, param_groups=[{"named_params": named}])
        return module

    def test_changes_only_promoted_rows_and_preserves_moments(self):
        for tied in (False, True):
            with self.subTest(tied=tied):
                module = self.make_module(tied)
                before = {name: value.clone() for name, value in module.optim.states.items()}
                params = module.optim.param_groups[0]["named_params"]
                model_before = {name: param.detach().clone() for name, param in params.items()}
                keep = torch.ones(VOCAB, dtype=torch.bool)
                keep[[row for row, _ in ROWS]] = False
                olmo_core_utils.initialize_promoted_token_embeddings(module, _StubTokenizer(ROWS))
                for name, param in params.items():
                    master = module.optim.states[f"{name}.main"].view(VOCAB, HIDDEN)
                    original = before[f"{name}.main"].view(VOCAB, HIDDEN)
                    self.assertTrue(torch.equal(master[keep], original[keep]))
                    self.assertTrue(torch.equal(param.detach()[keep], model_before[name][keep]))
                    for target, sources in ROWS:
                        expected = model_before[name][list(sources)].float().mean(0).bfloat16()
                        self.assertTrue(torch.equal(master[target], expected.float()))
                        self.assertTrue(torch.equal(param[target], expected))
                    # The old whole-matrix copy fails the exact preservation assertion.
                    self.assertFalse(torch.equal(original[keep], param.detach()[keep].float()))
                    # Simulate the optimizer's copy-back: initialization survives.
                    param.data.copy_(master.bfloat16())
                    self.assertTrue(torch.equal(param.detach()[keep], model_before[name][keep]))
                for name, value in before.items():
                    if not name.endswith(".main"):
                        self.assertTrue(torch.equal(module.optim.states[name], value))

    def test_disabled_preserves_all_optimizer_values(self):
        module = self.make_module()
        before = {name: value.clone() for name, value in module.optim.states.items()}
        self.assertEqual(olmo_core_utils.initialize_promoted_token_embeddings(module, _StubTokenizer([])), 0)
        for name, value in before.items():
            self.assertTrue(torch.equal(module.optim.states[name], value))

    def test_missing_master_fails_before_any_master_changes(self):
        module = self.make_module()
        before = module.optim.states["embedding.main"].clone()
        del module.optim.states["head.main"]
        with self.assertRaisesRegex(ValueError, "Missing optimizer master"):
            olmo_core_utils.initialize_promoted_token_embeddings(module, _StubTokenizer(ROWS))
        self.assertTrue(torch.equal(module.optim.states["embedding.main"], before))

    def test_unmatched_parameter_fails(self):
        module = self.make_module()
        del module.optim.param_groups[0]["named_params"]["head"]
        with self.assertRaisesRegex(ValueError, "1 of 2"):
            olmo_core_utils.initialize_promoted_token_embeddings(module, _StubTokenizer(ROWS))

    def test_standard_optimizer_needs_no_master_sync(self):
        module = self.make_module()
        module.optim = torch.optim.AdamW(module.optim.param_groups[0]["named_params"].values())
        self.assertEqual(olmo_core_utils.initialize_promoted_token_embeddings(module, _StubTokenizer(ROWS)), len(ROWS))


class _StubDDPOptimizer:
    """An OLMoDDPOptimizer's bf16 layout: fp32 main copies that are separate from the params.

    Deliberately has no all-param check: the sync must not rely on one, since on a bf16 model
    every unsynced param differs from its fp32 main by rounding.
    """

    def __init__(self, named_params, mesh):
        self.param_groups = [{"named_params": named_params}]
        self.states = {
            f"{name}.main": distribute_tensor(param.detach().float().reshape(-1).clone(), mesh, [Replicate()])
            for name, param in named_params.items()
        }


class TestOptimizerMainParamSync(unittest.TestCase):
    ROWS = [(3, (0, 1))]

    @classmethod
    def setUpClass(cls):
        cls.init_dir = tempfile.TemporaryDirectory()
        init_file = os.path.join(cls.init_dir.name, "init")
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
        cls.mesh = init_device_mesh("cpu", (1,))

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.init_dir.cleanup()

    def _train_module(self, include_head: bool):
        train_module = _StubTrainModule(tie_word_embeddings=False)
        named = {"embeddings.weight": train_module.model.embeddings.weight}
        if include_head:
            named["lm_head.w_out.weight"] = train_module.model.lm_head.w_out.weight
        train_module.optim = _StubDDPOptimizer(named, self.mesh)
        return train_module

    def test_separate_main_copies_receive_the_seed(self):
        # Without the sync, step() would copy these stale mains back over the seeded rows.
        train_module = self._train_module(include_head=True)
        olmo_core_utils.initialize_promoted_token_embeddings(train_module, _StubTokenizer(self.ROWS))
        expected = reference_matrix()[[0, 1]].mean(dim=0)
        for name in ("embeddings.weight", "lm_head.w_out.weight"):
            main = train_module.optim.states[f"{name}.main"].full_tensor().reshape(reference_matrix().shape)
            torch.testing.assert_close(main[3], expected)

    def test_unsynced_params_that_differ_from_their_mains_by_rounding_do_not_fail(self):
        # The H015 gate (01M38HZ936D3WNWS25A9GYSG55): a bf16 embedding_norm differs from its
        # fp32 main by rounding, which an all-param check rejected before step 1.
        train_module = self._train_module(include_head=True)
        norm = torch.full((4,), 0.0014, dtype=torch.bfloat16)
        train_module.optim.param_groups[0]["named_params"]["embedding_norm.weight"] = norm
        train_module.optim.states["embedding_norm.weight.main"] = distribute_tensor(
            torch.full((4,), 0.0013), self.mesh, [Replicate()]
        )
        olmo_core_utils.initialize_promoted_token_embeddings(train_module, _StubTokenizer(self.ROWS))

    def test_a_seeded_matrix_the_optimizer_does_not_own_raises(self):
        train_module = self._train_module(include_head=False)
        with self.assertRaisesRegex(ValueError, "1 of 2 promoted matrices"):
            olmo_core_utils.initialize_promoted_token_embeddings(train_module, _StubTokenizer(self.ROWS))


class TestPromotedRowStepCheck(unittest.TestCase):
    ROWS = [(3, (0, 1))]

    def _seeded_check(self):
        train_module = _StubTrainModule(tie_word_embeddings=False)
        check = olmo_core_utils.PromotedRowStepCheck.before_seeding(train_module, _StubTokenizer(self.ROWS))
        pre_seed = [w.clone() for w in (train_module.model.embeddings.weight, train_module.model.lm_head.w_out.weight)]
        olmo_core_utils.initialize_promoted_token_embeddings(train_module, _StubTokenizer(self.ROWS))
        check.record_seeded()
        check.trainer = types.SimpleNamespace(global_step=1)
        return train_module, check, pre_seed

    def test_a_step_that_trains_from_the_seed_passes(self):
        train_module, check, _ = self._seeded_check()
        for weight in check.matrices:
            weight.data[3] += 1e-3
        check.post_step()

    def test_a_step_that_reverts_the_seed_raises(self):
        # What a stale master copy does: step() writes the pre-seed row back, plus an update.
        train_module, check, pre_seed = self._seeded_check()
        for weight, original in zip(check.matrices, pre_seed):
            weight.data[3] = original[3] + 1e-3
        with self.assertRaisesRegex(RuntimeError, "not strictly closer"):
            check.post_step()

    def test_a_tie_raises(self):
        # Halfway between seed and pre-seed value: equidistant, so not evidence either way.
        train_module, check, pre_seed = self._seeded_check()
        for weight, original, seeded in zip(check.matrices, pre_seed, check.seeded):
            weight.data[3] = (original[3] + seeded[0]) / 2
        with self.assertRaisesRegex(RuntimeError, "not strictly closer"):
            check.post_step()

    def test_a_non_finite_row_raises(self):
        train_module, check, _ = self._seeded_check()
        check.matrices[0].data[3] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "not strictly closer"):
            check.post_step()

    def test_nothing_promoted_builds_no_check(self):
        train_module = _StubTrainModule(tie_word_embeddings=False)
        self.assertIsNone(olmo_core_utils.PromotedRowStepCheck.before_seeding(train_module, _StubTokenizer([])))


if __name__ == "__main__":
    unittest.main()
