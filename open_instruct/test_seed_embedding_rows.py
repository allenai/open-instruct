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
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import Shard, distribute_tensor, init_device_mesh

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


class TestZeroGatherOnlyWhenPromoted(unittest.TestCase):
    def _model(self):
        embed, head = torch.nn.Embedding(10, 4), torch.nn.Linear(4, 10, bias=False)
        embed.weight.data.copy_(reference_matrix())
        head.weight.data.copy_(reference_matrix())
        return types.SimpleNamespace(get_input_embeddings=lambda: embed, get_output_embeddings=lambda: head)

    def test_nothing_promoted_gathers_nothing(self):
        with mock.patch.object(model_utils.deepspeed.zero, "GatheredParameters") as gathered:
            self.assertEqual(
                model_utils.initialize_promoted_token_embeddings_under_zero(self._model(), _StubTokenizer([])), 0
            )
        gathered.assert_not_called()

    def test_promoted_rows_gather_both_matrices_once(self):
        model = self._model()
        with mock.patch.object(model_utils.deepspeed.zero, "GatheredParameters") as gathered:
            written = model_utils.initialize_promoted_token_embeddings_under_zero(model, _StubTokenizer([(3, (0, 1))]))
        self.assertEqual(written, 1)
        gathered.assert_called_once()
        params = gathered.call_args.args[0]
        self.assertIs(params[0], model.get_input_embeddings().weight)
        self.assertIs(params[1], model.get_output_embeddings().weight)
        self.assertEqual(gathered.call_args.kwargs, {"modifier_rank": 0})


if __name__ == "__main__":
    unittest.main()
