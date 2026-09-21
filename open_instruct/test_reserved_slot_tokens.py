"""Tests for promoting multi-token tags into reserved vocabulary slots.

The tokenizer here is synthetic so the tests are hermetic, but it reproduces the property that
matters: a byte-level BPE that merges `>` with what follows it, exactly as the Olmo vocabularies
merge `>Ċ`, `>ĊĊ` and `></`.
"""

import dataclasses
import tempfile
import unittest

import torch
from parameterized import parameterized
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from open_instruct import dataset_transformation, model_utils

THINK_TAGS = ["<think>", "</think>"]

# Merges over the byte-level alphabet, in rank order. The first is the one that makes a
# reasoning turn starting on the line after the tag unreachable from a prompt ending in `>`;
# the last two do the same for an empty `<think></think>` block.
MERGES = [(">", "Ċ"), (">", "<"), ("><", "/")]

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|im_start|>", "<|im_end|>"]

THINK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>' }}{% endif %}"
)


def build_tokenizer(num_reserved_slots: int = 4, chat_template: str = THINK_CHAT_TEMPLATE):
    """A byte-level BPE with `>`-merges, four special tokens and N reserved slots."""
    vocab = {char: index for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for left, right in MERGES:
        vocab[left + right] = len(vocab)
    reserved = [f"<|extra_id_{index}|>" for index in range(num_reserved_slots)]
    # Reserved slots live in the BPE vocab *and* in added_tokens, as they do in the real
    # Olmo tokenizers; promotion has to keep the two agreeing on the id.
    for content in SPECIAL_TOKENS + reserved:
        vocab[content] = len(vocab)

    backend = Tokenizer(models.BPE(vocab=vocab, merges=MERGES))
    # use_regex=False keeps `>` and the newline in one pre-token, so the merge can apply. The
    # real cl100k regex does the same; the plain ByteLevel one does not (see #1896).
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens([AddedToken(content, normalized=False, special=True) for content in SPECIAL_TOKENS])
    backend.add_tokens([AddedToken(content, normalized=False, special=False) for content in reserved])

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|endoftext|>", pad_token="<|pad|>", unk_token=None
    )
    tokenizer.chat_template = chat_template
    return tokenizer


def is_token_prefix(tokenizer, prompt: str, continuation: str) -> bool:
    """Whether `continuation` is reachable from a model that has just emitted `prompt`.

    It is reachable only if tokenizing the two together leaves the prompt's tokens untouched.
    If the last prompt token merges with the start of the continuation, the model is never in
    the state the prompt puts it in, and no labelling scheme can repair that.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = tokenizer.encode(prompt + continuation, add_special_tokens=False)
    return full_ids[: len(prompt_ids)] == prompt_ids


# Openers taken from the Dolci-Think distribution measured in #1869, plus the empty think block
# the olmo35 template renders for every non-reasoning turn.
CONTINUATIONS = [" Okay, so", "\nOkay, so", "\n\nOkay", "Okay", "</think>The answer", "\n</think>\n\nThe answer"]

GENERATION_PROMPT = "<|im_start|>assistant\n<think>"


class TestReachabilityWithoutPromotion(unittest.TestCase):
    """The failure this change exists to remove, reproduced on the synthetic vocabulary."""

    def setUp(self):
        self.tokenizer = build_tokenizer()

    def test_think_tags_are_multi_token(self):
        for tag in THINK_TAGS:
            self.assertGreater(len(self.tokenizer.encode(tag, add_special_tokens=False)), 1)

    @parameterized.expand([("newline", "\nOkay, so"), ("blank_line", "\n\nOkay"), ("empty_block", "</think>The")])
    def test_continuation_is_unreachable(self, _name, continuation):
        self.assertFalse(is_token_prefix(self.tokenizer, GENERATION_PROMPT, continuation))

    @parameterized.expand([("space", " Okay, so"), ("bare_word", "Okay")])
    def test_continuation_is_reachable(self, _name, continuation):
        # Not everything is broken: only continuations whose first character merges with `>`.
        self.assertTrue(is_token_prefix(self.tokenizer, GENERATION_PROMPT, continuation))


class TestPromoteTokensIntoReservedSlots(unittest.TestCase):
    def setUp(self):
        self.tokenizer = build_tokenizer()
        self.vocab_size_before = self.tokenizer.vocab_size
        self.length_before = len(self.tokenizer)

    def promote(self, tokens=None):
        return dataset_transformation.promote_tokens_into_reserved_slots(self.tokenizer, tokens or THINK_TAGS)

    def test_vocabulary_does_not_grow(self):
        self.promote()
        self.assertEqual(self.tokenizer.vocab_size, self.vocab_size_before)
        self.assertEqual(len(self.tokenizer), self.length_before)

    def test_tags_become_single_tokens(self):
        promoted = self.promote()
        self.assertEqual([token.content for token in promoted], THINK_TAGS)
        for token in promoted:
            self.assertEqual(self.tokenizer.encode(token.content, add_special_tokens=False), [token.token_id])

    def test_slots_are_taken_lowest_id_first(self):
        promoted = self.promote()
        slot_ids = sorted(
            token_id
            for token_id, token in build_tokenizer().added_tokens_decoder.items()
            if dataset_transformation.RESERVED_SLOT_RE.fullmatch(token.content)
        )
        self.assertEqual([token.token_id for token in promoted], slot_ids[: len(THINK_TAGS)])

    def test_source_ids_are_the_pre_promotion_pieces(self):
        expected = [tuple(self.tokenizer.encode(tag, add_special_tokens=False)) for tag in THINK_TAGS]
        promoted = self.promote()
        self.assertEqual([token.source_ids for token in promoted], expected)

    @parameterized.expand([(f"case_{index}", text) for index, text in enumerate(CONTINUATIONS)])
    def test_every_continuation_becomes_reachable(self, _name, continuation):
        self.promote()
        self.assertTrue(is_token_prefix(self.tokenizer, GENERATION_PROMPT, continuation))

    def test_promotion_is_idempotent(self):
        first = self.promote()
        second = self.promote()
        self.assertEqual(len(first), len(THINK_TAGS))
        self.assertEqual(second, [])
        self.assertEqual(len(self.tokenizer), self.length_before)

    def test_promoted_tokens_survive_skip_special_tokens(self):
        # vLLM detokenizes with skip_special_tokens=True by default, and the verifiers in
        # ground_truth_utils split rollouts on a literal `</think>`.
        self.promote()
        ids = self.tokenizer.encode("<think>reason</think>answer", add_special_tokens=False)
        self.assertEqual(self.tokenizer.decode(ids, skip_special_tokens=True), "<think>reason</think>answer")

    def test_round_trip_is_unchanged_for_other_text(self):
        text = "<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n"
        before = self.tokenizer.encode(text, add_special_tokens=False)
        self.promote()
        self.assertEqual(self.tokenizer.encode(text, add_special_tokens=False), before)

    def test_promotion_survives_save_and_reload(self):
        # Both SFT paths save the tokenizer next to the checkpoint, so a promoted vocabulary
        # that did not round-trip would ship a model whose tags the tokenizer re-splits.
        promoted = self.promote()
        with tempfile.TemporaryDirectory() as directory:
            self.tokenizer.save_pretrained(directory)
            reloaded = AutoTokenizer.from_pretrained(directory)
        self.assertEqual(len(reloaded), self.length_before)
        for token in promoted:
            self.assertEqual(reloaded.encode(token.content, add_special_tokens=False), [token.token_id])

    def test_slot_used_by_the_chat_template_is_left_alone(self):
        self.tokenizer = build_tokenizer(chat_template=THINK_CHAT_TEMPLATE.replace("<think>", "<|extra_id_0|>"))
        promoted = self.promote()
        self.assertNotIn(self.tokenizer.convert_tokens_to_ids("<|extra_id_0|>"), [t.token_id for t in promoted])

    def test_raises_when_slots_run_out(self):
        self.tokenizer = build_tokenizer(num_reserved_slots=1)
        with self.assertRaisesRegex(ValueError, "1 unused reserved slots but 2 are needed"):
            self.promote()

    def test_raises_when_there_are_no_slots(self):
        self.tokenizer = build_tokenizer(num_reserved_slots=0)
        with self.assertRaisesRegex(ValueError, "0 unused reserved slots"):
            self.promote()


class TestTokenizerConfigWiring(unittest.TestCase):
    """`--reserved_slot_tokens` has to reach the tokenizer, and only then change the cache key."""

    @classmethod
    def setUpClass(cls):
        cls._directory = tempfile.TemporaryDirectory()
        build_tokenizer().save_pretrained(cls._directory.name)
        cls.path = cls._directory.name

    @classmethod
    def tearDownClass(cls):
        cls._directory.cleanup()

    def make_config(self, **kwargs):
        return dataset_transformation.TokenizerConfig(tokenizer_name_or_path=self.path, **kwargs)

    def test_unset_leaves_the_tags_multi_token(self):
        tokenizer = self.make_config().tokenizer
        self.assertEqual(tokenizer.promoted_reserved_slot_tokens, [])
        self.assertGreater(len(tokenizer.encode("<think>", add_special_tokens=False)), 1)

    def test_set_promotes_the_tags(self):
        tokenizer = self.make_config(reserved_slot_tokens=THINK_TAGS).tokenizer
        self.assertEqual([token.content for token in tokenizer.promoted_reserved_slot_tokens], THINK_TAGS)
        for tag in THINK_TAGS:
            self.assertEqual(len(tokenizer.encode(tag, add_special_tokens=False)), 1)

    def test_cache_hash_ignores_the_field_when_unset(self):
        # A new field must not invalidate every existing dataset cache, so it defaults to None,
        # which is the value `compute_config_hash` drops.
        config = self.make_config()
        self.assertIsNone(config.reserved_slot_tokens)
        hashed = {key: value for key, value in dataclasses.asdict(config).items() if value is not None}
        self.assertNotIn("reserved_slot_tokens", hashed)

    def test_cache_hash_changes_when_set(self):
        plain = dataset_transformation.compute_config_hash([], self.make_config())
        promoted = dataset_transformation.compute_config_hash([], self.make_config(reserved_slot_tokens=THINK_TAGS))
        self.assertNotEqual(plain, promoted)


class TinyModel(torch.nn.Module):
    """The narrowest thing `initialize_promoted_token_embeddings` accepts."""

    def __init__(self, vocab_size: int, hidden_size: int = 8, tie_weights: bool = False):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        torch.nn.init.normal_(self.embed.weight)
        if not tie_weights:
            torch.nn.init.normal_(self.lm_head.weight)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


class TestInitializePromotedTokenEmbeddings(unittest.TestCase):
    def setUp(self):
        self.tokenizer = build_tokenizer()
        self.tokenizer.promoted_reserved_slot_tokens = dataset_transformation.promote_tokens_into_reserved_slots(
            self.tokenizer, THINK_TAGS
        )
        self.model = TinyModel(vocab_size=len(self.tokenizer))

    def test_rows_are_set_to_the_mean_of_their_pieces(self):
        expected = {
            token.token_id: self.model.embed.weight[list(token.source_ids)].mean(dim=0).clone()
            for token in self.tokenizer.promoted_reserved_slot_tokens
        }
        written = model_utils.initialize_promoted_token_embeddings(self.model, self.tokenizer)
        self.assertEqual(written, len(THINK_TAGS))
        for token_id, row in expected.items():
            torch.testing.assert_close(self.model.embed.weight[token_id], row)

    def test_output_head_is_initialized_too(self):
        expected = {
            token.token_id: self.model.lm_head.weight[list(token.source_ids)].mean(dim=0).clone()
            for token in self.tokenizer.promoted_reserved_slot_tokens
        }
        model_utils.initialize_promoted_token_embeddings(self.model, self.tokenizer)
        for token_id, row in expected.items():
            torch.testing.assert_close(self.model.lm_head.weight[token_id], row)

    def test_tied_weights_are_written_once(self):
        model = TinyModel(vocab_size=len(self.tokenizer), tie_weights=True)
        token = self.tokenizer.promoted_reserved_slot_tokens[0]
        expected = model.embed.weight[list(token.source_ids)].mean(dim=0).clone()
        model_utils.initialize_promoted_token_embeddings(model, self.tokenizer)
        torch.testing.assert_close(model.embed.weight[token.token_id], expected)

    def test_other_rows_are_untouched(self):
        promoted_ids = {token.token_id for token in self.tokenizer.promoted_reserved_slot_tokens}
        before = self.model.embed.weight.clone()
        model_utils.initialize_promoted_token_embeddings(self.model, self.tokenizer)
        for token_id in range(len(self.tokenizer)):
            if token_id not in promoted_ids:
                torch.testing.assert_close(self.model.embed.weight[token_id], before[token_id])

    def test_no_promoted_tokens_is_a_noop(self):
        before = self.model.embed.weight.clone()
        self.assertEqual(model_utils.initialize_promoted_token_embeddings(self.model, build_tokenizer()), 0)
        torch.testing.assert_close(self.model.embed.weight, before)

    def test_raises_when_the_slot_is_outside_the_embedding_matrix(self):
        model = TinyModel(vocab_size=4)
        with self.assertRaisesRegex(ValueError, "embedding matrix has only 4 rows"):
            model_utils.initialize_promoted_token_embeddings(model, self.tokenizer)


if __name__ == "__main__":
    unittest.main()
