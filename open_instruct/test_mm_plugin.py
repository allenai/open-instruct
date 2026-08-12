"""Tests for multimodal SFT support.

These deliberately avoid downloading a VLM checkpoint: the image processor is constructed
directly, which is all the plugin needs to compute image token counts.
"""

import os
import pickle
import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace

from PIL import Image
from transformers import Qwen2VLImageProcessor

from open_instruct import dataset_transformation, mm_plugin
from open_instruct.dataset_transformation import _resolve_media_paths


def _make_processor(image_max_pixels=None, image_min_pixels=None):
    processor = SimpleNamespace(image_processor=Qwen2VLImageProcessor(), image_token="<|image_pad|>")
    return mm_plugin.attach_processor_config(processor, image_max_pixels, image_min_pixels)


class TestQwen2VLPlugin(unittest.TestCase):
    def setUp(self):
        self.processor = _make_processor()
        self.plugin = mm_plugin.get_mm_plugin("qwen2_5_vl", processor=self.processor)

    def test_token_count_matches_image_processor(self):
        """The count we bake into input_ids must equal what the vision tower actually emits.

        This is the invariant the whole design rests on: if it drifts, training silently
        misaligns image features against their placeholders.
        """
        merge_length = self.processor.image_processor.merge_size**2
        for size in [(64, 64), (224, 224), (256, 256), (640, 480), (1024, 768), (333, 977)]:
            with self.subTest(size=size):
                images = self.plugin.regularize_images([Image.new("RGB", size)], self.processor)
                counts = self.plugin.image_token_counts(images, self.processor)
                grid = self.processor.image_processor(images, return_tensors="pt")["image_grid_thw"]
                expected = [int(g.prod()) // merge_length for g in grid]
                self.assertEqual(counts, expected)

    def test_process_messages_expands_every_placeholder(self):
        messages = [
            {"role": "user", "content": f"{mm_plugin.IMAGE_PLACEHOLDER}what is this?"},
            {"role": "assistant", "content": "a square"},
            {"role": "user", "content": f"and this? {mm_plugin.IMAGE_PLACEHOLDER}"},
        ]
        images = [Image.new("RGB", (224, 224)), Image.new("RGB", (448, 224))]
        out = self.plugin.process_messages(messages, images, self.processor)

        self.assertNotIn(mm_plugin.IMAGE_PLACEHOLDER, "".join(m["content"] for m in out))
        regularized = self.plugin.regularize_images(images, self.processor)
        counts = self.plugin.image_token_counts(regularized, self.processor)
        total = sum(m["content"].count(self.plugin.image_token) for m in out)
        self.assertEqual(total, sum(counts))
        # Each block is delimited, and the assistant turn is untouched.
        self.assertEqual(out[0]["content"].count(self.plugin.vision_bos_token), 1)
        self.assertEqual(out[1]["content"], "a square")

    def test_placeholder_image_count_mismatch_raises(self):
        messages = [{"role": "user", "content": mm_plugin.IMAGE_PLACEHOLDER}]
        with self.assertRaises(ValueError):
            self.plugin.process_messages(messages, [Image.new("RGB", (64, 64))] * 2, self.processor)
        with self.assertRaises(ValueError):
            self.plugin.process_messages(
                [{"role": "user", "content": "no image"}], [Image.new("RGB", (64, 64))], self.processor
            )

    def test_regularize_images_applies_pixel_budget(self):
        processor = _make_processor(image_max_pixels=128 * 128, image_min_pixels=32 * 32)
        plugin = mm_plugin.get_mm_plugin("qwen2_5_vl", processor=processor)

        (big,) = plugin.regularize_images([Image.new("RGB", (2048, 2048))], processor)
        self.assertLessEqual(big.width * big.height, 128 * 128)

        (small,) = plugin.regularize_images([Image.new("RGB", (8, 8))], processor)
        self.assertGreaterEqual(small.width * small.height, 32 * 32)

        (grayscale,) = plugin.regularize_images([Image.new("L", (64, 64))], processor)
        self.assertEqual(grayscale.mode, "RGB")

    def test_pixel_budget_changes_token_count(self):
        """Guards the cache-key requirement: image_max_pixels genuinely changes tokenization."""
        image = Image.new("RGB", (1024, 1024))
        counts = []
        for max_pixels in [256 * 256, 768 * 768]:
            processor = _make_processor(image_max_pixels=max_pixels)
            plugin = mm_plugin.get_mm_plugin("qwen2_5_vl", processor=processor)
            regularized = plugin.regularize_images([image], processor)
            counts.append(plugin.image_token_counts(regularized, processor)[0])
        self.assertNotEqual(counts[0], counts[1])

    def test_pixel_budget_survives_pickling(self):
        """`dataset.map(num_proc>1)` pickles the processor to each worker.

        If the budget were lost in transit, workers would tokenize at a different resolution than
        the collator later renders at, and image tokens would silently stop matching features.
        """
        processor = _make_processor(image_max_pixels=123 * 123, image_min_pixels=17 * 17)
        restored = pickle.loads(pickle.dumps(processor))
        self.assertEqual(restored.image_max_pixels, 123 * 123)
        self.assertEqual(restored.image_min_pixels, 17 * 17)

    def test_registry_and_detection(self):
        self.assertTrue(mm_plugin.is_multimodal_model_type("qwen2_5_vl"))
        self.assertFalse(mm_plugin.is_multimodal_model_type("llama"))
        self.assertIsNone(mm_plugin.get_mm_plugin("llama"))

    def test_every_registered_family_has_module_keys(self):
        """A plugin without a COMPOSITE_MODULES entry silently freezes nothing and filters no
        LoRA targets, so the two registries must stay in step."""
        self.assertEqual(sorted(mm_plugin.MM_PLUGIN_REGISTRY), sorted(mm_plugin.COMPOSITE_MODULES))

    def test_qwen3_5_uses_the_qwen2_vl_image_path(self):
        """Qwen3.5's image branch is identical to Qwen2-VL's; only the video branch differs."""
        self.assertTrue(mm_plugin.is_multimodal_model_type("qwen3_5"))
        plugin = mm_plugin.get_mm_plugin("qwen3_5")
        self.assertIsInstance(plugin, mm_plugin.Qwen2VLPlugin)
        keys = mm_plugin.COMPOSITE_MODULES["qwen3_5"]
        self.assertIn("visual.blocks", keys.vision_model_keys)
        self.assertEqual(keys.projector_keys, ["visual.merger"])

    def test_image_token_taken_from_processor(self):
        processor = _make_processor()
        processor.image_token = "<|custom_pad|>"
        plugin = mm_plugin.get_mm_plugin("qwen2_5_vl", processor=processor)
        self.assertEqual(plugin.image_token, "<|custom_pad|>")


class TestFreezingAndLora(unittest.TestCase):
    def test_freeze_vision_tower_keeps_projector_trainable(self):
        """Qwen nests the projector inside the vision tower; freezing must not catch it."""
        keys = mm_plugin.get_frozen_module_keys(
            "qwen2_5_vl", freeze_vision_tower=True, freeze_multi_modal_projector=False, freeze_language_model=False
        )
        self.assertTrue(any("visual.blocks" in k for k in keys))
        self.assertFalse(any("merger" in k for k in keys))

    def test_freeze_projector_included_when_requested(self):
        keys = mm_plugin.get_frozen_module_keys(
            "qwen2_5_vl", freeze_vision_tower=False, freeze_multi_modal_projector=True, freeze_language_model=False
        )
        self.assertEqual(keys, ["visual.merger"])

    def test_unknown_model_type_freezes_nothing(self):
        self.assertEqual(mm_plugin.get_frozen_module_keys("llama", True, True, True), [])

    def test_lora_targets_exclude_vision_tower(self):
        """The default LoRA target names match vision-tower MLPs too; they must be filtered out."""
        module_names = [
            "model.language_model.layers.0.self_attn.q_proj",
            "model.language_model.layers.0.mlp.gate_proj",
            "model.visual.blocks.0.mlp.gate_proj",
            "model.visual.blocks.0.attn.proj",
            "model.visual.patch_embed.proj",
            "model.visual.merger.mlp.0",
        ]
        model = SimpleNamespace(named_modules=lambda: [(n, None) for n in module_names])
        frozen = mm_plugin.get_frozen_module_keys("qwen2_5_vl", True, False, False)

        targets = mm_plugin.filter_lora_target_modules(
            model, "qwen2_5_vl", ["q_proj", "gate_proj", "up_proj", "down_proj"], frozen
        )
        self.assertIn("model.language_model.layers.0.self_attn.q_proj", targets)
        self.assertIn("model.language_model.layers.0.mlp.gate_proj", targets)
        self.assertFalse([t for t in targets if "visual" in t])


class TestCacheKeys(unittest.TestCase):
    """The multimodal fields must not disturb existing text-only dataset caches."""

    def _tc(self, **kwargs):
        tc = dataset_transformation.TokenizerConfig(
            tokenizer_name_or_path="dummy/tokenizer", tokenizer_revision="main", chat_template_name="tulu", **kwargs
        )
        # `tokenizer` is a cached_property and is only consulted for the chat template here;
        # priming the cache keeps this test independent of any downloadable checkpoint.
        tc.__dict__["tokenizer"] = SimpleNamespace(chat_template="{{ messages }}")
        return tc

    def test_none_valued_multimodal_fields_are_not_hashed(self):
        """Adding these fields must not silently invalidate every cached text dataset."""
        tc = self._tc()
        hashed = {k: v for k, v in asdict(tc).items() if v is not None}
        for field_name in ["processor_name_or_path", "image_max_pixels", "image_min_pixels"]:
            self.assertIsNone(getattr(tc, field_name))
            self.assertNotIn(field_name, hashed)

    def test_image_max_pixels_changes_the_cache_key(self):
        """It changes the number of image tokens, so it must change the key."""
        base = dataset_transformation.compute_config_hash([], self._tc())
        smaller = dataset_transformation.compute_config_hash([], self._tc(image_max_pixels=256 * 256))
        bigger = dataset_transformation.compute_config_hash([], self._tc(image_max_pixels=768 * 768))
        self.assertNotEqual(base, smaller)
        self.assertNotEqual(smaller, bigger)


class TestMmTokenTypeIds(unittest.TestCase):
    """Qwen3.5 refuses to compute M-RoPE without a per-token modality map.

    We build input_ids with the tokenizer rather than the full processor, so nothing produces this
    for free -- the collator has to.
    """

    def test_qwen3_5_requires_it_and_qwen2_5_vl_does_not(self):
        self.assertTrue(mm_plugin.get_mm_plugin("qwen3_5").requires_mm_token_type_ids)
        self.assertFalse(mm_plugin.get_mm_plugin("qwen2_5_vl").requires_mm_token_type_ids)

    def test_built_from_processor_when_available(self):
        """Prefer the processor's own implementation so values track the transformers version."""
        import torch  # noqa: PLC0415

        plugin = mm_plugin.get_mm_plugin("qwen3_5")
        processor = SimpleNamespace(create_mm_token_type_ids=lambda ids: [[0, 1, 1, 0] for _ in ids])
        out = plugin.build_mm_token_type_ids(torch.tensor([[5, 9, 9, 5]]), processor, None)
        self.assertEqual(out.tolist(), [[0, 1, 1, 0]])

    def test_falls_back_to_marking_the_image_token(self):
        import torch  # noqa: PLC0415

        plugin = mm_plugin.get_mm_plugin("qwen3_5")
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda t: 9)
        out = plugin.build_mm_token_type_ids(torch.tensor([[5, 9, 9, 5]]), SimpleNamespace(), tokenizer)
        self.assertEqual(out.tolist(), [[0, 1, 1, 0]])


class TestVisionGradientCheckpointing(unittest.TestCase):
    """A frozen vision tower must not be activation-checkpointed under ZeRO-3.

    ZeRO-3 releases frozen params after forward and never re-gathers them, so recomputation sees
    shape-[0] tensors and torch raises CheckpointError.
    """

    def _model(self):
        """Mirror the real tree: transformers sets the flag on the vision root AND every block."""
        modules = [("model.visual", SimpleNamespace(gradient_checkpointing=True))]
        modules += [(f"model.visual.blocks.{i}", SimpleNamespace(gradient_checkpointing=True)) for i in range(3)]
        modules += [("model.visual.merger", SimpleNamespace(gradient_checkpointing=True))]
        modules += [("model.language_model", SimpleNamespace(gradient_checkpointing=True))]
        modules += [
            (f"model.language_model.layers.{i}", SimpleNamespace(gradient_checkpointing=True)) for i in range(3)
        ]
        return SimpleNamespace(named_modules=lambda: modules), dict(modules)

    def test_disables_every_vision_module_including_nested_blocks(self):
        model, by_name = self._model()
        disabled = mm_plugin.disable_vision_gradient_checkpointing(model, "qwen2_5_vl")

        # The blocks are where recomputation actually happens; disabling only the root is not enough.
        self.assertIn("model.visual", disabled)
        for i in range(3):
            self.assertIn(f"model.visual.blocks.{i}", disabled)
        self.assertFalse(any(by_name[n].gradient_checkpointing for n in by_name if "visual" in n))

    def test_language_model_checkpointing_preserved(self):
        model, by_name = self._model()
        mm_plugin.disable_vision_gradient_checkpointing(model, "qwen2_5_vl")
        self.assertTrue(by_name["model.language_model"].gradient_checkpointing)
        for i in range(3):
            self.assertTrue(by_name[f"model.language_model.layers.{i}"].gradient_checkpointing)

    def test_unknown_model_type_is_a_noop(self):
        model, by_name = self._model()
        self.assertEqual(mm_plugin.disable_vision_gradient_checkpointing(model, "llama"), [])
        self.assertTrue(by_name["model.visual"].gradient_checkpointing)


class TestVisualConfig(unittest.TestCase):
    def test_hidden_size_hoisted_from_text_config(self):
        """DeepSpeed ZeRO-3 reads model.config.hidden_size and errors if it is absent."""
        config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=3584))
        mm_plugin.configure_visual_config(config)
        self.assertEqual(config.hidden_size, 3584)

    def test_existing_hidden_size_not_overwritten(self):
        config = SimpleNamespace(hidden_size=111, text_config=SimpleNamespace(hidden_size=3584))
        mm_plugin.configure_visual_config(config)
        self.assertEqual(config.hidden_size, 111)

    def test_text_only_config_untouched(self):
        config = SimpleNamespace(hidden_size=4096)
        mm_plugin.configure_visual_config(config)
        self.assertEqual(config.hidden_size, 4096)


class TestArgumentParsing(unittest.TestCase):
    """`FlatArguments` and `TokenizerConfig` share one argparse parser.

    A field defined on both raises `conflicting option strings` at startup — which only shows up
    when the script is actually invoked, so unit tests that build the dataclasses directly miss it.
    """

    def test_multimodal_flags_parse(self):
        from open_instruct.finetune import FlatArguments  # noqa: PLC0415
        from open_instruct.utils import ArgumentParserPlus  # noqa: PLC0415

        parser = ArgumentParserPlus((FlatArguments, dataset_transformation.TokenizerConfig))
        args, tc = parser.parse_args_into_dataclasses(
            [
                "--model_name_or_path",
                "some/vlm",
                "--tokenizer_name_or_path",
                "some/vlm",
                "--dataset_mixer_list",
                "some/dataset",
                "1.0",
                "--image_max_pixels",
                "589824",
                "--image_min_pixels",
                "1024",
                "--media_dir",
                "/tmp/media",
                "--freeze_vision_tower",
            ]
        )
        # The pixel budget belongs to the tokenizer config, since it changes the cache key.
        self.assertEqual(tc.image_max_pixels, 589824)
        self.assertEqual(tc.image_min_pixels, 1024)
        self.assertEqual(args.media_dir, "/tmp/media")
        self.assertTrue(args.freeze_vision_tower)
        self.assertFalse(hasattr(args, "image_max_pixels"))


class TestCheckpointKeyValidation(unittest.TestCase):
    """transformers' reverse key mapping corrupts Qwen3.5 checkpoints on save.

    It emits e.g. `model.language_model.language_model.language_model.layers.0...` and folds the
    vision tower under the language-model prefix, so nothing can load the result.
    """

    def test_detects_repeated_path_component(self):
        from open_instruct.model_utils import _has_mangled_keys  # noqa: PLC0415

        self.assertTrue(_has_mangled_keys(["model.language_model.language_model.layers.0.x"]))
        self.assertTrue(_has_mangled_keys(["a.b.b.c"]))

    def test_leaves_valid_layouts_alone(self):
        from open_instruct.model_utils import _has_mangled_keys  # noqa: PLC0415

        # Both the native and the legacy layouts must pass untouched.
        self.assertFalse(
            _has_mangled_keys(
                [
                    "model.language_model.layers.0.mlp.up_proj.weight",
                    "model.visual.blocks.0.attn.qkv.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                    "lm_head.weight",
                ]
            )
        )


class TestMissingWeightsGuard(unittest.TestCase):
    """A checkpoint that silently fails to load trains from noise with a healthy-looking curve."""

    def test_raises_on_missing_keys(self):
        from open_instruct.finetune import _check_checkpoint_fully_loaded  # noqa: PLC0415

        args = SimpleNamespace(allow_missing_checkpoint_keys=False)
        with self.assertRaises(RuntimeError) as ctx:
            _check_checkpoint_fully_loaded({"missing_keys": ["model.language_model.layers.0.x"]}, args)
        self.assertIn("ZeRO-2", str(ctx.exception))

    def test_passes_when_nothing_missing(self):
        from open_instruct.finetune import _check_checkpoint_fully_loaded  # noqa: PLC0415

        args = SimpleNamespace(allow_missing_checkpoint_keys=False)
        _check_checkpoint_fully_loaded({"missing_keys": []}, args)
        _check_checkpoint_fully_loaded({}, args)


class TestMediaPathResolution(unittest.TestCase):
    def test_relative_paths_resolved_against_media_dir(self):
        with tempfile.TemporaryDirectory() as media_dir:
            os.makedirs(os.path.join(media_dir, "sub"))
            rel = os.path.join("sub", "a.png")
            Image.new("RGB", (8, 8)).save(os.path.join(media_dir, rel))

            self.assertEqual(_resolve_media_paths([rel], media_dir), [os.path.join(media_dir, rel)])
            # Missing files pass through untouched so the error names the original path.
            self.assertEqual(_resolve_media_paths(["nope.png"], media_dir), ["nope.png"])
            # Absolute paths are never rewritten.
            self.assertEqual(_resolve_media_paths(["/abs/b.png"], media_dir), ["/abs/b.png"])
            # No media_dir configured is a no-op.
            self.assertEqual(_resolve_media_paths([rel], None), [rel])


if __name__ == "__main__":
    unittest.main()
