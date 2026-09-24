# RL export template

`olmo_3_2_think_dev.jinja` is the unmodified `chat_template` string from
[`allenai/olmo-3.2-tokenizer-think-dev`](https://huggingface.co/allenai/olmo-3.2-tokenizer-think-dev/blob/cf6d9298ed2f5d53f098f1785b58a23a24dff8ed/tokenizer_config.json),
revision `cf6d9298ed2f5d53f098f1785b58a23a24dff8ed`.
SHA-256: `43b0c225dd327d4af450809bfb3abfbd828eb19d6546d3e9ae33782464874d6a`.
It ends generation prompts with `<|im_start|>assistant\n<think>`.
This is temporary for [dolci-think-sft-anchor (#1895)](https://github.com/allenai/open-instruct/issues/1895)
until Kevin Farhat's Olmo 3.5 template is ready. Replace the pinned template and
its rendering tests when migrating; do not change the SFT tokenization recipe.

## Existing HF export (CPU only)

Run from the repository root, targeting the intended RL export directory:

```bash
python -m open_instruct.export_chat_template \
    --checkpoint-dir /path/to/hf_step11768-think \
    --export-chat-template scripts/tokenizers/templates/olmo_3_2_think_dev.jinja
```

This updates an existing directory in place. To preserve the SFT artifact, first
prepare a separate copy. The helper replaces metadata directory entries, so
linked metadata in a sibling export does not modify the source. It does not
load model weights, reserialize the tokenizer, or contact Hugging Face. Exports
with additional named templates are rejected to avoid leaving another active
template behind.
The Jinja template is compiled before any files are changed, using Transformers'
compiler so `{% generation %}` blocks remain supported. Syntax validation does
not guarantee that every conversation shape can be rendered.

## New MoE conversion

`scripts/train/debug/convert_moe_checkpoint_to_hf.py` accepts
`--export-chat-template PATH`. Omit it to retain the existing conversion behavior.
With an override, `--tokenizer` must point to a saved training tokenizer directory
containing `tokenizer.json`, not a Hub ID. NumPy dataset conversion saves this at
`<numpy-cache-directory>/tokenizer/`; use the cache actually used by the SFT run.
Do not substitute the think-dev tokenizer: its pre-tokenizer and post-processor differ.

The converter compiles the template and snapshots the reference `tokenizer.json`
and loaded special-token roles and IDs before conversion. It requires JSON equality
and identical special-token mappings after conversion, before installing the
template or reporting success. The metadata check catches changes such as an EOS
override in `tokenizer_config.json` even when `tokenizer.json` is unchanged.
The backend check catches both reconstruction by Transformers and
OLMo-core preferring `$CKPT_ROOT/tokenizer` over the explicit tokenizer argument.
A mismatch fails the conversion qualification; it does not repair segmentation
or remove the written weights. Resolve the tokenizer mismatch before using that
export for RL. Model/logit parity remains a separate check.

The `oc_sft_olmoe3_kda_think.sh` launcher has a `convert_rl` mode that writes
`$CKPT_ROOT/hf_$STEP-think` and selects this pinned template. It requires
`EXPORT_TOKENIZER` to name the saved training tokenizer directory. Its image must
contain this change. `EXPORT_CHAT_TEMPLATE` can select another in-image or mounted
Jinja file. Both variables also work with ordinary `convert`, whose defaults and
output directory are unchanged. The launcher prints the destination, tokenizer,
template, and timeout before submission so inherited environment overrides are visible.
`CONVERT_TIMEOUT` defaults to an explicitly supplied `JOB_TIMEOUT`, otherwise `2h`,
independently of the gate's `45m` default. Set a longer timeout for CPU conversions,
which can exceed two hours. Launches use the normal image-build/launch workflow
and require the usual per-launch compute approval.

For RL, load the tokenizer from the resulting export and leave
`chat_template_name` unset so rollout prompts and saved RL checkpoints retain
this template. Reload the exported tokenizer and check
`apply_chat_template(..., tokenize=False, add_generation_prompt=True)` before use.
