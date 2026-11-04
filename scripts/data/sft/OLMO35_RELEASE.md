# Olmo 3.5 release-data preparation

Implements the CPU preparation for [release-data-fixes (#1964)](https://github.com/allenai/open-instruct/issues/1964): exact audited exclusions, replacement identity examples, and a check that the release tokenizer states a real cutoff. It does not retrain a model or establish a knowledge cutoff. **The cutoff is filled upstream in the tokenizer's chat template ([template-cutoff-placeholder (#2171)](https://github.com/allenai/open-instruct/issues/2171)); the final identity dataset remains pending that revision.** Do not use the Dolci creation date as a substitute for training-data provenance.

## Audited exclusions

Run where the audit files are available. `AUDIT_ROOT` is their common parent containing `k3-audit/`. The exporter checks the frozen evidence hashes and counts before writing a manifest; private row lists are not checked into this repository.

```bash
uv run python scripts/data/sft/export_olmo35_audit.py \
  --audit-root "$AUDIT_ROOT" --output "$PREP/audit_manifest.json"

uv run python scripts/data/sft/prepare_olmo35_release_data.py \
  --data-files "$SOURCE"/*.parquet \
  --audit-manifest "$PREP/audit_manifest.json" \
  --output-dir "$PREP/filtered"
```

The manifest is tied to `allenai/Dolci-Think-SFT@7668c638cc84100951973b456069a1a462d6d915`. It contains 1,021 dated-cutoff rows, 58 old identity rows, 21 additional stale-identity rows, 8,323 hidden-safety-prompt rows, and 15,593 Aya wrong-language rows. Reasons overlap; their union has 24,906 IDs. The safety disposition is **drop pending regeneration**, not a claim that regeneration has happened. Other baseline hygiene filters are outside this change.

For an already-filtered input such as C12, add `--allow-missing-ids`. The receipt records absent IDs separately from new drops; it must not count prior cleanup as newly removed data. All input columns and row order are preserved. Input, manifest and output hashes are recorded, and the output directory is published only after validation. Use a fresh directory for every build. Keep the manifest and receipt with the derived dataset in a new private `allenai` HF repository; do not overwrite a published Dolci dataset.

## Cutoff, tokenizer and identity

The cutoff is filled once, upstream, as a new commit of `allenai/dolma2-tokenizer-olmo35` (#2171); nothing here edits a template. `$CUTOFF_REVISION` below is that commit's full SHA. Check it, then build the identity rows from it:

```bash
uv run python scripts/tokenizers/check_olmo35_release.py \
  --tokenizer allenai/dolma2-tokenizer-olmo35 \
  --revision "$CUTOFF_REVISION" --output "$PREP/tokenizer_receipt.json"

uv run python scripts/data/sft/build_olmo35_identity.py \
  --tokenizer allenai/dolma2-tokenizer-olmo35 --revision "$CUTOFF_REVISION" --output "$PREP/identity" \
  --context-length "$VERIFIED_CONTEXT_LENGTH" --context-source "$CONTEXT_SOURCE"
```

The check refuses a revision whose templates still contain `[CUTOFF_DATE]`, and requires the default system prompt to say `knowledge cutoff is <Month YYYY>` exactly once. Its receipt records the revision, the cutoff, the rendered default system prompt, and template and backend hashes. `VERIFIED_CONTEXT_LENGTH` and `CONTEXT_SOURCE` record the exact release checkpoint's supported token count and a pinned model-card/config reference; the issue's original 64K target must not silently carry into a different checkpoint.

The identity bank supplies new concise answers to all 58 audited prompts, with no copied reasoning or citations to hidden system instructions. A canonical hash checks the original IDs and prompts against the audit. The explicit Spanish request gets a Spanish answer; English capability questions get English answers with qualified language claims, consistent with C12's exception for requested languages. Each answer has a no-system-turn variant and an explicit-default-system variant. The default template injects its system turn when none is supplied, so these variants intentionally render identically; they are not two distinct conditioning contexts. Include the 116 replacement rows once in the release mix after excluding the old IDs, and record their mixing weight as two copies of 58 unique prompts.

For SFT use `--tokenizer_name_or_path allenai/dolma2-tokenizer-olmo35 --tokenizer_revision "$CUTOFF_REVISION" --chat_template_name tokenizer_default`. The historical `olmo35` template name is no longer registered. Keep the recipe's existing reserved-slot-token settings. Tokenization hashes include the resolved template, so the filled cutoff produces a new cache key. Data tokenized outside open-instruct, such as MaxText caches, must be rebuilt from the same revision. Do not reuse a forced `--dataset_config_hash` or an old NumPy output directory. The conversion saves the exact resolved tokenizer under the NumPy cache's `tokenizer/` directory; use that directory for subsequent export when reserved slots were promoted during tokenization.

For `scripts/train/convert_olmo_core_to_hf.py`, pass `--tokenizer-name` pointing at that saved training tokenizer. Serving must use the tokenizer saved with the exported checkpoint. A training-only template override is insufficient because this exporter otherwise reloads its own tokenizer. A MILES `hf_template` directory override must likewise use the saved training tokenizer. The cutoff changes only tokenizer assets; preserve the checkpoint's generation configuration and stop-token settings.

## Release checks still required

Run the 32 prompts in `olmo35_cutoff_probe.json` on the actual release candidate after training. Save responses, the model and tokenizer revisions, generation settings, and the cutoff from the tokenizer receipt with the judgments. No probe results are claimed here.

For the safety change, report XSTest benign compliance and harmful refusal separately, WJ-benign compliance, and WildGuardTest harmful compliance/refusal. WJ-harmful supplies a jailbreak regression check. Use explicit distinct seeds for repeated runs; WJ-benign can be near its ceiling. Follow the ledger workflow before launching a comparison meant to establish a model-quality result. The exact-ID cleanup receipt alone does not establish safety or multilingual quality.
