# OLMo 3

For details on reproducing OLMo 3 models, see the [OLMo 3 training scripts README](https://github.com/allenai/open-instruct/blob/main/scripts/train/olmo3/README.md).

## Tokenizer Settings

When releasing multiple models (instruct and think) and using two different codebases for post-training (allenai/olmo-core for SFT and allenai/open-instruct for DPO and RL), there are many steps needed to get exact chat templates right. The final step is getting the chat templates right for public release, which can entail different system prompts to maintain model identity.

This document is a reference for the settings used for Olmo 3, based on the best available information.

**Olmo 3 Instruct Models ([7b](https://huggingface.co/allenai/Olmo-3-7B-Instruct), [32b](https://huggingface.co/allenai/Olmo-3.1-32B-Instruct)):** Tokenized AND intermediately evaluated with [`allenai/olmo-3-tokenizer-instruct-dev`](https://huggingface.co/allenai/olmo-3-tokenizer-instruct-dev), released with:

- 7b instruct: the above tokenizer on HuggingFace (no Olmo identity modified system prompt, appropriate tool use chat template/special tokens), the above tokenizer and Olmo identity system prompt modification for demos such as Ai2's playground.
- 32b instruct: the above tokenizer on HuggingFace for SFT and DPO, and for RL/Playground the final model has the chat template with modified Olmo system prompt on playground.
- Reason for the above discrepancy: We needed to match the chat templates for the final models relative to what was used in demos, which in the case for these models involved system prompt edits to improve model reliability. Matching models in demos and on HuggingFace is ideal.

**Olmo 3 Thinking Models:**

- [7b](https://huggingface.co/allenai/Olmo-3-7B-Think): Training data tokenized with the `olmo_thinker_no_think_7b` chat template (that has the olmo identity in the prompt), but there was a minor miscommunication in transition to the next training stages, so the DPO and RL models have a **slightly** different chat template, all reflected in the final released models.
- [32b](https://huggingface.co/allenai/Olmo-3-32B-Think): Training data tokenized with the `olmo_thinker_no_think_sft_tokenization` chat template (otherwise identical, doesn't have olmo identity in the prompt), released with that chat template + the think token in `add_generation_prompt`.
- Reason for the difference between 7b and 32b: we learned as we went to not have the identity baked into the prompt (so it was easier to fix at the time of the demo in the form of a system prompt) but couldn't afford to retrain 7b thinking model at that point.

**Olmo 3.X and future models:**

- **Think SFT data** is tokenized with the Instruct chat template [`allenai/olmo-3-tokenizer-instruct-dev`](https://huggingface.co/allenai/olmo-3-tokenizer-instruct-dev). This template does not include `<think>`, which prevents `<think>` from being masked out during tokenization so the model learns to generate it. (We plan to fix the underlying masking bug so this workaround is no longer needed.)
- **Think evaluation** should use [`allenai/olmo-3-tokenizer-think-dev`](https://huggingface.co/allenai/olmo-3-tokenizer-think-dev), which is the instruct chat template plus `<think>` in `add_generation_prompt` (new models should combine tool use abilities from the instruct template with `<think>` for reasoning). (TODO: check if this tokenizer should be renamed to `olmo-3.X-tokenizer-think-dev` since it includes function calling in the template, which differs from the original OLMo 3 think tokenizers.)
- **Think release models** should use [`allenai/olmo-3-tokenizer-think-release`](https://huggingface.co/allenai/olmo-3-tokenizer-think-release), which is the same as the think-dev template but with the Olmo identity system prompt.
- **Instruct release models** should use [`allenai/olmo-3-tokenizer-instruct-release`](https://huggingface.co/allenai/olmo-3-tokenizer-instruct-release), which is the same as `instruct-dev` but with the Olmo identity system prompt. This is analogous to how `think-release` differs from `think-dev`.

---

There are two main issues that lead to all the floating chat templates: one, the <think> token chopping in the tokenization script where our code incorrectly masks the first <think> token as part of the prompt, and two, the identity issue which means we should train and release with different system prompts.

**TLDR until these two issues are resolved:**

-  [`allenai/olmo-3-tokenizer-instruct-dev`](https://huggingface.co/allenai/olmo-3-tokenizer-instruct-dev) is the primary chat template for tokenizing both instruct and think models that have tool use abilities.
- For Instruct evaluation/training, use [`allenai/olmo-3-tokenizer-instruct-dev`](https://huggingface.co/allenai/olmo-3-tokenizer-instruct-dev). For release, use [`allenai/olmo-3-tokenizer-instruct-release`](https://huggingface.co/allenai/olmo-3-tokenizer-instruct-release) (adds Olmo identity).
- For Think evaluation/training, use [`allenai/olmo-3-tokenizer-think-dev`](https://huggingface.co/allenai/olmo-3-tokenizer-think-dev) (adds `<think>` to `add_generation_prompt`). For release, use [`allenai/olmo-3-tokenizer-think-release`](https://huggingface.co/allenai/olmo-3-tokenizer-think-release) (adds Olmo identity).
