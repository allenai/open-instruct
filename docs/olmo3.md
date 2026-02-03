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

- Tokenized with the same Instruct chat template  [`allenai/olmo-3-tokenizer-instruct-dev`](https://huggingface.co/allenai/olmo-3-tokenizer-instruct-dev) (since no <think>, which is our hack for tokenization to make sure the model learns how to generate <think> and it isn't masked out), ideally evaluated with a chat template [here](https://huggingface.co/allenai/dolma2-tokenizer-special-tokens-reasoner-hybrid) that is the instruct chat template + think tokens (because new models should combine the tool use abilities from instruct chat template with the <think> for thinking models). Final models, if successfully trained, should likely release with [this](https://huggingface.co/allenai/dolma2-tokenizer-special-tokens-v5-lc-reasoner) which is the same as the evaluated one but with olmo identity.

---

There are two main issues that lead to all the floating chat templates: one, the <think> token chopping in the tokenization script where our code incorrectly masks the first <think> token as part of the prompt, and two, the identity issue which means we should train and release with different system prompts.

**TLDR until these two issues are resolved:**

-  [`allenai/olmo-3-tokenizer-instruct-dev`](https://huggingface.co/allenai/olmo-3-tokenizer-instruct-dev) is the primary chat template for tokenizing both instruct and think models that have tool use abilities.
- for Instruct, copy that chat template back after training (can be done before or after conversion to HuggingFace format, but confirm it is correct before evaluation and continued training).
- for Think, copy [this](https://huggingface.co/allenai/dolma2-tokenizer-special-tokens-reasoner-hybrid) chat template back after training since we want the <think>. Consider releasing with [this](https://huggingface.co/allenai/dolma2-tokenizer-special-tokens-v5-lc-reasoner) chat template that also includes the "You are Olmo" identity.
