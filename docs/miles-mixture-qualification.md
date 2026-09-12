# Mixed-source qualification

`scripts.miles.mixture_trials` exercises the existing MILES datasource and per-sample reward dispatcher with GSM8K, math, and legacy IFEval in the same update. Each of two updates contains two independent prompts from each source and four samples per prompt: six prompt groups and 24 responses. Initial and final held-out evaluations contain two prompts per source, one response each.

Prepare the three single-source inputs using the same HF descriptor and explicit chat template. GSM8K accepts the canonical `scripts.miles.prepare_gsm8k_parity` output. Math and legacy IFEval accept `scripts.miles.datasource_trials prepare` outputs. Input preparation manifests must contain immutable prepared train/eval file hashes and either a pinned dataset revision or an explicitly recorded local snapshot hash. Older SFT preparations without those hashes must be regenerated.

```bash
python -m scripts.miles.mixture_trials prepare /path/to/new-mixture \
  --gsm8k-root /path/to/canonical-gsm8k \
  --math-root /path/to/prepared-math \
  --ifeval-root /path/to/prepared-ifeval \
  --hf /path/to/hf --seed 17
python -m scripts.miles.mixture_trials validate /path/to/new-mixture
python -m scripts.miles.mixture_trials run /path/to/new-mixture
python -m scripts.miles.mixture_trials audit /path/to/new-mixture
```

Use `--local` on preparation for the existing resident tiny-model profile; otherwise the harness inherits the SFT EP2 trainer plus one SGLang GPU profile. These commands run inside the qualified MILES/Core environment. Beaker launches must still use the repository's public image/launch wrapper; this tool does not submit jobs.

Selection is deterministic per source, split, original identity, and seed. The materialized files preserve each source row's prompt, target, verifier metadata, and prepared identity, adding a namespaced mixture identity and token/provenance hashes. Identical prompt strings remain separate groups when they have different original identities. Selected train/evaluation prompt overlap is rejected. The manifest records the actual ordered selection, source preparation manifests, and file hashes; runtime validation refuses changed snapshots, templates, registries, or shuffled/retemplated input.

The audit checks all response memberships, group indices, four-sample multiplicity, prompt tokens, unchanged targets, behavior log probabilities, policy versions, two optimizer steps, and the diagnostic publication sequence. Each reward must agree with both direct verifier execution and the reward bridge. Results report reward, mixed-reward groups, response-cap counts, and sample counts separately for every source and evaluation boundary. Two prompts per source measure integration integrity, not learning or source-level accuracy.

Fourteen focused CPU tests passed with the real MILES dataset/grouping implementation and real GSM8K, math, and legacy IFEval verifiers. They include duplicate-prompt identities and adversarial target, source, group, token, version, multiplicity, and reward corruption. The subsequent [three-B300 mixed-source run](https://beaker.org/ex/01M270TQ57VM13CMYAP8AYAZA8) passed two updates (48 training responses) on EP2 plus one TP1 engine, with an 8,192-token response cap and initial/final held-out evaluation. The [retained audit](measurements/miles-mixture-20260910.json) checks source/reward/group contracts. All math training responses reached the cap; this is execution evidence, not a meaningful learning result from six held-out prompts. Code, judged general chat, and production topology remain separate work; see the [Dolci production proposal](miles-dolci-production-proposal.md).
