# SLR-Bench: Scalable Logical Reasoning Benchmark

[SLR-Bench](https://huggingface.co/datasets/AIML-TUDA/SLR-Bench) is a scalable benchmark for improving logical reasoning in LLMs via **structured curricula and verifiable rewards**.

## Why SLR-Bench?

Logical reasoning is a core capability for language models, yet it is difficult to train and evaluate reliably. SLR-Bench provides:

- **Structured curricula** — 20 difficulty levels (basic → hard) so models learn reasoning incrementally.
- **Verifiable rewards** — every task comes with a deterministic symbolic verifier that scores model outputs automatically, enabling RLVR training with precise, partial-credit feedback.
- **Scalable task generation** — over 19,000 tasks with controllable complexity; new tasks can be synthesised automatically.

Each task asks the model to discover a general rule that correctly classifies a set of labelled examples. Solutions are verified by executing them as logic programs, giving exact correctness signals rather than relying on LLM judges or exact string matching.

For more details see the [paper (arXiv:2506.15787)](https://arxiv.org/abs/2506.15787) and the [SLR framework on GitHub](https://github.com/ml-research/ScalableLogicalReasoning).

## Dataset

Hosted on Hugging Face: [`AIML-TUDA/SLR-Bench`](https://huggingface.co/datasets/AIML-TUDA/SLR-Bench)

Use the colon separator syntax to specify a config:

```
AIML-TUDA/SLR-Bench:v1-All
```

### Requirements

- **SWI-Prolog** (`swipl`) must be installed and available on `$PATH`. The Docker image includes it by default.
- No additional Python packages are required beyond core dependencies.

## Verifier

The `SLRBenchVerifier` evaluates model outputs by:

1. **Extracting** the predicted rule (supports `[RULE]...[/RULE]` tags, fenced code blocks, etc.).
2. **Executing** it against the task's validation program via SWI-Prolog.
3. **Scoring** with partial credit (fraction of examples classified correctly) plus a simplicity bonus.

Evaluation uses isomorphic matching (constant names are normalised), so the model must learn the underlying logical structure.

## Usage with GRPO

```bash
python open_instruct/grpo_fast.py \
    --dataset_mixer_list "AIML-TUDA/SLR-Bench:v1-All" 1.0 \
    --dataset_transform_fn slr_bench_prepare_v1 rlvr_tokenize_v1 rlvr_max_length_filter_v1 \
    --max_token_length 4096 \
    --max_prompt_token_length 2048 \
    ...
```

See [scripts/train/slr/example.sh](https://github.com/allenai/open-instruct/blob/main/scripts/train/slr/example.sh) for a complete example.
