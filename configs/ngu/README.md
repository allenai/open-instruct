# NGU comparison recipes

These comparisons extend the standard
[Qwen3 Deepscaler example](../miles/benchmarks/qwen3_4b_deepscaler.toml) and require
the NGU runtime integration. Data preparation and benchmark settings are documented
in the [benchmark README](../miles/benchmarks/README.md).

`qwen3_4b_base_deepscaler_ngu.toml` uses probabilistic retry (0.875).
`qwen3_4b_base_deepscaler_ngu_budget4_k8n16.toml` uses 16 prompts, eight samples
per attempt and up to four attempts. Both use barrier publication and the
`never_give_up` handler. See [NGU controls and metrics](../../docs/miles/data-and-evaluation.md#never-give-up-and-rejected-groups).
