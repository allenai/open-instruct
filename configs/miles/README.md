# MILES run configurations

There are four maintained run examples in [examples/](examples/README.md):
**dev**, **small**, **medium**, and **large**. Copy one into Git-ignored `runs/`
before customizing it. Do not add personal runs, qualification snapshots or sweep
variants to this directory.

Historical configurations were moved locally to
`runs/miles-archive-20260915/configs/miles/`. They also remain in Git history at
`fe4d9f2bd`. Existing Beaker runs retain their submitted configuration and source
revision; this cleanup does not change those jobs.

Tokenizer templates, diagnostic inputs and datasource definitions used by helper
scripts live in `scripts/miles/assets/`; they are not additional run examples.
