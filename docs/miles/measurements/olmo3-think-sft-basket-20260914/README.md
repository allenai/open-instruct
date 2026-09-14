# Olmo 3 Think-SFT: 200-update Dolci comparison

This compares the published dense `allenai/Olmo-3-7B-Think-SFT` checkpoint,
revision `6ff857587e040d6d523a3d5f3a56e918f5401d66`, with the ongoing fully
SFT MoE basket. Earlier dense lifecycle qualification used Think-DPO; this
experiment intentionally uses the earlier SFT stage requested by the user.

Configuration: `configs/miles/qualification/olmo3-think-sft-basket-200.toml`.
Launch uses the committed baseline source-overlay launcher, with
`--prepare-module scripts.miles.prepare_olmo3_basket` for the CPU preparation.
Training uses the common frozen-data verifier.

The preparer pins the existing baseline's five JSONL files and reward registry
by SHA256. It preserves row order, training duplicates, rendered prompt text,
labels, verifier targets and held-out identities. It validates every prompt
against the new tokenizer and updates token measurements only. Overlong prompts
or overlapping identities fail preparation; no questions are silently dropped.
The original rendered system prompt is retained for comparison, rather than
applying the SFT checkpoint's default chat template a second time. The receipt
reports changed tokenizations and includes positive/negative code canaries.

Matched settings: 200 updates; 64 prompts × four responses; learning rate 1e-6;
Adam beta2 0.95; clipping 0.2/0.28; no reference KL; response limit 4096;
context 6144; packed training; initial/every-20 evaluation on the same 512
held-out questions; saves at 100 and 200; final HF export; async lag two;
seven inference engines plus the same Qwen3-32B judge on the second node.
This is a short-context comparison, not a reproduction of the original 32K
Olmo 3 RL recipe.

Differences: eight FSDP trainer ranks instead of EP8; no router replay or router
auxiliary loss for this dense model. Dense serving uses a 131072-token KV pool
(approximately 64 GiB) and disables decode graphs and radix caching initially,
following the qualified dense path. Scoring-skip comparison guards remain on.
Packing, async refresh and the sustained nonzero-gradient workload are being
qualified together here; prior dense lifecycle tests alone do not establish
that this combination works. Current source also includes sibling admission,
engine-forward and completion timing; the already-running MoE baseline does not.

Local validation: nine focused tests passed for fixed splits, duplicates,
retokenization, changed-source rejection, overlong-prompt rejection, overlap
rejection, immutable receipt verification and CPU/GPU placement. Ruff passed.
Another 26 dense-model, YaRN and packing tests passed inside the runtime image
(CPU execution; these do not substitute for GPU qualification).

Preparation [01M2FBAC57BPZCP48GMWYKGHR5](https://beaker.org/ex/01M2FBAC57BPZCP48GMWYKGHR5)
finished successfully. All 101434 training rows and 512 held-out rows passed;
counts, verifier counts and ordered held-out identities match the MoE receipt.
All prompts have different tokenization hashes under the Olmo 3 tokenizer;
rendered text is unchanged. Code canaries returned [1, 0, 1, 0].

Training [01M2FBGGE8K8XJ7WJKCTE4KHMB](https://beaker.org/ex/01M2FBGGE8K8XJ7WJKCTE4KHMB)
was submitted at 07:01 UTC on September 14 and both replicas were scheduled.
Source is `59c0700e3`; immutable base image `01M2CJG5RQQ93GEYNYAS7ASCQJ`, with
the committed runtime overlay. The full source and dataset receipts are adjacent.
Training completion and performance remain pending.

Both GPU replicas started at 07:02:55–56 UTC. FlashAttention preflight and
frozen-data verification passed; the managed judge was loading normally.
No optimizer result was available at this startup check.

The concurrent MoE comparator stopped at 06:58:57 UTC after 18 optimizer
updates. The immediate cause was exhausted code-verifier HTTP read-timeout
retries (30 seconds per attempt, eight retries plus backoff), not the fleet
router. Logs show repeated attempts beginning around 06:50. Whether a particular
program/test payload or execution-service load caused this remains unresolved.
The dense run uses the same service, so sustained code verification remains a
qualification risk. No timeout-to-zero reward substitution was introduced.
