# Experimental adaptive data scheduling

This branch implements an ADS-inspired sampler for the frozen H121 atomic-token
control pool. It is an exploratory 100-update comparison, not a reproduction of
the ADS paper's reported results or a full Dolci mixture experiment.

The initial policy embeds every candidate once. `scripts/data/ads/extract.py`
concatenates the training query, two newlines, and its final reference answer;
it mean-pools the final post-normalization hidden states and computes mean answer
token negative log likelihood. This deliberately uses final answers consistently:
H121's 4,000 MATH records do not contain worked solutions. The original ADS work
uses reference solutions. Representation quality and difficulty rankings may
therefore differ. Training inputs and labels are never changed by this step.

`cluster.py` L2-normalizes vectors, fits randomized PCA with 128 components, then
64-cluster KMeans (`n_init="auto"`, seed 42). The output table contains stable
prepared prompt identities, row indices, cluster IDs and difficulty values.
Vectors, PCA parameters, centers, input fingerprints and extraction timings remain
alongside the table on WEKA. Runtime verifies its SHA-256 and exact coverage of
the filtered MILES training pool; mismatches fail before rollout.

Enable with an absolute `core.ads_table`, its `core.ads_sha256`,
`core.ads_mini_size=8` and `core.ads_active_clusters=4`. The pilot supports fully
async refresh publication with retry handling, binary rewards, and no frozen
exclusion table. It leaves the native reward/filter, optimizer, and policy-lag
logic intact.

Four distinct clusters are selected using smoothed success-rate weights. Each
contributes its current eight candidates (or all candidates when smaller than
eight). Candidates start at the easiest ranks. After every selected group in a
round has been scored, success below .33 moves a candidate toward an easier
neighbor; above .67 moves it toward a harder neighbor; other candidates stay.
Collision resolution preserves distinct candidates. Cluster weights approach
normalized latest observed cluster success rates with smoothing alpha .3.

Feedback is captured before zero-variance filtering. This preserves the learning
signal from all-wrong and all-right groups, although those groups still do not
enter the GRPO gradient batch. Aborted or missing-reward groups do not provide
curriculum feedback. Asynchronous prefetch means selection rounds and optimizer
updates are different clocks. Reports must distinguish dispatched groups,
completed feedback rounds, and accepted training batches.

The schedule resets after ceil(pool size / 32) completed selection rounds.
Already outstanding work can finish after a reset, but its feedback cannot move
the new epoch's candidates. The release-inspired implementation does not disable
a cluster after visiting every example, and does not implement graph traversal.
Robert's proposed graph of easier/harder and other relationships remains a
separate design direction; this pilot provides frozen representations to explore it.

Sampler state is committed atomically with the native prompt cursor and pristine
pending groups. It includes RNG state, queued selections, incomplete feedback
rounds, cluster probabilities and candidate positions. Completed feedback is
marked in pending prompts so regenerated work after preemption cannot update the
schedule twice. Resuming with a different table or sampler geometry is rejected.

The historical comparison is H121 plain seed17 retry2:
https://beaker.org/ex/01M5P40BR7BZ479JT9NAF4TQ4W . Both arms start at the H077
atomic-safe step11768 4T non-EMO SFT export (roughly 794M active / 12.5B total
parameters). They use the same 6,000 prompts, 32 accepted groups per update,
16 responses per group and 100 optimizer updates. The primary readout is the
matched final bare/cued MATH500 and GSM8K evaluation, eight draws per question.
Also report generated tokens, group filtering, coverage and GPU time including
embedding preparation. Historical one-seed comparisons do not measure seed
variance or establish an efficiency win merely by completing equal updates.

References: [ADS paper](https://arxiv.org/html/2606.22305v2) and
[released implementation](https://github.com/Richard-zrx/ADS/tree/8a32745d58f7b7f57baae51c2cd69f7b63cecdae).
