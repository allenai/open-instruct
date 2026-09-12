# MILES/Core project working branches

> Historical evidence. For current operating instructions, start at the [MILES guide](../index.md).

Updated September 11, 2026 after consolidation and the researcher-workflow exercise.
These are the project integration branches; the unrelated root checkouts still
have their existing `qmoe-int` branches. No branch was renamed to Git `main`.

| Repository | Working branch | Working directory |
| --- | --- | --- |
| open-instruct | `robertb/miles-olmo-core` | `~/proj/open-instruct/.worktrees/miles-integration` |
| MILES | `robertb/olmo-core-backend` | `~/proj/open-instruct/.worktrees/miles-runtime` |
| OLMo-core | `robertb/miles-rl-adapter` | `~/proj/open-instruct/.worktrees/miles-core-adapter` |

The [runtime lock](../../../runtime/miles/runtime.lock.json) and checked-in source
patches reconstruct the dependency revisions. Local sibling worktrees are not
required by a built runtime image. Both dependency patches were reconstructed
in temporary clones and checked against their complete consolidated trees.

## Incorporated work and cleanup

- Open-instruct's earlier `miles-olmo-core` and checkpoint-performance branches
  were already ancestors of the working branch; their worktrees and refs were removed.
- Core's checkpoint-performance branch pointed at the working branch's commit;
  its worktree and ref were removed.
- Core's original SwiGLU prototype kernel matches the integrated dynamic kernel
  structurally. The current implementation retains the explicit static/dynamic
  selector. Its three historical evidence files were preserved on the working
  branch in `779b183d8`; the prototype worktree/ref were then removed.
- MILES already contained the router-padding implementation. Missing CP mask
  assertions were incorporated in `df24d2ed5` and all five targeted CPU tests passed.
  Both the padding branch and its formatting backup were removed. This does not
  establish that every custom Olmo model consumes the padding mask in its
  auxiliary objective; that remains a separate semantic question.
- Clean incorporated launch snapshots were removed. The durable-continuation
  snapshot had a different commit ID but was patch-equivalent to incorporated work.

The cleanup removed **34 worktrees and six local branch refs**. Complete Git
bundles for the superseded padding and SwiGLU branches, and the earlier runtime
source clones with their staged patches, are preserved under
`~/proj/open-instruct/.artifacts/worktree-cleanup/20260911/`.
No staged nested-repository changes were discarded. Remote branches were unchanged.

## Retained work is not a pending verified-feature merge

| Retained material | Disposition |
| --- | --- |
| Core `robertb/miles-adapter` / `.worktrees/miles-core` | Earlier gdn2 architecture lineage. Its RL hook/interchange work was ported into the hero adapter; retain historical provenance rather than merging the entire older architecture tree. |
| Core `router-fp32-export`, `router-bf16-autocast`, and related historical branches | Earlier precision/export investigations on another lineage. Keep for provenance and targeted comparisons; no whole-branch merge is justified by this review. |
| Older MILES weight-pipeline/teardown snapshots | The current runtime already includes chunk pipelining and W&B teardown. Historical combined snapshots have different patches and remain references, not required merges identified by this audit. |
| Dirty MILES patch-validation and teardown worktrees | Preserved. They contain local changes from the separate olmo-miles work; this cleanup does not certify or discard those changes. |
| `docs/miles-core-optimization-targets-20260911.md` | Existing uncommitted discussion draft, containing superseded estimates and proposals. Kept intact; current reviewed decisions live in the parity and measurement reports. |
| Root `qmoe-int`, old SFT/replay branches and the stale hybrid-reproduction worktree registration | Separate historical work, outside this integration cleanup. |

There is no known unmerged **verified capability required by the current Core RL
path** left among the consolidated checkpoint, padding, dynamic-row or replay
branches. Unreviewed historical changes are not covered by that statement.

See [current feature parity](../feature-parity.md), the
[detailed audit](feature-parity-audit-20260911.md), and the
[136-field inventory](knob-inventory.md) for actual remaining implementation
and qualification work. The confirmed unsupported native controls found by the
audit now fail in both CPU planning and direct runtime argument validation.

The structured researcher workflow, 8 × 8 starter defaults and async TIS are now incorporated on the open-instruct working branch. The [four-update config launch and independent audit](researcher-workflow-20260911.md) passed. Dependency branches were unchanged by this addition.
