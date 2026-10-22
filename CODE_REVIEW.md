# Code review criteria

- Find consequential bugs, not reasons to block progress. Put nitpicks in a separate section at the end.
- Call out changes in behavior to unrelated paths through the code.
- Call out security concerns, with a concrete explanation of the risk and when it applies.
- Respect existing architecture. For a generic issue, refer to how other places in the code already handle it.
- Help the author move forward: plainly state the issue, when it occurs, its impact, and a practical fix when one is clear.

Explicitly identify a concern as blocking only when it is a significant correctness problem.
For other concerns, defer to the author's judgment and make the recommendation clear.

## Review output

For each finding, give a severity, the repository path and line at the PR head,
the concrete trigger, the impact, and a practical fix when clear. Separate
optional suggestions from correctness findings.

Describe what you checked and any important uncertainty, missing context, or
verification limits. If there are no consequential findings, say so briefly.
