# Code review criteria

- Find consequential bugs without blocking progress over minor issues.
- Report all issues that reasonably deserve consideration or a fix, including non-blocking bugs, edge cases, maintainability concerns, and gaps in tests or documentation. Include these comments even when approving; the approval threshold does not limit what the review should report.
- Call out changes in behavior to unrelated paths through the code.
- Call out security concerns, with a concrete explanation of the risk and when it applies.
- Respect existing architecture. For a generic issue, refer to how other places in the code already handle it.
- Help the author move forward: plainly state the issue, when it occurs, its impact, and a practical fix when one is clear.

Explicitly identify a concern as blocking only when it is a significant correctness problem.
Approve a completed review when there are no blocking concerns, even if non-blocking
comments remain. For those comments, defer to the author's judgment and make the
recommendation clear. Minor issues and stylistic preferences must not hold up approval.

## Review output

For each finding, give a severity, the repository path and line at the PR head,
the concrete trigger, the impact, and a practical fix when clear. Group blocking
findings and non-blocking comments separately, with minor suggestions at the end.

Describe what you checked and any important uncertainty, missing context, or
verification limits. If there are no blocking findings, say so briefly and still
include any non-blocking comments that reasonably deserve attention.
