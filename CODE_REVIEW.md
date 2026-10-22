# Code review criteria

- Find consequential bugs, not reasons to block progress. Put nitpicks in a separate section at the end.
- Call out changes in behavior to unrelated paths through the code.
- Call out security concerns, with a concrete explanation of the risk and when it applies.
- Respect existing architecture. For a generic issue, refer to how other places in the code already handle it.
- Help the author move forward: plainly state the issue, when it occurs, its impact, and a practical fix when one is clear.

Explicitly identify a concern as blocking only when it is a significant correctness problem.
For other concerns, defer to the author's judgment and make the recommendation clear.

## Run a Codex review on GitHub

Add `OPENAI_API_KEY` as a repository Actions secret (or an organization secret
with access to this repository). After the workflow is merged to the default
branch, open **Actions → Codex review → Run workflow**, select the default branch,
and enter the pull request number. This also supports PRs from forks. The action
requires a triggering user with repository write access and uses OpenAI API
billing. It does not use `CLAUDE_CODE_OAUTH_TOKEN`.

Reviews are on demand, with a 20-minute job timeout. Codex reads these criteria
and `.github/codex/review-prompt.md` from the selected workflow revision, inspects
the PR's exact commits without checking out or running its code, and returns an
advisory GitHub review. A separate job posts the result with Codex attribution;
it skips publishing if the PR has closed or its head has changed. Reviews do not
approve PRs or request changes. Rerun the workflow after new commits when needed.
