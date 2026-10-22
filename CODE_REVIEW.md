# Code review criteria

- Find consequential bugs, not reasons to block progress. Put nitpicks in a separate section at the end.
- Call out changes in behavior to unrelated paths through the code.
- Call out security concerns, with a concrete explanation of the risk and when it applies.
- Respect existing architecture. For a generic issue, refer to how other places in the code already handle it.
- Help the author move forward: plainly state the issue, when it occurs, its impact, and a practical fix when one is clear.

Explicitly identify a concern as blocking only when it is a significant correctness problem.
For other concerns, defer to the author's judgment and make the recommendation clear.

## Run an agent review on GitHub

Add `OPENAI_API_KEY` as a repository Actions secret (or an organization secret
with access to this repository). After the workflow is merged to the default
branch, open **Actions → Codex review → Run workflow**, select the default branch,
and enter the pull request number. This also supports PRs from forks. The action
requires a triggering user with repository write access and uses OpenAI API
billing.

For Claude, use the existing `CLAUDE_CODE_OAUTH_TOKEN` Actions secret and select
**Actions → Claude review → Run workflow** with the PR number. This uses the
token owner's Claude subscription; the token must still be valid. The workflow
passes GitHub's built-in read-only token to the Claude action, so it does not
require the Claude GitHub App token exchange or an `ANTHROPIC_API_KEY`.

Both workflows run automatically on PR #1986 when opened, updated, reopened, or marked ready
for review, provided it is not a draft and its branch belongs to this repository.
This lets us validate the integration before merging; other PRs remain on demand.
The setup PR uses its own workflow and instructions for this bootstrap run.

Reviews have a 20-minute job timeout. Each agent reads these criteria and its
prompt in `.github/codex/` or `.github/claude/` from the workflow revision.
Codex inspects Git objects in a read-only sandbox. Claude gets a diff and regular
source-file snapshots, with only Read, Glob, and Grep tools in restricted mode
and a 30-turn limit. Neither executes repository code. A separate job posts
each advisory review with the agent's attribution;
it skips publishing if the PR has closed or its head has changed. Reviews do not
approve PRs or request changes. Rerun the workflow after new commits when needed.
