---
name: update-pr-body
description: Update the body of a GitHub pull request. Use when the user asks to update, edit, or modify a PR description/body.
allowed-tools: Bash(gh:*)
---

# Update GitHub PR Body

## Important

**Do NOT use `gh pr edit --body`** - it fails with a GraphQL error about Projects (classic) deprecation. Always use the REST API approach below instead.

## Instructions

When updating a GitHub PR body:

1. Get the current PR number (if not provided):
   ```bash
   gh pr list --head "$(git branch --show-current)" --json number --jq '.[0].number'
   ```

2. Get the current PR body to review existing content:
   ```bash
   gh pr view <pr-number> --json body -q '.body'
   ```

3. Update the PR body using the REST API:
   ```bash
   gh api -X PATCH /repos/{owner}/{repo}/pulls/<pr-number> -f body="New PR body content here."
   ```

## Examples

Get current PR number from branch:
```bash
gh pr list --head "$(git branch --show-current)" --json number,url --jq '.[0]'
```

View current PR body:
```bash
gh pr view 1372 --json body -q '.body'
```

Update PR body with new content:
```bash
gh api -X PATCH /repos/allenai/open-instruct/pulls/1372 -f body="## Summary
- Updated vllm to 0.13.0
- Fixed tool_grpo_fast.sh script

## Test Plan
- [x] Single GPU GRPO
- [x] Tool GRPO
- [x] Multi-node GRPO"
```

Add to existing body (read first, then append):
```bash
# Read the current PR body
CURRENT_BODY=$(gh pr view 1372 --json body -q '.body')

# Append new content
gh api -X PATCH /repos/allenai/open-instruct/pulls/1372 -f body="${CURRENT_BODY}

## Additional Notes
New content appended here."
```
