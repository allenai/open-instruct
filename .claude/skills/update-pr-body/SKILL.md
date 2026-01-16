---
name: update-pr-body
description: Update the body of a GitHub pull request. Use when the user asks to update, edit, or modify a PR description/body.
allowed-tools: Bash(gh:*)
---

# Update GitHub PR Body

## Instructions

When updating a GitHub PR body:

1. Get the current PR number (if not provided):
   ```bash
   gh pr list --head "$(git branch --show-current)" --json number --jq '.[0].number'
   ```

2. Get the current PR body to review existing content:
   ```bash
   gh pr view <pr-number> --json body --jq -r '.body'
   ```

3. Update the PR body using `gh pr edit`:
   ```bash
   gh pr edit <pr-number> --body "$(cat <<'EOF'
   New PR body content here.

   Use markdown formatting as needed.
   EOF
   )"
   ```

## Examples

Get current PR number from branch:
```bash
gh pr list --head "$(git branch --show-current)" --json number,url --jq '.[0]'
```

View current PR body:
```bash
gh pr view 1372 --json body --jq -r '.body'
```

Update PR body with new content:
```bash
gh pr edit 1372 --body "$(cat <<'EOF'
## Summary
- Updated vllm to 0.13.0
- Fixed tool_grpo_fast.sh script

## Test Plan
- [x] Single GPU GRPO
- [x] Tool GRPO
- [x] Multi-node GRPO
EOF
)"
```

Add to existing body (read first, then append):
```bash
# Read the current PR body, ensuring to get the raw string
CURRENT_BODY=$(gh pr view 1372 --json body --jq -r '.body')

# Append new content safely using a double-quoted string
gh pr edit 1372 --body "${CURRENT_BODY}

## Additional Notes
New content appended here.
"
```
