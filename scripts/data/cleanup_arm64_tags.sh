#!/usr/bin/env bash
#
# Delete all Docker Hub tags in a repository whose manifest is arm64 / aarch64.
# Run in dry-run mode by default; set DRY_RUN=0 to actually delete.
#
# Requires: curl, jq, and env vars DOCKERHUB_USERNAME and DOCKER_PAT.
#
# Usage:
#   DRY_RUN=1 bash scripts/data/cleanup_arm64_tags.sh                # audit
#   DRY_RUN=0 bash scripts/data/cleanup_arm64_tags.sh                # delete
#   REPO=hamishi740/swerl-tmax-10k DRY_RUN=0 bash scripts/data/cleanup_arm64_tags.sh
#
# Caveats:
#   * Multi-arch manifest lists that include arm64 WILL be deleted in full
#     (including their amd64 side). The rebuild flow here is amd64-only, so
#     that is what we want.
#   * Hub tag metadata can lag a few seconds after a push; if arch shows as
#     "unknown" for a just-pushed tag, re-run.

set -euo pipefail

REPO="${REPO:-hamishi740/swerl-tmax}"
USER="${DOCKERHUB_USERNAME:?set DOCKERHUB_USERNAME}"
PAT="${DOCKER_PAT:?set DOCKER_PAT}"
DRY_RUN="${DRY_RUN:-1}"

if ! command -v jq >/dev/null; then
  echo "jq is required" >&2
  exit 1
fi

echo "Repo: $REPO"
echo "Dry run: $DRY_RUN (set DRY_RUN=0 to actually delete)"

TOKEN=$(curl -sS -H 'Content-Type: application/json' \
  -d "{\"username\":\"$USER\",\"password\":\"$PAT\"}" \
  https://hub.docker.com/v2/users/login/ | jq -r .token)

if [ -z "$TOKEN" ] || [ "$TOKEN" = "null" ]; then
  echo "Failed to obtain Docker Hub JWT (check DOCKERHUB_USERNAME / DOCKER_PAT)." >&2
  exit 1
fi

deleted=0
kept=0
inspected=0

next="https://hub.docker.com/v2/repositories/${REPO}/tags/?page_size=100"
while [ -n "$next" ] && [ "$next" != "null" ]; do
  page=$(curl -sS -H "Authorization: JWT $TOKEN" "$next")

  # Emit one row per tag: "tag <tab> arch1,arch2,..."
  tags_tsv=$(echo "$page" | jq -r '
    .results[]
    | [ .name,
        ( [ .images[]?.architecture // "unknown" ] | unique | join(",") )
      ]
    | @tsv
  ')

  while IFS=$'\t' read -r tag archs; do
    inspected=$((inspected + 1))
    case ",$archs," in
      *,arm64,*|*,arm,*|*,aarch64,*)
        echo "DELETE $REPO:$tag (archs=$archs)"
        if [ "$DRY_RUN" = "0" ]; then
          curl -sS -o /dev/null -w "  status=%{http_code}\n" -X DELETE \
            -H "Authorization: JWT $TOKEN" \
            "https://hub.docker.com/v2/repositories/${REPO}/tags/${tag}/"
        fi
        deleted=$((deleted + 1))
        ;;
      *)
        kept=$((kept + 1))
        ;;
    esac
  done <<< "$tags_tsv"

  next=$(echo "$page" | jq -r '.next')
done

echo
echo "Inspected: $inspected"
echo "Flagged for deletion: $deleted"
echo "Kept: $kept"
if [ "$DRY_RUN" = "1" ]; then
  echo "Re-run with DRY_RUN=0 to actually delete."
fi
