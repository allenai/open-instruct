#!/bin/bash
# Write an image directory's patches/ from the MaxText fork, before build.sh hashes it into the image tag.
# maxtext_source holds "<fork url> <base sha> <head sha> <patch file>"; the patch is `git diff base head -- src`.
# core.abbrev=8 reproduces, byte for byte, the patches the published images were built from, so their tags still match.
#   ../fetch_patches.sh   (from an image directory; build.sh calls it)
set -euo pipefail
read -r FORK BASE HEAD NAME < maxtext_source
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
git clone -q --filter=blob:none --no-checkout "$FORK" "$tmp"
git -C "$tmp" fetch -q origin "$HEAD"
mkdir -p patches
git -C "$tmp" --no-pager -c core.abbrev=8 diff --no-color --no-ext-diff "$BASE" "$HEAD" -- src > "patches/$NAME"
echo "patches/$NAME: $BASE..$HEAD from $FORK ($(wc -l < "patches/$NAME") lines)"
