#!/bin/bash
# Sourced by the debug launch scripts that need a Beaker image. Sets BEAKER_IMAGE from the
# caller's first argument. Without one it explains how to build or find an image and exits,
# instead of launching on a default image that holds some other commit.
if [ -z "${1:-}" ]; then
    cat >&2 <<EOF
Error: no Beaker image given, so there is nothing to run this job on.

Usage: $0 BEAKER_IMAGE [ARGS...]

To build an image from your current commit and launch this script on it:

    ./scripts/train/build_image_and_launch.sh $0

  Commit your changes first: the launcher refuses a checkout with uncommitted changes.
  It reuses the image when one already exists for the commit.

To launch on an image that already exists, pass its name as the first argument:

    $0 <beaker-user>/<image-name>

  The launcher names its images <beaker-user>/open-instruct-integration-test-<branch>-cuda12
  ("beaker account whoami" prints your Beaker user name). To list yours:

    beaker workspace images ai2/<beaker-user> --text open-instruct-integration-test

  The job runs the code in the image, not your checkout, so use an image built from the
  commit you want to test. To see which commit an image holds:

    beaker image get <beaker-user>/<image-name> --format json | jq -r '.[0].description'
EOF
    exit 1
fi
BEAKER_IMAGE="$1"
