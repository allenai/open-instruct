The training image for Olmo 3.5 SFT: `../image/` (same Dockerfile and build.sh) built on MaxText's `olmo35-rebase`
branch (AI-Hypercomputer/maxtext@a7f34953, the `olmoe3` model) instead of `e259ed61`, with this directory's
`patches/maxtext.patch` (the same fixes, cherry-picked onto that branch).

    cp ../image/build.sh . && ./build.sh a7f34953d595ed8b43e02afa2ccd66acd4b08b5e   # -> maxtext-posttrain:a7f34953-p1499e63d
