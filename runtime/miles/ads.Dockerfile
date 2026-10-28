# Keep the completed H121 control's exact runtime and overlay this branch only.
FROM gcr.io/ai2-beaker-core/public/dbdq64ppchrs73a68rpg@sha256:bbc409704afc7d246668f52629173fc9230bb2e3920ac82b18faf43c3dc83cd7
ARG SOURCE_REVISION
LABEL org.opencontainers.image.revision=${SOURCE_REVISION}
RUN printf '%s\n' "$SOURCE_REVISION" > /opt/core-rl/source-revision
COPY open_instruct /opt/core-rl/open_instruct
COPY tests/miles /opt/core-rl/tests/miles
COPY scripts/data/ads /opt/core-rl/scripts/data/ads
COPY docs/miles/ads.md /opt/core-rl/docs/miles/ads.md
COPY runtime/miles/ads.Dockerfile /opt/core-rl/runtime/miles/ads.Dockerfile
RUN python -m pytest tests/miles/test_ads.py tests/miles/test_ads_preparation.py tests/miles/test_async.py tests/miles/test_selection_data_source.py tests/miles/test_options.py -q
