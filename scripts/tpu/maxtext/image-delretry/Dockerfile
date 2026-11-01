# Any published maxtext-posttrain image plus one change: Orbax's retention delete of an old checkpoint step retries
# with backoff and, if every try fails, logs ORBAX-DELETE-LEAKED and continues instead of stopping the run.
# Saves and restores are untouched. Built on the published image so nothing else can move.
ARG BASE_TAG
FROM us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain:${BASE_TAG}
COPY patch_orbax_delete_retry.py test_orbax_delete_retry.py /tmp/delretry/
RUN python3 /tmp/delretry/patch_orbax_delete_retry.py && python3 /tmp/delretry/test_orbax_delete_retry.py
