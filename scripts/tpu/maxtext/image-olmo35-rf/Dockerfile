# r2's exact training image (maxtext 9f55e7b6 via open-instruct 432c8d43's image-olmo35 patch) plus (1) logging-only: MAXTEXT_LOG_BATCH_HASH=1 batch hashes and HBM lines; (2) the optimizer-state restore
# targets taken from the live shardings (maxtext c469d980). Built on the
# published r2 image rather than from BASE_REF so nothing else (pip-resolved dependencies included) can move.
FROM us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain:51f8cc4b-p907d86f3
COPY patches/ /tmp/patches/
RUN cd /deps && patch -p1 --forward < /tmp/patches/posttrain-fixes.patch
RUN python3 -c "import maxtext.input_pipeline.multihost_dataloading as m, inspect; assert '_log_batch_hash' in inspect.getsource(m); import maxtext.trainers.post_train.sft.train_sft as t; assert hasattr(t, '_restore_targets_from_live_shardings'); print(m.__file__)"
