# The hero image (H161, maxtext 1e612a76) plus an optional spawned pool for the SFT chat-template map
# (MAXTEXT_SFT_TEMPLATE_WORKERS > 0), which speeds up replaying already-trained batches on resume. With the variable
# unset the pipeline is unchanged. Built on the published image so nothing else can move.
FROM us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3-dc88c6c7ea
COPY patches/ /tmp/patches/
RUN cd /deps && patch -p1 --forward --fuzz=0 < /tmp/patches/fastskip.patch
RUN python3 -c "import maxtext.input_pipeline.input_pipeline_utils as u, maxtext.input_pipeline.hf_data_processing as h, inspect; assert hasattr(u, 'apply_chat_template_batched'); assert 'MAXTEXT_SFT_TEMPLATE_WORKERS' in inspect.getsource(h); print(u.__file__)"
