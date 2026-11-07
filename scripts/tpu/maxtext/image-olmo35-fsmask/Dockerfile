# H200: the fast-skip image (image-olmo35-fs: the hero image plus the pooled SFT chat-template map) plus a per-message
# loss mask for SFT: a `masked_turns` dataset column (message indices of assistant turns) whose spans are excluded from
# the loss under MAXTEXT_SFT_LABELS=prefix. Without the column the pipeline is unchanged. Built on the published image
# so nothing else can move. Patch: olmo-post-training-ledger experiments/h200/loader/maskturns.patch.
FROM us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3-dc88c6c7ea-fse2555fa0
COPY patches/ /tmp/patches/
RUN cd /deps && patch -p1 --forward --fuzz=0 < /tmp/patches/maskturns.patch
RUN python3 -c "import maxtext.input_pipeline.input_pipeline_utils as u, maxtext.input_pipeline.hf_data_processing as h, inspect; assert 'masked_turns' in inspect.getsource(u._split_conversation_into_segments); assert 'mask_column' in inspect.getsource(h.preprocessing_pipeline); assert 'mask_column' in inspect.signature(u.apply_chat_template_batched).parameters; print(u.__file__)"
