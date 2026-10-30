# H149's c12-b2 image (maxtext 1e612a76) plus a document-role branch in the SFT chat segmenter: a conversation of one
# `document` message is tokenized without the chat template, ends with one EOS and is trained on every token (H149
# Amendment 2's long-context replay rows). Chat conversations are unchanged. Built on the published image so nothing
# else can move.
FROM us-docker.pkg.dev/ai2-olmo/olmo/maxtext-posttrain:51f8cc4b-p907d86f3-rfb46e215e-bs3ec936c3-dc88c6c7ea
COPY patches/ /tmp/patches/
RUN cd /deps && patch -p1 --forward --fuzz=0 < /tmp/patches/docrole.patch
RUN python3 -c "import maxtext.input_pipeline.input_pipeline_utils as u, inspect; assert 'conversation[0][\"role\"] == \"document\"' in inspect.getsource(u._split_conversation_into_segments); print(u.__file__)"
