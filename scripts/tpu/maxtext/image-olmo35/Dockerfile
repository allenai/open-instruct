# MaxText TPU image for Olmo post-training: the upstream image built by olmo-gcp-infra
# (examples/olmo3-maxtext/cloudbuild.yaml) at BASE_REF, plus
#   - patches/maxtext.patch (made against BASE_REF), fixes not yet upstream:
#       * Olmo 3 YaRN on full-attention layers only (AI-Hypercomputer/maxtext#5544)
#       * multi-turn SFT conversations templated as a whole (AI-Hypercomputer/maxtext#5522)
#   - Tunix (at the commit MaxText pins in src/dependencies/extra_deps), google-metrax and perfetto,
#     which train_sft imports and the pre-training image leaves out, installed without letting them
#     move the JAX/Flax/libtpu stack.
ARG BASE_REF
FROM us-docker.pkg.dev/ai2-olmo/olmo/maxtext:${BASE_REF}
ARG BASE_REF
COPY patches/ /tmp/patches/
RUN cd /deps && patch -p1 --forward < /tmp/patches/maxtext.patch
# Tunix pins jax<0.11.1, which the image is past, so (as MaxText's own setup does) it goes in with
# --no-deps and its other requirements (tunix pyproject.toml at that commit, minus jax) go in
# constrained to the image's JAX/Flax stack. Tunix caps safetensors<0.8 for its own safetensors loading,
# which this path does not use; the image keeps its 0.8.
RUN python3 -m pip freeze | grep -iE '^(jax|jaxlib|libtpu|flax|orbax-checkpoint|optax|numpy|transformers|datasets|grain|safetensors)==' > /tmp/pins.txt \
 && cat /tmp/pins.txt \
 && python3 -m pip install --no-cache-dir --no-deps \
    "$(grep -h '^google-tunix @' /deps/src/dependencies/extra_deps/*github_deps.txt | head -1)" \
 && python3 -m pip install --no-cache-dir -c /tmp/pins.txt google-metrax perfetto \
    datasets "flax>=0.12.5" fsspec grain hf_transfer huggingface_hub importlib_resources jaxtyping jinja2 \
    kagglehub numba omegaconf "orbax-checkpoint>=0.12.0" pillow pylatexenc python-dotenv "qwix>=0.1.6" \
    sentencepiece safetensors sympy tensorflow_datasets tqdm "transformers!=4.57.2" tenacity
# Report (do not fail on) dependency conflicts; the import check below is the real gate.
RUN python3 -m pip check || true
# Fail the build if the patched source is not the one Python imports.
RUN python3 -c "import inspect, metrax, perfetto, tunix; import maxtext.trainers.post_train.sft.train_sft; \
from maxtext.models import olmo3; from maxtext.input_pipeline import input_pipeline_utils as u; \
assert 'current_rope_type = \"default\"' in inspect.getsource(olmo3), olmo3.__file__; \
assert hasattr(u, '_split_conversation_into_segments'), u.__file__; \
print('patched:', olmo3.__file__, u.__file__)"
