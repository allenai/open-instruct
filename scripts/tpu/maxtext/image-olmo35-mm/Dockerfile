# The layer that lets MaxText's dataset_type=olmo_mm run OLMo-core's multimodal data loader (MixtureDataLoader,
# MultimodalCollator) in-process on every TPU host: CPU torch and torchvision, pypdfium2 for olmOCR pages, the image
# sources' readers, and OLMo-core itself. BASE is MaxText at the pinned commit, built by build.sh with MaxText's own
# TPU dependencies Dockerfile. From allenai/olmo-gcp-infra@3575547e82325258583ff7f83e3e7e123d1e7826
# examples/olmo35-maxtext/scripts/vision-align-image/cloudbuild.yaml (the Dockerfile it writes inline), which built
# the reference run's image maxtext:217c84582d2f985c9b2e9d3a7067e8423035c19e-olmocore-fd9bba1ba.
#   - The constraints pin every package the MaxText image already has, so pip fails rather than move the JAX stack.
#     The exceptions are what torch needs different: sympy >= 1.13.3 (the image has 1.12), with it mpmath < 1.4,
#     and setuptools.
#   - torch comes from the CPU index alone; with PyPI in reach pip resolves the CUDA build and its nvidia wheels.
#   - torchvision is required: without it OLMo-core's loader skips every image example as a data error and trains
#     text only (MaxText's adapter raises at startup since jason718/maxtext@2b8446d3).
#   - OLMo-core and cached-path go in without their dependencies, which are installed by name on the line before or
#     are already in the image; cached-path's upper bounds would otherwise fight the image's pins.
#   - The RUN checks fail the build if anything the loader or MaxText imports is missing.
ARG BASE
FROM ${BASE}
ARG OLMO_CORE_REF
RUN python3 -m pip list --format=freeze | grep -viE '^(sympy|mpmath|setuptools)==' > /tmp/image-packages.txt \
 && python3 -m pip install --no-cache-dir --root-user-action=ignore -c /tmp/image-packages.txt \
      --index-url https://download.pytorch.org/whl/cpu 'torch>=2.13,<2.14' torchvision \
 && python3 -m pip freeze | grep -iE '^(torch|torchvision|sympy|mpmath)==' >> /tmp/image-packages.txt \
 && python3 -m pip install --no-cache-dir --root-user-action=ignore -c /tmp/image-packages.txt \
      dataclass-extensions bettermap pandas importlib_resources boto3 pypdfium2 h5py 'datasets>=4,<6' \
 && python3 -m pip install --no-cache-dir --root-user-action=ignore --no-deps cached-path \
      "ai2-olmo-core @ https://github.com/allenai/OLMo-core/archive/${OLMO_CORE_REF}.tar.gz"
RUN JAX_PLATFORMS=cpu python3 -c "import jax, torch, torchvision, pypdfium2; \
import olmo_core.data.multimodal.mixture_data_loader, olmo_core.data.multimodal.alignment, olmo_core.data.multimodal.collator; \
from olmo_core.data.multimodal.mixture_data_loader import MixtureDataLoaderConfig as C; assert 'image_dtype' in C.__dataclass_fields__; \
import maxtext.input_pipeline.olmo_mm_data_processing; \
print('jax', jax.__version__, 'torch', torch.__version__, 'torchvision', torchvision.__version__)"
RUN JAX_PLATFORMS=cpu python3 -c "import importlib, pkgutil, olmo_core.data.multimodal as mm; \
names = [m.name for m in pkgutil.walk_packages(mm.__path__, mm.__name__ + '.') if 'mm_olmo_loader' not in m.name]; \
[importlib.import_module(n) for n in names]; \
import datasets, h5py, pyarrow, transformers; \
print('olmo_core.data.multimodal: imported', len(names), 'modules; datasets', datasets.__version__, 'transformers', transformers.__version__)"
RUN JAX_PLATFORMS=cpu python3 -c "import olmo_core.internal.olmo35_mm_sft" \
 || echo "WARNING: olmo_core.internal.olmo35_mm_sft does not import (only the data modules are required)"
