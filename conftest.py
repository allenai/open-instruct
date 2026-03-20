import pathlib

import torch

try:
    import vllm  # noqa: F401

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

collect_ignore = []
if not VLLM_AVAILABLE:
    collect_ignore.extend([
        "open_instruct/test_vllm_utils.py",
        "open_instruct/test_data_loader.py",
        "open_instruct/test_grpo_fast.py",
    ])
if not torch.cuda.is_available():
    collect_ignore.extend(str(p) for p in pathlib.Path("open_instruct").glob("*_gpu.py"))
