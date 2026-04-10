"""Patch vLLM gpu_worker.py to add weight validation after layerwise reload."""
import pathlib
import sys

gpu_worker_path = pathlib.Path(sys.argv[1])
src = gpu_worker_path.read_text()
target = "finalize_layerwise_reload(model, self.model_config)"
idx = src.find(target)
if idx == -1:
    print("WARNING: target not found in gpu_worker.py", file=sys.stderr)
    sys.exit(0)
end = src.index("\n", idx) + 1
patch = (
    "            import logging as _logging\n"
    "            _wv_logger = _logging.getLogger('weight_validation')\n"
    "            _bad = []\n"
    "            for _n, _p in model.named_parameters():\n"
    "                if _p.isnan().any():\n"
    "                    _bad.append((_n, 'NaN', list(_p.shape)))\n"
    "                elif _p.isinf().any():\n"
    "                    _bad.append((_n, 'Inf', list(_p.shape)))\n"
    "            if _bad:\n"
    "                _wv_logger.error('WEIGHT VALIDATION FAILED after layerwise reload: %s', _bad[:10])\n"
    "            else:\n"
    "                _wv_logger.warning('WEIGHT VALIDATION OK: all %d params clean after layerwise reload', sum(1 for _ in model.named_parameters()))\n"
)
gpu_worker_path.write_text(src[:end] + patch + src[end:])
print("Patched gpu_worker.py with weight validation")
