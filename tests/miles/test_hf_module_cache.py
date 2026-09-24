"""Serving children must only read custom modules completed by their parent."""

import json
import os
import subprocess
import sys
from concurrent import futures

from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast, dynamic_module_utils

from open_instruct.miles import hf_module_cache


def test_primed_custom_modules_need_no_copy_in_concurrent_children(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    cache = tmp_path / "modules"
    monkeypatch.setattr(dynamic_module_utils, "HF_MODULES_CACHE", str(cache))
    source = (
        "from transformers import PretrainedConfig\n"
        "class CacheProbeConfig(PretrainedConfig):\n"
        "    model_type = 'cache_probe'\n"
        "    def __init__(self, dense_mlp_intermediate_size=128, **kwargs):\n"
        "        super().__init__(**kwargs)\n"
        "        self.dense_mlp_intermediate_size = dense_mlp_intermediate_size\n"
    )
    (model / "configuration_cache_probe.py").write_text(source)
    (model / "config.json").write_text(
        json.dumps(
            {"model_type": "cache_probe", "auto_map": {"AutoConfig": "configuration_cache_probe.CacheProbeConfig"}}
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]")), unk_token="[UNK]"
    )
    tokenizer.save_pretrained(model)
    hf_module_cache.prime_serving_modules(["--model-path", str(model), "--trust-remote-code"])
    assert [p.read_text() for p in cache.rglob("configuration_cache_probe.py")] == [source]
    # A fresh interpreter is essential: in-process module reuse would hide a
    # missing priming step. Refuse copies so this checks the read-only child path.
    script = """
import sys
from unittest import mock
from transformers import AutoConfig, AutoTokenizer, dynamic_module_utils

with mock.patch.object(dynamic_module_utils.shutil, 'copyfile', side_effect=AssertionError('child copied module')):
    config = AutoConfig.from_pretrained(sys.argv[1], trust_remote_code=True)
    assert config.dense_mlp_intermediate_size == 128
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
    assert tokenizer.encode('hello', add_special_tokens=False) == [1]
"""
    env = dict(os.environ, HF_MODULES_CACHE=str(cache), HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")

    def child(_):
        result = subprocess.run(
            [sys.executable, "-c", script, str(model)], env=env, capture_output=True, text=True, timeout=90
        )
        assert result.returncode == 0, result.stdout + result.stderr

    with futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(child, range(4)))
    assert [p.read_text() for p in cache.rglob("configuration_cache_probe.py")] == [source]
