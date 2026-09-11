"""Read named Megatron DCP model tensors in verified canonical HF coordinates.

Optimizer buckets are deliberately excluded: their flat offsets require a
separate native layout proof and must never be guessed from parameter sizes.
"""

import re
from collections import OrderedDict
from collections.abc import Mapping

import torch
from olmo_miles.runtime.olmo_weight_export import OlmoDirectExportSpec, OlmoDirectWeightExporter
from torch.distributed import checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner, FileSystemReader


class MegatronCheckpointState(Mapping):
    def __init__(self, path, hf_config, *, max_cache_bytes=4 * 1024**3):
        self.path = str(path)
        self.metadata = FileSystemReader(self.path).read_metadata()
        self.exporter = OlmoDirectWeightExporter(OlmoDirectExportSpec.from_hf_config(hf_config))
        self.aliases = {}
        self.canonical = {}
        for name, entry in self.metadata.state_dict_metadata.items():
            if name.startswith("optimizer.") or not hasattr(entry, "size"):
                continue
            match = re.fullmatch(r"(decoder\.layers\.\d+\.mlp\.experts)\.experts\.linear_fc([12])\.weight", name)
            if match:
                if len(entry.size) != 3 or entry.size[0] != self.exporter.spec.num_experts:
                    raise ValueError("Checkpoint expert axis differs from HF expert count")
                aliases = [
                    (f"{match[1]}.linear_fc{match[2]}.weight{expert}", expert) for expert in range(entry.size[0])
                ]
            else:
                aliases = [(name, None)]
            for alias, expert in aliases:
                self.aliases[alias] = (name, expert)
                for canonical in self.exporter.output_names(alias):
                    if canonical in self.canonical:
                        raise ValueError(f"Duplicate canonical parameter: {canonical}")
                    self.canonical[canonical] = alias
        self.exporter.validate_inventory(self.aliases)
        self.max_cache_bytes = max_cache_bytes
        self.cache = OrderedDict()
        self.cache_bytes = 0
        self.bytes_read = 0
        self.tensor_reads = 0
        self.converted_alias = None
        self.converted = {}

    def __iter__(self):
        return iter(self.canonical)

    def __len__(self):
        return len(self.canonical)

    def _tensor(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        metadata = self.metadata.state_dict_metadata[key]
        size = metadata.size.numel() * metadata.properties.dtype.itemsize
        if size > self.max_cache_bytes:
            raise ValueError(f"Checkpoint tensor exceeds declared cache budget: {key}")
        while self.cache and self.cache_bytes + size > self.max_cache_bytes:
            _, previous = self.cache.popitem(last=False)
            self.cache_bytes -= previous.numel() * previous.element_size()
        value = torch.empty(metadata.size, dtype=metadata.properties.dtype, device="cpu")
        destination = {key: value}
        dcp.load(
            destination,
            storage_reader=FileSystemReader(self.path),
            planner=DefaultLoadPlanner(flatten_state_dict=False, allow_partial_load=True),
        )
        if value.device.type != "cpu" or value.dtype != metadata.properties.dtype or value.shape != metadata.size:
            raise ValueError(f"Loaded checkpoint tensor differs from metadata: {key}")
        self.cache[key] = value
        self.cache_bytes += size
        self.bytes_read += size
        self.tensor_reads += 1
        return value

    def __getitem__(self, name):
        alias = self.canonical[name]
        if alias != self.converted_alias:
            key, expert = self.aliases[alias]
            value = self._tensor(key)
            if expert is not None:
                value = value[expert]
            self.exporter.begin()
            self.converted = dict(self.exporter.convert(alias, value))
            self.converted_alias = alias
        return self.converted[name]

    def storage_dtype(self, name):
        key, _ = self.aliases[self.canonical[name]]
        return self.metadata.state_dict_metadata[key].properties.dtype
