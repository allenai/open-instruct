"""Checkpoint-compatible, streaming weight export for live vLLM updates."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class ExportedWeightSpec:
    """Metadata for one tensor emitted to vLLM's checkpoint weight loader."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


class WeightExportAdapter(Protocol):
    """Expand learner parameters into zero or more checkpoint-format tensors."""

    name: str

    def export_specs(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> Iterator[ExportedWeightSpec]: ...

    def export_tensors(self, name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]: ...


class PassthroughWeightExportAdapter:
    """Export every learner parameter without model-specific changes."""

    name = "passthrough"

    def export_specs(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> Iterator[ExportedWeightSpec]:
        yield ExportedWeightSpec(name=name, shape=shape, dtype=dtype)

    def export_tensors(self, name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        yield name, tensor


class Qwen3MoeWeightExportAdapter(PassthroughWeightExportAdapter):
    """Convert Transformers 5 fused Qwen3-MoE experts to vLLM loader names."""

    name = "qwen3_moe"
    _GATE_UP_SUFFIX = ".experts.gate_up_proj"
    _DOWN_SUFFIX = ".experts.down_proj"

    @staticmethod
    def _validate_rank(name: str, shape: tuple[int, ...]) -> None:
        if len(shape) != 3:
            raise ValueError(f"Qwen3-MoE fused parameter {name!r} must be rank 3, got shape {shape}")

    @classmethod
    def _gate_up_dimensions(cls, name: str, shape: tuple[int, ...]) -> tuple[int, int, int]:
        cls._validate_rank(name, shape)
        num_experts, doubled_intermediate_size, hidden_size = shape
        if doubled_intermediate_size % 2:
            raise ValueError(
                f"Qwen3-MoE fused gate/up parameter {name!r} has an odd projection dimension: shape {shape}"
            )
        return num_experts, doubled_intermediate_size // 2, hidden_size

    def export_specs(self, name: str, shape: tuple[int, ...], dtype: torch.dtype) -> Iterator[ExportedWeightSpec]:
        if name.endswith(self._GATE_UP_SUFFIX):
            num_experts, intermediate_size, hidden_size = self._gate_up_dimensions(name, shape)
            prefix = name[: -len(".gate_up_proj")]
            for expert_idx in range(num_experts):
                expert_prefix = f"{prefix}.{expert_idx}"
                expert_shape = (intermediate_size, hidden_size)
                yield ExportedWeightSpec(f"{expert_prefix}.gate_proj.weight", expert_shape, dtype)
                yield ExportedWeightSpec(f"{expert_prefix}.up_proj.weight", expert_shape, dtype)
            return

        if name.endswith(self._DOWN_SUFFIX):
            self._validate_rank(name, shape)
            num_experts, hidden_size, intermediate_size = shape
            prefix = name[: -len(".down_proj")]
            for expert_idx in range(num_experts):
                yield ExportedWeightSpec(
                    f"{prefix}.{expert_idx}.down_proj.weight", (hidden_size, intermediate_size), dtype
                )
            return

        yield from super().export_specs(name, shape, dtype)

    def export_tensors(self, name: str, tensor: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        shape = tuple(tensor.shape)
        if name.endswith(self._GATE_UP_SUFFIX):
            num_experts, intermediate_size, _ = self._gate_up_dimensions(name, shape)
            prefix = name[: -len(".gate_up_proj")]
            for expert_idx in range(num_experts):
                expert = tensor[expert_idx]
                expert_prefix = f"{prefix}.{expert_idx}"
                yield f"{expert_prefix}.gate_proj.weight", expert[:intermediate_size].contiguous()
                yield f"{expert_prefix}.up_proj.weight", expert[intermediate_size:].contiguous()
            return

        if name.endswith(self._DOWN_SUFFIX):
            self._validate_rank(name, shape)
            prefix = name[: -len(".down_proj")]
            for expert_idx in range(shape[0]):
                yield f"{prefix}.{expert_idx}.down_proj.weight", tensor[expert_idx].contiguous()
            return

        yield from super().export_tensors(name, tensor)


def map_weight_name(name: str, name_mapper: Callable[[str], str] | None) -> str:
    """Apply wrapper/name compatibility mapping before model-specific expansion."""

    return name_mapper(name) if name_mapper is not None else name


def resolve_weight_export_adapter(model: torch.nn.Module) -> WeightExportAdapter:
    """Select an adapter explicitly from the unwrapped learner configuration."""

    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        config = getattr(current, "config", None)
        if config is not None:
            model_type = getattr(config, "model_type", None)
            if model_type == "qwen3_moe":
                return Qwen3MoeWeightExportAdapter()
            text_config = getattr(config, "text_config", None)
            if getattr(text_config, "model_type", None) == "qwen3_moe":
                return Qwen3MoeWeightExportAdapter()
        wrapped = getattr(current, "module", None)
        if not isinstance(wrapped, torch.nn.Module):
            break
        current = wrapped
    return PassthroughWeightExportAdapter()
