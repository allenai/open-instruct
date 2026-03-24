# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

from dataclasses import asdict, dataclass, field
from enum import Enum

import torch


class SpecialTerm(Enum):
    SQRT = "sqrt"
    EXP = "exp"


# serialize dtypes as strings, map to torch.dtype for runtime packing
DTYPE_NAME_TO_TORCH = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}
TORCH_DTYPE_TO_NAME = {v: k for k, v in DTYPE_NAME_TO_TORCH.items()}


def parse_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        if dtype not in TORCH_DTYPE_TO_NAME:
            raise ValueError(f"Unsupported torch dtype: {dtype}")
        return dtype
    if dtype not in DTYPE_NAME_TO_TORCH:
        raise ValueError(f"Unsupported dtype name: {dtype}")
    return DTYPE_NAME_TO_TORCH[dtype]


def torch_dtype_to_name(dtype: torch.dtype) -> str:
    if dtype not in TORCH_DTYPE_TO_NAME:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    return TORCH_DTYPE_TO_NAME[dtype]


def torch_dtype_bit_width(dtype: torch.dtype) -> int:
    if dtype not in TORCH_DTYPE_TO_NAME:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    # compute dynamically from torch metadata instead of using a manual bit-width map
    return torch.finfo(dtype).bits


@dataclass
class QuantizationBin:
    scale_dtype: torch.dtype
    element_bits: int
    num_elements: int

    def __post_init__(self) -> None:
        self.scale_dtype = parse_torch_dtype(self.scale_dtype)


@dataclass
class DistributionQuantizationConfig:
    d: int
    k: int
    exact_k: int
    exact_dtype: torch.dtype = torch.float32
    polynomial_terms: list[SpecialTerm | int] | None = None
    term_dtype: torch.dtype = torch.float32
    residual_bins: list[QuantizationBin] = field(default_factory=list)
    delta_encoding: bool = True
    error_diffusion: bool = False
    normalize_t: bool = False

    def __post_init__(self) -> None:
        self.exact_dtype = parse_torch_dtype(self.exact_dtype)
        self.term_dtype = parse_torch_dtype(self.term_dtype)

    @classmethod
    def from_dict(cls, data: dict) -> "DistributionQuantizationConfig":
        # normalize loosely typed YAML polynomial_terms into stricter terms used by
        # polynomial fitting code (ints or SpecialTerm enums).
        terms_raw = data.get("polynomial_terms")
        terms: list[SpecialTerm | int] | None
        if terms_raw is None:
            terms = None
        else:
            terms = []
            for term in terms_raw:
                # keep numeric terms
                if isinstance(term, int):
                    terms.append(term)
                # map symbolic terms
                elif isinstance(term, str) and term in {SpecialTerm.SQRT.value, SpecialTerm.EXP.value}:
                    terms.append(SpecialTerm(term))
                else:
                    # numeric strings from YAML -> ints.
                    terms.append(int(term))

        # build strongly typed config from loosely typed YAML input
        cfg = cls(
            d=int(data["d"]),
            k=int(data["k"]),
            exact_k=int(data["exact_k"]),
            exact_dtype=parse_torch_dtype(data.get("exact_dtype", "float32")),
            polynomial_terms=terms,
            term_dtype=parse_torch_dtype(data.get("term_dtype", "float32")),
            residual_bins=[
                QuantizationBin(
                    scale_dtype=b["scale_dtype"],
                    element_bits=int(b["element_bits"]),
                    num_elements=int(b["num_elements"]),
                )
                for b in data.get("residual_bins", [])
            ],
            delta_encoding=bool(data.get("delta_encoding", True)),
            error_diffusion=bool(data.get("error_diffusion", False)),
            normalize_t=bool(data.get("normalize_t", False)),
        )

        # validate cross-field constraints (k/exact_k/bin sizes) before use
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.k < self.exact_k:
            raise ValueError("k must be >= exact_k")
        if self.exact_k < self.k and (not self.polynomial_terms) and (not self.residual_bins):
            raise ValueError("If exact_k < k, polynomial_terms or residual_bins must be provided.")
        approx_terms = self.k - self.exact_k
        bin_elems = sum(bin_.num_elements for bin_ in self.residual_bins)
        if bin_elems > approx_terms:
            raise ValueError("residual_bins total num_elements must be <= k - exact_k")

    def to_dict(self) -> dict:
        out = asdict(self)
        out["exact_dtype"] = torch_dtype_to_name(self.exact_dtype)
        out["term_dtype"] = torch_dtype_to_name(self.term_dtype)
        if out["polynomial_terms"] is not None:
            out["polynomial_terms"] = [t.value if isinstance(t, SpecialTerm) else t for t in out["polynomial_terms"]]
        for bin_cfg in out["residual_bins"]:
            bin_cfg["scale_dtype"] = torch_dtype_to_name(bin_cfg["scale_dtype"])
        return out
