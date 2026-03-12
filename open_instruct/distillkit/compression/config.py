# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass, field
from enum import Enum

import torch


class SpecialTerm(Enum):
    SQRT = "sqrt"
    EXP = "exp"


class TermDtype(Enum):
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    def bit_width(self) -> int:
        return 16 if self in {TermDtype.FLOAT16, TermDtype.BFLOAT16} else (32 if self == TermDtype.FLOAT32 else 64)

    def dtype(self) -> torch.dtype:
        mapping = {
            TermDtype.FLOAT16: torch.float16,
            TermDtype.BFLOAT16: torch.bfloat16,
            TermDtype.FLOAT32: torch.float32,
            TermDtype.FLOAT64: torch.float64,
        }
        return mapping[self]


@dataclass
class QuantizationBin:
    scale_dtype: TermDtype
    element_bits: int
    num_elements: int

    @classmethod
    def from_dict(cls, data: dict) -> "QuantizationBin":
        return cls(
            scale_dtype=TermDtype(data["scale_dtype"]),
            element_bits=int(data["element_bits"]),
            num_elements=int(data["num_elements"]),
        )


@dataclass
class DistributionQuantizationConfig:
    d: int
    k: int
    exact_k: int
    exact_dtype: TermDtype = TermDtype.FLOAT32
    polynomial_terms: list[SpecialTerm | int] | None = None
    term_dtype: TermDtype = TermDtype.FLOAT32
    residual_bins: list[QuantizationBin] = field(default_factory=list)
    delta_encoding: bool = True
    error_diffusion: bool = False
    normalize_t: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "DistributionQuantizationConfig":
        terms_raw = data.get("polynomial_terms")
        terms: list[SpecialTerm | int] | None
        if terms_raw is None:
            terms = None
        else:
            terms = []
            for term in terms_raw:
                if isinstance(term, int):
                    terms.append(term)
                elif isinstance(term, str) and term in {SpecialTerm.SQRT.value, SpecialTerm.EXP.value}:
                    terms.append(SpecialTerm(term))
                else:
                    terms.append(int(term))

        cfg = cls(
            d=int(data["d"]),
            k=int(data["k"]),
            exact_k=int(data["exact_k"]),
            exact_dtype=TermDtype(data.get("exact_dtype", TermDtype.FLOAT32.value)),
            polynomial_terms=terms,
            term_dtype=TermDtype(data.get("term_dtype", TermDtype.FLOAT32.value)),
            residual_bins=[QuantizationBin.from_dict(b) for b in data.get("residual_bins", [])],
            delta_encoding=bool(data.get("delta_encoding", True)),
            error_diffusion=bool(data.get("error_diffusion", False)),
            normalize_t=bool(data.get("normalize_t", False)),
        )
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
        return {
            "d": self.d,
            "k": self.k,
            "exact_k": self.exact_k,
            "exact_dtype": self.exact_dtype.value,
            "polynomial_terms": [t.value if isinstance(t, SpecialTerm) else t for t in (self.polynomial_terms or [])],
            "term_dtype": self.term_dtype.value,
            "residual_bins": [
                {"scale_dtype": b.scale_dtype.value, "element_bits": b.element_bits, "num_elements": b.num_elements}
                for b in self.residual_bins
            ],
            "delta_encoding": self.delta_encoding,
            "error_diffusion": self.error_diffusion,
            "normalize_t": self.normalize_t,
        }
