"""Implement CuTe's argument protocols on the pinned FA4 AuxData NamedTuple.

FA4 4.0.0b27 passes a NamedTuple across the JIT boundary, but CuTe DSL 4.6.2
does not recognize the empty NamedTuple as a supported JIT argument. Delegate to
CuTe's tuple helpers to flatten and reconstruct empty and non-empty arguments,
preserving the NamedTuple representation used by the TVM-FFI calling convention.
Fail closed if the pinned wheel's source changes; do not silently patch new FA4.
"""

import hashlib
from importlib import metadata
from pathlib import Path

VERSION = "4.0.0b27"
SOURCE_SHA256 = "c80d6ef6a62d005ac259829cf69c675f5f570ed93768d296e46e210768194c08"
OLD_IMPORT = "from cutlass.cute.runtime import from_dlpack"
NEW_IMPORT = OLD_IMPORT + "\nfrom cutlass.base_dsl import typing as dsl_typing"
OLD_CLASS = """class AuxData(NamedTuple):
    tensors: tuple | list | None = None
    scalars: tuple | None = None
"""
NEW_CLASS = (
    OLD_CLASS
    + """
    def __c_pointers__(self):
        return dsl_typing.get_c_pointers(tuple(self))

    def __get_mlir_types__(self):
        return dsl_typing.get_mlir_types(tuple(self))

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(tuple(self))

    def __new_from_mlir_values__(self, values):
        return AuxData(*cutlass.new_from_mlir_values(tuple(self), values))
"""
)


def patch(path: Path, version: str) -> None:
    source = path.read_text()
    if version != VERSION or hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA256:
        raise ValueError(
            f"Unexpected flash-attn-4 source/version at {path}; review the AuxData patch before rebuilding"
        )
    if source.count(OLD_IMPORT) != 1 or source.count(OLD_CLASS) != 1:
        raise ValueError("Expected exactly one AuxData declaration and typing import")
    updated = source.replace(OLD_IMPORT, NEW_IMPORT).replace(OLD_CLASS, NEW_CLASS)
    compile(updated, str(path), "exec")
    path.write_text(updated)


def main() -> None:
    distribution = metadata.distribution("flash-attn-4")
    path = Path(distribution.locate_file("flash_attn/cute/utils.py"))
    patch(path, distribution.version)
    print(f"Patched flash-attn-4 {distribution.version} AuxData with CuTe argument protocols")


if __name__ == "__main__":
    main()
