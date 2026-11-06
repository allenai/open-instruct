"""Compatibility entry point for source plans using the former module name.

New commands and imports should use ``source_preparation``.
"""

from open_instruct.miles.datasets import source_preparation


def __getattr__(name):
    return getattr(source_preparation, name)


if __name__ == "__main__":
    source_preparation.main()
