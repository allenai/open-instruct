"""Compatibility imports for verifier factories in existing immutable bundles.

New registries should use ``deterministic_verifiers.DeterministicVerifier``.
"""

from open_instruct.miles.rewards import deterministic_verifiers

CandidateVerifier = deterministic_verifiers.DeterministicVerifier


def __getattr__(name):
    return getattr(deterministic_verifiers, name)
