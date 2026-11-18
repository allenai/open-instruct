"""Validate prepared reward registries without importing verifier factories."""

from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import grading


def validate(registry):
    if not isinstance(registry, dict) or not registry:
        raise InputError("Trusted reward registry must be a nonempty mapping")
    for name, spec in registry.items():
        if (
            not isinstance(name, str)
            or not isinstance(spec, dict)
            or not isinstance(spec.get("factory"), str)
            or "." not in spec["factory"]
        ):
            raise InputError("Trusted registry entries require explicit factory paths")
        try:
            grading.validate_spec(spec)
        except ValueError as error:
            raise InputError(f"Verifier {name!r}: {error}") from error
