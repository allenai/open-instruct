"""Build ``ExamplesGRPOStreamingConfig`` from ``StreamingDataLoaderConfig`` + registered verifier configs.

Registered with :func:`open_instruct.ground_truth_registry.register_verifier_config` on each
``VerifierConfig`` subclass (see ``examples/manufactoria/verifier.py``). Fields are merged into
one dataclass so ``build_all_verifiers`` reads task fields from ``streaming_config``—same path
as ``code_*``—without ``verifier_extra_sources``.
"""

from __future__ import annotations

# Import order: each verifier module registers its VerifierConfig before we build.
import examples.ballsim.verifier  # noqa: F401
import examples.manufactoria.verifier  # noqa: F401
from open_instruct.ground_truth_registry import build_streaming_config_with_registered_verifier_fields

ExamplesGRPOStreamingConfig = build_streaming_config_with_registered_verifier_fields(
    class_name="ExamplesGRPOStreamingConfig"
)
