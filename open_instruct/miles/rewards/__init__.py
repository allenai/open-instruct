"""Turn generated responses into training rewards using Open Instruct verifiers.
These modules connect task metadata to trusted verifier implementations, external
code execution and language-model judges, with shared failure diagnostics and
optional postprocessing of truncated responses.
"""
