# Copied from allenai/scaling-ladders@59534d7bda695c8dcbd7eb8e873472cba05c4993 ladders/olmoe3/transformers_plugin/src/olmoe3_transformers_plugin/hf_vision/__init__.py
"""Hugging Face config/model files for a multimodal (vision + OLMoE3 LM) export.

The two modules here are importable from the installed plugin and are also copied verbatim
into every multimodal export, next to the text team's ``*_olmo3moe.py`` files, so the export
reloads with ``trust_remote_code=True`` in a process that has no olmo-core.
"""
