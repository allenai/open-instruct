"""Translate run files into validated MILES and Core settings before submission.
The planning helpers describe options, GPU placement and requested pipeline
capacities without importing the GPU runtime; a separate argument bridge connects
the resulting configuration to the native parser during execution.
"""
