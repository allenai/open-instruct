"""Prepare reproducible training inputs and connect them to inference records.
These modules render and validate datasets, identify the starting checkpoint for
recording, and build frozen prompt-exclusion tables that later runs can reuse.
"""
