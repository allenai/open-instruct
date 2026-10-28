"""Evaluate policy snapshots through separate Beaker jobs while training continues.
These modules plan evaluation milestones, submit bounded background requests and
run olmo-eval with resumable task outputs and tracking publication.
"""
