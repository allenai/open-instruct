"""Does the reward agree with the probe that produced it?

Run on a GPU node:

    python -m pytest projects/pedagogy_rm/test_plugin.py -x -q

The head is fitted offline on states from extract_hidden.py and read online from
states the plugin builds itself. Those two paths reconstruct the chat context
separately, so they can drift - a different template call, a different token
boundary - and nothing would fail. The reward would just quietly stop meaning what
was measured. This scores real labelled turns through the plugin and checks the
numbers land where the probe said they would.
"""

from __future__ import annotations

import asyncio
import glob
import json
import os
import statistics

import pytest

from open_instruct.scored_rewards import Sample
from open_instruct.scored_rewards.types import TRANSCRIPT_KEY
from projects.pedagogy_rm.plugin import SIGNS, PedagogyHead

HEAD = "data/head.npz"
UNITS = "data/label_slices/slice_1.json"
needs_data = pytest.mark.skipif(not os.path.exists(HEAD), reason="fit_head.py has not been run")


def sample_for(unit: dict) -> Sample:
    """A Sample shaped the way PartnerModelEnv would leave one."""
    transcript = [{"who": "partner", "text": unit["student_before"]}, {"who": "policy", "text": unit["tutor_turn"]}]
    return Sample(
        completion=unit["tutor_turn"],
        prompt=unit["question"],
        label=json.dumps({"question": unit["question"], "student_before": unit["student_before"]}),
        rollout={"info": {TRANSCRIPT_KEY: json.dumps(transcript)}},
    )


@needs_data
def test_context_matches_extraction():
    """The plugin must rebuild the exact context the head was fitted on."""
    from projects.pedagogy_rm.extract_hidden import context_messages  # noqa: PLC0415

    with open(UNITS) as handle:
        unit = json.load(handle)["units"][0]
    messages, turn = PedagogyHead.context(PedagogyHead.__new__(PedagogyHead), sample_for(unit))
    assert messages == context_messages(unit)
    assert turn == unit["tutor_turn"]


@needs_data
def test_falls_back_without_a_transcript():
    """A bare completion still scores, using the item for context."""
    with open(UNITS) as handle:
        unit = json.load(handle)["units"][0]
    sample = Sample(
        completion="What happens if you add the cents first?",
        label=json.dumps({"question": unit["question"], "student_before": unit["student_before"]}),
    )
    messages, turn = PedagogyHead.context(PedagogyHead.__new__(PedagogyHead), sample)
    assert turn == "What happens if you add the cents first?"
    assert unit["question"] in messages[1]["content"]


@needs_data
def test_signs_point_the_right_way():
    """leak is the one dimension where a high raw score is bad."""
    assert SIGNS["leak"] < 0
    assert all(v > 0 for k, v in SIGNS.items() if k != "leak")


@needs_data
@pytest.mark.skipif(os.environ.get("PEDAGOGY_GPU") != "1", reason="loads a 7B model; set PEDAGOGY_GPU=1")
def test_scores_track_the_labels():
    """End to end: the reward should correlate with the labels it was fitted on.

    Not a re-measurement of the probe - the head has seen these turns. It is a
    check that the online path reproduces the offline one, so a low number here
    means the context or pooling drifted, not that the probe is bad.
    """
    from projects.pedagogy_rm.agreement import load as load_labels  # noqa: PLC0415
    from projects.pedagogy_rm.agreement import pearson  # noqa: PLC0415

    with open(UNITS) as handle:
        units = json.load(handle)["units"][:64]
    by_unit = load_labels(sorted(glob.glob("data/labels/*.json")))
    scorer = PedagogyHead(head=HEAD)
    results = asyncio.run(scorer.score_group([sample_for(u) for u in units]))

    for dim in scorer.dims:
        truth, got = [], []
        for unit, result in zip(units, results, strict=True):
            scores = [r[dim] for r in by_unit.get(unit["id"], {}).values() if isinstance(r.get(dim), int)]
            if scores:
                truth.append(statistics.fmean(scores))
                got.append(result.info["raw"][dim])
        r = pearson(got, truth)
        assert r > 0.6, f"{dim}: online scores correlate at {r:.2f} with the labels; the paths have drifted"
