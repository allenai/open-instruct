"""Physical packing must preserve MILES' trainers-then-engines GPU order."""

from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration import topology


@pytest.mark.parametrize(
    "trainer_nodes, trainer_gpus, rollout, tp, expected, unused",
    [
        (1, 4, 11, 1, [(4, 4), (0, 7)], 0),
        (1, 8, 7, 1, [(8, 0), (0, 7)], 0),
        (2, 4, 11, 1, [(4, 0), (4, 4), (0, 7)], 4),
        (1, 6, 8, 4, [(6, 0), (0, 8), (0, 0)], 9),
    ],
)
def test_packing_preserves_trainer_order_and_whole_engines(
    monkeypatch, trainer_nodes, trainer_gpus, rollout, tp, expected, unused
):
    monkeypatch.setattr(
        topology.judging, "registry", lambda _: {"judges": {"general": {"mode": "managed", "gpus": 1}}}
    )
    spec = SimpleNamespace(
        launch={"gpus_per_replica": 8},
        output={"root": "/weka/run"},
        judges={},
        compile=lambda: SimpleNamespace(
            miles={
                "actor_num_nodes": trainer_nodes,
                "actor_num_gpus_per_node": trainer_gpus,
                "rollout_num_gpus": rollout,
                "rollout_num_gpus_per_engine": tp,
                "num_gpus_per_node": 8,
                "colocate": False,
            }
        ),
    )
    plan = topology.plan(spec)
    assert [(n["trainer_gpus"], n["rollout_gpus"]) for n in plan["nodes"]] == expected
    assert plan["unused_gpus"] == unused
    assert sum(n["rollout_gpus"] for n in plan["nodes"]) == rollout
    assert all(n["rollout_gpus"] % tp == 0 for n in plan["nodes"])
    # MILES sorts node IPs numerically, then takes the first world-size GPUs.
    addresses = [f"10.0.0.{i + 2}" for i in range(len(plan["nodes"]))]
    assigned = topology.assign(plan, list(reversed(addresses)))
    roles = []
    for address in addresses:
        node = assigned[address]
        visible, judges = topology.devices(node, list(map(str, range(8))))
        assert not (set(visible) & {gpu for group in judges.values() for gpu in group})
        roles.extend(["trainer"] * node["trainer_gpus"] + ["inference"] * node["rollout_gpus"])
    world = trainer_nodes * trainer_gpus
    assert roles == ["trainer"] * world + ["inference"] * rollout
    assert assigned[addresses[-1]]["judges"] == {"general": 1}
