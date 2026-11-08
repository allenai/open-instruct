"""Fixed-routing samples used by the Core/MILES integration checks."""

import json
from types import SimpleNamespace

import numpy as np
from miles.backends.core_utils import expert_schedule as schedule

from open_instruct.test_miles_expert_schedule import configuration


def sample_groups(count=16, sample_type=SimpleNamespace):
    samples = []
    for i in range(count):
        hot = [0, 1] if i % 4 < 2 else [2, 3]
        routes = np.broadcast_to(np.array(hot, dtype=np.int32), (5, 2, 2)).copy()
        routes[:, 0] = -1  # Dense layer: never dispatched or counted.
        samples.append(
            sample_type(
                index=i,
                group_index=i // 4,
                rollout_id=None,
                tokens=[i % 11] * 6,
                response_length=2,
                reward=float(i % 3),
                rollout_routed_experts=routes,
                weight_versions=[str(i // 16)],
                rollout_log_probs=[-1.0, -1.0],
                loss_mask=[1, 1],
                remove_sample=False,
            )
        )
    return [samples[i : i + 4] for i in range(0, count, 4)]


def hook_args(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                model_type="olmo3moe",
                n_routed_experts=4,
                num_hidden_layers=2,
                num_experts_per_tok=2,
                dense_layers_indices=[0],
            )
        )
    )
    config = configuration()
    return SimpleNamespace(
        **{**config.miles, "hf_checkpoint": str(tmp_path)}, actor_num_nodes=1, olmo_core=config.core
    )


def histograms(samples):
    return [
        schedule.destination_histogram(
            s.rollout_routed_experts, len(s.tokens), num_experts=4, ep_degree=2, num_layers=2, top_k=2, layers=(1,)
        )
        for s in samples
    ]
