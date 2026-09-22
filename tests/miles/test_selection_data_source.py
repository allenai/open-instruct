"""The real MILES data source skips excluded prompts, keeps them out of the ledger, and resumes past them."""

import json
from types import SimpleNamespace

import torch
from miles.utils.types import Sample

from open_instruct.miles import inference_records, record_selection
from open_instruct.miles.config import CoreConfig
from open_instruct.miles.data_source import DashboardDrainingRolloutDataSource


def metadata(index):
    return {
        "query": f"q{index}",
        "verifiers": [{"name": "math", "target": str(index)}],
        "run_prompt_token_ids_sha256": f"t{index}",
    }


def make_source(root, table=None, digest=None, *, load=None):
    checkpoint = root / "policy"
    checkpoint.mkdir(exist_ok=True)
    (checkpoint / "workflow-model.json").write_text(json.dumps({"identity": {"source": {"path": "/p", "files": []}}}))
    args = SimpleNamespace(
        rollout_global_dataset=False,
        use_miles_dashboard=False,
        fully_async=True,
        buffer_filter_path=None,
        n_samples_per_prompt=2,
        save=str(root),
        load=str(load) if load else None,
        rollout_shuffle=False,
        start_rollout_id=1 if load else 0,
        hf_checkpoint=str(checkpoint),
        rollout_temperature=1.0,
        sglang_enable_deterministic_inference=False,
        olmo_core=CoreConfig(selection_table=str(table) if table else None, selection_sha256=digest),
    )
    if table is None:
        return args
    source = DashboardDrainingRolloutDataSource(args)
    args.rollout_global_dataset = True

    class Dataset:
        samples = [Sample(prompt=f"prompt-{i}", metadata=metadata(i)) for i in range(32)]

        def __len__(self):
            return len(self.samples)

    source._delegate.dataset = Dataset()
    return source


def write_table(root, excluded):
    args = make_source(root)
    protocol = inference_records.sha256(inference_records.protocol_identity(args))
    keys = [
        inference_records.sha256(
            {
                "task": inference_records.task_identity(metadata(i), f"prompt-{i}")["task_key"],
                "tokens": f"t{i}",
                "protocol": protocol,
            }
        )
        for i in excluded
    ]
    table = {
        "schema_version": record_selection.TABLE_SCHEMA_VERSION,
        "lineage": inference_records.lineage_identity(args.hf_checkpoint)["inventory_sha256"],
        "protocol_sha256": protocol,
        "excluded": [{"input_key": key, "domain": "math"} for key in keys],
    }
    path = root / "table.json"
    return path, record_selection.write_table(table, path)


def test_excluded_prompts_are_skipped_outside_the_ledger_and_after_resume(tmp_path):
    table, digest = write_table(tmp_path, excluded=[1, 3, 6])
    source = make_source(tmp_path, table, digest)
    groups = source.get_samples(3)
    assert [g[0].prompt for g in groups] == ["prompt-0", "prompt-2", "prompt-4"]
    assert sorted(source._pending_groups) == sorted(g[0].group_index for g in groups)
    source.acknowledge_groups(groups)
    source.save(0)
    state = torch.load(tmp_path / "rollout/global_dataset_state_dict_0.pt", weights_only=True)
    assert state["sample_offset"] == 5
    restored = make_source(tmp_path, table, digest, load=tmp_path)
    restored.load(0)
    assert [g[0].prompt for g in restored.get_samples(2)] == ["prompt-5", "prompt-7"]
    assert restored._selection.skipped == {"math": 1}
