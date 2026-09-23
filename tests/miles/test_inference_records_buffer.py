"""The completed buffer records kept, filtered and aborted groups without changing their fate."""

import asyncio
import json
from types import SimpleNamespace

from miles.rollout.fully_async_data_buffer import DataBufferConstructorInput, DataBufferInput
from miles.utils.types import Sample, WeightVersionSpan, WeightVersionsPerCall

from open_instruct.miles.async_buffer import HomogeneousPolicyDataBuffer
from open_instruct.miles.config import ZERO_STD_FILTER, CoreConfig


def test_buffer_records_every_scored_group_and_keeps_filter_semantics(tmp_path):
    checkpoint = tmp_path / "policy"
    checkpoint.mkdir()
    source = {"path": "/weka/source", "files": []}
    (checkpoint / "workflow-model.json").write_text(json.dumps({"identity": {"source": source}}))

    async def exercise():
        recycled = []
        args = SimpleNamespace(
            rollout_batch_size=2,
            n_samples_per_prompt=2,
            global_batch_size=2,
            max_weight_staleness=2,
            async_data_buffer_capacity_factor=2,
            dynamic_sampling_filter_path=ZERO_STD_FILTER,
            reward_key=None,
            olmo_core=CoreConfig(records_root=str(tmp_path / "records")),
            hf_checkpoint=str(checkpoint),
            wandb_run_name="buffer-test",
        )
        buffer = HomogeneousPolicyDataBuffer(DataBufferConstructorInput(args, recycled.append))

        def entry(rewards, status=Sample.Status.COMPLETED):
            group = [
                Sample(
                    group_index=len(rewards) * 10 + index,
                    index=index,
                    reward=reward,
                    weight_versions=[WeightVersionsPerCall([WeightVersionSpan("1", 1, 2)])],
                    status=status,
                    metadata={"query": "q", "verifiers": [{"name": "math", "target": "1"}]},
                )
                for index, reward in enumerate(rewards)
            ]
            return DataBufferInput(prompt_group=group, group=group)

        filtered = entry([0.0, 0.0])
        kept = entry([0.0, 1.0])
        aborted = entry([1.0, 0.0], Sample.Status.ABORTED)
        assert await buffer.put(filtered) is False
        assert await buffer.put(kept) is True
        await buffer.put(aborted)
        assert await asyncio.wait_for(buffer.get(current_version=1), 1) is kept
        assert recycled == [aborted.prompt_group]
        buffer._delegate._records.close()
        metrics = buffer.get_metrics()
        assert metrics["rollout/records/written_total"] == 4
        assert metrics["rollout/records/failed_total"] == metrics["rollout/records/dropped_total"] == 0

    asyncio.run(exercise())
    lines = next((tmp_path / "records").glob("*/buffer-test-*/records-*.jsonl")).read_text().splitlines()
    rows = [json.loads(line) for line in lines]
    groups = [row for row in rows if row["kind"] == "group"]
    assert [(row["filter_decision"], row["filter_reason"]) for row in groups] == [
        ("filtered", "zero_std_0.0"),
        ("passed", None),
        ("aborted", None),
    ]
    assert [response["reward"] for response in groups[1]["responses"]] == [0.0, 1.0]
    [consumed] = [row for row in rows if row["kind"] == "disposition"]
    assert (consumed["observation_id"], consumed["disposition"]) == (groups[1]["observation_id"], "consumed")
