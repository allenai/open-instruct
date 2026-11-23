import json
import subprocess
import sys

import pyarrow as pa
import pytest
from pyarrow import parquet as pq
from scripts.data.sft import prepare_olmo35_release_data as cleanup


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "source_dataset": "allenai/Dolci-Think-SFT",
                "source_revision": "pinned-revision",
                "groups": {"dated_cutoff": ["a", "b"], "stale_identity": ["b", "c"]},
                "evidence": ["audit reference"],
            }
        )
    )
    return path


def shard(path, ids, *, source=True):
    data = {
        "extra": list(range(len(ids))),
        "id": ids,
        "messages": [[{"role": "user", "content": f"text {index}"}] for index in range(len(ids))],
    }
    if source:
        data["source"] = ["same-source"] * len(ids)
    table = pa.table(data)
    messages_index = table.schema.get_field_index("messages")
    content = pa.field("content", pa.string(), metadata={b"nested": b"preserve-me"})
    messages_type = pa.list_(pa.field("item", pa.struct([content, pa.field("role", pa.string())])))
    schema = table.schema.set(messages_index, pa.field("messages", messages_type, metadata={b"column": b"keep"}))
    table = table.cast(schema).replace_schema_metadata({b"provenance": b"preserve-me"})
    pq.write_table(table, path, row_group_size=2, use_compliant_nested_type=False)
    return table


def test_multiple_shards_overlap_schema_order_and_receipt(tmp_path, manifest):
    first, second = tmp_path / "one.parquet", tmp_path / "two.parquet"
    tables = [shard(first, ["keep-1", "b", "a", "keep-2"]), shard(second, ["c", "b", "keep-3"])]
    hashes = [cleanup.sha256(path) for path in (first, second)]
    output = tmp_path / "filtered"
    receipt = cleanup.prepare_release_data([str(tmp_path / "*.parquet")], manifest, output, batch_size=1)
    assert (receipt["rows_in"], receipt["rows_out"], receipt["rows_dropped"]) == (7, 3, 4)
    assert receipt["matched_unique_ids"] == 3
    assert receipt["audit_source"] == {"dataset": "allenai/Dolci-Think-SFT", "revision": "pinned-revision"}
    assert receipt["groups"] == {
        "dated_cutoff": {"requested": 2, "matched": 2, "absent": 0, "absent_ids": []},
        "stale_identity": {"requested": 2, "matched": 2, "absent": 0, "absent_ids": []},
    }
    assert receipt["per_source_dropped"] == [{"source": "same-source", "rows_dropped": 4}]
    for index, entry in enumerate(receipt["shards"]):
        result = pq.read_table(output / entry["output"])
        expected = tables[index].take(pa.array([0, 3] if index == 0 else [2]))
        assert result.equals(expected, check_metadata=True)
        assert entry["input_sha256"] == hashes[index]
        assert entry["output_sha256"] == cleanup.sha256(output / entry["output"])
    assert hashes == [cleanup.sha256(path) for path in (first, second)]
    assert receipt["audit_manifest"]["sha256"] == cleanup.sha256(manifest)
    assert json.loads((output / "receipt.json").read_text()) == receipt


def test_missing_ids_fail_without_publishing_and_explicit_allow_records_them(tmp_path, manifest):
    path = tmp_path / "input.parquet"
    shard(path, ["a", "keep"], source=False)
    output = tmp_path / "filtered"
    with pytest.raises(ValueError, match="requested IDs absent"):
        cleanup.prepare_release_data([path], manifest, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".filtered-*"))
    receipt = cleanup.prepare_release_data([path], manifest, output, allow_missing_ids=True)
    assert receipt["allow_missing_ids"] is True
    assert receipt["groups"]["stale_identity"]["absent_ids"] == ["b", "c"]
    assert "per_source_dropped" not in receipt
    assert pq.read_table(output / receipt["shards"][0]["output"])["id"].to_pylist() == ["keep"]


@pytest.mark.parametrize("ids", [[1, 2], ["a", None]])
def test_wrong_type_or_null_ids_abort_without_publishing(tmp_path, manifest, ids):
    path = tmp_path / "input.parquet"
    shard(path, ids)
    output = tmp_path / "filtered"
    with pytest.raises(ValueError, match="string id|null id"):
        cleanup.prepare_release_data([path], manifest, output, allow_missing_ids=True, batch_size=1)
    assert not output.exists()
    assert not list(tmp_path.glob(".filtered-*"))


def test_absent_id_column_aborts(tmp_path, manifest):
    path = tmp_path / "input.parquet"
    pq.write_table(pa.table({"text": ["hello"]}), path)
    with pytest.raises(ValueError, match="string id"):
        cleanup.prepare_release_data([path], manifest, tmp_path / "filtered")


@pytest.mark.parametrize("empty_kind", ["no-files", "unmatched-glob", "zero-rows", "duplicate"])
def test_invalid_inputs_are_rejected(tmp_path, manifest, empty_kind):
    path = tmp_path / "input.parquet"
    shard(path, ["a"])
    inputs = {
        "no-files": [],
        "unmatched-glob": [tmp_path / "absent-*.parquet"],
        "zero-rows": [path],
        "duplicate": [path, path.parent / "." / path.name],
    }[empty_kind]
    if empty_kind == "zero-rows":
        pq.write_table(pa.table({"id": pa.array([], type=pa.string())}), path)
    with pytest.raises(ValueError, match="Input files|No input files|Empty input"):
        cleanup.prepare_release_data(inputs, manifest, tmp_path / "filtered")
    assert not (tmp_path / "filtered").exists()


@pytest.mark.parametrize("existing_kind", ["directory", "file", "input", "dangling-symlink"])
def test_output_never_clobbers(tmp_path, manifest, existing_kind):
    path = tmp_path / "input.parquet"
    shard(path, ["a", "b", "c"])
    output = tmp_path / "filtered"
    if existing_kind == "directory":
        output.mkdir()
    elif existing_kind == "file":
        output.write_text("keep this")
    elif existing_kind == "input":
        output = path
    else:
        output.symlink_to(tmp_path / "missing")
    source_hash = cleanup.sha256(path)
    with pytest.raises(ValueError, match="Output already exists"):
        cleanup.prepare_release_data([path], manifest, output)
    assert cleanup.sha256(path) == source_hash
    if existing_kind == "file":
        assert output.read_text() == "keep this"
    assert output.exists() or output.is_symlink()


@pytest.mark.parametrize("bad_ids", [[1], [None], "a"])
def test_manifest_does_not_cast_ids(tmp_path, manifest, bad_ids):
    audit = json.loads(manifest.read_text())
    audit["groups"]["dated_cutoff"] = bad_ids
    manifest.write_text(json.dumps(audit))
    path = tmp_path / "input.parquet"
    shard(path, ["a"])
    with pytest.raises(ValueError, match="only string IDs"):
        cleanup.prepare_release_data([path], manifest, tmp_path / "filtered")


def test_all_rows_filtered_still_preserves_schema(tmp_path, manifest):
    path = tmp_path / "input.parquet"
    table = shard(path, ["a", "b", "c"])
    output = tmp_path / "filtered"
    receipt = cleanup.prepare_release_data([path], manifest, output)
    result = pq.read_table(output / receipt["shards"][0]["output"])
    assert result.num_rows == 0
    assert result.schema.equals(table.schema, check_metadata=True)


def test_cli_explicitly_permits_previously_filtered_inputs(tmp_path, manifest):
    path = tmp_path / "input.parquet"
    shard(path, ["a", "keep"])
    output = tmp_path / "filtered"
    assert cleanup.__file__ is not None
    result = subprocess.run(
        [
            sys.executable,
            cleanup.__file__,
            "--data-files",
            str(path),
            "--audit-manifest",
            str(manifest),
            "--output-dir",
            str(output),
            "--allow-missing-ids",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == {"rows_in": 2, "rows_out": 1, "rows_dropped": 1}
    assert json.loads((output / "receipt.json").read_text())["allow_missing_ids"] is True
