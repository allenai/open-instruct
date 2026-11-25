"""scripts/data/check_replay_eos_mask.py passes the masked dir that mask_replay_final_eos.py writes, and fails each
kind of wrong one."""

import gzip
import hashlib
import importlib.util
import json
import os
import pathlib

import numpy as np

from open_instruct import numpy_dataset_conversion

CHECK_PATH = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "data" / "check_replay_eos_mask.py"
spec = importlib.util.spec_from_file_location("check_replay_eos_mask", CHECK_PATH)
assert spec is not None and spec.loader is not None
check_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_script)

EOS, IM = 2, 3
REPLAY = [[[20, 21, EOS], [22, EOS, 23, EOS]], [[24, 25, 26, 27, EOS]]]  # two parts, as the c11 build writes them


def write_dir(numpy_dir, parts, mask=True, manifest=True):
    os.makedirs(numpy_dir)
    records = []
    for index, docs in enumerate(parts):
        tokens = np.array([t for doc in docs for t in doc], dtype=np.uint32)
        tokens.tofile(os.path.join(numpy_dir, f"token_ids_part_{index:04d}.npy"))
        np.full(len(tokens), mask, dtype=np.bool_).tofile(os.path.join(numpy_dir, f"labels_mask_part_{index:04d}.npy"))
        with gzip.open(os.path.join(numpy_dir, f"token_ids_part_{index:04d}.csv.gz"), "wt") as f:
            start = 0
            for doc in docs:
                f.write(f"{start},{start + len(doc)}\n")
                sha = hashlib.sha256(np.array(doc[:-1], dtype=np.uint32).tobytes()).hexdigest()
                records.append(
                    {
                        "part": index,
                        "start": start,
                        "end": start + len(doc),
                        "id": f"doc{len(records)}",
                        "length_with_eos": len(doc),
                        "tokens_sha256": sha,
                    }
                )
                start += len(doc)
    if manifest:
        with open(os.path.join(numpy_dir, "manifest.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)


def pins_for(replay):
    with open(os.path.join(replay, "manifest.jsonl")) as f:
        records = [json.loads(line) for line in f]
    return {
        "manifest_sha256": check_script.sha256(os.path.join(replay, "manifest.jsonl")),
        "ids_sha256": hashlib.sha256(json.dumps([r["id"] for r in records]).encode()).hexdigest(),
        "rows": len(records),
        "tokens": sum(r["end"] - r["start"] for r in records),
    }


def setup_dirs(tmp_path, untrained_final=False):
    replay, chat, masked = (str(tmp_path / name) for name in ("replay", "chat", "masked"))
    write_dir(replay, REPLAY)
    if untrained_final:
        set_label(replay, 2, False)
    write_dir(chat, [[[IM, 10, EOS], [IM, 11, EOS, IM, 12, EOS]]], mask=False, manifest=False)
    pinned = check_script.sha256(os.path.join(replay, "manifest.jsonl"))
    numpy_dataset_conversion.write_replay_eos_masked_dir(replay, masked, pinned, EOS, IM)
    return replay, chat, masked


def run(replay, masked, chat, pins=None):
    pins = pins or pins_for(replay)
    return check_script.check(replay, masked, pins, [chat], max_seq_length=4, eos=EOS, im_start=IM)


def set_label(numpy_dir, position, value, part="0000"):
    path = os.path.join(numpy_dir, f"labels_mask_part_{part}.npy")
    mask = np.fromfile(path, dtype=np.bool_)
    mask[position] = value
    mask.tofile(path)


def test_passes_the_written_dir(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    result = run(replay, masked, chat)
    assert result["pass"], result["totals"]
    totals = result["totals"]
    assert totals["found"] == totals["dropped_at_final"] == 3
    assert totals["chat_docs"] == 2
    assert totals["inventory_over_max_seq_length"] == 1


def test_an_already_untrained_final_label_is_valid(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path, untrained_final=True)
    result = run(replay, masked, chat)
    assert result["pass"], result["totals"]
    assert (result["totals"]["dropped_at_final"], result["totals"]["already_untrained_at_final"]) == (2, 1)


def test_fails_a_label_cleared_elsewhere(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    set_label(masked, 4, False)  # the interior EOS of the second document
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["cleared_elsewhere"] == 1


def test_fails_a_replay_final_eos_left_trained(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    set_label(masked, 2, True)
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["final_still_trained"] == 1


def test_fails_an_unmasked_copy(tmp_path):
    replay, chat, _ = setup_dirs(tmp_path)
    result = run(replay, replay, chat)
    assert not result["pass"] and result["totals"]["final_still_trained"] == 3


def test_fails_a_document_that_does_not_match_the_inventory(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    pins = pins_for(replay)
    tokens = np.fromfile(os.path.join(replay, "token_ids_part_0001.npy"), dtype=np.uint32)
    tokens[1] = 99
    tokens.tofile(os.path.join(replay, "token_ids_part_0001.npy"))  # the masked dir's symlink sees it too
    result = run(replay, masked, chat, pins)
    assert not result["pass"] and result["totals"]["not_found"] == 1


def test_fails_a_manifest_that_does_not_match_its_pins(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    pins = {**pins_for(replay), "manifest_sha256": "0" * 64}
    assert not run(replay, masked, chat, pins)["pass"]


def test_fails_orphan_and_missing_parts(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    np.ones(3, dtype=np.bool_).tofile(os.path.join(masked, "labels_mask_part_0002.npy"))
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["part_set_mismatch"] == 1
    os.remove(os.path.join(masked, "labels_mask_part_0002.npy"))
    os.remove(os.path.join(masked, "token_ids_part_0001.csv.gz"))
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["part_set_mismatch"] == 1
    open(os.path.join(replay, "token_ids_part_backup.npy"), "w").close()
    assert run(replay, masked, chat)["totals"]["orphan_part_files"] == 1


def test_fails_a_chat_cache_row_without_im_start(tmp_path):
    replay, _, masked = setup_dirs(tmp_path)
    chat = str(tmp_path / "chat-bad")
    write_dir(chat, [[[IM, 10, EOS], [11, EOS]]], mask=False, manifest=False)
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["chat_not_im_start"] == 1


def test_fails_a_blank_metadata_line(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    with gzip.open(os.path.join(replay, "token_ids_part_0001.csv.gz"), "at") as f:  # OUT's symlink sees it too
        f.write("\n")
    result = run(replay, masked, chat)
    assert not result["pass"]
    assert result["totals"]["metadata_malformed_lines"] == result["totals"]["statistics_mismatch"] == 1


def test_fails_missing_output_statistics(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    os.remove(os.path.join(masked, "dataset_statistics.json"))
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["statistics_missing"] == 1


def test_fails_a_chat_part_whose_rows_do_not_tile_it(tmp_path):
    replay, _, masked = setup_dirs(tmp_path)
    chat = str(tmp_path / "chat-short")
    write_dir(chat, [[[IM, 10, EOS, IM, 11, EOS]]], mask=False, manifest=False)
    with gzip.open(os.path.join(chat, "token_ids_part_0000.csv.gz"), "wt") as f:
        f.write("0,3\n")
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["untiled_parts"] == 1


def test_fails_an_incomplete_or_empty_chat_dir(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    os.remove(os.path.join(chat, "labels_mask_part_0000.npy"))
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["part_set_mismatch"] == 1
    empty = str(tmp_path / "chat-empty")
    os.makedirs(empty)
    result = run(replay, masked, empty)
    assert not result["pass"] and result["totals"]["part_set_mismatch"] == 1
