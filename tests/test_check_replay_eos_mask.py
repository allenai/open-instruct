"""scripts/data/check_replay_eos_mask.py passes the masked dir that mask_replay_final_eos.py writes, and fails each
kind of wrong one."""

import gzip
import importlib.util
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


def write_dir(numpy_dir, rows, mask=True):
    os.makedirs(numpy_dir)
    tokens = np.array([t for row in rows for t in row], dtype=np.uint32)
    tokens.tofile(os.path.join(numpy_dir, "token_ids_part_0000.npy"))
    np.full(len(tokens), mask, dtype=np.bool_).tofile(os.path.join(numpy_dir, "labels_mask_part_0000.npy"))
    with gzip.open(os.path.join(numpy_dir, "token_ids_part_0000.csv.gz"), "wt") as f:
        start = 0
        for row in rows:
            f.write(f"{start},{start + len(row)}\n")
            start += len(row)


def setup_dirs(tmp_path):
    replay, chat, masked = (str(tmp_path / name) for name in ("replay", "chat", "masked"))
    write_dir(replay, [[20, 21, EOS], [22, EOS, 23, EOS], [24, 25, 26, 27, EOS]])
    write_dir(chat, [[IM, 10, EOS], [IM, 11, EOS, IM, 12, EOS]], mask=False)
    numpy_dataset_conversion.write_replay_eos_masked_dir(replay, masked, EOS, IM)
    return replay, chat, masked


def run(replay, masked, chat):
    return check_script.check(replay, masked, [chat], max_seq_length=4, eos=EOS, im_start=IM)


def set_label(masked, position, value):
    path = os.path.join(masked, "labels_mask_part_0000.npy")
    mask = np.fromfile(path, dtype=np.bool_)
    mask[position] = value
    mask.tofile(path)


def test_passes_the_written_dir(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    result = run(replay, masked, chat)
    assert result["pass"], result["totals"]
    totals = result["totals"]
    assert totals["replay_docs"] == totals["dropped"] == totals["dropped_at_replay_final"] == 3
    assert totals["chat_docs"] == 2
    assert totals["replay_over_max_seq_length"] == 1


def test_fails_a_label_cleared_elsewhere(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    set_label(masked, 4, False)  # the interior EOS of the second document
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["dropped_elsewhere"] == 1


def test_fails_a_replay_final_eos_left_trained(tmp_path):
    replay, chat, masked = setup_dirs(tmp_path)
    set_label(masked, 2, True)
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["replay_final_kept"] == 1


def test_fails_a_chat_cache_row_without_im_start(tmp_path):
    replay, _, masked = setup_dirs(tmp_path)
    chat = str(tmp_path / "chat-bad")
    write_dir(chat, [[IM, 10, EOS], [11, EOS]], mask=False)
    result = run(replay, masked, chat)
    assert not result["pass"] and result["totals"]["chat_not_im_start"] == 1


def test_fails_an_unmasked_copy(tmp_path):
    replay, chat, _ = setup_dirs(tmp_path)
    result = run(replay, replay, chat)
    assert not result["pass"] and result["totals"]["dropped"] == 0
