import sys

from scripts.data.rlvr import gsm8k_pass_at_32_dataset


def test_parse_args_accepts_num_engines_aliases(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["gsm8k_pass_at_32_dataset.py", "--num_engines", "4"])
    args = gsm8k_pass_at_32_dataset.parse_args()
    assert args.num_engines == 4

    monkeypatch.setattr(sys, "argv", ["gsm8k_pass_at_32_dataset.py", "--num-engines", "3"])
    args = gsm8k_pass_at_32_dataset.parse_args()
    assert args.num_engines == 3


def test_split_evenly_balances_prompt_shards():
    chunks = gsm8k_pass_at_32_dataset._split_evenly(["p0", "p1", "p2", "p3", "p4"], 3)
    assert chunks == [(0, ["p0", "p1"]), (2, ["p2", "p3"]), (4, ["p4"])]
