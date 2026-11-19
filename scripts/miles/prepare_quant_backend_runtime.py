"""Apply the isolated Olmo adapter correction used by the backend investigation."""

import argparse
from pathlib import Path


def patch_top_k(source):
    before = "self.experts = expert_class(\n            num_experts=self.num_experts,\n"
    after = before + "            top_k=config.num_experts_per_tok,\n"
    if source.count(before) != 1 or after in source:
        raise ValueError("Unexpected Olmo expert constructor; inspect before patching")
    return source.replace(before, after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    source = patch_top_k(args.model.read_text())
    compile(source, str(args.model), "exec")
    args.model.write_text(source)


if __name__ == "__main__":
    main()
