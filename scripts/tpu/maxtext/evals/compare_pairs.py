"""Paired comparison of released vs TPU eval cells: per-item scores joined on native_id, paired Wald 95% CI.

  python compare_pairs.py <results dir> <prefix, e.g. o3i7b-dpo> [tpu prefix, if different]

Each cell has two result datasets, <prefix>-released-<cell>-t10 and <prefix>-tpu-<cell>-t10 (beaker dataset fetch).
The item score is the task's primary metric from instance_metrics (pass_at_1 where the task has no primary, as LCB's
summary is empty when sandbox scoring partly fails); items whose score is missing on either side are dropped and
counted. Multi-subtask cells (BFCL, RULER) pool all items. Also reports mean output tokens per sample.
"""

import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path.home() / "repos/olmo-post-training-ledger"))
from ledger.stats import paired_wald  # noqa: E402

CELLS = ["ifeval", "ifbench", "ifeval_mt_ood", "gpqa", "aime25", "aime26", "hmmt26", "lcb", "bfcl_single", "ruler64k"]


def load(cell_dir):
    m = json.loads(pathlib.Path(f"{cell_dir}/metrics.json").read_text())
    primary = {t["task"]: (t.get("primary_metric") or "pass_at_1").split(":")[0] for t in m["tasks"]}
    items, toks = {}, []
    for f in glob.glob(f"{cell_dir}/predictions/*/*-predictions.jsonl"):
        for line in pathlib.Path(f).read_text().splitlines():
            r = json.loads(line)
            im = r["instance_metrics"]
            task = next((t for t in primary if t.replace(":", "_") in f), None)
            key = primary.get(task, "pass_at_1") if task else "pass_at_1"
            if key not in im:
                key = next((k for k in ("pass_at_1", "exact_match", "accuracy", "recall") if k in im), None)
            v = None if key is None else next(iter(im[key].values()), None)
            items[(pathlib.Path(f).name.split("-predictions")[0].rsplit("_", 1)[0], str(r["native_id"]))] = v
            outs = r["model_output"] if isinstance(r["model_output"], list) else [r["model_output"]]
            toks += [float(o.get("num_tokens", 0)) for o in outs]
    return items, sum(toks) / max(len(toks), 1)


root, prefix = sys.argv[1], sys.argv[2]
tpu_prefix = sys.argv[3] if len(sys.argv) > 3 else prefix
print(f"{'cell':14s} {'n':>5s} {'released':>9s} {'tpu':>7s} {'tpu-rel':>8s} {'95% CI':>18s}  tokens rel/tpu  dropped")
for cell in CELLS:
    a_dir, b_dir = f"{root}/{prefix}-released-{cell}-t10", f"{root}/{tpu_prefix}-tpu-{cell}-t10"
    if not (glob.glob(f"{a_dir}/metrics.json") and glob.glob(f"{b_dir}/metrics.json")):
        print(f"{cell:14s} missing")
        continue
    (a, ta), (b, tb) = load(a_dir), load(b_dir)
    keys = [k for k in a if k in b and a[k] is not None and b[k] is not None]
    dropped = len(set(a) | set(b)) - len(keys)
    e = paired_wald([a[k] for k in keys], [b[k] for k in keys], cell)
    ma, mb = sum(a[k] for k in keys) / len(keys), sum(b[k] for k in keys) / len(keys)
    sig = "*" if e.ci_low > 0 or e.ci_high < 0 else " "
    print(
        f"{cell:14s} {len(keys):5d} {ma:9.4f} {mb:7.4f} {e.delta:+8.4f} [{e.ci_low:+.4f}, {e.ci_high:+.4f}]{sig} {ta:7.0f}/{tb:<7.0f} {dropped}"
    )
