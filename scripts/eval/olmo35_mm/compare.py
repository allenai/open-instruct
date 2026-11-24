"""Compare multimodal SFT eval results across runs (e.g. TPU vs GPU, or a run against two GPU seeds).

    python compare.py --run GPU=jobs/olmo35-mmsft-4n-emo256k-s10962.log --run TPU=jobs/olmo35-...-tpu-s10962.log
    python compare.py --run G1=A.log --run G2=B.log --run TPU=C.log --seed-pair G1 G2 [--out TABLE.md]

Each ``--run NAME=LOG[@STEP]`` names a job log written by run_eval.sh (lines ``... vision core: <id>``,
``... text cell <name>: <id>``; later lines win, so relaunches replace earlier ids). ``@STEP`` reads only the
``stepSTEP`` lines of the 2026-11 session's per-arm sweep logs (``/root/mmsft-eval/sweep-<arm>.log``), whose lines
are ``stepSTEP vision core: <id>``.

Scores come from each job's results table in its Beaker log (``beaker experiment logs``); IFEval is prompt-level
strict, read from the job's results (``beaker experiment results``, cached under --cache). For RULER only
``ruler_all`` is shown. SE is binomial: sqrt(p(1-p)/n) per run (an upper bound for ANLS / VQA-score metrics).

Columns: the first run is the reference. Every other run gets ``run − ref`` and a ``>2SE`` flag when
|diff| > 2 SE(diff). With ``--seed-pair A B`` (two seeds of the same recipe; A must be the reference), the table
also has ``seed Δ`` = B − A and, for each remaining run, ``run − Ḡ`` (Ḡ = mean of A and B) flagged when
|run − Ḡ| > |seed Δ| + 2 SE. Metrics without an item count get no SE and no flag.

From H271's compare_tpu_gpu.py / compare_seeds.py (olmo-post-training-ledger experiments/mmsft/tpu/), generalized
from the emo256k arm's fixed logs to any runs. Reads Beaker only; launches nothing.
"""

import argparse
import glob
import math
import os
import re
import subprocess
from pathlib import Path

TASKS = [  # (task key in the results table, label), in table order
    ("doc_qa", "DocVQA (ANLS, 24 crops)"),
    ("text_vqa", "TextVQA (24 crops)"),
    ("chart_qa", "ChartQA (24 crops)"),
    ("info_qa", "InfoVQA (ANLS, 24 crops)"),
    ("vqa2", "VQAv2"),
    ("ai2d", "AI2D"),
    ("countbench_qa", "CountBench"),
    ("real_world_qa", "RealWorldQA"),
    ("mmmu", "MMMU"),
    ("muir_bench", "MuirBench"),
    ("mmiu", "MMIU"),
    ("blink", "BLINK"),
    ("mmmu_pro", "MMMU-Pro"),
    ("ruler_all__65536", "RULER 64K (text)"),
    ("ruler_all__131072", "RULER 128K (text)"),
    ("ifeval", "IFEval strict (text)"),
    ("gpqa", "GPQA (text)"),
    ("ifbench", "IFBench (text)"),
]
LABELS = dict(TASKS)
ORDER = [task for task, _ in TASKS]
RULER_ITEMS = 1300  # ruler_all: 13 sub-tasks x 100 items; the table prints no count for it
LINE = r"(vision \w+|text cell [^:]+?)(?: \([^)]*\))?: (?:beaker\.org/ex/)?(01[A-Z0-9]{24})"


def job_ids(log: str, step: int | None) -> dict[str, str]:
    pattern = re.compile((rf"step{step} " if step is not None else "") + LINE)
    ids: dict[str, str] = {}
    if not os.path.exists(log):
        return ids
    with open(log) as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ids[match.group(1)] = match.group(2)
    return ids


def ifeval_strict(job_id: str, cache: Path) -> float | None:
    out = cache / f"ifeval-{job_id}"
    if not glob.glob(str(out / "**" / "*metrics*.json"), recursive=True):
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["beaker", "experiment", "results", job_id, "-o", str(out)], capture_output=True, timeout=600)
    for path in glob.glob(str(out / "**" / "*.json"), recursive=True):
        try:
            text = Path(path).read_text()
        except OSError:
            continue
        match = re.search(r'"prompt_level_strict_acc"\s*:\s*\{\s*"ifeval"\s*:\s*([0-9.]+)', text)
        if match:
            return float(match.group(1))
    return None


class Scores:
    def __init__(self, cache: Path):
        self.cache = cache
        self._jobs: dict[str, dict] = {}

    def job(self, job_id: str) -> dict[str, tuple[float | None, int | None]]:
        """{task: (score, n)} from the job's results table."""
        if job_id in self._jobs:
            return self._jobs[job_id]
        out = subprocess.run(["beaker", "experiment", "logs", job_id], capture_output=True, text=True).stdout
        out = re.sub(r"\x1b\[[0-9;]*m", "", out)
        res: dict[str, tuple[float | None, int | None]] = {}
        for line in out.splitlines():
            cells = [c.strip() for c in line.split("│")]
            if len(cells) >= 6 and cells[2] == "Success":
                match = re.match(r"(\d+)/(\d+)", cells[3])
                n = int(match.group(1)) if match else (RULER_ITEMS if cells[1].startswith("ruler_all__") else None)
                res[cells[1]] = (float(cells[-2]), n)
        if "ifeval" in res:
            res["ifeval"] = (ifeval_strict(job_id, self.cache), res["ifeval"][1])
        self._jobs[job_id] = res
        return res

    def run(self, log: str, step: int | None) -> dict[str, tuple[float | None, int | None]]:
        merged: dict[str, tuple[float | None, int | None]] = {}
        for job_id in job_ids(log, step).values():
            merged.update(self.job(job_id))
        return merged


def se(p: float | None, n: int | None) -> float | None:
    return math.sqrt(max(p * (1 - p), 0.0) / n) if (p is not None and n) else None


def fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:+.1f}" if signed else f"{value:.1f}"


def table(runs: dict[str, dict], seed_pair: tuple[str, str] | None) -> list[str]:
    names = list(runs)
    ref = names[0]
    others = [n for n in names[1:] if not seed_pair or n != seed_pair[1]]
    keys = set().union(*(set(r) for r in runs.values()))
    tasks = [k for k in ORDER if k in keys] + sorted(
        k for k in keys if k not in ORDER and (k.startswith("ruler_all") or "__" not in k)
    )
    head = ["task", *names]
    if seed_pair:
        head.append(f"seed Δ ({seed_pair[1]}−{seed_pair[0]})")
    for name in others:
        head += [f"{name}−{ref}", "SE", ""]
        if seed_pair:
            head += [f"{name}−Ḡ", "SE", ""]
    head.append("n")
    rows = ["| " + " | ".join(head) + " |", "|---" + "|---:" * (len(head) - 1) + "|"]
    flags: dict[str, list[str]] = {name: [] for name in others}
    diffs: dict[str, list[float]] = {name: [] for name in others}
    for task in tasks:
        score = {name: runs[name].get(task, (None, None)) for name in names}
        n = next((score[name][1] for name in names if score[name][1]), None)
        row = [LABELS.get(task, task)] + [fmt(None if p is None else 100 * p) for p, _ in score.values()]
        seed_delta = None
        if seed_pair:
            (a, _), (b, _) = score[seed_pair[0]], score[seed_pair[1]]
            seed_delta = 100 * (b - a) if a is not None and b is not None else None
            row.append(fmt(seed_delta, signed=True))
        for name in others:
            (p, pn), (r, rn) = score[name], score[ref]
            d = 100 * (p - r) if p is not None and r is not None else None
            s_p, s_r = se(p, pn), se(r, rn)
            s = 100 * math.sqrt(s_p**2 + s_r**2) if s_p is not None and s_r is not None else None
            flag = "**>2SE**" if d is not None and s and abs(d) > 2 * s else ""
            if d is not None:
                diffs[name].append(d)
            row += [fmt(d, signed=True), fmt(s), flag]
            if seed_pair:
                (b, bn) = score[seed_pair[1]]
                dm = 100 * (p - (r + b) / 2) if None not in (p, r, b) else None
                s_b = se(b, bn)
                sm = (
                    100 * math.sqrt(s_p**2 + (s_r**2 + s_b**2) / 4)
                    if None not in (s_p, s_r, s_b) and dm is not None
                    else None
                )
                mflag = ""
                if dm is not None and sm is not None and seed_delta is not None and abs(dm) > abs(seed_delta) + 2 * sm:
                    mflag = "**FLAG**"
                    flags[name].append(LABELS.get(task, task))
                row += [fmt(dm, signed=True), fmt(sm), mflag]
        row.append(str(n) if n else "—")
        rows.append("| " + " | ".join(row) + " |")
    rows.append("")
    for name in others:
        if diffs[name]:
            m = len(diffs[name])
            rows.append(
                f"{name} − {ref}: mean {sum(diffs[name]) / m:+.2f} points, mean |diff| "
                f"{sum(abs(x) for x in diffs[name]) / m:.2f} over {m} tasks"
                + (f"; seed-control flags: {', '.join(flags[name]) or 'none'}" if seed_pair else "")
                + "."
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=LOG[@STEP]")
    parser.add_argument("--seed-pair", nargs=2, metavar=("REF", "SEED2"))
    parser.add_argument("--title", default="Olmo 3.5 multimodal SFT: same eval pipeline, scores in points")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "olmo35-mm-eval" / "results")
    args = parser.parse_args()

    runs_spec = {}
    for item in args.run:
        name, _, rest = item.partition("=")
        log, _, step = rest.partition("@")
        if not name or not log:
            parser.error(f"--run {item!r}: expected NAME=LOG[@STEP]")
        runs_spec[name] = (log, int(step) if step else None)
    names = list(runs_spec)
    if args.seed_pair and (args.seed_pair[0] != names[0] or args.seed_pair[1] not in names):
        parser.error("--seed-pair REF SEED2: REF must be the first --run and SEED2 another --run")
    scores = Scores(args.cache)
    runs = {name: scores.run(log, step) for name, (log, step) in runs_spec.items()}
    lines = [
        f"# {args.title}",
        "",
        *(f"- {n}: {log}" + (f" @ step {s}" if s else "") for n, (log, s) in runs_spec.items()),
    ]
    lines += ["", *table(runs, tuple(args.seed_pair) if args.seed_pair else None)]
    text = "\n".join(lines)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
